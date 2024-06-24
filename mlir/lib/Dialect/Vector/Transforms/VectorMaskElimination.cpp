#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/ScalableValueBoundsConstraintSet.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;

namespace {

std::optional<int64_t> getConstantScalableMultiplier(Value value) {
  auto mul = value.getDefiningOp<arith::MulIOp>();
  if (!mul)
    return {};
  auto lhs = mul.getLhs();
  auto rhs = mul.getRhs();
  if (lhs.getDefiningOp<vector::VectorScaleOp>())
    return getConstantIntValue(rhs);
  if (rhs.getDefiningOp<vector::VectorScaleOp>())
    return getConstantIntValue(lhs);
  return {};
}

LogicalResult elimateOneMask(IRRewriter &rewriter, vector::CreateMaskOp mask) {
  auto maskType = mask.getVectorType();
  SmallVector<std::pair<int64_t, Value>> unknownValues;
  for (auto [i, dimSize] : llvm::enumerate(mask.getOperands())) {
    if (auto intValue = getConstantIntValue(dimSize)) {
      if (maskType.getScalableDims()[i])
        return failure();
      if (*intValue < maskType.getDimSize(i))
        return failure();
    } else if (auto scalableValue = getConstantScalableMultiplier(dimSize)) {
      if (*intValue < maskType.getDimSize(i))
        return failure();
    } else {
      unknownValues.push_back(std::make_pair(i, dimSize));
    }
  }

  for (auto [i, dimSize] : unknownValues) {
    auto bound = vector::ScalableValueBoundsConstraintSet::computeScalableBound(
        dimSize, {}, 1, 16, presburger::BoundType::LB);
    if (failed(bound))
      return failure();
    auto size = bound->getSize();
    if (failed(size))
      return failure();
    if (size->scalable) {
      if (size->baseSize < maskType.getDimSize(i))
        return failure();
    } else {
      if (maskType.getScalableDims()[i])
        return failure();
      if (size->baseSize < maskType.getDimSize(i))
        return failure();
    }
  }

  auto allTrue = rewriter.create<arith::ConstantOp>(
      mask.getLoc(), maskType, DenseElementsAttr::get(maskType, true));
  rewriter.replaceAllUsesWith(mask, allTrue);
  return success();
}

} // namespace

namespace mlir::vector {

void elimateVectorMasks(IRRewriter &rewriter, FunctionOpInterface func) {
  OpBuilder::InsertionGuard g(rewriter);
  SmallVector<vector::CreateMaskOp> createMasks;
  func.walk([&](vector::CreateMaskOp createMask) {
    createMasks.push_back(createMask);
  });
  rewriter.setInsertionPointToStart(&func.front());
  for (auto mask : createMasks)
    (void)elimateOneMask(rewriter, mask);
}

} // namespace mlir::vector
