#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/ArmSME/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/OneToNTypeConversion.h"

#define DEBUG_TYPE "arm-sme-vector-legalization"

namespace mlir::arm_sme {
#define GEN_PASS_DEF_VECTORLEGALIZATION
#include "mlir/Dialect/ArmSME/Transforms/Passes.h.inc"
} // namespace mlir::arm_sme

using namespace mlir;
using namespace mlir::arm_sme;

namespace {

struct SubTile {
  // Note: The units of (row, col) value (as SME tiles are scalable).
  int row{0};
  int col{0};
  VectorType type;
};

SmallVector<Value, 2> add2DScalableOffsetToIndices(OpBuilder &builder,
                                                   Location loc,
                                                   ValueRange indices,
                                                   int scalableOffset[2]) {
  auto vscale = builder.create<vector::VectorScaleOp>(loc);
  auto rowIndex = builder.create<arith::MulIOp>(
      loc, builder.create<arith::ConstantIndexOp>(loc, scalableOffset[0]),
      vscale);
  auto colIndex = builder.create<arith::MulIOp>(
      loc, builder.create<arith::ConstantIndexOp>(loc, scalableOffset[1]),
      vscale);
  auto subTileRowIndex =
      builder.create<arith::AddIOp>(loc, rowIndex, indices[0]);
  auto subTileColIndex =
      builder.create<arith::AddIOp>(loc, colIndex, indices[1]);
  return {subTileRowIndex, subTileColIndex};
}

SmallVector<Value, 2> remapIndicesForSubTile(OpBuilder &builder, Location loc,
                                             ValueRange indices,
                                             SubTile subTile) {
  int offset[2] = {subTile.row, subTile.col};
  return add2DScalableOffsetToIndices(builder, loc, indices, offset);
}

bool isSupportedMask(Value mask) {
  return !mask || mask.getDefiningOp<vector::CreateMaskOp>();
}

Value extractSubMask(OpBuilder &builder, Location loc, Value mask,
                     SubTile subTile) {
  assert(isSupportedMask(mask));
  if (!mask)
    return Value{};
  auto createMask = mask.getDefiningOp<vector::CreateMaskOp>();
  // The the operands of `vector.create_mask` (from a 2D perspective) are the
  // coordinates where the mask ends. So we subtract where this tile starts,
  // from the mask operands to get the parameters for this sub tile.
  int offset[2] = {-subTile.row, -subTile.col};
  auto subMaskDims = add2DScalableOffsetToIndices(
      builder, loc, createMask.getOperands(), offset);
  auto createSubTileMask = builder.create<vector::CreateMaskOp>(
      loc, subTile.type.clone(builder.getI1Type()), subMaskDims);
  return createSubTileMask.getResult();
}

auto decompose2DVectorType(OpBuilder &builder, VectorType type,
                           VectorType subTileType,
                           bool transposeSubTileOrder = false) {
  assert(isMultipleOfSMETileVectorType(type));
  return llvm::map_range(
      StaticTileOffsetRange(
          type.getShape(),
          {subTileType.getDimSize(0), subTileType.getDimSize(1)},
          transposeSubTileOrder ? ArrayRef<int64_t>{1, 0}
                                : ArrayRef<int64_t>{0, 1}),
      [=](auto indices) {
        return SubTile{int(indices[0]), int(indices[1]), subTileType};
      });
}

int getNumberOfSMESubTilesForVectorType(VectorType type) {
  int64_t vectorRows = type.getDimSize(0);
  int64_t vectorCols = type.getDimSize(1);
  auto elementType = type.getElementType();
  unsigned minNumElts = getSMETileSliceMinNumElts(elementType);
  return (vectorRows * vectorCols) / (minNumElts * minNumElts);
}

struct LegalizeVectorOuterProductOp
    : public OneToNOpConversionPattern<vector::OuterProductOp> {
  using OneToNOpConversionPattern<
      vector::OuterProductOp>::OneToNOpConversionPattern;
  LogicalResult
  matchAndRewrite(vector::OuterProductOp outerProductOp, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto vectorType = outerProductOp.getResultVectorType();
    if (!isMultipleOfSMETileVectorType(vectorType))
      return failure();

    Value mask;
    Operation *rootOp = outerProductOp;
    auto loc = outerProductOp.getLoc();
    if (outerProductOp.isMasked()) {
      auto maskOp = outerProductOp.getMaskingOp();
      mask = maskOp.getMask();
      rewriter.setInsertionPoint(maskOp);
      rootOp = maskOp;
    }

    if (!isSupportedMask(mask))
      return failure();

    // FIXME: This is a workaround for `vector.mask`; without this the
    // unrealized_conversion_casts to the SME tile types are placed within
    // the `vector.mask` region, which results in incorrect IR. This moves
    // the unrealized_conversion_cast to just before the `vector.mask` op
    // (if present).
    ValueRange accSMETiles = adaptor.getAcc();
    if (!accSMETiles.empty())
      accSMETiles[0].getDefiningOp<UnrealizedConversionCastOp>()->moveBefore(
          rootOp);

    auto tileType = getSMETileTypeForElement(vectorType.getElementType());
    VectorType sliceType = VectorType::Builder(tileType).dropDim(0);

    SmallVector<Value> resultSMETiles;
    for (auto [index, subTile] : llvm::enumerate(
             decompose2DVectorType(rewriter, vectorType, tileType))) {

      auto subMask = extractSubMask(rewriter, loc, mask, subTile);
      auto lhs = rewriter.create<vector::ScalableExtractOp>(
          loc, sliceType, outerProductOp.getLhs(), subTile.row);
      auto rhs = rewriter.create<vector::ScalableExtractOp>(
          loc, sliceType, outerProductOp.getRhs(), subTile.col);

      auto subOuterProduct = rewriter.create<vector::OuterProductOp>(
          loc, tileType, lhs, rhs,
          !accSMETiles.empty() ? accSMETiles[index] : Value{},
          outerProductOp.getKind());

      auto maskedOuterProduct =
          vector::maskOperation(rewriter, subOuterProduct, subMask);
      resultSMETiles.push_back(maskedOuterProduct->getResult(0));
    }

    rewriter.replaceOp(rootOp, resultSMETiles, adaptor.getResultMapping());
    return success();
  }
};

struct LegalizeTransferReadOp
    : public OneToNOpConversionPattern<vector::TransferReadOp> {
  using OneToNOpConversionPattern<
      vector::TransferReadOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::TransferReadOp readOp, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto vectorType = readOp.getVectorType();
    if (!isMultipleOfSMETileVectorType(vectorType))
      return failure();

    auto mask = readOp.getMask();
    if (!isSupportedMask(mask))
      return failure();

    auto permutationMap = readOp.getPermutationMap();
    if (!permutationMap.isPermutation())
      return failure();

    // Note: For 2D vector types the only non-identity permutation is a simple
    // tranpose [1, 0].
    bool transposeTiles = !permutationMap.isIdentity();

    auto loc = readOp.getLoc();
    auto tileType = getSMETileTypeForElement(vectorType.getElementType());

    SmallVector<Value> resultSMETiles;
    for (SubTile subTile : decompose2DVectorType(rewriter, vectorType, tileType,
                                                 transposeTiles)) {
      auto subMask = extractSubMask(rewriter, loc, mask, subTile);
      auto transferRead = rewriter.create<vector::TransferReadOp>(
          loc, tileType, readOp.getSource(),
          remapIndicesForSubTile(rewriter, loc, readOp.getIndices(), subTile),
          readOp.getPermutationMapAttr(), readOp.getPadding(), subMask,
          readOp.getInBoundsAttr());
      resultSMETiles.push_back(transferRead);
    }

    rewriter.replaceOp(readOp, resultSMETiles, adaptor.getResultMapping());
    return success();
  }
};

struct LegalizeTransferWriteOp
    : public OneToNOpConversionPattern<vector::TransferWriteOp> {
  using OneToNOpConversionPattern<
      vector::TransferWriteOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::TransferWriteOp writeOp, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    auto vectorType = writeOp.getVectorType();
    if (!isMultipleOfSMETileVectorType(vectorType))
      return failure();

    auto mask = writeOp.getMask();
    if (!isSupportedMask(mask))
      return failure();

    auto permutationMap = writeOp.getPermutationMap();
    if (!permutationMap.isPermutation())
      return failure();

    // Note: For 2D vector types the only non-identity permutation is a simple
    // tranpose [1, 0].
    bool transposeTiles = !permutationMap.isIdentity();

    auto loc = writeOp.getLoc();
    auto tileType = getSMETileTypeForElement(vectorType.getElementType());
    auto inputSMETiles = adaptor.getVector();

    Value destTensorOrMemref = writeOp.getSource();
    for (auto [index, subTile] : llvm::enumerate(decompose2DVectorType(
             rewriter, vectorType, tileType, transposeTiles))) {
      auto subMask = extractSubMask(rewriter, loc, mask, subTile);
      auto subWrite = rewriter.create<vector::TransferWriteOp>(
          loc, inputSMETiles[index], destTensorOrMemref,
          remapIndicesForSubTile(rewriter, loc, writeOp.getIndices(), subTile),
          writeOp.getPermutationMapAttr(), subMask, writeOp.getInBoundsAttr());
      if (writeOp.hasPureTensorSemantics())
        destTensorOrMemref = subWrite.getResult();
    }

    if (writeOp.hasPureTensorSemantics())
      rewriter.replaceOp(writeOp, destTensorOrMemref);
    else
      rewriter.eraseOp(writeOp);

    return success();
  }
};

struct VectorLegalizationPass
    : public arm_sme::impl::VectorLegalizationBase<VectorLegalizationPass> {
  void runOnOperation() override {
    auto *context = &getContext();
    {
      OneToNTypeConverter typeConverter;
      RewritePatternSet patterns(context);

      typeConverter.addConversion([](Type type) { return type; });
      typeConverter.addConversion(
          [](VectorType vectorType,
             SmallVectorImpl<Type> &types) -> std::optional<LogicalResult> {
            if (!isMultipleOfSMETileVectorType(vectorType))
              return std::nullopt;
            auto subTileCount = getNumberOfSMESubTilesForVectorType(vectorType);
            auto tileType =
                getSMETileTypeForElement(vectorType.getElementType());
            types = SmallVector<Type>(subTileCount, tileType);
            return success();
          });

      patterns.add<LegalizeVectorOuterProductOp, LegalizeTransferReadOp,
                   LegalizeTransferWriteOp>(typeConverter, context);
      scf::populateSCFStructuralOneToNTypeConversions(typeConverter, patterns);

      if (failed(applyPartialOneToNConversion(getOperation(), typeConverter,
                                              std::move(patterns))))
        return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::arm_sme::createVectorLegalizationPass() {
  return std::make_unique<VectorLegalizationPass>();
}
