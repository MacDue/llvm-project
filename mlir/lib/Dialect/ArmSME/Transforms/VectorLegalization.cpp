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

struct SubTileBuilder {
  SubTileBuilder(OpBuilder &builder, VectorType tileType, int row, int col)
      : builder(builder), tileType(tileType), row(row), col(col) {}

  int getRow() const { return row; }
  int getCol() const { return col; }

  SmallVector<Value, 2> remapIndicesForSubTile(Location loc,
                                               ValueRange parentTileIndices) {
    auto vscale = builder.create<vector::VectorScaleOp>(loc);
    auto rowIndex = builder.create<arith::MulIOp>(
        loc, builder.create<arith::ConstantIndexOp>(loc, getRow()), vscale);
    auto colIndex = builder.create<arith::MulIOp>(
        loc, builder.create<arith::ConstantIndexOp>(loc, getCol()), vscale);
    auto subTileRowIndex =
        builder.create<arith::AddIOp>(loc, rowIndex, parentTileIndices[0]);
    auto subTileColIndex =
        builder.create<arith::AddIOp>(loc, colIndex, parentTileIndices[1]);
    return {subTileRowIndex, subTileColIndex};
  }

  FailureOr<Value> extractMaskForSubTile(Location loc, Value mask) {
    if (!mask)
      return Value{};
    auto createMask = mask.getDefiningOp<vector::CreateMaskOp>();
    if (!createMask)
      return failure();
    // The the operands of `vector.create_mask` (from a 2D perspective) are the
    // coordinates where the mask ends. So we subtract where this tile starts,
    // from the mask operands to get the parameters for this sub tile.
    auto subMaskDims =
        SubTileBuilder(builder, tileType, -getRow(), -getCol())
            .remapIndicesForSubTile(loc, createMask.getOperands());
    auto createSubTileMask = builder.create<vector::CreateMaskOp>(
        loc, tileType.clone(builder.getI1Type()), subMaskDims);
    return createSubTileMask.getResult();
  }

private:
  OpBuilder &builder;
  VectorType tileType;
  int row{0};
  int col{0};
};

bool isVectorTypeMultipleOfSMETileSize(VectorType type) {
  if (type.getRank() != 2 || !type.allDimsScalable())
    return false;

  auto elementType = type.getElementType();
  if (!isValidSMETileElementType(elementType))
    return false;

  unsigned minNumElts = getSMETileSliceMinNumElts(elementType);

  int64_t vectorRows = type.getDimSize(0);
  int64_t vectorCols = type.getDimSize(1);

  return (vectorRows > minNumElts || vectorCols > minNumElts) &&
         vectorRows % minNumElts == 0 && vectorCols % minNumElts == 0;
}

VectorType getSMETileTypeForElement(Type elementType) {
  unsigned minNumElts = getSMETileSliceMinNumElts(elementType);
  return VectorType::get({minNumElts, minNumElts}, elementType, {true, true});
}

auto decompose2DVectorType(OpBuilder &builder, VectorType type,
                           VectorType subTileType,
                           bool transposeSubTileOrder = false) {
  assert(isVectorTypeMultipleOfSMETileSize(type));
  return llvm::map_range(
      StaticTileOffsetRange(
          type.getShape(),
          {subTileType.getDimSize(0), subTileType.getDimSize(1)},
          transposeSubTileOrder ? ArrayRef<int64_t>{1, 0}
                                : ArrayRef<int64_t>{0, 1}),
      [&, subTileType = subTileType](auto indices) {
        return SubTileBuilder(builder, subTileType, indices[0], indices[1]);
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
    if (!isVectorTypeMultipleOfSMETileSize(vectorType))
      return failure();

    Value mask = {};
    Operation *rootOp = outerProductOp;
    auto loc = outerProductOp.getLoc();
    if (outerProductOp.isMasked()) {
      auto maskOp = outerProductOp.getMaskingOp();
      mask = maskOp.getMask();
      rewriter.setInsertionPoint(maskOp);
      rootOp = maskOp;
    }

    auto tileType = getSMETileTypeForElement(vectorType.getElementType());
    VectorType sliceType = VectorType::Builder(tileType).dropDim(0);

    SmallVector<Value> resultSMETiles;
    for (auto [index, subTile] : llvm::enumerate(
             decompose2DVectorType(rewriter, vectorType, tileType))) {

      auto subMask = subTile.extractMaskForSubTile(loc, mask);
      if (failed(subMask))
        return failure();

      auto lhs = rewriter.create<vector::ScalableExtractOp>(
          loc, sliceType, outerProductOp.getLhs(), subTile.getRow());
      auto rhs = rewriter.create<vector::ScalableExtractOp>(
          loc, sliceType, outerProductOp.getRhs(), subTile.getCol());

      Operation *subOuterProduct = rewriter.create<vector::OuterProductOp>(
          loc, tileType, lhs, rhs,
          !adaptor.getAcc().empty() ? adaptor.getAcc()[index] : Value{},
          outerProductOp.getKind());

      if (*subMask)
        subOuterProduct =
            vector::maskOperation(rewriter, subOuterProduct, *subMask);

      resultSMETiles.push_back(subOuterProduct->getResult(0));
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
    if (!isVectorTypeMultipleOfSMETileSize(vectorType))
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
    for (SubTileBuilder subTile : decompose2DVectorType(
             rewriter, vectorType, tileType, transposeTiles)) {
      auto mask = subTile.extractMaskForSubTile(loc, readOp.getMask());
      if (failed(mask))
        return failure();

      auto transferRead = rewriter.create<vector::TransferReadOp>(
          loc, tileType, readOp.getSource(),
          subTile.remapIndicesForSubTile(loc, readOp.getIndices()),
          readOp.getPermutationMapAttr(), readOp.getPadding(), *mask,
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
    if (!isVectorTypeMultipleOfSMETileSize(vectorType))
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
      auto mask = subTile.extractMaskForSubTile(loc, writeOp.getMask());
      if (failed(mask))
        return failure();

      auto subWrite = rewriter.create<vector::TransferWriteOp>(
          loc, inputSMETiles[index], destTensorOrMemref,
          subTile.remapIndicesForSubTile(loc, writeOp.getIndices()),
          writeOp.getPermutationMapAttr(), *mask, writeOp.getInBoundsAttr());

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
            if (!isVectorTypeMultipleOfSMETileSize(vectorType))
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
