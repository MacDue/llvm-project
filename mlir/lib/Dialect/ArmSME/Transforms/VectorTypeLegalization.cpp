#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/ArmSME/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "arm-sme-vector-type-legalization"

namespace mlir::arm_sme {
#define GEN_PASS_DEF_VECTORTYPELEGALIZATION
#include "mlir/Dialect/ArmSME/Transforms/Passes.h.inc"
} // namespace mlir::arm_sme

using namespace mlir;
using namespace mlir::arm_sme;

namespace {

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

VectorType legalTileType(Type elementType) {
  unsigned minNumElts = getSMETileSliceMinNumElts(elementType);
  return VectorType::get({minNumElts, minNumElts}, elementType, {true, true});
}

template <typename Callback>
void forEachDecomposedVectorType(VectorType type, Callback callback,
                                 ArrayRef<int64_t> perm = {0, 1}) {
  auto elementType = type.getElementType();
  unsigned minNumElts = getSMETileSliceMinNumElts(elementType);
  for (SmallVector<int64_t> index :
       StaticTileOffsetRange(type.getShape(), {minNumElts, minNumElts}, perm)) {
    callback(index[0], index[1]);
  }
}

SmallVector<Value, 2> remapIndices(PatternRewriter &rewriter, Location loc,
                                   ValueRange indices, int i, int j) {
  auto vscale = rewriter.create<vector::VectorScaleOp>(loc);
  auto rowIndex = rewriter.create<arith::MulIOp>(
      loc, rewriter.create<arith::ConstantIndexOp>(loc, i), vscale);
  auto colIndex = rewriter.create<arith::MulIOp>(
      loc, rewriter.create<arith::ConstantIndexOp>(loc, j), vscale);
  auto subTileRowIndex =
      rewriter.create<arith::AddIOp>(loc, rowIndex, indices[0]);
  auto subTileColIndex =
      rewriter.create<arith::AddIOp>(loc, colIndex, indices[1]);
  return {subTileRowIndex, subTileColIndex};
}

int getNumDecomposedTiles(VectorType type) {
  int64_t vectorRows = type.getDimSize(0);
  int64_t vectorCols = type.getDimSize(1);
  auto elementType = type.getElementType();
  unsigned minNumElts = getSMETileSliceMinNumElts(elementType);
  return (vectorRows * vectorCols) / (minNumElts * minNumElts);
}

auto createUnrealizedConversionToSMETiles(PatternRewriter &rewriter,
                                          Location loc, Value input) {
  auto inputType = llvm::cast<VectorType>(input.getType());
  VectorType tileType = legalTileType(inputType.getElementType());
  SmallVector<Type> resultTiles(getNumDecomposedTiles(inputType), tileType);
  return rewriter.create<UnrealizedConversionCastOp>(loc, resultTiles, input);
}

auto createUnrealizedConversionFromSMETiles(PatternRewriter &rewriter,
                                            Location loc, ValueRange input,
                                            VectorType targetType) {
  return rewriter.create<UnrealizedConversionCastOp>(loc, targetType, input);
}

Value extractSubMask(PatternRewriter &rewriter, Location loc, Value mask, int i,
                     int j, VectorType tileType) {
  if (!mask)
    return {};
  auto createMask = mask.getDefiningOp<vector::CreateMaskOp>();
  if (!createMask)
    llvm_unreachable("TODO");
  auto newMaskDims =
      remapIndices(rewriter, loc, createMask.getOperands(), -i, -j);
  return rewriter.create<vector::CreateMaskOp>(
      loc, tileType.clone(rewriter.getI1Type()), newMaskDims);
}

struct LegalizeVectorLoad : public OpRewritePattern<vector::LoadOp> {
  using OpRewritePattern<vector::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::LoadOp load,
                                PatternRewriter &rewriter) const override {
    auto vectorType = load.getVectorType();

    if (!isVectorTypeMultipleOfSMETileSize(vectorType))
      return failure();

    auto tileType = legalTileType(vectorType.getElementType());
    auto loc = load.getLoc();

    SmallVector<Value> smeTiles;
    forEachDecomposedVectorType(vectorType, [&](int i, int j) {
      auto subTileLoad = rewriter.create<vector::LoadOp>(
          loc, tileType, load.getBase(),
          remapIndices(rewriter, loc, load.getIndices(), i, j));
      smeTiles.push_back(subTileLoad);
    });

    rewriter.replaceOp(load, createUnrealizedConversionFromSMETiles(
                                 rewriter, loc, smeTiles, vectorType));

    return success();
  }
};

// VectorTranpose -> tranpose each tile + transpose the order of subtiles
struct LegalizeVectorTranspose : public OpRewritePattern<vector::TransposeOp> {
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;
};

struct LegalizeVectorStore : public OpRewritePattern<vector::StoreOp> {
  using OpRewritePattern<vector::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::StoreOp store,
                                PatternRewriter &rewriter) const override {
    auto vectorType = store.getVectorType();
    if (!isVectorTypeMultipleOfSMETileSize(vectorType))
      return failure();

    auto loc = store.getLoc();
    auto smeTiles = createUnrealizedConversionToSMETiles(
        rewriter, loc, store.getValueToStore());

    int tileIndex = 0;
    forEachDecomposedVectorType(vectorType, [&](int i, int j) {
      rewriter.create<vector::StoreOp>(
          loc, smeTiles.getResult(tileIndex++), store.getBase(),
          remapIndices(rewriter, loc, store.getIndices(), i, j));
    });

    rewriter.eraseOp(store);
    return success();
  }
};

struct LegalizeVectorOuterProduct
    : public OpRewritePattern<vector::OuterProductOp> {
  using OpRewritePattern<vector::OuterProductOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::OuterProductOp outerProduct,
                                PatternRewriter &rewriter) const override {
    auto vectorType = outerProduct.getResultVectorType();

    if (!isVectorTypeMultipleOfSMETileSize(vectorType))
      return failure();

    Operation *rootOp = outerProduct;
    auto loc = outerProduct.getLoc();
    Value mask = {};
    if (outerProduct.isMasked()) {
      auto maskOp = outerProduct.getMaskingOp();
      mask = maskOp.getMask();
      rewriter.setInsertionPoint(maskOp);
      rootOp = maskOp;
    }

    auto tileType = legalTileType(vectorType.getElementType());
    VectorType sliceType = VectorType::Builder(tileType).dropDim(0);

    ValueRange accTiles{};
    if (outerProduct.getAcc())
      accTiles = createUnrealizedConversionToSMETiles(rewriter, loc,
                                                      outerProduct.getAcc())
                     .getResults();

    int tileIndex = 0;
    SmallVector<Value> smeTiles;
    forEachDecomposedVectorType(vectorType, [&](int i, int j) {
      auto subMask = extractSubMask(rewriter, loc, mask, i, j, tileType);
      auto lhs = rewriter.create<vector::ScalableExtractOp>(
          loc, sliceType, outerProduct.getLhs(), i);
      auto rhs = rewriter.create<vector::ScalableExtractOp>(
          loc, sliceType, outerProduct.getRhs(), j);
      Operation *subTileOuterproduct = rewriter.create<vector::OuterProductOp>(
          loc, tileType, lhs, rhs,
          !accTiles.empty() ? accTiles[tileIndex] : Value{},
          outerProduct.getKind());
      if (subMask)
        subTileOuterproduct =
            vector::maskOperation(rewriter, subTileOuterproduct, subMask);
      smeTiles.push_back(subTileOuterproduct->getResult(0));
      ++tileIndex;
    });

    rewriter.replaceOp(rootOp, createUnrealizedConversionFromSMETiles(
                                   rewriter, loc, smeTiles, vectorType));

    return success();
  }
};

struct LegalizeTransferRead : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp read,
                                PatternRewriter &rewriter) const override {
    auto vectorType = read.getVectorType();

    if (!isVectorTypeMultipleOfSMETileSize(vectorType))
      return failure();

    auto tileType = legalTileType(vectorType.getElementType());
    auto loc = read.getLoc();

    SmallVector<Value> smeTiles;
    forEachDecomposedVectorType(
        vectorType,
        [&](int i, int j) {
          auto subTileLoad = rewriter.create<vector::TransferReadOp>(
              loc, tileType, read.getSource(),
              remapIndices(rewriter, loc, read.getIndices(), i, j),
              read.getPermutationMapAttr(), read.getPadding(),
              extractSubMask(rewriter, loc, read.getMask(), i, j, tileType),
              read.getInBoundsAttr());
          smeTiles.push_back(subTileLoad);
        },
        read.getPermutationMap().isIdentity() ? ArrayRef<int64_t>{0, 1}
                                              : ArrayRef<int64_t>{1, 0});

    rewriter.replaceOp(read, createUnrealizedConversionFromSMETiles(
                                 rewriter, loc, smeTiles, vectorType));

    return success();
  }
};

struct LegalizeTransferWrite
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp write,
                                PatternRewriter &rewriter) const override {
    auto vectorType = write.getVectorType();

    if (!isVectorTypeMultipleOfSMETileSize(vectorType))
      return failure();

    auto tileType = legalTileType(vectorType.getElementType());
    auto loc = write.getLoc();

    int tileIndex = 0;
    auto smeTiles =
        createUnrealizedConversionToSMETiles(rewriter, loc, write.getVector());
    forEachDecomposedVectorType(
        vectorType,
        [&](int i, int j) {
          rewriter.create<vector::TransferWriteOp>(
              loc, smeTiles.getResult(tileIndex++), write.getSource(),
              remapIndices(rewriter, loc, write.getIndices(), i, j),
              write.getPermutationMapAttr(),
              extractSubMask(rewriter, loc, write.getMask(), i, j, tileType),
              write.getInBoundsAttr());
        },
        write.getPermutationMap().isIdentity() ? ArrayRef<int64_t>{0, 1}
                                               : ArrayRef<int64_t>{1, 0});

    rewriter.eraseOp(write);

    return success();
  }
};

struct VectorTypeLegalizationPass
    : public arm_sme::impl::VectorTypeLegalizationBase<
          VectorTypeLegalizationPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LegalizeVectorLoad, LegalizeVectorStore,
                 LegalizeVectorOuterProduct, LegalizeTransferRead,
                 LegalizeTransferWrite>(patterns.getContext());
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::arm_sme::createVectorTypeLegalizationPass() {
  return std::make_unique<VectorTypeLegalizationPass>();
}
