#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/ArmSME/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
void forEachDecomposedVectorType(VectorType type, Callback callback) {
  auto elementType = type.getElementType();
  unsigned minNumElts = getSMETileSliceMinNumElts(elementType);
  int64_t vectorRows = type.getDimSize(0);
  int64_t vectorCols = type.getDimSize(1);
  for (int i = 0; i < vectorRows; i += minNumElts) {
    for (int j = 0; j < vectorCols; j += minNumElts) {
      callback(i, j);
    }
  }
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

struct LegalizeVectorLoad : public OpRewritePattern<vector::LoadOp> {
  using OpRewritePattern<vector::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::LoadOp load,
                                PatternRewriter &rewriter) const override {
    auto vectorType = load.getVectorType();

    if (!isVectorTypeMultipleOfSMETileSize(vectorType))
      return failure();

    auto tileType = legalTileType(vectorType.getElementType());
    auto base = load.getBase();
    auto indices = load.getIndices();
    auto loc = load.getLoc();

    SmallVector<Value> smeTiles;
    forEachDecomposedVectorType(vectorType, [&](int i, int j) {
      auto rowIndex = rewriter.create<arith::ConstantIndexOp>(loc, i);
      auto colIndex = rewriter.create<arith::ConstantIndexOp>(loc, j);

      auto subTileRowIndex =
          rewriter.create<arith::AddIOp>(loc, rowIndex, indices[0]);
      auto subTileColIndex =
          rewriter.create<arith::AddIOp>(loc, colIndex, indices[1]);

      auto subTileLoad = rewriter.create<vector::LoadOp>(
          loc, tileType, base, ValueRange{subTileRowIndex, subTileColIndex});

      smeTiles.push_back(subTileLoad);
    });

    rewriter.replaceOp(load, createUnrealizedConversionFromSMETiles(
                                 rewriter, loc, smeTiles, vectorType));

    return success();
  }
};

struct LegalizeVectorStore : public OpRewritePattern<vector::StoreOp> {
  using OpRewritePattern<vector::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::StoreOp store,
                                PatternRewriter &rewriter) const override {
    auto vectorType = store.getVectorType();

    if (!isVectorTypeMultipleOfSMETileSize(vectorType))
      return failure();

    auto base = store.getBase();
    auto indices = store.getIndices();
    auto loc = store.getLoc();

    auto smeTiles = createUnrealizedConversionToSMETiles(
        rewriter, loc, store.getValueToStore());

    int tileIndex = 0;
    forEachDecomposedVectorType(vectorType, [&](int i, int j) {
      auto rowIndex = rewriter.create<arith::ConstantIndexOp>(loc, i);
      auto colIndex = rewriter.create<arith::ConstantIndexOp>(loc, j);

      auto subTileRowIndex =
          rewriter.create<arith::AddIOp>(loc, rowIndex, indices[0]);
      auto subTileColIndex =
          rewriter.create<arith::AddIOp>(loc, colIndex, indices[1]);

      rewriter.create<vector::StoreOp>(
          loc, smeTiles.getResult(tileIndex++), base,
          ValueRange{subTileRowIndex, subTileColIndex});
    });

    rewriter.eraseOp(store);

    return success();
  }
};

// struct LegalizeVectorOuterProduct
//     : public OpRewritePattern<vector::OuterProductOp> {
//   using OpRewritePattern<vector::OuterProductOp>::OpRewritePattern;

// };

struct VectorTypeLegalizationPass
    : public arm_sme::impl::VectorTypeLegalizationBase<
          VectorTypeLegalizationPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LegalizeVectorLoad, LegalizeVectorStore>(
        patterns.getContext());
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
