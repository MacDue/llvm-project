#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/ArmSME/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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
      [&](auto indices) {
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

ValueRange createUnrealizedConversionToSMETiles(PatternRewriter &rewriter,
                                                Location loc, Value input) {
  auto inputType = llvm::cast<VectorType>(input.getType());
  VectorType tileType = getSMETileTypeForElement(inputType.getElementType());
  SmallVector<Type> resultTiles(getNumberOfSMESubTilesForVectorType(inputType),
                                tileType);
  return rewriter.create<UnrealizedConversionCastOp>(loc, resultTiles, input)
      .getResults();
}

Value createUnrealizedConversionFromSMETiles(PatternRewriter &rewriter,
                                             Location loc, ValueRange input,
                                             VectorType targetType) {
  return rewriter.create<UnrealizedConversionCastOp>(loc, targetType, input)
      .getResult(0);
}

struct LegalizeVectorLoad : public OpRewritePattern<vector::LoadOp> {
  using OpRewritePattern<vector::LoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto vectorType = loadOp.getVectorType();
    if (!isVectorTypeMultipleOfSMETileSize(vectorType))
      return failure();

    auto loc = loadOp.getLoc();
    auto tileType = getSMETileTypeForElement(vectorType.getElementType());
    auto resultSMETiles = llvm::map_to_vector(
        decompose2DVectorType(rewriter, vectorType, tileType),
        [&](SubTileBuilder subTile) -> Value {
          return rewriter.create<vector::LoadOp>(
              loc, tileType, loadOp.getBase(),
              subTile.remapIndicesForSubTile(loc, loadOp.getIndices()));
        });

    rewriter.replaceOp(loadOp, createUnrealizedConversionFromSMETiles(
                                   rewriter, loc, resultSMETiles, vectorType));
    return success();
  }
};

struct LegalizeVectorStore : public OpRewritePattern<vector::StoreOp> {
  using OpRewritePattern<vector::StoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto vectorType = storeOp.getVectorType();
    if (!isVectorTypeMultipleOfSMETileSize(vectorType))
      return failure();

    auto loc = storeOp.getLoc();
    auto tileType = getSMETileTypeForElement(vectorType.getElementType());
    auto inputSMETiles = createUnrealizedConversionToSMETiles(
        rewriter, loc, storeOp.getValueToStore());

    for (auto [index, subTile] : llvm::enumerate(
             decompose2DVectorType(rewriter, vectorType, tileType))) {
      rewriter.create<vector::StoreOp>(
          loc, inputSMETiles[index], storeOp.getBase(),
          subTile.remapIndicesForSubTile(loc, storeOp.getIndices()));
    }

    rewriter.eraseOp(storeOp);
    return success();
  }
};

struct LegalizeVectorOuterProduct
    : public OpRewritePattern<vector::OuterProductOp> {
  using OpRewritePattern<vector::OuterProductOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::OuterProductOp outerProductOp,
                                PatternRewriter &rewriter) const override {
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

    ValueRange accSMETiles{};
    if (outerProductOp.getAcc())
      accSMETiles = createUnrealizedConversionToSMETiles(
          rewriter, loc, outerProductOp.getAcc());

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
          !accSMETiles.empty() ? accSMETiles[index] : Value{},
          outerProductOp.getKind());

      if (*subMask)
        subOuterProduct =
            vector::maskOperation(rewriter, subOuterProduct, *subMask);

      resultSMETiles.push_back(subOuterProduct->getResult(0));
    }

    rewriter.replaceOp(rootOp, createUnrealizedConversionFromSMETiles(
                                   rewriter, loc, resultSMETiles, vectorType));

    return success();
  }
};

struct LegalizeTransferRead : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
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

    rewriter.replaceOp(readOp, createUnrealizedConversionFromSMETiles(
                                   rewriter, loc, resultSMETiles, vectorType));

    return success();
  }
};

struct LegalizeTransferWrite
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern<vector::TransferWriteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
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
    auto inputSMETiles = createUnrealizedConversionToSMETiles(
        rewriter, loc, writeOp.getVector());

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

struct LegalizeSCFYield : public OpRewritePattern<scf::YieldOp> {

  LogicalResult matchAndRewrite(scf::YieldOp yieldOp,
                                PatternRewriter &rewriter) const override {
    auto loc = yieldOp.getLoc();
    SmallVector<Value> expandedOperands;
    expandedOperands.reserve(yieldOp.getNumOperands());
    for (Value operand : yieldOp.getOperands()) {
      auto vectorType = llvm::dyn_cast<VectorType>(operand.getType());
      if (!vectorType || !isVectorTypeMultipleOfSMETileSize(vectorType)) {
        expandedOperands.push_back(operand);
        continue;
      }
      auto newOperands =
          createUnrealizedConversionToSMETiles(rewriter, loc, operand);
      expandedOperands.append(std::begin(newOperands), std::end(newOperands));
    }
    if (expandedOperands.size() == yieldOp.getNumOperands())
      return failure();
    rewriter.create<scf::YieldOp>(loc, expandedOperands);
  }
};

struct FoldNonConstantExtractsOf2DScalableMasks
    : public OpRewritePattern<vector::ExtractOp> {
  using OpRewritePattern<vector::ExtractOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    auto loc = extractOp.getLoc();
    auto createMaskOp =
        extractOp.getVector().getDefiningOp<vector::CreateMaskOp>();
    if (!createMaskOp)
      return failure();

    VectorType extractedMaskType =
        llvm::dyn_cast<VectorType>(extractOp.getResult().getType());

    if (!extractedMaskType)
      return failure();

    if (extractOp.hasDynamicPosition())
      return failure();

    auto sourceVectorType = extractOp.getSourceVectorType();
    ArrayRef<int64_t> extractOpPos = extractOp.getStaticPosition();

    // TODO
    if (extractOpPos.size() != 1)
      return failure();

    for (auto i = 0U; i < extractOpPos.size(); i++) {
      if (sourceVectorType.getScalableDims()[i])
        return failure();
    }
    auto numScalable = llvm::count(extractedMaskType.getScalableDims(), true);

    if (numScalable != 2)
      return failure();

    auto operand = createMaskOp.getOperand(0);
    if (operand.getDefiningOp<arith::ConstantOp>())
      return failure();

    auto index = rewriter.create<arith::ConstantIndexOp>(loc, extractOpPos[0]);
    auto cond = rewriter.create<arith::CmpIOp>(
        loc, rewriter.getI1Type(), arith::CmpIPredicate::slt, index, operand);
    auto activeRows = rewriter.create<arith::SelectOp>(
        loc, cond, createMaskOp.getOperand(1),
        rewriter.create<arith::ConstantIndexOp>(loc, 0));
    rewriter.replaceOpWithNewOp<vector::CreateMaskOp>(
        extractOp, extractedMaskType,
        ValueRange{activeRows.getResult(), createMaskOp.getOperand(2)});
    return success();
  }
};

struct FoldIllegalTransposeIntoTransferRead
    : public OpRewritePattern<vector::TransposeOp> {
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp tranposeOp,
                                PatternRewriter &rewriter) const override {
    auto sourceType = tranposeOp.getSourceVectorType();

    bool seenSD = false;
    bool iST = false;
    for (bool s : sourceType.getScalableDims()) {
      if (s)
        seenSD = true;
      if (!s && seenSD) {
        iST = true;
        break;
      }
    }

    if (!iST)
      return failure();
    auto read = tranposeOp.getVector().getDefiningOp<vector::TransferReadOp>();
    if (!read)
      return failure();

    if (!read.getPermutationMap().isIdentity())
      return failure();

    auto loc = tranposeOp.getLoc();
    mlir::AffineMap map =
        AffineMap::getPermutationMap(tranposeOp.getPermutation(), getContext());

    Value mask = read.getMask();
    if (mask)
      mask = rewriter.create<vector::TransposeOp>(loc, mask,
                                                  tranposeOp.getPermutation());

    auto source = rewriter.create<memref::TransposeOp>(loc, read.getSource(),
                                                       AffineMapAttr::get(map));
    auto test = SmallVector{read.getIndices()[1], read.getIndices()[0]};

    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        tranposeOp, tranposeOp.getResultVectorType(), source, test,
        read.getPermutationMapAttr(), read.getPadding(), mask,
        read.getInBoundsAttr());

    return success();
  }
};

struct VectorLegalizationPass
    : public arm_sme::impl::VectorLegalizationBase<VectorLegalizationPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns
        .add<LegalizeVectorLoad, LegalizeVectorStore,
             LegalizeVectorOuterProduct, LegalizeTransferRead,
             LegalizeTransferWrite, FoldNonConstantExtractsOf2DScalableMasks,
             FoldIllegalTransposeIntoTransferRead>(patterns.getContext());
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                        std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::arm_sme::createVectorLegalizationPass() {
  return std::make_unique<VectorLegalizationPass>();
}
