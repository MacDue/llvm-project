//===- TileAllocation.cpp - Allocate SME ZA tiles -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass allocates SME tiles at the 'func.func' op level for ArmSME
// operations. It does this using a 16-bit tile mask that has a bit for each
// 128-bit element tile (ZA0.Q-ZA15.Q), the smallest ZA tile granule.
//
// The 128-bit tiles overlap with other element tiles as follows (see section
// B2.3.2 of SME spec [1]):
//
//   Tile    Overlaps
//   ---------------------------------------------------------------------------
//   ZA0.B   ZA0.Q, ZA1.Q, ZA2.Q, ZA3.Q, ZA4.Q, ZA5.Q, ZA6.Q, ZA7.Q, ZA8.Q,
//           ZA9.Q, ZA10.Q, ZA11.Q, ZA12.Q, ZA13.Q, ZA14.Q, ZA15.Q
//   ZA0.H   ZA0.Q, ZA2.Q, ZA4.Q, ZA6.Q, ZA8.Q, ZA10.Q, ZA12.Q, ZA14.Q
//   ZA1.H   ZA1.Q, ZA3.Q, ZA5.Q, ZA7.Q, ZA9.Q, ZA11.Q, ZA13.Q, ZA15.Q
//   ZA0.S   ZA0.Q, ZA4.Q, ZA8.Q, ZA12.Q
//   ZA1.S   ZA1.Q, ZA5.Q, ZA9.Q, ZA13.Q
//   ZA2.S   ZA2.Q, ZA6.Q, ZA10.Q, ZA14.Q
//   ZA3.S   ZA3.Q, ZA7.Q, ZA11.Q, ZA15.Q
//   ZA0.D   ZA0.Q, ZA8.Q
//   ZA1.D   ZA1.Q, ZA9.Q
//   ZA2.D   ZA2.Q, ZA10.Q
//   ZA3.D   ZA3.Q, ZA11.Q
//   ZA4.D   ZA4.Q, ZA12.Q
//   ZA5.D   ZA5.Q, ZA13.Q
//   ZA6.D   ZA6.Q, ZA14.Q
//   ZA7.D   ZA7.Q, ZA15.Q
//
// The tiles in use are tracked via a function attribute 'arm_sme.tiles_in_use'
// that is initalized during the first tile allocation within a function and
// updated on each subsequent allocation.
//
// [1] https://developer.arm.com/documentation/ddi0616/aa
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "allocate-arm-sme-tiles"

namespace mlir {
namespace arm_sme {
#define GEN_PASS_DEF_TILEALLOCATION
#include "mlir/Dialect/ArmSME/Transforms/Passes.h.inc"
} // namespace arm_sme
} // namespace mlir

using namespace mlir;
using namespace mlir::arm_sme;

namespace {

static constexpr StringLiteral kTilesInUseAttr("arm_sme.tiles_in_use");
static constexpr StringLiteral
    kNextInMemoryTileIdAttr("arm_sme.next_in_memory_tile_id");

enum class TileMask : unsigned {
  // clang-format off
  kZA0B  = 0xffff, // 1111 1111 1111 1111

  kZA0H  = 0xaaaa, // 1010 1010 1010 1010
  kZA1H  = 0x5555, // 0101 0101 0101 0101

  kZA0S  = 0x8888, // 1000 1000 1000 1000
  kZA1S  = 0x4444, // 0100 0100 0100 0100
  kZA2S  = 0x2222, // 0010 0010 0010 0010
  kZA3S  = 0x1111, // 0001 0001 0001 0001

  kZA0D  = 0x8080, // 1000 0000 1000 0000
  kZA1D  = 0x4040, // 0100 0000 0100 0000
  kZA2D  = 0x2020, // 0010 0000 0010 0000
  kZA3D  = 0x1010, // 0001 0000 0001 0000
  kZA4D  = 0x808,  // 0000 1000 0000 1000
  kZA5D  = 0x404,  // 0000 0100 0000 0100
  kZA6D  = 0x202,  // 0000 0010 0000 0010
  kZA7D  = 0x101,  // 0000 0001 0000 0001

  kZA0Q  = 0x8000, // 1000 0000 0000 0000
  kZA1Q  = 0x4000, // 0100 0000 0000 0000
  kZA2Q  = 0x2000, // 0010 0000 0000 0000
  kZA3Q  = 0x1000, // 0001 0000 0000 0000
  kZA4Q  = 0x800,  // 0000 1000 0000 0000
  kZA5Q  = 0x400,  // 0000 0100 0000 0000
  kZA6Q  = 0x200,  // 0000 0010 0000 0000
  kZA7Q  = 0x100,  // 0000 0001 0000 0000
  kZA8Q  = 0x80,   // 0000 0000 1000 0000
  kZA9Q  = 0x40,   // 0000 0000 0100 0000
  kZA10Q = 0x20,   // 0000 0000 0010 0000
  kZA11Q = 0x10,   // 0000 0000 0001 0000
  kZA12Q = 0x8,    // 0000 0000 0000 1000
  kZA13Q = 0x4,    // 0000 0000 0000 0100
  kZA14Q = 0x2,    // 0000 0000 0000 0010
  kZA15Q = 0x1,    // 0000 0000 0000 0001

  kNone = 0x0,     // 0000 0000 0000 0000
  // clang-format on

  LLVM_MARK_AS_BITMASK_ENUM(kZA0B)
};

/// Returns the set of masks relevant for the given type.
static ArrayRef<TileMask> getMasks(ArmSMETileType type) {
  static constexpr std::array ZA_B_MASKS = {TileMask::kZA0B};
  static constexpr std::array ZA_H_MASKS = {TileMask::kZA0H, TileMask::kZA1H};
  static constexpr std::array ZA_S_MASKS = {TileMask::kZA0S, TileMask::kZA1S,
                                            TileMask::kZA2S, TileMask::kZA3S};
  static constexpr std::array ZA_D_MASKS = {
      TileMask::kZA0D, TileMask::kZA1D, TileMask::kZA2D, TileMask::kZA3D,
      TileMask::kZA4D, TileMask::kZA5D, TileMask::kZA6D, TileMask::kZA7D};
  static constexpr std::array ZA_Q_MASKS = {
      TileMask::kZA0Q,  TileMask::kZA1Q,  TileMask::kZA2Q,  TileMask::kZA3Q,
      TileMask::kZA4Q,  TileMask::kZA5Q,  TileMask::kZA6Q,  TileMask::kZA7Q,
      TileMask::kZA8Q,  TileMask::kZA9Q,  TileMask::kZA10Q, TileMask::kZA11Q,
      TileMask::kZA12Q, TileMask::kZA13Q, TileMask::kZA14Q, TileMask::kZA15Q};
  switch (type) {
  case ArmSMETileType::ZAB:
    return ZA_B_MASKS;
  case ArmSMETileType::ZAH:
    return ZA_H_MASKS;
  case ArmSMETileType::ZAS:
    return ZA_S_MASKS;
  case ArmSMETileType::ZAD:
    return ZA_D_MASKS;
  case ArmSMETileType::ZAQ:
    return ZA_Q_MASKS;
  }
}

/// Allocates and returns a tile ID. Returns an error if there are no tiles
/// left.
static FailureOr<unsigned> allocateTileId(ArmSMETileType tileType,
                                          TileMask &tilesInUse) {
  auto masks = getMasks(tileType);
  for (auto [tileId, tileMask] : llvm::enumerate(masks)) {
    if ((tilesInUse & tileMask) == TileMask::kNone) {
      tilesInUse |= tileMask;
      return tileId;
    }
  }
  return failure();
}

struct HandleTileMoves
    : public OpInterfaceRewritePattern<ArmSMETileOpInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(ArmSMETileOpInterface tileOp,
                                PatternRewriter &rewriter) const override {

    auto tileId = tileOp.getTileId();
    if (!tileId)
      return failure();
    bool updated = false;
    for (auto &arg : tileOp->getOpOperands()) {
      if (auto zeroOp = arg.get().getDefiningOp<arm_sme::ZeroOp>()) {
        if (zeroOp.getTileId() && zeroOp.getTileId() != tileId) {
          auto newZero = zeroOp.clone();
          newZero.setTileId(tileId);
          rewriter.insert(newZero);
          rewriter.modifyOpInPlace(tileOp,
                                   [&] { arg.assign(newZero.getResult()); });
          updated = true;
        }
      }
    }

    if (updated)
      return success();
    return failure();
  }
};

struct TileAllocationPass
    : public arm_sme::impl::TileAllocationBase<TileAllocationPass> {
  void runOnOperation() override {
    auto &liveness = getAnalysis<Liveness>();

    DenseMap<Operation *, unsigned> opToFirstIndex;
    DenseMap<Operation *, unsigned> opToLastIndex;

    unsigned index = 0;
    llvm::unique_function<void(Operation *)> walk = [&](Operation *op) {
      opToFirstIndex.try_emplace(op, index++);
      for (Region &region : op->getRegions())
        for (Block &block : region.getBlocks())
          for (Operation &nested : block)
            walk(&nested);
      opToLastIndex.try_emplace(op, index - 1);
    };
    walk(getOperation());

    // FIXME: This is all dumb hacks.

    struct LiveRange {
      struct Range {
        unsigned start;
        unsigned end;

        bool overlaps(Range other) const {
          return std::max(start, other.start) < std::min(end, other.end);
        }
      };

      unsigned getEnd() const {
        unsigned end = 0;
        for (auto r : ranges) {
          end = std::max(end, r.end);
        }
        return end;
      }

      bool isLive(unsigned programPoint) const {
        for (auto range : ranges) {
          if (programPoint >= range.start && programPoint <= range.end)
            return true;
        }
        return false;
      }

      bool overlaps(LiveRange const &other) const {
        for (auto rangeA : ranges) {
          for (auto rangeB : other.ranges) {
            if (rangeA.overlaps(rangeB))
              return true;
          }
        }
        return false;
      }

      SmallVector<Value> values;
      SmallVector<Range> ranges;
      int tileId = -1;
    };

    DenseMap<Value, LiveRange> liveRanges;
    getOperation()->walk([&](Block *block) {
      const LivenessBlockInfo *info = liveness.getLiveness(block);
      assert(info && "expected liveness info for block");
      auto processValue = [&](Value value, Operation *firstUseOrDef) {
        auto vType = dyn_cast<VectorType>(value.getType());
        if (!vType || !isValidSMETileVectorType(vType))
          return;
        auto liveRange = liveRanges.try_emplace(value).first;
        liveRange->second.values.push_back(value);
        liveRange->second.ranges.push_back(
            {opToLastIndex[firstUseOrDef],
             opToFirstIndex[info->getEndOperation(value, firstUseOrDef)]});
      };

      // Process the live-ins of this block.
      for (Value liveIn : info->in()) {
        // Only process the value if it has been defined in the current region.
        if (liveIn.getParentRegion() == block->getParent())
          processValue(liveIn, &block->front());
      }

      // Process the block arguments for the entry block (those are not
      // live-in).
      if (block->isEntryBlock()) {
        for (Value argument : block->getArguments())
          processValue(argument, &block->front());
      }

      // Process any new defs within this block.
      for (Operation &op : *block)
        for (Value result : op.getResults())
          processValue(result, &op);
    });

    llvm::EquivalenceClasses<void *> valueMerges;
    for (auto [value, _] : liveRanges) {
      valueMerges.insert(value.getAsOpaquePointer());
    }

    auto tryMergeRanges = [&](Value a, Value b) {
      if (!liveRanges[a].overlaps(liveRanges[b])) {
        valueMerges.unionSets(a.getAsOpaquePointer(), b.getAsOpaquePointer());
      }
    };

    for (auto [value, range] : liveRanges) {
      if (auto op = value.getDefiningOp<arm_sme::ArmSMETileOpInterface>()) {
        for (auto arg : op->getOperands()) {
          auto vType = dyn_cast<VectorType>(arg.getType());
          if (vType && arm_sme::isValidSMETileVectorType(vType)) {
            tryMergeRanges(value, arg);
          }
        }
      }
      if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        Block *block = blockArg.getOwner();

        for (Block *pred : block->getPredecessors()) {
          auto branch = pred->getTerminator();
          if (auto br = dyn_cast<cf::BranchOp>(branch)) {
            Value operand =
                *(br.getDestOperands().begin() + blockArg.getArgNumber());
            tryMergeRanges(value, operand);
          } else if (auto condBr = dyn_cast<cf::CondBranchOp>(branch)) {
            if (condBr.getFalseDest() == block) {
              Value operand = *(condBr.getFalseDestOperands().begin() +
                                blockArg.getArgNumber());
              tryMergeRanges(value, operand);
            }
            if (condBr.getTrueDest() == block) {
              Value operand = *(condBr.getTrueDestOperands().begin() +
                                blockArg.getArgNumber());
              tryMergeRanges(value, operand);
            }
          }
        }
      }
    }

    SmallVector<LiveRange> finalLiveRanges;
    for (llvm::EquivalenceClasses<void *>::iterator I = valueMerges.begin(),
                                                    E = valueMerges.end();
         I != E; ++I) {
      if (!I->isLeader())
        continue; // Ignore non-leader sets.
      LiveRange mergedRange;
      for (llvm::EquivalenceClasses<void *>::member_iterator MI =
               valueMerges.member_begin(I);
           MI != valueMerges.member_end(); ++MI) {
        Value value = Value::getFromOpaquePointer(*MI);
        mergedRange.values.push_back(value);
        auto &smallerRange = liveRanges[value];
        mergedRange.ranges.append(smallerRange.ranges.begin(),
                                  smallerRange.ranges.end());
      }
      std::sort(mergedRange.ranges.begin(), mergedRange.ranges.end(),
                [&](auto &a, auto &b) { return a.start < b.start; });
      finalLiveRanges.push_back(mergedRange);
    }

    std::sort(finalLiveRanges.begin(), finalLiveRanges.end(),
              [&](auto &a, auto &b) {
                return a.ranges[0].start < b.ranges[0].start;
              });

    // TODO: Proper allocation
    std::array<bool, 4> freeTiles = {true, true, true, true};

    SmallVector<LiveRange *> allocatedRanges;
    for (auto &liveRange : finalLiveRanges) {
      allocatedRanges.erase(
          std::remove_if(allocatedRanges.begin(), allocatedRanges.end(),
                         [&](auto *range) {
                           if (range->getEnd() <= liveRange.ranges[0].start) {
                             freeTiles[range->tileId] = true;
                             return true;
                           }
                           return false;
                         }),
          allocatedRanges.end());

      int rangeTileId = -1;
      for (auto [tileId, free] : llvm::enumerate(freeTiles)) {
        if (free) {
          rangeTileId = tileId;
          freeTiles[tileId] = false;
          break;
        }
      }

      if (rangeTileId < 0) {
        llvm::dbgs() << "Here?\n";
        signalPassFailure();
        break;
      }

      for (auto value : liveRange.values) {
        auto smeOp = value.getDefiningOp<arm_sme::ArmSMETileOpInterface>();
        if (smeOp) {
          smeOp.setTileId(IntegerAttr::get(IntegerType::get(&getContext(), 32),
                                           rangeTileId));
        }
      }
      liveRange.tileId = rangeTileId;
      allocatedRanges.push_back(&liveRange);
    }

    for (auto &liveRange : finalLiveRanges) {
      auto tileIdAttr = IntegerAttr::get(IntegerType::get(&getContext(), 32),
                                         liveRange.tileId);
      for (auto value : liveRange.values) {
        for (auto *user : value.getUsers()) {
          if (auto armSmeOp = dyn_cast<arm_sme::ArmSMETileOpInterface>(user)) {
            if (!armSmeOp.getTileId()) {
              armSmeOp.setTileId(tileIdAttr);
            }
          }
        }
      }
    }

    // return;

    llvm::unique_function<void(Operation *, int)> walk2 = [&](Operation *op,
                                                              int level) {
      auto idxHere = opToFirstIndex[op];
      for (auto &range : finalLiveRanges) {
        if (range.isLive(idxHere)) {
          if (!range.isLive(idxHere - 1))
            llvm::dbgs() << "S";
          else if (!range.isLive(idxHere + 1))
            llvm::dbgs() << "E";
          else
            llvm::dbgs() << "|";
        } else {
          llvm::dbgs() << " ";
        }
      }
      llvm::dbgs() << " ";
      for (int i = 0; i < level; i++)
        llvm::dbgs() << ' ';
      llvm::dbgs() << op->getName();
      llvm::dbgs() << " | firstIndex = " << opToFirstIndex[op];
      llvm::dbgs() << ", lastIndex = " << opToLastIndex[op] << '\n';
      for (auto [regionIdx, region] : llvm::enumerate(op->getRegions())) {
        for (int i = 0; i < level; i++)
          llvm::dbgs() << ' ';
        llvm::dbgs() << "START NESTED REGION\n";
        for (auto [blockIdx, block] : llvm::enumerate(region.getBlocks())) {
          for (int i = 0; i < level; i++)
            llvm::dbgs() << ' ';
          llvm::dbgs() << "^bb" << blockIdx++ << ":\n";
          for (Operation &nested : block)
            walk2(&nested, level + 2);
        }
        for (int i = 0; i < level; i++)
          llvm::dbgs() << ' ';
        llvm::dbgs() << "END NESTED REGION\n";
      }
    };
    walk2(getOperation(), 0);
    RewritePatternSet patterns(&getContext());
    patterns.add<HandleTileMoves>(patterns.getContext());
    GreedyRewriteConfig config;
    // Setting useTopDownTraversal ensures tiles are allocated in program
    // order.
    config.useTopDownTraversal = true;
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
            getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::arm_sme::createTileAllocationPass() {
  return std::make_unique<TileAllocationPass>();
}
