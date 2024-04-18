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
#include "mlir/Dialect/ArmSME/Transforms/Transforms.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include <algorithm>
#include <vector>

#define DEBUG_TYPE "allocate-arm-sme-tiles"

namespace mlir::arm_sme {
#define GEN_PASS_DEF_TILEALLOCATION
#include "mlir/Dialect/ArmSME/Transforms/Passes.h.inc"
} // namespace mlir::arm_sme

using namespace mlir;
using namespace mlir::arm_sme;

namespace {

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

class TileAllocator {
public:
  /// Allocates and returns a tile ID.
  /// Returns an error if there are no tiles left.
  FailureOr<unsigned> allocateTileId(ArmSMETileType tileType) {
    auto masks = getMasks(tileType);
    for (auto [tileId, tileMask] : llvm::enumerate(masks)) {
      if ((tilesInUse & tileMask) == TileMask::kNone) {
        tilesInUse |= tileMask;
        return tileId;
      }
    }
    return failure();
  }

  /// Releases a previously allocated tile ID.
  void releaseTileId(ArmSMETileType tileType, unsigned tileId) {
    if (tileId > kInMemoryTileIdBase)
      return;
    TileMask tileMask = getMasks(tileType)[tileId];
    assert((tilesInUse & tileMask) != TileMask::kNone &&
           "cannot release unallocated tile!");
    tilesInUse ^= tileMask;
  }

  /// Allocates an in-memory tile ID.
  unsigned allocateInMemoryTileId() {
    // Note: We never release in-memory tile IDs. We could, which may allow
    // reusing an allocation, but as we _never_ want to spill an SME tile this
    // is not optimized.
    return nextInMemoryTileId++;
  }

private:
  TileMask tilesInUse = {TileMask::kNone};
  unsigned nextInMemoryTileId = kInMemoryTileIdBase;
};

void insertCopies(IRRewriter &rewriter, FunctionOpInterface function) {
  auto insertCopy = [&](Location loc, OpOperand &operand) {
    auto copy = rewriter.create<arm_sme::CopyTileOp>(loc, operand.get());
    operand.assign(copy);
  };
  for (Block &block : function.getBlocks()) {
    Operation *terminator = block.getTerminator();
    if (!isa<cf::BranchOp, cf::CondBranchOp>(terminator))
      continue;
    rewriter.setInsertionPoint(terminator);
    for (OpOperand &operand : terminator->getOpOperands()) {
      if (isValidSMETileVectorType(operand.get().getType()))
        insertCopy(terminator->getLoc(), operand);
    }
  }
}

struct LiveRange {
  using RangeSet = llvm::IntervalMap<uint64_t, uint8_t, 16,
                                     llvm::IntervalMapHalfOpenInfo<unsigned>>;
  using Allocator = RangeSet::Allocator;

  LiveRange(Allocator &allocator)
      : ranges(std::make_unique<RangeSet>(allocator)) {}

  bool isLive(unsigned pos) const { return ranges->lookup(pos, 0) != 0; }

  bool overlaps(LiveRange const &other) const {
    return llvm::IntervalMapOverlaps<RangeSet, RangeSet>(*ranges, *other.ranges)
        .valid();
  }

  void unionWith(LiveRange const &other) {
    for (auto it = other.ranges->begin(); it != other.ranges->end(); ++it) {
      ranges->insert(it.start(), it.stop(), /*dummy*/ 0xFF);
    }
    values.set_union(other.values);
  }

  bool empty() const { return ranges->empty(); }
  unsigned start() const { return ranges->start(); }
  unsigned end() const { return ranges->stop(); }
  unsigned length() const { return end() - start(); }

  ArmSMETileType getTileType() const {
    return *arm_sme::getSMETileType(cast<VectorType>(values[0].getType()));
  }

  void insert(Value value, unsigned start, unsigned end) {
    values.insert(value);
    ranges->insert(start, end, /*dummy*/ 0xFF);
  }

  bool operator<(LiveRange const &other) { return start() < other.start(); }

  std::unique_ptr<RangeSet> ranges;
  SetVector<Value> values;
  std::optional<unsigned> tileId;
};

DenseMap<Operation *, unsigned>
generateOperationNumbering(FunctionOpInterface func) {
  unsigned index = 0;
  DenseMap<Operation *, unsigned> operationToIndexMap;
  for (Block &block : func.getBlocks()) {
    for (Operation &op : block.getOperations()) {
      assert(op.getNumRegions() == 0 && "expected flat control flow");
      operationToIndexMap.try_emplace(&op, index++);
    }
  }
  return operationToIndexMap;
}

DenseMap<Value, LiveRange>
gatherLiveRanges(LiveRange::Allocator &liveRangeAllocator, Liveness &liveness,
                 FunctionOpInterface function) {
  auto operationToIndexMap = generateOperationNumbering(function);
  DenseMap<Value, LiveRange> liveRanges;
  auto updateLiveRanges = [&](Value value, Operation *firstUseOrDef,
                              LivenessBlockInfo const &livenessInfo) {
    if (!arm_sme::isValidSMETileVectorType(value.getType()))
      return;
    auto it = liveRanges.try_emplace(value, liveRangeAllocator).first;
    auto lastUseInBlock = livenessInfo.getEndOperation(value, firstUseOrDef);
    unsigned start = operationToIndexMap[firstUseOrDef];
    unsigned end = operationToIndexMap[lastUseInBlock];
    if (start == end) {
      ++end;
    }
    it->second.insert(value, start, end);
  };

  for (Block &block : function.getBlocks()) {
    LivenessBlockInfo const *livenessInfo = liveness.getLiveness(&block);
    if (block.isEntryBlock()) {
      for (Value argument : block.getArguments())
        updateLiveRanges(argument, &block.front(), *livenessInfo);
    }

    for (Value liveIn : livenessInfo->in()) {
      updateLiveRanges(liveIn, &block.front(), *livenessInfo);
    }

    for (Operation &op : block)
      for (Value result : op.getResults())
        updateLiveRanges(result, &op, *livenessInfo);
  }

  return liveRanges;
}

class ValueUnionFind {
public:
  struct Comparator {
    bool operator()(Value const &a, Value const &b) const {
      return a.getImpl() < b.getImpl();
    }
  };
  using EquivalenceClasses = llvm::EquivalenceClasses<Value, Comparator>;
  using MemberIterator = EquivalenceClasses::member_iterator;
  using MembersCallback = function_ref<void(MemberIterator, MemberIterator)>;

  void insert(Value value) { equivalenceClasses.insert(value); }
  void unionSets(Value a, Value b) { equivalenceClasses.unionSets(a, b); }

  void forEachEquivalenceClass(MembersCallback callback) {
    for (EquivalenceClasses::iterator it = equivalenceClasses.begin(),
                                      end = equivalenceClasses.end();
         it != end; ++it) {
      if (!it->isLeader())
        continue;
      callback(equivalenceClasses.member_begin(it),
               equivalenceClasses.member_end());
    }
  };

private:
  EquivalenceClasses equivalenceClasses;
};

SmallVector<LiveRange>
coalesceLiveRanges(LiveRange::Allocator &liveRangeAllocator,
                   DenseMap<Value, LiveRange> const &initialLiveRanges) {
  ValueUnionFind valueMerges;
  for (auto &[value, _] : initialLiveRanges) {
    valueMerges.insert(value);
  }

  auto mergeValuesIfNonOverlapping = [&](Value a, Value b) {
    LiveRange const &aLiveRange = initialLiveRanges.at(a);
    LiveRange const &bLiveRange = initialLiveRanges.at(b);
    if (!aLiveRange.overlaps(bLiveRange)) {
      valueMerges.unionSets(a, b);
    }
  };

  auto unifyDefinitionsWithOperands = [&](Value value) {
    auto armSMEOp = value.getDefiningOp<arm_sme::ArmSMETileOpInterface>();
    if (!armSMEOp)
      return;
    for (auto operand : armSMEOp->getOperands()) {
      if (arm_sme::isValidSMETileVectorType(operand.getType()))
        mergeValuesIfNonOverlapping(value, operand);
    }
  };

  auto unifyBlockArgumentsWithPredecessors = [&](Value value) {
    auto blockArg = dyn_cast<BlockArgument>(value);
    if (!blockArg)
      return;
    Block *block = blockArg.getOwner();
    unsigned argNumber = blockArg.getArgNumber();
    for (Block *pred : block->getPredecessors()) {
      TypeSwitch<Operation *>(pred->getTerminator())
          .Case<cf::BranchOp>([&](auto branch) {
            Value precedingOperand = branch.getDestOperands()[argNumber];
            mergeValuesIfNonOverlapping(value, precedingOperand);
          })
          .Case<cf::CondBranchOp>([&](auto condBranch) {
            if (condBranch.getFalseDest() == block) {
              Value precedingOperand =
                  condBranch.getFalseDestOperands()[argNumber];
              mergeValuesIfNonOverlapping(value, precedingOperand);
            }
            if (condBranch.getTrueDest() == block) {
              Value precedingOperand =
                  condBranch.getTrueDestOperands()[argNumber];
              mergeValuesIfNonOverlapping(value, precedingOperand);
            }
          });
    }
  };

  for (auto &[value, _] : initialLiveRanges) {
    unifyDefinitionsWithOperands(value);
    unifyBlockArgumentsWithPredecessors(value);
  }

  SmallVector<LiveRange> coalescedLiveRanges;
  valueMerges.forEachEquivalenceClass([&](auto memberBegin, auto memberEnd) {
    LiveRange coalescedLiveRange(liveRangeAllocator);
    for (Value value : llvm::make_range(memberBegin, memberEnd)) {
      coalescedLiveRange.unionWith(initialLiveRanges.at(value));
    }
    coalescedLiveRanges.emplace_back(std::move(coalescedLiveRange));
  });

  std::sort(coalescedLiveRanges.begin(), coalescedLiveRanges.end());
  return coalescedLiveRanges;
}

void allocateLiveRanges(MutableArrayRef<LiveRange> liveRanges) {
  TileAllocator tileAllocator;
  SetVector<LiveRange *> allocatedRanges;
  for (auto &newRange : liveRanges) {
    allocatedRanges.remove_if([&](LiveRange *allocatedRange) {
      if (allocatedRange->end() <= newRange.start()) {
        tileAllocator.releaseTileId(allocatedRange->getTileType(),
                                    *allocatedRange->tileId);
        return true;
      }
      return false;
    });

    auto tileId = tileAllocator.allocateTileId(newRange.getTileType());
    if (failed(tileId)) {
      LiveRange *longestActiveRange = *std::max_element(
          allocatedRanges.begin(), allocatedRanges.end(),
          [](LiveRange *a, LiveRange *b) { return a->length() < b->length(); });
      tileId = *longestActiveRange->tileId;
      longestActiveRange->tileId = tileAllocator.allocateInMemoryTileId();
      allocatedRanges.remove(longestActiveRange);
    }

    newRange.tileId = *tileId;
    allocatedRanges.insert(&newRange);
  }
}

void assignTileIdsAndFoldCopies(IRRewriter &rewriter,
                                FunctionOpInterface function,
                                ArrayRef<LiveRange> allocatedLiveRanges) {
  auto tryFoldCopy = [&](LiveRange const &copyLiveness,
                         arm_sme::CopyTileOp copyOp) {
    Value copySourceTile = copyOp.getTile();
    if (copyLiveness.values.contains(copyOp.getTile()))
      return rewriter.replaceAllUsesWith(copyOp, copySourceTile);
    if (auto zeroOp = copySourceTile.getDefiningOp<arm_sme::ZeroOp>()) {
      rewriter.setInsertionPoint(copyOp);
      auto clonedZeroOp = zeroOp.clone();
      clonedZeroOp.setTileId(copyOp.getTileId());
      rewriter.insert(clonedZeroOp);
      rewriter.replaceAllUsesWith(copyOp, clonedZeroOp);
    }
  };
  auto assignTileId = [&](unsigned tileId,
                          arm_sme::ArmSMETileOpInterface tileOp) {
    if (tileId >= kInMemoryTileIdBase) {
      tileOp->emitWarning(
          "failed to allocate SME virtual tile to operation, all tile "
          "operations will go through memory, expect degraded performance");
    }
    auto tileIdAttr = rewriter.getI32IntegerAttr(tileId);
    tileOp.setTileId(tileIdAttr);
  };
  for (LiveRange const &liveRange : allocatedLiveRanges) {
    unsigned tileId = *liveRange.tileId;
    for (Value value : liveRange.values) {
      if (auto tileOp = value.getDefiningOp<arm_sme::ArmSMETileOpInterface>())
        assignTileId(tileId, tileOp);
      for (Operation *user : value.getUsers()) {
        if (auto tileOp = dyn_cast<arm_sme::ArmSMETileOpInterface>(user))
          assignTileId(tileId, tileOp);
      }
      if (auto copyOp = value.getDefiningOp<arm_sme::CopyTileOp>())
        tryFoldCopy(liveRange, copyOp);
    }
  }
}

void eraseTriviallyDeadArmSMEOps(IRRewriter &rewriter,
                                 FunctionOpInterface function) {
  bool stillOpsToCheck = true;
  while (stillOpsToCheck) {
    stillOpsToCheck = false;
    function->walk([&](Operation *op) {
      auto armSMEOp = dyn_cast<arm_sme::ArmSMETileOpInterface>(op);
      if (armSMEOp && armSMEOp.use_empty() &&
          !mlir::hasEffect<MemoryEffects::Write>(op)) {
        rewriter.eraseOp(armSMEOp);
        stillOpsToCheck = true;
      }
    });
  }
}

struct TileAllocationPass
    : public arm_sme::impl::TileAllocationBase<TileAllocationPass> {
  void runOnOperation() override {
    if (failed(arm_sme::allocateSMETiles(getOperation())))
      signalPassFailure();
  }
};
} // namespace

LogicalResult mlir::arm_sme::allocateSMETiles(FunctionOpInterface function) {
  LiveRange::Allocator liveRangeAllocator;
  IRRewriter rewriter(function.getContext());

  // 1. Insert copy operations at branch operations (i.e. the predecessors to
  // block arguments).
  insertCopies(rewriter, function);

  // 2. Gather live ranges for each ArmSME tile within the function.
  Liveness liveness(function);
  auto initialLiveRanges =
      gatherLiveRanges(liveRangeAllocator, liveness, function);

  // 3. Coalesce (non-overlapping) live ranges where it would be beneficial
  // for tile allocation. E.g. Unify the result of an operation with it's
  // operands.
  auto coalescedLiveRanges =
      coalesceLiveRanges(liveRangeAllocator, initialLiveRanges);

  // 4. Allocate tile IDs to live ranges.
  allocateLiveRanges(coalescedLiveRanges);

  // 5. Assign the tile IDs back to the ArmSME operations (and fold way
  // redundant copies).
  assignTileIdsAndFoldCopies(rewriter, function, coalescedLiveRanges);

  /// 6. Erase trivially dead ArmSME operations (e.g. a ZeroOp with no
  /// users).
  eraseTriviallyDeadArmSMEOps(rewriter, function);
  return success();
}

std::unique_ptr<Pass> mlir::arm_sme::createTileAllocationPass() {
  return std::make_unique<TileAllocationPass>();
}
