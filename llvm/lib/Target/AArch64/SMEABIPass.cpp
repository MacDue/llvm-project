//===--------- SMEABI - SME  ABI-------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements parts of the the SME ABI, such as:
// * Using the lazy-save mechanism before enabling the use of ZA.
// * Setting up the lazy-save mechanism around invokes.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "Utils/AArch64SMEAttributes.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-sme-abi"

namespace {

struct LiveRange {
  using RangeSet =
      llvm::IntervalMap<uint64_t, uint8_t, 16, llvm::IntervalMapInfo<unsigned>>;
  using Allocator = RangeSet::Allocator;
  static constexpr uint8_t kValidLiveRange = 0xff;

  LiveRange(Allocator &allocator)
      : Ranges(std::make_unique<RangeSet>(allocator)) {}

  bool overlaps(LiveRange const &Other) const {
    return llvm::IntervalMapOverlaps<RangeSet, RangeSet>(*Ranges, *Other.Ranges)
        .valid();
  }

  bool overlaps(uint64_t Point) const {
    return Ranges->lookup(Point) == kValidLiveRange;
  }

  void mark(unsigned Start, unsigned End) {
    if (Start < End)
      Ranges->insert(Start, End, kValidLiveRange);
  }

  bool empty() const { return Ranges->empty(); }
  unsigned start() const { return Ranges->start(); }
  unsigned end() const { return Ranges->stop(); }

  std::unique_ptr<RangeSet> Ranges;
};

class ZALiveness {
public:
  struct BlockInfo {
    using ValueSet = SmallPtrSet<const Value *, 8>;

    BlockInfo() = default;

    BlockInfo(const BasicBlock *Block, Type *ZaType) : Block(Block) {

      // TODO: Need to pre-process the IR to handle PHI nodes correctly
      // (as their uses don't make sense for live ranges -- really the
      // arguments are copies/uses in a predecessor).
      // -> split cond blocks -> insert uses before branches
      // (then ignore PHI users elsewhere)
      for (auto It = Block->begin(), E = Block->end(); It != E; ++It) {
        const Instruction &Inst = *It;

        // Collect uses of ZA.
        if (!isa<PHINode>(Inst)) {
          for (size_t I = 0, N = Inst.getNumOperands(); I < N; ++I) {
            auto *Op = Inst.getOperand(I);
            if (Op->getType() != ZaType)
              continue;
            ZaUseVals.insert(Op);
          }
        }

        if (Inst.getType() != ZaType)
          continue;

        // Collect definitions of ZA.
        auto *ZaDef = cast<Value>(&Inst);
        ZaDefVals.insert(ZaDef);

        // Collect out values for the current block.
        for (auto *User : ZaDef->users()) {
          if (isa<PHINode>(User))
            continue;
          auto UserInst = cast<Instruction>(User);
          if (UserInst->getParent() != Block)
            OutZa.insert(ZaDef);
        }
      }
      set_subtract(ZaUseVals, ZaDefVals);
    }

    bool updateLiveIn() {
      ValueSet NewIn = ZaUseVals;
      set_union(NewIn, OutZa);
      set_subtract(NewIn, ZaDefVals);

      if (NewIn.size() == InZa.size())
        return false;

      InZa = std::move(NewIn);
      return true;
    }

    void
    updateLiveOut(const DenseMap<const BasicBlock *, BlockInfo> &BlockInfoMap) {
      for (const BasicBlock *Succ : successors(Block)) {
        const BlockInfo &Info = BlockInfoMap.at(Succ);
        set_union(OutZa, Info.InZa);
      }
    }

    const BasicBlock *Block{nullptr};
    ValueSet InZa;
    ValueSet OutZa;
    ValueSet ZaDefVals;
    ValueSet ZaUseVals;
  };

  DenseMap<const BasicBlock *, BlockInfo> BlockInfoMap;

  // Order valid within basic blocks (arbitrary between basic blocks).
  DenseMap<const Instruction *, unsigned> InstructionOrder;

public:
  ZALiveness(Function *F, Type *ZaType) {
    unsigned NextInstructionId = 0;
    SetVector<const BasicBlock *> Worklist;
    for (auto *Block : depth_first(F)) {
      auto &Info = BlockInfoMap.try_emplace(Block, Block, ZaType).first->second;

      if (Info.updateLiveIn()) {
        Worklist.insert(pred_begin(Block), pred_end(Block));
      }

      for (auto It = Block->begin(), E = Block->end(); It != E; ++It)
        InstructionOrder.try_emplace(&*It, NextInstructionId++);
    }

    while (!Worklist.empty()) {
      const BasicBlock *Block = Worklist.pop_back_val();
      BlockInfo &Info = const_cast<BlockInfo &>(BlockInfoMap.at(Block));
      Info.updateLiveOut(BlockInfoMap);
      if (Info.updateLiveIn()) {
        Worklist.insert(pred_begin(Block), pred_end(Block));
      }
    }
  }

  const BlockInfo &getBlockLiveness(const BasicBlock *Block) const {
    return BlockInfoMap.at(Block);
  }

  const Instruction *getEndInstruction(const BlockInfo &Info, const Value *V,
                                       const Instruction *Start) {
    if (Info.OutZa.contains(V))
      return &Info.Block->back();

    if (isa<Constant>(V))
      return Start;

    const Instruction *End = Start;
    for (auto User : V->users()) {
      if (isa<PHINode>(User))
        continue;
      auto &Inst = cast<Instruction>(*User);
      if (Inst.getParent() == Info.Block &&
          InstructionOrder.at(End) < InstructionOrder.at(&Inst))
        End = &Inst;
    }

    return End;
  }
};

static void insertLazySaveAndRestores(Function *F) {
  Type *ZaType = TargetExtType::get(F->getContext(), "aarch64.za.generation");
  ZALiveness Liveness(F, ZaType);
  LiveRange::Allocator LiveRangeAllocator;
  DenseMap<const Value *, LiveRange> LiveRanges;

  auto defineOrUpdateValueLiveRange = [&](const Value *V,
                                          const Instruction *FirstUseOrDef,
                                          ZALiveness::BlockInfo const &Info,
                                          bool Def = false) {
    // Find or create a live range for `value`.
    auto [It, _] = LiveRanges.try_emplace(V, LiveRangeAllocator);
    LiveRange &LiveRange = It->second;
    auto LastUseInBlock = Liveness.getEndInstruction(Info, V, FirstUseOrDef);
    unsigned Start = Liveness.InstructionOrder.at(FirstUseOrDef);
    unsigned End = Liveness.InstructionOrder.at(LastUseInBlock);
    LiveRange.mark(Start + (Def ? 1 : 0), End);
  };

  // Compute live ranges for ZA state and collect clobbers.
  SmallVector<const IntrinsicInst *> Clobbers;
  for (auto It = F->begin(), E = F->end(); It != E; ++It) {
    const BasicBlock *Block = &*It;
    auto &Info = Liveness.getBlockLiveness(Block);
    for (const Value *LiveIn : Info.InZa)
      defineOrUpdateValueLiveRange(LiveIn, &Block->front(), Info);

    for (auto It = Block->begin(), E = Block->end(); It != E; ++It) {
      const Instruction *Inst = &*It;
      if (auto *Intr = dyn_cast<IntrinsicInst>(Inst)) {
        if (Intr->getIntrinsicID() == Intrinsic::aarch64_sme_clobber_za_state)
          Clobbers.push_back(Intr);
      }
      if (Inst->getType() != ZaType)
        continue;
      const Value *Def = cast<Value>(Inst);
      defineOrUpdateValueLiveRange(Def, Inst, Info, true);
    }
  }

  DominatorTree DT(*F);
  SmallVector<const Instruction *> SavePoints;
  SmallVector<const Instruction *> ReloadPoints;
  LiveRange ClobberPoints(LiveRangeAllocator);
  for (auto &[V, Range] : LiveRanges) {
    for (auto *Clobber : Clobbers) {
      unsigned ClobberPoint = Liveness.InstructionOrder.at(Clobber);
      if (ClobberPoints.overlaps(ClobberPoint) || !Range.overlaps(ClobberPoint))
        continue;

      Instruction *SpillPoint = const_cast<IntrinsicInst *>(Clobber);
      DenseMap<const BasicBlock *, unsigned> BlockToMinReloadIndex;
      SmallPtrSet<const Instruction *, 8> ReloadCandidates;
      for (auto *User : V->users()) {
        if (isa<PHINode>(User))
          continue;
        if (!isPotentiallyReachable(SpillPoint, cast<Instruction>(User),
                                    nullptr, &DT))
          continue;
        // Find a common dominator of all reload points as the spill (save)
        // point. It dominating all users means that it's safe to mark the paths
        // to the users as "clobbered" (preventing additional saves/reloads).
        auto *Inst = const_cast<Instruction *>(cast<Instruction>(User));
        SpillPoint = DT.findNearestCommonDominator(SpillPoint, Inst);
        unsigned ReloadIndex = Liveness.InstructionOrder.at(Inst);
        auto [It, Inserted] = BlockToMinReloadIndex.insert(
            std::make_pair(Inst->getParent(), ReloadIndex));
        if (!Inserted)
          It->second = std::min(It->second, ReloadIndex);
        ReloadCandidates.insert(Inst);
      }
      SavePoints.push_back(SpillPoint);

      auto *SpillBlock = SpillPoint->getParent();
      SmallPtrSet<const BasicBlock *, 8> ClobberedBlocks;
      for (auto *Candidate : ReloadCandidates) {
        bool IsDominatedByReload = false;
        SmallVector<const BasicBlock *> ClobberPath;
        const BasicBlock *CandidateBlock = Candidate->getParent();
        const BasicBlock *Block = CandidateBlock;
        unsigned ReloadIndex = Liveness.InstructionOrder.at(Candidate);
        while (true) {
          // Check for any other reloads that might dominate this reload. If
          // this reload is dominated by another, we can ignore this candidate
          // (and not clobber its second of the live range). If another clobber
          // exists before this reload point an additional save/restore will
          // still be inserted.
          auto It = BlockToMinReloadIndex.find(Block);
          if (It != BlockToMinReloadIndex.end()) {
            if (CandidateBlock != Block || ReloadIndex > It->second) {
              IsDominatedByReload = true;
              break;
            }
          }
          ClobberPath.push_back(Block);
          if (Block == SpillBlock)
            break;
          auto *DomNode = DT.getNode(Block);
          if (!DomNode)
            break;
          auto *IDomBlock = DomNode->getIDom()->getBlock();
          Block = IDomBlock;
        }
        if (!IsDominatedByReload) {
          ReloadPoints.push_back(Candidate);
          ClobberedBlocks.insert_range(ClobberPath);
        }
      }

      // Mark the ranges of ZA that are 'clobbered'. Any additional clobbers in
      // in these ranges will not incur additional save/reloads.
      unsigned SpillIndex = Liveness.InstructionOrder.at(SpillPoint);
      for (auto *Block : ClobberedBlocks) {
        unsigned BlockEndIndex = Liveness.InstructionOrder.at(&Block->back());
        unsigned ClobberEndIndex =
            BlockToMinReloadIndex.lookup_or(Block, BlockEndIndex);
        if (Block == SpillBlock) {
          ClobberPoints.mark(SpillIndex, ClobberEndIndex);
        } else {
          unsigned BlockStartIndex =
              Liveness.InstructionOrder.at(&Block->front());
          ClobberPoints.mark(BlockStartIndex, ClobberEndIndex);
        }
      }
    }
  }

#ifndef NDEBUG
  // Debug print.
  llvm::dbgs() << "========== ZA liveness and clobers:\n";
  LiveRanges.try_emplace(nullptr, std::move(ClobberPoints));
  unsigned BlockIdx = 0;
  for (auto It = F->begin(), E = F->end(); It != E; ++It) {
    const BasicBlock *Block = &*It;
    llvm::errs() << "^bb" << BlockIdx++ << ":\n";
    for (auto It = Block->begin(), E = Block->end(); It != E; ++It) {
      const Instruction *Inst = &*It;
      unsigned Index = Liveness.InstructionOrder.at(Inst);
      for (auto &[V, Range] : LiveRanges) {
        char liveness = [Index, V = V, &Range = Range] {
          bool InRange = Range.overlaps(Index);
          // ZA value:
          if (V)
            return InRange ? '|' : ' ';
          // ZA clobber:
          return InRange ? 'x' : ' ';
        }();

        llvm::errs() << liveness;
      }
      llvm::errs() << ' ';
      Inst->dump();
    }
    llvm::errs() << "==========\n";
  }
#endif

  IRBuilder<> Builder(F->getContext());
  Module *M = F->getParent();

  Function *LazySaveIntr = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::aarch64_sme_lazy_save_za_state);
  Function *RestoreIntr = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::aarch64_sme_restore_za_state);

  for (auto *SavePoint : SavePoints) {
    Builder.SetInsertPoint(const_cast<Instruction *>(SavePoint));
    Builder.CreateCall(LazySaveIntr->getFunctionType(), LazySaveIntr);
  }

  for (auto *Restore : ReloadPoints) {
    Builder.SetInsertPoint(const_cast<Instruction *>(Restore));
    Builder.CreateCall(LazySaveIntr->getFunctionType(), RestoreIntr);
  }
}

struct SMEABI : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  SMEABI() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

private:
  bool updateNewStateFunctions(Module *M, Function *F, IRBuilder<> &Builder,
                               SMEAttrs FnAttrs);
};
} // end anonymous namespace

char SMEABI::ID = 0;
static const char *name = "SME ABI Pass";
INITIALIZE_PASS_BEGIN(SMEABI, DEBUG_TYPE, name, false, false)
INITIALIZE_PASS_END(SMEABI, DEBUG_TYPE, name, false, false)

FunctionPass *llvm::createSMEABIPass() { return new SMEABI(); }

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

// Utility function to emit a call to __arm_tpidr2_save and clear TPIDR2_EL0.
void emitTPIDR2Save(Module *M, IRBuilder<> &Builder, bool ZT0IsUndef = false) {
  auto &Ctx = M->getContext();
  auto *TPIDR2SaveTy =
      FunctionType::get(Builder.getVoidTy(), {}, /*IsVarArgs=*/false);
  auto Attrs =
      AttributeList().addFnAttribute(Ctx, "aarch64_pstate_sm_compatible");
  FunctionCallee Callee =
      M->getOrInsertFunction("__arm_tpidr2_save", TPIDR2SaveTy, Attrs);
  CallInst *Call = Builder.CreateCall(Callee);

  // If ZT0 is undefined (i.e. we're at the entry of a "new_zt0" function), mark
  // that on the __arm_tpidr2_save call. This prevents an unnecessary spill of
  // ZT0 that can occur before ZA is enabled.
  if (ZT0IsUndef)
    Call->addFnAttr(Attribute::get(Ctx, "aarch64_zt0_undef"));

  Call->setCallingConv(
      CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X0);

  // A save to TPIDR2 should be followed by clearing TPIDR2_EL0.
  Function *WriteIntr =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::aarch64_sme_set_tpidr2);
  Builder.CreateCall(WriteIntr->getFunctionType(), WriteIntr,
                     Builder.getInt64(0));
}

/// This function generates code at the beginning and end of a function marked
/// with either `aarch64_new_za` or `aarch64_new_zt0`.
/// At the beginning of the function, the following code is generated:
///  - Commit lazy-save if active   [Private-ZA Interface*]
///  - Enable PSTATE.ZA             [Private-ZA Interface]
///  - Zero ZA                      [Has New ZA State]
///  - Zero ZT0                     [Has New ZT0 State]
///
/// * A function with new ZT0 state will not change ZA, so committing the
/// lazy-save is not strictly necessary. However, the lazy-save mechanism
/// may be active on entry to the function, with PSTATE.ZA set to 1. If
/// the new ZT0 function calls a function that does not share ZT0, we will
/// need to conditionally SMSTOP ZA before the call, setting PSTATE.ZA to 0.
/// For this reason, it's easier to always commit the lazy-save at the
/// beginning of the function regardless of whether it has ZA state.
///
/// At the end of the function, PSTATE.ZA is disabled if the function has a
/// Private-ZA Interface. A function is considered to have a Private-ZA
/// interface if it does not share ZA or ZT0.
///
bool SMEABI::updateNewStateFunctions(Module *M, Function *F,
                                     IRBuilder<> &Builder, SMEAttrs FnAttrs) {
  LLVMContext &Context = F->getContext();
  BasicBlock *OrigBB = &F->getEntryBlock();
  Builder.SetInsertPoint(&OrigBB->front());

  // Commit any active lazy-saves if this is a Private-ZA function. If the
  // value read from TPIDR2_EL0 is not null on entry to the function then
  // the lazy-saving scheme is active and we should call __arm_tpidr2_save
  // to commit the lazy save.
  if (FnAttrs.hasPrivateZAInterface()) {
    // Create the new blocks for reading TPIDR2_EL0 & enabling ZA state.
    auto *SaveBB = OrigBB->splitBasicBlock(OrigBB->begin(), "save.za", true);
    auto *PreludeBB = BasicBlock::Create(Context, "prelude", F, SaveBB);

    // Read TPIDR2_EL0 in PreludeBB & branch to SaveBB if not 0.
    Builder.SetInsertPoint(PreludeBB);
    Function *TPIDR2Intr =
        Intrinsic::getOrInsertDeclaration(M, Intrinsic::aarch64_sme_get_tpidr2);
    auto *TPIDR2 = Builder.CreateCall(TPIDR2Intr->getFunctionType(), TPIDR2Intr,
                                      {}, "tpidr2");
    auto *Cmp = Builder.CreateCmp(ICmpInst::ICMP_NE, TPIDR2,
                                  Builder.getInt64(0), "cmp");
    Builder.CreateCondBr(Cmp, SaveBB, OrigBB);

    // Create a call __arm_tpidr2_save, which commits the lazy save.
    Builder.SetInsertPoint(&SaveBB->back());
    emitTPIDR2Save(M, Builder, /*ZT0IsUndef=*/FnAttrs.isNewZT0());

    // Enable pstate.za at the start of the function.
    Builder.SetInsertPoint(&OrigBB->front());
    Function *EnableZAIntr =
        Intrinsic::getOrInsertDeclaration(M, Intrinsic::aarch64_sme_za_enable);
    Builder.CreateCall(EnableZAIntr->getFunctionType(), EnableZAIntr);
  }

  if (FnAttrs.isNewZA()) {
    Function *ZeroIntr =
        Intrinsic::getOrInsertDeclaration(M, Intrinsic::aarch64_sme_zero);
    Builder.CreateCall(ZeroIntr->getFunctionType(), ZeroIntr,
                       Builder.getInt32(0xff));
  }

  if (FnAttrs.isNewZT0()) {
    Function *ClearZT0Intr =
        Intrinsic::getOrInsertDeclaration(M, Intrinsic::aarch64_sme_zero_zt);
    Builder.CreateCall(ClearZT0Intr->getFunctionType(), ClearZT0Intr,
                       {Builder.getInt32(0)});
  }

  if (FnAttrs.hasPrivateZAInterface()) {
    // Before returning, disable pstate.za
    for (BasicBlock &BB : *F) {
      Instruction *T = BB.getTerminator();
      if (!T || !isa<ReturnInst>(T))
        continue;
      Builder.SetInsertPoint(T);
      Function *DisableZAIntr = Intrinsic::getOrInsertDeclaration(
          M, Intrinsic::aarch64_sme_za_disable);
      Builder.CreateCall(DisableZAIntr->getFunctionType(), DisableZAIntr);
    }
  }

  F->addFnAttr("aarch64_expanded_pstate_za");
  return true;
}

bool SMEABI::runOnFunction(Function &F) {
  Module *M = F.getParent();
  LLVMContext &Context = F.getContext();
  IRBuilder<> Builder(Context);

  if (F.isDeclaration() || F.hasFnAttribute("aarch64_expanded_pstate_za"))
    return false;

  insertLazySaveAndRestores(&F);

  bool Changed = false;
  // SMEAttrs FnAttrs(F);
  // if (FnAttrs.isNewZA() || FnAttrs.isNewZT0())
  //   Changed |= updateNewStateFunctions(M, &F, Builder, FnAttrs);

  return Changed;
}
