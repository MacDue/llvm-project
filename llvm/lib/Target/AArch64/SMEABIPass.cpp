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
#include "AArch64TargetMachine.h"
#include "Utils/AArch64SMEAttributes.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-sme-abi"

namespace {

struct LiveRange {
  using RangeSet =
      llvm::IntervalMap<uint64_t, Value *, 16, llvm::IntervalMapInfo<unsigned>>;
  using Allocator = RangeSet::Allocator;

  LiveRange(Allocator &allocator)
      : Ranges(std::make_unique<RangeSet>(allocator)) {}

  void unionWith(LiveRange const &Other) {
    for (auto It = Other.Ranges->begin(); It != Other.Ranges->end(); ++It)
      Ranges->insert(It.start(), It.stop(), It.value());
  }

  bool overlaps(LiveRange const &Other) const {
    return llvm::IntervalMapOverlaps<RangeSet, RangeSet>(*Ranges, *Other.Ranges)
        .valid();
  }

  Value *findLiveValue(uint64_t Point) const { return Ranges->lookup(Point); }

  bool overlaps(uint64_t Start, uint64_t End) {
    return Start <= End && Ranges->overlaps(Start, End);
  }

  void insert(uint64_t Start, uint64_t End, Value *V) {
    if (Start <= End)
      Ranges->insert(Start, End, V);
  }

  bool empty() const { return Ranges->empty(); }
  uint64_t start() const { return Ranges->start(); }
  uint64_t end() const { return Ranges->stop(); }

  std::unique_ptr<RangeSet> Ranges;
};

class TypeLiveness {
public:
  struct BlockInfo {
    using ValueSet = SmallPtrSet<Value *, 8>;

    BlockInfo() = default;

    BlockInfo(BasicBlock *Block, Type *Type) : Block(Block) {
      // TODO: Need to pre-process the IR to handle PHI nodes correctly
      // (as their uses don't make sense for live ranges -- really the
      // arguments are copies/uses in a predecessor).
      // -> split cond blocks -> insert uses before branches
      // (then ignore PHI users elsewhere)
      for (Instruction &Inst : *Block) {
        // Collect uses of `Type`.
        // Note: We ignore phi uses -- these require preprocessing as they are
        // not dominated by the definition.
        if (!isa<PHINode>(Inst)) {
          for (size_t I = 0, N = Inst.getNumOperands(); I < N; ++I) {
            auto *Op = Inst.getOperand(I);
            if (Op->getType() != Type)
              continue;
            UseVals.insert(Op);
          }
        }

        if (Inst.getType() != Type)
          continue;

        // Collect definitions of `Type`.
        auto *Def = cast<Value>(&Inst);
        DefVals.insert(Def);
        // Collect out values for the current block.
        for (auto *User : Def->users()) {
          auto UserInst = cast<Instruction>(User);
          if (!isa<PHINode>(UserInst) && UserInst->getParent() != Block)
            LiveOut.insert(Def);
        }
      }
      set_subtract(UseVals, DefVals);
    }

    bool updateLiveIn() {
      ValueSet NewIn = UseVals;
      set_union(NewIn, LiveOut);
      set_subtract(NewIn, DefVals);

      if (NewIn.size() == LiveIn.size())
        return false;

      LiveIn = std::move(NewIn);
      return true;
    }

    void
    updateLiveOut(const DenseMap<const BasicBlock *, BlockInfo> &BlockInfoMap) {
      for (const BasicBlock *Succ : successors(Block)) {
        const BlockInfo &Info = BlockInfoMap.at(Succ);
        set_union(LiveOut, Info.LiveIn);
      }
    }

    const BasicBlock *Block{nullptr};
    ValueSet LiveIn;
    ValueSet LiveOut;
    ValueSet DefVals;
    ValueSet UseVals;
  };

  // A map of basic block to liveness information.
  DenseMap<const BasicBlock *, BlockInfo> BlockInfoMap;
  // An order for instructions based on dominance.
  DenseMap<const Instruction *, unsigned> InstructionOrder;

public:
  TypeLiveness(Function *F, Type *Type) {
    unsigned NextInstructionId = 0;
    SetVector<const BasicBlock *> Worklist;
    for (auto *Block : depth_first(F)) {
      auto &Info = BlockInfoMap.try_emplace(Block, Block, Type).first->second;
      if (Info.updateLiveIn())
        Worklist.insert(pred_begin(Block), pred_end(Block));
      for (Instruction &Inst : *Block)
        InstructionOrder.try_emplace(&Inst, NextInstructionId++);
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

  const Instruction *getEndInstruction(const BlockInfo &Info, Value *V,
                                       Instruction *Start) {
    if (Info.LiveOut.contains(V))
      return &Info.Block->back();
    if (isa<Constant>(V))
      return Start;
    const Instruction *End = Start;
    for (auto User : V->users()) {
      auto &Inst = cast<Instruction>(*User);
      if (!isa<PHINode>(Inst) && Inst.getParent() == Info.Block &&
          InstructionOrder.at(End) < InstructionOrder.at(&Inst))
        End = &Inst;
    }
    return End;
  }
};

// We cannot compute live ranges of phi uses directly as the operands of phi
// nodes do not necessarily dominate the phi. Instead, we model ZA phi's by
// inserting "za.phi" blocks along predecessor edges. These blocks contain a
// single ssa.copy instruction of the phi's operand -- the phi is then updated
// to use the ssa.copy (and we then ignore phi uses for live range calculation).
// After this pass, the "za.phi" blocks will either be empty (and optimize
// away), or contain a ZA restore.
static bool preprocessPhisForZASaveRestore(DominatorTree &DT, Module *M,
                                           Function *F, Type *ZaType,
                                           IRBuilder<> &Builder) {
  SmallVector<PHINode *> ZAPhis;
  for (BasicBlock &Block : *F) {
    for (PHINode &Phi : Block.phis()) {
      if (Phi.getType() != ZaType)
        continue;
      ZAPhis.push_back(&Phi);
    }
  }
  Function *SSACopy =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::ssa_copy, ZaType);
  DomTreeUpdater DTU(&DT, DomTreeUpdater::UpdateStrategy::Eager);
  for (PHINode *Phi : ZAPhis) {
    auto *PhiBlock = Phi->getParent();
    unsigned OpIdx = 0;
    for (auto [Predecessor, V] :
         zip_equal(Phi->blocks(), Phi->incoming_values())) {
      auto *PredBlock = Predecessor;
      // Update IR:
      auto *CopyBlock =
          BasicBlock::Create(F->getContext(), "za.phi", F, PhiBlock);
      Builder.SetInsertPoint(CopyBlock);
      auto *Copy =
          Builder.CreateCall(SSACopy->getFunctionType(), SSACopy, V.get());
      Builder.CreateBr(PhiBlock);
      PredBlock->getTerminator()->replaceSuccessorWith(PhiBlock, CopyBlock);
      PhiBlock->replacePhiUsesWith(PredBlock, CopyBlock);
      Phi->setOperand(OpIdx, Copy);
      // Update dominator tree:
      DTU.applyUpdates({{DominatorTree::Delete, PredBlock, PhiBlock},
                        {DominatorTree::Insert, PredBlock, CopyBlock},
                        {DominatorTree::Insert, CopyBlock, PhiBlock}});
      ++OpIdx;
    }
  }
  assert(DT.verify());
  return ZAPhis.size() > 0;
}

static Value *emitLazySaveBuffer(Module *M, Function *F, IRBuilder<> &Builder) {
  Builder.SetInsertPoint(&F->getEntryBlock().front());
  Function *ReadSVLIntr =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::aarch64_sme_cntsb);

  Value *SVL = Builder.CreateCall(ReadSVLIntr->getFunctionType(), ReadSVLIntr);
  Value *BufferSize = Builder.CreateMul(SVL, SVL, "za.buffer.size",
                                        /*HasNUW=*/true, /*HasNSW=*/true);

  Type *I8Type = Builder.getInt8Ty();
  Type *I64Type = Type::getInt64Ty(F->getContext());
  AllocaInst *Buffer = Builder.CreateAlloca(I8Type, BufferSize, "za.buffer");
  Buffer->setAlignment(Align(16));

  Value *TPIDR2Block = Builder.CreateAlloca(I64Type, Builder.getInt64(2));
  Value *TPIDR2Meta = Builder.CreatePtrAdd(TPIDR2Block, Builder.getInt64(8));

  // NumSaveSlices is a 16-bit value (stored as an i64 for implicit zeroing).
  Value *NumSaveSlices = SVL;
  if (M->getDataLayout().isBigEndian())
    NumSaveSlices = Builder.CreateShl(NumSaveSlices, 48);

  Builder.CreateStore(Buffer, TPIDR2Block);
  Builder.CreateStore(NumSaveSlices, TPIDR2Meta);

  return TPIDR2Block;
}

static Value *emitFullZASaveBuffer(Module *M, Function *F,
                                   IRBuilder<> &Builder) {
  Builder.SetInsertPoint(&F->getEntryBlock().front());
  auto &Ctx = M->getContext();
  auto *SMEStateSizeTy =
      FunctionType::get(Builder.getInt64Ty(), {}, /*IsVarArgs=*/false);
  auto Attrs =
      AttributeList().addFnAttribute(Ctx, "aarch64_pstate_sm_compatible");
  FunctionCallee SMEStateSizeDecl =
      M->getOrInsertFunction("__arm_sme_state_size", SMEStateSizeTy, Attrs);

  CallBase *StateSize = Builder.CreateCall(SMEStateSizeDecl);
  StateSize->setCallingConv(
      CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X1);

  Type *I8Type = Builder.getInt8Ty();
  AllocaInst *Buffer = Builder.CreateAlloca(I8Type, StateSize, "za.buffer");
  Buffer->setAlignment(Align(16));

  return Buffer;
}

static void emitLazySaveZAState(Module *M, IRBuilder<> &Builder,
                                Value *TPIDR2Block) {
  Function *SetTPIDR2Intr =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::aarch64_sme_set_tpidr2);
  Builder.CreateCall(
      SetTPIDR2Intr->getFunctionType(), SetTPIDR2Intr,
      {Builder.CreatePtrToInt(TPIDR2Block, Builder.getInt64Ty())});
}

static void emitFullSaveRestoreZAState(Module *M, IRBuilder<> &Builder,
                                       Value *Buffer, bool IsSave) {
  auto &Ctx = M->getContext();
  auto *CalleeTy = FunctionType::get(Builder.getVoidTy(), {Builder.getPtrTy()},
                                     /*IsVarArgs=*/false);
  auto Attrs =
      AttributeList().addFnAttribute(Ctx, "aarch64_pstate_sm_compatible");
  FunctionCallee CalleeDecl = M->getOrInsertFunction(
      IsSave ? "__arm_sme_save" : "__arm_sme_restore", CalleeTy, Attrs);

  CallBase *Call = Builder.CreateCall(CalleeDecl, {Buffer});
  Call->setCallingConv(
      CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X1);
}

static void emitZAOffAroundClobber(Module *M, Instruction *Clobber,
                                   IRBuilder<> &Builder) {
  Value *Null = Builder.getInt64(0);
  Function *DisableZAIntr =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::aarch64_sme_za_disable);
  Function *EnableZAIntr =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::aarch64_sme_za_enable);
  Builder.SetInsertPoint(Clobber);
  emitLazySaveZAState(M, Builder, Null);
  Builder.CreateCall(DisableZAIntr->getFunctionType(), DisableZAIntr);
  Builder.SetInsertPoint(Clobber->getNextNode());
  Builder.CreateCall(EnableZAIntr->getFunctionType(), EnableZAIntr);
}

static void emitLazyRestoreZAState(Module *M, Function *F, IRBuilder<> &Builder,
                                   Value *TPIDR2Block) {
  auto &Ctx = M->getContext();
  auto *TPIDR2RestoreTy = FunctionType::get(
      Builder.getVoidTy(), {Builder.getPtrTy()}, /*IsVarArgs=*/false);
  auto Attrs =
      AttributeList().addFnAttribute(Ctx, "aarch64_pstate_sm_compatible");
  FunctionCallee RestoreDecl =
      M->getOrInsertFunction("__arm_tpidr2_restore", TPIDR2RestoreTy, Attrs);

  Function *GetTPIDR2Intr =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::aarch64_sme_get_tpidr2);
  Function *SetTPIDR2Intr =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::aarch64_sme_set_tpidr2);
  Function *EnableZAIntr =
      Intrinsic::getOrInsertDeclaration(M, Intrinsic::aarch64_sme_za_enable);

  Value *TPIDR2 =
      Builder.CreateCall(GetTPIDR2Intr->getFunctionType(), GetTPIDR2Intr);
  Value *IsZero = Builder.CreateICmpEQ(TPIDR2, Builder.getInt64(0));
  Builder.CreateCall(EnableZAIntr->getFunctionType(), EnableZAIntr);

  auto *CurrentBB = Builder.GetInsertBlock();
  auto *AfterRestore =
      CurrentBB->splitBasicBlock(Builder.GetInsertPoint(), "after.restore.za");
  BasicBlock *RestoreZA =
      BasicBlock::Create(F->getContext(), "restore.za", F, AfterRestore);

  auto *PrevBR = CurrentBB->getTerminator();
  Builder.SetInsertPoint(PrevBR);
  Builder.CreateCondBr(IsZero, RestoreZA, AfterRestore);
  PrevBR->eraseFromParent();

  Builder.SetInsertPoint(RestoreZA);
  CallBase *RestoreCall = Builder.CreateCall(RestoreDecl, {TPIDR2Block});
  RestoreCall->setCallingConv(
      CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X1);
  Builder.CreateBr(AfterRestore);

  Builder.SetInsertPoint(&AfterRestore->front());
  Builder.CreateCall(SetTPIDR2Intr->getFunctionType(), SetTPIDR2Intr,
                     {Builder.getInt64(0)});
}

static Value *emitZASaveBuffer(Module *M, Function *F, IRBuilder<> &Builder,
                               SMEAttrs FnAttrs) {
  if (FnAttrs.hasZAState())
    return emitLazySaveBuffer(M, F, Builder);
  if (FnAttrs.hasAgnosticZAInterface())
    return emitFullZASaveBuffer(M, F, Builder);
  llvm_unreachable("Don't know how to allocate ZA save buffer");
}

static void emitSaveZAState(Module *M, Function *F, IRBuilder<> &Builder,
                            Value *Buffer, SMEAttrs FnAttrs) {
  bool IsLazySave = FnAttrs.hasZAState();
  OptimizationRemarkEmitter ORE(F);
  ORE.emit([&] {
    auto Clobber = Builder.GetInsertPoint();
    OptimizationRemarkAnalysis R("sme", "SMEABI", &*Clobber);
    R << "in function '" << F->getName() << "' "
      << (IsLazySave ? "lazy save for ZA" : "full ZA save")
      << " required before ";
    if (CallBase *Call = dyn_cast<CallBase>(Clobber))
      if (auto *Callee = Call->getCalledFunction())
        R << "call to '" << Callee->getName() << "'";
      else
        R << "indirect function call";
    else
      R << "clobber";
    return R;
  });
  if (IsLazySave)
    return emitLazySaveZAState(M, Builder, Buffer);
  if (FnAttrs.hasAgnosticZAInterface())
    return emitFullSaveRestoreZAState(M, Builder, Buffer, /*IsSave=*/true);
  llvm_unreachable("Don't know how to save ZA state");
}

static void emitRestoreZAState(Module *M, Function *F, IRBuilder<> &Builder,
                               Value *Buffer, SMEAttrs FnAttrs) {
  if (FnAttrs.hasZAState())
    return emitLazyRestoreZAState(M, F, Builder, Buffer);
  if (FnAttrs.hasAgnosticZAInterface())
    return emitFullSaveRestoreZAState(M, Builder, Buffer, /*IsSave=*/false);
  llvm_unreachable("Don't know how to restore ZA state");
}

static bool eraseZALivenessAnnotations(Function *F, Type *ZaType) {
  bool Changed = false;
  auto *UndefZA = UndefValue::get(ZaType);
  auto IsZAType = [&](Value *V) { return V->getType() == ZaType; };
  for (BasicBlock &Block : *F) {
    for (Instruction &Inst : make_early_inc_range(Block)) {
      if (IsZAType(&Inst) || any_of(Inst.operands(), IsZAType)) {
        for (Use &U : make_early_inc_range(Inst.uses()))
          U.set(UndefZA);
        [[maybe_unused]] bool IsLoadOrStore = isa<LoadInst, StoreInst>(Inst);
        assert(!IsLoadOrStore &&
               "ZA loads and stores should have been eliminated");
        Inst.eraseFromParent();
        Changed |= true;
      }
    }
  }
  return Changed;
}

static bool insertZASavesAndRestores(Module *M, Function *F,
                                     IRBuilder<> &Builder,
                                     bool EnableZALiveness) {
  // TODO: Exit early if there are no clobbers.
  bool Changed = false;
  Type *ZaType = TargetExtType::get(F->getContext(), "aarch64.za.generation");
  // TODO: Look up existing dominator tree?
  DominatorTree DT(*F);

  {
    // Eliminate any allocas of ZA annotations at this point. For -O1 and above
    // this will have already been done by now, but we still need to do this for
    // the -O0 case.
    SmallVector<AllocaInst *> ZaAllocas;
    auto &EntryBlock = F->getEntryBlock();
    for (BasicBlock::iterator I = EntryBlock.begin(), E = EntryBlock.end();
         I != E; ++I) {
      if (AllocaInst *AI = dyn_cast<AllocaInst>(I);
          AI && AI->getAllocatedType() == ZaType)
        ZaAllocas.push_back(AI);
    }

    if (!ZaAllocas.empty()) {
      PromoteMemToReg(ZaAllocas, DT);
      Changed |= true;
    }
  }

  if (!EnableZALiveness) {
    Changed |= eraseZALivenessAnnotations(F, ZaType);
    return Changed;
  }

  // We need to pre-process phis to correctly compute their liveness, since a
  // phi is a copy in a predecessor, its uses cannot be considered live-in.
  Changed |= preprocessPhisForZASaveRestore(DT, M, F, ZaType, Builder);

  TypeLiveness Liveness(F, ZaType);
  LiveRange::Allocator LiveRangeAllocator;
  LiveRange ZALiveness(LiveRangeAllocator);
  auto defineOrUpdateValueLiveRange = [&](Value *V, Instruction *FirstUseOrDef,
                                          TypeLiveness::BlockInfo const &Info,
                                          bool Def = false) {
    // Find or create a live range for `value`.
    auto LastUseInBlock = Liveness.getEndInstruction(Info, V, FirstUseOrDef);
    unsigned Start =
        Liveness.InstructionOrder.at(FirstUseOrDef) + (Def ? 1 : 0);
    unsigned End = Liveness.InstructionOrder.at(LastUseInBlock);
    if (ZALiveness.overlaps(Start, End))
      reportFatalUsageError(
          "Expected at most one live AArch64 SME ZA value at any point!");
    ZALiveness.insert(Start, End, V);
  };

  // Compute live ranges for ZA state and collect clobbers.
  SmallVector<CallBase *> Clobbers;
  for (BasicBlock &Block : *F) {
    auto &Info = Liveness.getBlockLiveness(&Block);
    for (Value *LiveIn : Info.LiveIn) {
      assert(!isa<Constant>(LiveIn) && "Constant ZA values are not supported");
      defineOrUpdateValueLiveRange(LiveIn, &Block.front(), Info);
    }

    for (Instruction &Inst : Block) {
      if (auto *Call = dyn_cast<CallBase>(&Inst)) {
        if (!isa<IntrinsicInst>(Call) && SMECallAttrs(*Call).clobbersZAState())
          Clobbers.push_back(Call);
      }
      if (Inst.getType() != ZaType)
        continue;
      Value *Def = cast<Value>(&Inst);
      defineOrUpdateValueLiveRange(Def, &Inst, Info, true);
    }
  }

  SmallVector<CallBase *> OffPoints;
  SmallPtrSet<Instruction *, 8> SavePoints, ReloadPoints;

  // Sort clobbers by dominance.
  sort(Clobbers, [&](auto *A, auto *B) {
    return Liveness.InstructionOrder.at(A) < Liveness.InstructionOrder.at(B);
  });

  for (auto *Clobber : Clobbers) {
    unsigned ClobberPoint = Liveness.InstructionOrder.at(Clobber);
    Value *V = ZALiveness.findLiveValue(ClobberPoint);
    if (!V) {
      OffPoints.push_back(Clobber);
      continue;
    }

    unsigned MinReloadLevel = 0;
    DenseMap<BasicBlock *, unsigned> BlockToMinReloadIndex;
    SmallVector<Instruction *> ReloadCandidates;
    for (auto *User : V->users()) {
      auto *Inst = cast<Instruction>(User);
      if (!isPotentiallyReachable(Clobber, Inst,
                                  /*ExclusionSet=*/nullptr, &DT)) {
        continue;
      }
      // Phi's only use llvm.ssa.copy's of ZA which are live between the
      // "za.phi" blocks and the phi. It should not be possible for a clobber to
      // occur for a phi operand.
      assert(!isa<PHINode>(Inst) &&
             "Did not expect phi's operand to be clobbered");
      unsigned ReloadIndex = Liveness.InstructionOrder.at(Inst);
      auto *ReloadBlock = Inst->getParent();
      auto [It, Inserted] = BlockToMinReloadIndex.insert(
          std::make_pair(ReloadBlock, ReloadIndex));
      if (!Inserted)
        It->second = std::min(It->second, ReloadIndex);
      unsigned ReloadLevel = DT.getNode(ReloadBlock)->getLevel();
      if (ReloadCandidates.empty())
        MinReloadLevel = ReloadLevel;
      else
        MinReloadLevel = std::min(ReloadLevel, MinReloadLevel);
      ReloadCandidates.push_back(Inst);
    }

    // Setup a ZA save just after the definition of the current ZA state.
    Instruction *SavePoint = cast<Instruction>(V)->getNextNode();
    if (isa<PHINode>(SavePoint))
      SavePoint = &*SavePoint->getParent()->getFirstNonPHIIt();
    SavePoints.insert(SavePoint);

    for (auto *Candidate : ReloadCandidates) {
      bool IsDominatedByReload = false;
      BasicBlock *CandidateBlock = Candidate->getParent();
      BasicBlock *Block = CandidateBlock;
      unsigned ReloadIndex = Liveness.InstructionOrder.at(Candidate);
      while (Block) {
        // Check for any other reloads that might dominate this reload. If this
        // reload is dominated by another, we can ignore this candidate.
        auto It = BlockToMinReloadIndex.find(Block);
        if (It != BlockToMinReloadIndex.end()) {
          if (CandidateBlock != Block || ReloadIndex > It->second) {
            IsDominatedByReload = true;
            break;
          }
        }
        auto *DomNode = DT.getNode(Block);
        auto *IDom = DomNode->getIDom();
        if (!IDom || IDom->getLevel() < MinReloadLevel)
          break;
        Block = IDom->getBlock();
      }
      if (IsDominatedByReload)
        continue;
      ReloadPoints.insert(Candidate);
    }
  }

  LLVM_DEBUG({
    dbgs() << "========== @" << F->getName() << ": ZA Liveness\n"
           << "Key:\n"
           << "| - Live ZA value\n\n";
    for (BasicBlock &Block : *F) {
      dbgs() << Block.getNameOrAsOperand() << ":\n";
      for (Instruction &Inst : Block) {
        unsigned Index = Liveness.InstructionOrder.at(&Inst);
        dbgs() << (ZALiveness.findLiveValue(Index) ? '|' : ' ');
        Inst.dump();
      }
      dbgs() << "==========\n";
    }
  });

  Changed |= !OffPoints.empty() || !SavePoints.empty();

  // "OffPoints" are clobbers that occurred where ZA is not live. At these
  // points we need to ensure ZA state is off before the call and then re-enable
  // it after the call. FIXME: This may emit back-to-back SMSTOP/START ZA pairs.
  // Note: These should not exist in shared or agnostic ZA functions emitted by
  // Clang (but could occur in hand-crafted IR or private ZA functions).
  for (CallBase *Clobber : OffPoints)
    emitZAOffAroundClobber(M, Clobber, Builder);

  if (!SavePoints.empty()) {
    SMEAttrs FnAttrs(*F);
    Value *Buffer = emitZASaveBuffer(M, F, Builder, FnAttrs);

    for (auto *SavePoint : SavePoints) {
      Builder.SetInsertPoint(SavePoint);
      emitSaveZAState(M, F, Builder, Buffer, FnAttrs);
    }

    for (auto *Restore : ReloadPoints) {
      Builder.SetInsertPoint(Restore);
      emitRestoreZAState(M, F, Builder, Buffer, FnAttrs);
    }
  }

  Changed |= eraseZALivenessAnnotations(F, ZaType);
  return Changed;
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

  return true;
}

bool SMEABI::runOnFunction(Function &F) {
  Module *M = F.getParent();
  LLVMContext &Context = F.getContext();
  IRBuilder<> Builder(Context);

  if (F.isDeclaration() || F.hasFnAttribute("aarch64_expanded_pstate_za"))
    return false;

  bool Changed = false;
  SMEAttrs FnAttrs(F);

  bool EnableZALiveness = AArch64TargetMachine::usesZALiveness();
  if (FnAttrs.hasZAState() || FnAttrs.hasAgnosticZAInterface())
    Changed |= insertZASavesAndRestores(M, &F, Builder, EnableZALiveness);

  if (FnAttrs.isNewZA() || FnAttrs.isNewZT0())
    Changed |= updateNewStateFunctions(M, &F, Builder, FnAttrs);

  if (Changed)
    F.addFnAttr("aarch64_expanded_pstate_za");

  return Changed;
}
