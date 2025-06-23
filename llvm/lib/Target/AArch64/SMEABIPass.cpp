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
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-sme-abi"

namespace {

struct ValueRange {
  using RangeSet =
      llvm::IntervalMap<uint64_t, Value *, 16, llvm::IntervalMapInfo<unsigned>>;
  using Allocator = RangeSet::Allocator;

  ValueRange(Allocator &allocator)
      : Ranges(std::make_unique<RangeSet>(allocator)) {}

  void unionWith(ValueRange const &Other) {
    for (auto It = Other.Ranges->begin(); It != Other.Ranges->end(); ++It)
      Ranges->insert(It.start(), It.stop(), It.value());
  }

  bool overlaps(ValueRange const &Other) const {
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

/// Does this intrinsic update ZA state? E.g. Load a tile slice.
/// TODO: Somehow table-generate this?
static bool isZAUpdate(Intrinsic::ID IID) {
  switch (IID) {
  default:
    return false;
  // Tile loads:
  case Intrinsic::aarch64_sme_ld1b_horiz:
  case Intrinsic::aarch64_sme_ld1b_vert:
  case Intrinsic::aarch64_sme_ld1h_horiz:
  case Intrinsic::aarch64_sme_ld1h_vert:
  case Intrinsic::aarch64_sme_ld1w_horiz:
  case Intrinsic::aarch64_sme_ld1w_vert:
  case Intrinsic::aarch64_sme_ld1d_horiz:
  case Intrinsic::aarch64_sme_ld1d_vert:
  case Intrinsic::aarch64_sme_ld1q_horiz:
  case Intrinsic::aarch64_sme_ld1q_vert:
  // ZA fill:
  case Intrinsic::aarch64_sme_ldr:
  // Vector to tile:
  case Intrinsic::aarch64_sme_write_horiz:
  case Intrinsic::aarch64_sme_write_vert:
  case Intrinsic::aarch64_sme_writeq_horiz:
  case Intrinsic::aarch64_sme_writeq_vert:
  // Zero:
  case Intrinsic::aarch64_sme_zero:
  // MOPA:
  case Intrinsic::aarch64_sme_mopa:
  case Intrinsic::aarch64_sme_mops:
  case Intrinsic::aarch64_sme_mopa_wide:
  case Intrinsic::aarch64_sme_mops_wide:
  case Intrinsic::aarch64_sme_smopa_wide:
  case Intrinsic::aarch64_sme_smops_wide:
  case Intrinsic::aarch64_sme_umopa_wide:
  case Intrinsic::aarch64_sme_umops_wide:
  case Intrinsic::aarch64_sme_usmopa_wide:
  case Intrinsic::aarch64_sme_usmops_wide:
  case Intrinsic::aarch64_sme_sumopa_wide:
  case Intrinsic::aarch64_sme_sumops_wide:
    return true;
    // TODO: Finish list...
  }
}

/// Does this intrinsic use/read ZA state? E.g. Store a tile slice.
/// TODO: Somehow table-generate this?
static bool isZAUse(Intrinsic::ID IID) {
  switch (IID) {
  default:
    return false;
    // Tile stores:
  case Intrinsic::aarch64_sme_st1b_horiz:
  case Intrinsic::aarch64_sme_st1b_vert:
  case Intrinsic::aarch64_sme_st1h_horiz:
  case Intrinsic::aarch64_sme_st1h_vert:
  case Intrinsic::aarch64_sme_st1w_horiz:
  case Intrinsic::aarch64_sme_st1w_vert:
  case Intrinsic::aarch64_sme_st1d_horiz:
  case Intrinsic::aarch64_sme_st1d_vert:
  case Intrinsic::aarch64_sme_st1q_horiz:
  case Intrinsic::aarch64_sme_st1q_vert:
    // ZA spill:
  case Intrinsic::aarch64_sme_str:
  // Tile to vector:
  case Intrinsic::aarch64_sme_read_horiz:
  case Intrinsic::aarch64_sme_read_vert:
  case Intrinsic::aarch64_sme_readq_horiz:
  case Intrinsic::aarch64_sme_readq_vert:
  case Intrinsic::aarch64_sme_readz_horiz_x2:
  case Intrinsic::aarch64_sme_readz_vert_x2:
  case Intrinsic::aarch64_sme_readz_horiz_x4:
  case Intrinsic::aarch64_sme_readz_vert_x4:
  case Intrinsic::aarch64_sme_readz_horiz:
  case Intrinsic::aarch64_sme_readz_vert:
  case Intrinsic::aarch64_sme_readz_q_horiz:
  case Intrinsic::aarch64_sme_readz_q_vert:
  case Intrinsic::aarch64_sme_readz_x2:
  case Intrinsic::aarch64_sme_readz_x4:
    return true;
    // TODO: Finish list...
  }
}

static bool usesZAState(Instruction *Inst) {
  if (auto *Intr = dyn_cast<IntrinsicInst>(Inst)) {
    Intrinsic::ID IID = Intr->getIntrinsicID();
    return isZAUpdate(IID) || isZAUse(IID);
  }
  auto *CallInst = dyn_cast<CallBase>(Inst);
  return CallInst && !SMECallAttrs(*CallInst).clobbersZAState();
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

static bool insertZASavesAndRestores(Module *M, Function *F,
                                     IRBuilder<> &Builder) {
  unsigned NextInstructionId = 0;
  SmallVector<CallBase *> Clobbers;
  DenseMap<const Instruction *, unsigned> InstructionOrder;
  for (auto *Block : depth_first(F)) {
    for (Instruction &Inst : *Block) {
      if (auto *Call = dyn_cast<CallBase>(&Inst)) {
        if (!isa<IntrinsicInst>(Call) && SMECallAttrs(*Call).clobbersZAState())
          Clobbers.push_back(Call);
      }
      InstructionOrder.try_emplace(&Inst, NextInstructionId++);
    }
  }

  if (Clobbers.empty())
    return false;

  sort(Clobbers, [&](auto *A, auto *B) {
    return InstructionOrder.at(A) < InstructionOrder.at(B);
  });

  ValueRange::Allocator Allocator;
  ValueRange ClobberedRange(Allocator);
  SmallPtrSet<Instruction *, 8> SavePoints, ReloadPoints;
  SmallSet<std::pair<BasicBlock *, BasicBlock *>, 8> ReloadEdges;

  for (auto *Clobber : Clobbers) {
    unsigned ClobberPoint = InstructionOrder.at(Clobber);
    if (ClobberedRange.findLiveValue(ClobberPoint))
      continue;

    SavePoints.insert(Clobber);

    SmallVector<Instruction *> Worklist;
    Worklist.push_back(Clobber);

    while (!Worklist.empty()) {
      auto *StartInst = Worklist.pop_back_val();
      auto *Block = StartInst->getParent();
      unsigned StartPoint = InstructionOrder.at(StartInst);

      llvm::BasicBlock::iterator Start(StartInst);
      Instruction *ReloadPoint = nullptr;
      for (auto It = Start; It != Block->end(); ++It) {
        Instruction *Inst = &*It;
        if (!usesZAState(Inst))
          continue;

        ReloadPoint = Inst;
        break;
      }

      unsigned EndPoint =
          InstructionOrder.at(ReloadPoint ? ReloadPoint : &Block->back());

      ClobberedRange.insert(StartPoint, EndPoint, Clobber);

      if (ReloadPoint) {
        ReloadPoints.insert(ReloadPoint);
        continue;
      }

      // Avoid increasing code-size too much if all edges need a reload.
      auto PreferReloadInCurrentBlock = [&] {
        return all_of(successors(Block), [](BasicBlock *Succ) {
          return !Succ->getSinglePredecessor() ||
                 usesZAState(&*Succ->getFirstNonPHIIt());
        });
      };

      if (succ_empty(Block) ||
          (&Block->back() != Clobber && PreferReloadInCurrentBlock())) {
        ReloadPoints.insert(&Block->back());
        continue;
      }

      for (auto *Succ : successors(Block)) {
        if (!Succ->getSinglePredecessor()) {
          ReloadEdges.insert({Block, Succ});
        } else {
          Worklist.push_back(&Succ->front());
        }
      }
    }
  }

  if (SavePoints.empty())
    return false;

  LLVM_DEBUG({
    dbgs() << "========== @" << F->getName() << ": ZA Clobbers\n"
           << "Key:\n"
           << "x - Clobbered ZA value\n\n";
    for (BasicBlock &Block : *F) {
      dbgs() << Block.getNameOrAsOperand() << ":\n";
      for (Instruction &Inst : Block) {
        unsigned Index = InstructionOrder.at(&Inst);
        dbgs() << (ClobberedRange.findLiveValue(Index) ? 'x' : ' ');
        Inst.dump();
      }
      dbgs() << "==========\n";
    }
  });

  SmallPtrSet<BasicBlock *, 8> PredNeedsReload;
  SmallPtrSet<LandingPadInst *, 8> LandingPads;
  for (auto [Pred, Succ] : ReloadEdges) {
    if (auto *LandingPad =
            dyn_cast_or_null<LandingPadInst>(Succ->getFirstNonPHIIt())) {
      PredNeedsReload.insert(Pred);
      LandingPads.insert(LandingPad);
      continue;
    }
    auto *ReloadBlock = ehAwareSplitEdge(Pred, Succ);
    ReloadPoints.insert(ReloadBlock->getTerminator());
  }
  // EH edges to landing pads are unfortunately quite awkward to spilt. We need
  // to split the edges from every predecessor, not just the edge we wish to
  // reload at.
  for (LandingPadInst *LandingPad : LandingPads) {
    BasicBlock *Succ = LandingPad->getParent();
    PHINode *ReplPHI = PHINode::Create(LandingPad->getType(), 1, "");
    ReplPHI->insertBefore(LandingPad->getIterator());
    ReplPHI->takeName(LandingPad);
    LandingPad->replaceAllUsesWith(ReplPHI);

    for (BasicBlock *Pred : make_early_inc_range(predecessors(Succ))) {
      auto *SplitBB = ehAwareSplitEdge(Pred, Succ, LandingPad, ReplPHI);
      if (PredNeedsReload.contains(Pred))
        ReloadPoints.insert(SplitBB->getTerminator());
    }

    LandingPad->eraseFromParent();
  }

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

  return true;
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

  if (AArch64TargetMachine::hasGlobalZASaveRestore() &&
      (FnAttrs.hasZAState() || FnAttrs.hasAgnosticZAInterface()))
    Changed |= insertZASavesAndRestores(M, &F, Builder);

  if (FnAttrs.isNewZA() || FnAttrs.isNewZT0())
    Changed |= updateNewStateFunctions(M, &F, Builder, FnAttrs);

  if (Changed)
    F.addFnAttr("aarch64_expanded_pstate_za");

  return Changed;
}
