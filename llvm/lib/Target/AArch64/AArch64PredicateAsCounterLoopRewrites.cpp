//===- AArch64PredicateAsCounterLoopRewrites.cpp --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Prototype IR rewrite for loop-carried wide masks that can be represented as
// predicate-as-counter values on AArch64.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64Subtarget.h"
#include "AArch64TargetMachine.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Local.h"
#include <optional>

using namespace llvm;

#define DEBUG_TYPE "aarch64-predicate-as-counter-loop-rewrites"
namespace {

STATISTIC(LoopsRewritten, "Number of loops rewritten");

struct MaskRewriteCandidate {
  BasicBlock *Preheader = nullptr;
  BasicBlock *Latch = nullptr;
  PHINode *MaskPhi = nullptr;
  IntrinsicInst *StartMask = nullptr;
  IntrinsicInst *NextMask = nullptr;
  unsigned Scale = 0;
  unsigned ElementSizeInBytes = 0;
  unsigned LegalLanes = 0;
};

static bool isScalableMaskType(Type *Ty) {
  auto *SVTy = dyn_cast<ScalableVectorType>(Ty);
  return SVTy && SVTy->getElementType()->isIntegerTy(1);
}

static IntrinsicInst *getGetActiveLaneMask(Value *V) {
  auto *II = dyn_cast<IntrinsicInst>(V);
  return II && II->getIntrinsicID() == Intrinsic::get_active_lane_mask_for_type
             ? II
             : nullptr;
}

static unsigned getMaskElementCount(unsigned ElementSizeInBytes) {
  return 16 / ElementSizeInBytes;
}

static Intrinsic::ID getWhileLOIntrinsic(unsigned ElementSizeInBytes) {
  switch (ElementSizeInBytes) {
  case 1:
    return Intrinsic::aarch64_sve_whilelo_c8;
  case 2:
    return Intrinsic::aarch64_sve_whilelo_c16;
  case 4:
    return Intrinsic::aarch64_sve_whilelo_c32;
  case 8:
    return Intrinsic::aarch64_sve_whilelo_c64;
  default:
    llvm_unreachable("unsupported predicate-as-counter element size");
  }
}

static Value *buildWideMask(IRBuilder<> &Builder, const MaskRewriteCandidate &C,
                            Value *Count) {
  Module *M = Builder.GetInsertBlock()->getModule();
  Type *LegalMaskTy =
      ScalableVectorType::get(Builder.getInt1Ty(), C.LegalLanes);
  FunctionCallee PExtX2 = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::aarch64_sve_pext_x2, {LegalMaskTy});

  Value *WideMask = PoisonValue::get(C.MaskPhi->getType());
  for (unsigned PairOffset = 0; PairOffset != C.Scale / 2; ++PairOffset) {
    auto *Pair = Builder.CreateCall(
        PExtX2, {Count, Builder.getInt32(PairOffset)}, "pac.pext.pair");
    for (unsigned SliceInPair = 0; SliceInPair != 2; ++SliceInPair) {
      Value *Part = Builder.CreateExtractValue(Pair, SliceInPair, "pac.pext");
      unsigned Slice = PairOffset * 2 + SliceInPair;
      WideMask =
          Builder.CreateInsertVector(C.MaskPhi->getType(), WideMask, Part,
                                     Slice * C.LegalLanes, "pac.mask");
    }
  }

  return WideMask;
}

static bool hasOnlySupportedPhiUsers(Value *V, PHINode *AllowedPhi = nullptr) {
  for (User *U : V->users())
    if (auto *PN = dyn_cast<PHINode>(U); PN && PN != AllowedPhi)
      return false;
  return true;
}

static Value *createWhileLO(IRBuilder<> &Builder, unsigned MaskElemType,
                            Value *Start, Value *End, unsigned Scale) {
  if (Start->getType()->getIntegerBitWidth() < 64) {
    Start = Builder.CreateZExt(Start, Builder.getInt64Ty());
    End = Builder.CreateZExt(End, Builder.getInt64Ty());
  }
  Module *M = Builder.GetInsertBlock()->getModule();
  auto ID = getWhileLOIntrinsic(MaskElemType);
  FunctionCallee WhileLT = Intrinsic::getOrInsertDeclaration(M, ID);
  return Builder.CreateCall(WhileLT, {Start, End, Builder.getInt32(Scale)},
                            "pac.mask");
}

static Intrinsic::ID getPNLoadStoreIntrinsic(unsigned Scale, bool IsLoad) {
  if (Scale == 2)
    return IsLoad ? Intrinsic::aarch64_sve_ld1_pn_x2
                  : Intrinsic::aarch64_sve_st1_pn_x2;
  if (Scale == 4)
    return IsLoad ? Intrinsic::aarch64_sve_ld1_pn_x4
                  : Intrinsic::aarch64_sve_st1_pn_x4;
  llvm_unreachable("unsupported predicate-as-counter scale");
}

static void copyCallArgAttrs(CallBase &Dst, unsigned DstArgNo,
                             const CallBase &Src, unsigned SrcArgNo) {
  AttrBuilder AB(Src.getContext(), Src.getParamAttributes(SrcArgNo));
  Dst.addParamAttrs(DstArgNo, AB);
}

static bool isPoisonOrUndefValue(Value *V) {
  return isa<PoisonValue>(V) || isa<UndefValue>(V);
}

static Value *buildWideData(IRBuilder<> &Builder, Value *Tuple,
                            Type *WideDataTy, const MaskRewriteCandidate &C) {
  Value *WideData = PoisonValue::get(WideDataTy);
  for (unsigned Slice = 0; Slice != C.Scale; ++Slice) {
    Value *Part = Builder.CreateExtractValue(Tuple, Slice, "pac.data");
    WideData = Builder.CreateInsertVector(WideDataTy, WideData, Part,
                                          Slice * C.LegalLanes, "pac.vec");
  }
  return WideData;
}

static bool tryRewriteMaskedLoadUser(Instruction &UserI,
                                     const MaskRewriteCandidate &C,
                                     Value *Count) {
  auto *II = dyn_cast<IntrinsicInst>(&UserI);
  if (!II || II->getIntrinsicID() != Intrinsic::masked_load ||
      II->getType()->getScalarSizeInBits() / 8 != C.ElementSizeInBytes ||
      !isPoisonOrUndefValue(II->getArgOperand(2)))
    return false;

  IRBuilder<> Builder(II);
  Builder.SetCurrentDebugLocation(II->getDebugLoc());

  Type *ScalarType = II->getType()->getScalarType();
  auto *WideDataTy =
      ScalableVectorType::get(ScalarType, C.LegalLanes * C.Scale);
  auto *LegalDataTy = ScalableVectorType::get(ScalarType, C.LegalLanes);

  Module *M = II->getModule();
  FunctionCallee LD1 = Intrinsic::getOrInsertDeclaration(
      M, getPNLoadStoreIntrinsic(C.Scale, /*IsLoad=*/true),
      {LegalDataTy, II->getArgOperand(0)->getType()});
  auto *PNLoad =
      Builder.CreateCall(LD1, {Count, II->getArgOperand(0)}, "pac.ld1");
  PNLoad->setDebugLoc(II->getDebugLoc());
  PNLoad->copyMetadata(*II);
  copyCallArgAttrs(*PNLoad, /*DstArgNo=*/1, *II, /*SrcArgNo=*/0);

  Value *WideData = buildWideData(Builder, PNLoad, WideDataTy, C);
  cast<Instruction>(WideData)->takeName(II);
  II->replaceAllUsesWith(WideData);
  II->eraseFromParent();
  return true;
}

static bool tryRewriteMaskedStoreUser(Instruction &UserI,
                                      const MaskRewriteCandidate &C,
                                      Value *Count) {
  auto *II = dyn_cast<IntrinsicInst>(&UserI);
  if (!II || II->getIntrinsicID() != Intrinsic::masked_store ||
      II->getArgOperand(0)->getType()->getScalarSizeInBits() / 8 !=
          C.ElementSizeInBytes)
    return false;

  IRBuilder<> Builder(II);
  Builder.SetCurrentDebugLocation(II->getDebugLoc());

  Type *ScalarType = II->getArgOperand(0)->getType()->getScalarType();
  auto *LegalDataTy = ScalableVectorType::get(ScalarType, C.LegalLanes);

  SmallVector<Value *, 6> StoreArgs;
  for (unsigned Slice = 0; Slice != C.Scale; ++Slice)
    StoreArgs.push_back(Builder.CreateExtractVector(
        LegalDataTy, II->getArgOperand(0), Slice * C.LegalLanes, "pac.data"));
  StoreArgs.push_back(Count);
  StoreArgs.push_back(II->getArgOperand(1));

  Module *M = II->getModule();
  FunctionCallee ST1 = Intrinsic::getOrInsertDeclaration(
      M, getPNLoadStoreIntrinsic(C.Scale, /*IsLoad=*/false),
      {LegalDataTy, II->getArgOperand(1)->getType()});
  auto *PNStore = Builder.CreateCall(ST1, StoreArgs);
  PNStore->setDebugLoc(II->getDebugLoc());
  PNStore->copyMetadata(*II);
  copyCallArgAttrs(*PNStore, /*DstArgNo=*/C.Scale + 1, *II, /*SrcArgNo=*/1);

  II->eraseFromParent();
  return true;
}

static bool tryRewriteExtractElement(Instruction &UserI,
                                     const MaskRewriteCandidate &C,
                                     Value *Count) {
  auto *EEI = dyn_cast<ExtractElementInst>(&UserI);
  if (!EEI)
    return false;

  auto *Idx = dyn_cast<ConstantInt>(EEI->getIndexOperand());
  if (!Idx || Idx->getValue().uge(C.LegalLanes))
    return false;

  IRBuilder<> Builder(EEI);
  Builder.SetCurrentDebugLocation(EEI->getDebugLoc());

  Module *M = EEI->getModule();
  FunctionCallee PExt = Intrinsic::getOrInsertDeclaration(
      M, Intrinsic::aarch64_sve_pext,
      {ScalableVectorType::get(Builder.getInt1Ty(), C.LegalLanes)});
  auto *ExtractMask =
      Builder.CreateCall(PExt, {Count, Builder.getInt32(0)}, "pac.pext");
  ExtractMask->setDebugLoc(EEI->getDebugLoc());
  ExtractMask->copyMetadata(*EEI);

  Value *Extracted = Builder.CreateExtractElement(
      ExtractMask, EEI->getIndexOperand(), EEI->getName() + ".pac");
  auto *ExtractedI = cast<Instruction>(Extracted);
  ExtractedI->setDebugLoc(EEI->getDebugLoc());
  ExtractedI->copyMetadata(*EEI);
  ExtractedI->takeName(EEI);

  EEI->replaceAllUsesWith(Extracted);
  EEI->eraseFromParent();
  return true;
}

class AArch64PredicateAsCounterLoopRewrites : public LoopPass {
public:
  static char ID;

  AArch64PredicateAsCounterLoopRewrites() : LoopPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.setPreservesCFG();
  }

  bool runOnLoop(Loop *L, LPPassManager &) override;

private:
  std::optional<MaskRewriteCandidate> matchMaskPhi(Loop &L, PHINode &Phi) const;
  bool rewriteCandidate(const MaskRewriteCandidate &C) const;
};

} // end anonymous namespace

char AArch64PredicateAsCounterLoopRewrites::ID = 0;

INITIALIZE_PASS_BEGIN(AArch64PredicateAsCounterLoopRewrites, DEBUG_TYPE,
                      "AArch64 Predicate As Counter Loop Rewrites", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(AArch64PredicateAsCounterLoopRewrites, DEBUG_TYPE,
                    "AArch64 Predicate As Counter Loop Rewrites", false, false)

Pass *llvm::createAArch64PredicateAsCounterLoopRewritesPass() {
  return new AArch64PredicateAsCounterLoopRewrites();
}

bool AArch64PredicateAsCounterLoopRewrites::runOnLoop(Loop *L,
                                                      LPPassManager &) {
  if (skipLoop(L))
    return false;

  Function &F = *L->getHeader()->getParent();
  auto &TPC = getAnalysis<TargetPassConfig>();
  const AArch64Subtarget *ST =
      TPC.getTM<AArch64TargetMachine>().getSubtargetImpl(F);
  if (!ST->isSVEorStreamingSVEAvailable() ||
      !(ST->hasSVE2p1() || ST->hasSME2()))
    return false;

  bool Changed = false;
  BasicBlock *Header = L->getHeader();
  for (auto It = Header->begin(); It != Header->end() && isa<PHINode>(*It);) {
    auto *Phi = cast<PHINode>(&*It++);
    auto Candidate = matchMaskPhi(*L, *Phi);
    if (Candidate)
      Changed |= rewriteCandidate(*Candidate);
  }

  if (Changed)
    ++LoopsRewritten;

  return Changed;
}

std::optional<MaskRewriteCandidate>
AArch64PredicateAsCounterLoopRewrites::matchMaskPhi(Loop &L,
                                                    PHINode &Phi) const {
  BasicBlock *Preheader = L.getLoopPreheader();
  BasicBlock *Latch = L.getLoopLatch();
  if (!Preheader || !Latch || !isScalableMaskType(Phi.getType()) ||
      Phi.getNumIncomingValues() != 2)
    return std::nullopt;

  auto *StartMask =
      getGetActiveLaneMask(Phi.getIncomingValueForBlock(Preheader));
  auto *NextMask = getGetActiveLaneMask(Phi.getIncomingValueForBlock(Latch));
  if (!StartMask || !NextMask)
    return std::nullopt;

  if (StartMask->getArgOperand(0)->getType()->getIntegerBitWidth() > 64 ||
      NextMask->getArgOperand(0)->getType()->getIntegerBitWidth() > 64 ||
      StartMask->getArgOperand(2) != NextMask->getArgOperand(2))
    return std::nullopt;

  if (!hasOnlySupportedPhiUsers(&Phi))
    return std::nullopt;

  unsigned ElementSizeInBytes =
      cast<ConstantInt>(StartMask->getOperand(2))->getZExtValue();
  if (!is_contained({1u, 2u, 4u, 8u}, ElementSizeInBytes))
    return std::nullopt;

  unsigned WideMaskElements =
      cast<ScalableVectorType>(Phi.getType())->getMinNumElements();
  unsigned BaseMaskElements = getMaskElementCount(ElementSizeInBytes);

  if (!isPowerOf2_32(WideMaskElements) || WideMaskElements <= BaseMaskElements)
    return std::nullopt;

  unsigned Scale = WideMaskElements / BaseMaskElements;
  if (Scale != 2 && Scale != 4)
    return std::nullopt;

  return MaskRewriteCandidate{Preheader,          Latch,           &Phi,
                              StartMask,          NextMask,        Scale,
                              ElementSizeInBytes, BaseMaskElements};
}

bool AArch64PredicateAsCounterLoopRewrites::rewriteCandidate(
    const MaskRewriteCandidate &C) const {
  WeakTrackingVH OldPhi(C.MaskPhi);
  WeakTrackingVH OldStart(C.StartMask);
  WeakTrackingVH OldNext(C.NextMask);

  IRBuilder<> StartBuilder(C.StartMask);
  IRBuilder<> NextBuilder(C.NextMask);
  Value *NewStart = createWhileLO(StartBuilder, C.ElementSizeInBytes,
                                  C.StartMask->getArgOperand(0),
                                  C.StartMask->getArgOperand(1), C.Scale);
  Value *NewNext = createWhileLO(NextBuilder, C.ElementSizeInBytes,
                                 C.NextMask->getArgOperand(0),
                                 C.NextMask->getArgOperand(1), C.Scale);

  auto *NewPhi =
      PHINode::Create(NewStart->getType(), 2, C.MaskPhi->getName() + ".pn",
                      C.MaskPhi->getIterator());
  NewPhi->addIncoming(NewStart, C.Preheader);
  NewPhi->addIncoming(NewNext, C.Latch);

  auto RewriteUses = [&](Value *OldMask, Value *Count) {
    SmallVector<Use *, 8> UsesToRewrite;
    for (Use &U : OldMask->uses()) {
      // Don't rewrite PHI users. We restrict the "MaskPhi" to only have non-phi
      // users. The StartMask/NextMask could be used by other PHIs (e.g. due to
      // CSE). In which case, they're very likely to be replaced when we rewrite
      // another loop.
      if (isa<PHINode>(U.getUser()))
        continue;
      UsesToRewrite.push_back(&U);
    }

    for (Use *U : UsesToRewrite) {
      auto *UserI = cast<Instruction>(U->getUser());
      if (tryRewriteMaskedLoadUser(*UserI, C, Count) ||
          tryRewriteMaskedStoreUser(*UserI, C, Count) ||
          tryRewriteExtractElement(*UserI, C, Count))
        continue;

      IRBuilder<> Builder(UserI);
      Builder.SetCurrentDebugLocation(UserI->getDebugLoc());
      U->set(buildWideMask(Builder, C, Count));
    }
  };

  RewriteUses(C.MaskPhi, NewPhi);
  RewriteUses(C.StartMask, NewStart);
  RewriteUses(C.NextMask, NewNext);

  for (WeakTrackingVH DeadValue : {OldPhi, OldStart, OldNext})
    if (DeadValue)
      RecursivelyDeleteTriviallyDeadInstructions(cast<Instruction>(DeadValue));

  return true;
}
