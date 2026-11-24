//===- GuardedIntrinsicVerifier.cpp - Verify guarded intrinsics -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/GuardedIntrinsicVerifier.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GuardedIntrinsicInfo.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Type.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "guarded-intrinsic-verifier"

namespace {

static bool matchesType(Type *Actual, Type *Expected, const DataLayout &DL) {
  if (Actual == Expected)
    return true;
  if (Expected->isPointerTy())
    return Actual->isPointerTy();
  if (auto *ActualVector = dyn_cast<VectorType>(Actual)) {
    if (auto *ExpectedVector = dyn_cast<VectorType>(Expected)) {
      auto *Pointer = dyn_cast<PointerType>(ActualVector->getElementType());
      return Pointer &&
             ExpectedVector->getElementType()->isIntegerTy(
                 DL.getPointerSizeInBits(Pointer->getAddressSpace())) &&
             ActualVector->getElementCount() ==
                 ExpectedVector->getElementCount();
    }
  }
  if (auto *ActualStruct = dyn_cast<StructType>(Actual)) {
    if (auto *ExpectedStruct = dyn_cast<StructType>(Expected)) {
      if (ActualStruct->getNumElements() != ExpectedStruct->getNumElements())
        return false;
      for (unsigned I = 0; I < ActualStruct->getNumElements(); ++I)
        if (!matchesType(ActualStruct->getElementType(I),
                         ExpectedStruct->getElementType(I), DL))
          return false;
      return true;
    }
  }
  return false;
}

static bool matchesSignature(const IntrinsicInst &Call,
                             const IntrinsicFeatureVariant &Variant,
                             const GuardedIntrinsicInfo &Info) {
  if (!Intrinsic::isOverloaded(Call.getIntrinsicID()))
    return true;
  FunctionType *Expected =
      Info.getFunctionType(Call.getContext(), Call.getIntrinsicID(), Variant);
  FunctionType *Actual = Call.getFunctionType();
  if (Actual->getNumParams() != Expected->getNumParams() ||
      Actual->isVarArg() != Expected->isVarArg())
    return false;
  const DataLayout &DL = Call.getDataLayout();
  if (!matchesType(Actual->getReturnType(), Expected->getReturnType(), DL))
    return false;
  for (unsigned I = 0; I < Actual->getNumParams(); ++I)
    if (!matchesType(Actual->getParamType(I), Expected->getParamType(I), DL))
      return false;
  return true;
}

struct IntrinsicCheckResult {
  bool HasGuards = false;
  bool FoundSignature = false;
  bool Valid = false;
  const IntrinsicFeatureGuard *Failed = nullptr;
};

static IntrinsicCheckResult checkIntrinsic(const IntrinsicInst &II,
                                           const GuardedIntrinsicInfo &Info,
                                           const TargetSubtargetInfo &STI) {
  IntrinsicCheckResult Result;
  ArrayRef<IntrinsicFeatureVariant> Variants =
      Info.getVariants(II.getIntrinsicID());
  if (Variants.empty())
    return Result;
  Result.HasGuards = true;

  for (const IntrinsicFeatureVariant &Variant : Variants) {
    if (!matchesSignature(II, Variant, Info))
      continue;
    Result.FoundSignature = true;
    if (Info.isVariantAvailable(Variant, STI)) {
      Result.Valid = true;
      break;
    }
    for (unsigned I = 0; I < Variant.NumGuards; ++I) {
      const IntrinsicFeatureGuard &Guard = Info.getGuard(Variant, I);
      if (Info.checkGuard(Guard, STI) ==
          IntrinsicGuardCheckResult::MissingFeatures)
        Result.Failed = &Guard;
    }
  }
  return Result;
}

static void verifyGuardedIntrinsics(Function &F,
                                    const GuardedIntrinsicInfo &Info,
                                    const TargetSubtargetInfo &STI) {
  for (Instruction &I : instructions(F)) {
    auto *II = dyn_cast<IntrinsicInst>(&I);
    if (!II)
      continue;
    IntrinsicCheckResult Result = checkIntrinsic(*II, Info, STI);
    if (!Result.HasGuards || Result.Valid)
      continue;

    StringRef Name = II->getCalledFunction()->getName();
    std::string Message = Info.getTargetName().str() + " intrinsic '" +
                          Name.str() + "' in function '" + F.getName().str() +
                          "' ";
    if (!Result.FoundSignature)
      Message += "has an unsupported type signature";
    else if (!Result.Failed || !Result.Failed->NumClauses)
      Message += "is unavailable";
    else
      Message += "requires " + std::string(Result.Failed->Expression);
    if (Result.Failed && Result.Failed->Name[0] != '\0')
      Message += " in " + std::string(Result.Failed->Name) + " mode";
    reportFatalUsageError(StringRef(Message));
  }
}

class GuardedIntrinsicVerifierLegacyPass : public FunctionPass {
  const TargetMachine *TM = nullptr;

public:
  static char ID;

  GuardedIntrinsicVerifierLegacyPass() : FunctionPass(ID) {}
  explicit GuardedIntrinsicVerifierLegacyPass(const TargetMachine &TM)
      : FunctionPass(ID), TM(&TM) {}

  StringRef getPassName() const override {
    return "Guarded Intrinsic Verifier";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetTransformInfoWrapperPass>();
    AU.setPreservesAll();
  }

  bool runOnFunction(Function &F) override {
    assert(TM && "target machine is required");
    const TargetTransformInfo &TTI =
        getAnalysis<TargetTransformInfoWrapperPass>().getTTI(F);
    const GuardedIntrinsicInfo *Info = TTI.getGuardedIntrinsicInfo();
    if (!Info)
      return false;
    verifyGuardedIntrinsics(F, *Info, *TM->getSubtargetImpl(F));
    return false;
  }
};

} // namespace

PreservedAnalyses
GuardedIntrinsicVerifierPass::run(Function &F, FunctionAnalysisManager &FAM) {
  const TargetTransformInfo &TTI = FAM.getResult<TargetIRAnalysis>(F);
  const GuardedIntrinsicInfo *Info = TTI.getGuardedIntrinsicInfo();
  if (Info)
    verifyGuardedIntrinsics(F, *Info, *TM->getSubtargetImpl(F));
  return PreservedAnalyses::all();
}

char GuardedIntrinsicVerifierLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(GuardedIntrinsicVerifierLegacyPass, DEBUG_TYPE,
                      "Guarded Intrinsic Verifier", false, true)
INITIALIZE_PASS_DEPENDENCY(TargetTransformInfoWrapperPass)
INITIALIZE_PASS_END(GuardedIntrinsicVerifierLegacyPass, DEBUG_TYPE,
                    "Guarded Intrinsic Verifier", false, true)

FunctionPass *
llvm::createGuardedIntrinsicVerifierPass(const TargetMachine &TM) {
  return new GuardedIntrinsicVerifierLegacyPass(TM);
}
