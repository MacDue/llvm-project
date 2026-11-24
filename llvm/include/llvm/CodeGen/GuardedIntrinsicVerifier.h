//===- GuardedIntrinsicVerifier.h - Verify guarded intrinsics ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GUARDEDINTRINSICVERIFIER_H
#define LLVM_CODEGEN_GUARDEDINTRINSICVERIFIER_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class FunctionPass;
class TargetMachine;

class GuardedIntrinsicVerifierPass
    : public RequiredPassInfoMixin<GuardedIntrinsicVerifierPass> {
  const TargetMachine *TM;

public:
  explicit GuardedIntrinsicVerifierPass(const TargetMachine &TM) : TM(&TM) {}
  LLVM_ABI PreservedAnalyses run(Function &F, FunctionAnalysisManager &FAM);
};

LLVM_ABI FunctionPass *
createGuardedIntrinsicVerifierPass(const TargetMachine &TM);

} // namespace llvm

#endif // LLVM_CODEGEN_GUARDEDINTRINSICVERIFIER_H
