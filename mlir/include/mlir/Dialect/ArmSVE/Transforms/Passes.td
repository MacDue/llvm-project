//===-- Passes.td - ArmSVE pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARMSVE_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_ARMSVE_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def LegalizeVectorStorage
    : Pass<"arm-sve-legalize-vector-storage", "mlir::func::FuncOp"> {
  let summary = "Ensures stores of SVE vector types will be legal";
  let description = [{
    Sub-byte predicate types, such as vector<[4]xi1> cannot be directly stored,
    they need to be widened to a legal type first. This pass replaces
    allocations of these types to full predicates (that is vector<[16]xi1>), and
    inserts `arm_sve.convert_to_svbool` and `arm_sve.convert_from_svbool` as
    needed.
  }];
  let constructor = "mlir::arm_sve::createLegalizeVectorStoragePass()";
  let dependentDialects = ["func::FuncDialect",
    "memref::MemRefDialect", "vector::VectorDialect",
    "arm_sve::ArmSVEDialect"];
}

#endif // MLIR_DIALECT_ARMSVE_TRANSFORMS_PASSES_TD
