//===- CostModel.cpp ------ Cost Model Analysis ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the cost model analysis. It provides a very basic cost
// estimation for LLVM-IR. This analysis uses the services of the codegen
// to approximate the cost of any IR instruction when lowered to machine
// instructions. The cost results are unit-less and the cost number represents
// the throughput of the machine assuming that all loads hit the cache, all
// branches are predicted, etc. The cost numbers can be added in order to
// compare two or more transformation alternatives.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/CostModel.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

enum class OutputCostKind {
  RecipThroughput,
  Latency,
  CodeSize,
  SizeAndLatency,
  All,
};

static cl::opt<OutputCostKind> CostKind(
    "cost-kind", cl::desc("Target cost kind"),
    cl::init(OutputCostKind::RecipThroughput),
    cl::values(clEnumValN(OutputCostKind::RecipThroughput, "throughput",
                          "Reciprocal throughput"),
               clEnumValN(OutputCostKind::Latency, "latency",
                          "Instruction latency"),
               clEnumValN(OutputCostKind::CodeSize, "code-size", "Code size"),
               clEnumValN(OutputCostKind::SizeAndLatency, "size-latency",
                          "Code size and latency"),
               clEnumValN(OutputCostKind::All, "all", "Print all cost kinds")));

enum class IntrinsicCostStrategy {
  InstructionCost,
  IntrinsicCost,
  PatternMatchCost,
  TypeBasedIntrinsicCost,
};

static cl::opt<IntrinsicCostStrategy> IntrinsicCost(
    "intrinsic-cost-strategy",
    cl::desc("Costing strategy for intrinsic instructions"),
    cl::init(IntrinsicCostStrategy::InstructionCost),
    cl::values(
        clEnumValN(IntrinsicCostStrategy::InstructionCost, "instruction-cost",
                   "Use TargetTransformInfo::getInstructionCost"),
        clEnumValN(IntrinsicCostStrategy::IntrinsicCost, "intrinsic-cost",
                   "Use TargetTransformInfo::getIntrinsicInstrCost"),
        clEnumValN(IntrinsicCostStrategy::PatternMatchCost,
                   "pattern-match-cost",
                   "Pattern match and cost a group of instructions"),
        clEnumValN(
            IntrinsicCostStrategy::TypeBasedIntrinsicCost,
            "type-based-intrinsic-cost",
            "Calculate the intrinsic cost based only on argument types")));

#define CM_NAME "cost-model"
#define DEBUG_TYPE CM_NAME

static std::optional<InstructionCost>
getPatternCost(Instruction &Inst, TTI::TargetCostKind CostKind,
               TargetTransformInfo &TTI) {
  using namespace PatternMatch;

  // Match partial.reduce.add(acc, binop(extA, extB)).
  Value *Acc, *ExtA, *ExtB, *BinOp;
  if (match(&Inst,
            m_Intrinsic<Intrinsic::vector_partial_reduce_add>(
                m_Value(Acc),
                m_Value(BinOp,
                        m_BinOp(m_Value(ExtA, m_ZExtOrSExt(m_Value())),
                                m_Value(ExtB, m_ZExtOrSExt(m_Value()))))))) {
    auto *CastA = cast<CastInst>(ExtA);
    auto *CastB = cast<CastInst>(ExtB);
    return TTI.getPartialReductionCost(
        Instruction::Add, CastA->getSrcTy()->getScalarType(),
        CastB->getSrcTy()->getScalarType(), Acc->getType()->getScalarType(),
        cast<VectorType>(ExtA->getType())->getElementCount(),
        TTI.getPartialReductionExtendKind(CastA),
        TTI.getPartialReductionExtendKind(CastB),
        cast<Instruction>(BinOp)->getOpcode(), CostKind, std::nullopt);
  }

  // TODO: Match other patterns.

  return std::nullopt;
}

static InstructionCost getCost(Instruction &Inst, TTI::TargetCostKind CostKind,
                               TargetTransformInfo &TTI, bool &MatchedPattern) {
  if (IntrinsicCost == IntrinsicCostStrategy::PatternMatchCost) {
    if (auto Cost = getPatternCost(Inst, CostKind, TTI)) {
      MatchedPattern = true;
      return *Cost;
    }
  }

  auto *II = dyn_cast<IntrinsicInst>(&Inst);
  if (II && IntrinsicCost != IntrinsicCostStrategy::InstructionCost) {
    IntrinsicCostAttributes ICA(
        II->getIntrinsicID(), *II, InstructionCost::getInvalid(),
        /*TypeBasedOnly=*/IntrinsicCost ==
            IntrinsicCostStrategy::TypeBasedIntrinsicCost);
    return TTI.getIntrinsicInstrCost(ICA, CostKind);
  }

  return TTI.getInstructionCost(&Inst, CostKind);
}

static TTI::TargetCostKind
OutputCostKindToTargetCostKind(OutputCostKind CostKind) {
  switch (CostKind) {
  case OutputCostKind::RecipThroughput:
    return TTI::TCK_RecipThroughput;
  case OutputCostKind::Latency:
    return TTI::TCK_Latency;
  case OutputCostKind::CodeSize:
    return TTI::TCK_CodeSize;
  case OutputCostKind::SizeAndLatency:
    return TTI::TCK_SizeAndLatency;
  default:
    llvm_unreachable("Unexpected OutputCostKind!");
  };
}

PreservedAnalyses CostModelPrinterPass::run(Function &F,
                                            FunctionAnalysisManager &AM) {
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  OS << "Printing analysis 'Cost Model Analysis' for function '" << F.getName() << "':\n";
  for (BasicBlock &B : F) {
    for (Instruction &Inst : B) {
      bool MatchedPattern = false;
      if (CostKind == OutputCostKind::All) {
        InstructionCost RThru =
            getCost(Inst, TTI::TCK_RecipThroughput, TTI, MatchedPattern);
        InstructionCost CodeSize =
            getCost(Inst, TTI::TCK_CodeSize, TTI, MatchedPattern);
        InstructionCost Lat =
            getCost(Inst, TTI::TCK_Latency, TTI, MatchedPattern);
        InstructionCost SizeLat =
            getCost(Inst, TTI::TCK_SizeAndLatency, TTI, MatchedPattern);
        if (!MatchedPattern)
          continue;
        OS << "Cost Model: ";
        OS << "Found costs of ";
        if (RThru == CodeSize && RThru == Lat && RThru == SizeLat)
          OS << RThru;
        else
          OS << "RThru:" << RThru << " CodeSize:" << CodeSize << " Lat:" << Lat
             << " SizeLat:" << SizeLat;
        OS << " for: " << (MatchedPattern ? "pattern rooted at " : "") << Inst
           << "\n";
      } else {
        InstructionCost Cost =
            getCost(Inst, OutputCostKindToTargetCostKind(CostKind), TTI,
                    MatchedPattern);
        if (!MatchedPattern)
          continue;
        OS << "Cost Model: ";
        if (Cost.isValid())
          OS << "Found an estimated cost of " << Cost.getValue();
        else
          OS << "Invalid cost";
        OS << " for " << (MatchedPattern ? "pattern rooted at " : "")
           << "instruction: " << Inst << "\n";
      }
    }
  }
  return PreservedAnalyses::all();
}
