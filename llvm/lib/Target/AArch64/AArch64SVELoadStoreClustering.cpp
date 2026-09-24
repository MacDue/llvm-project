//===- AArch64SVELoadStoreClustering.cpp - Cluster SVE loads and stores ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "aarch64-sve-load-store-clustering"

namespace {

struct SVELoadStoreCluster {
  Register Base;
  int64_t LowOffset;
  SmallVector<Register, 4> OffsetToReg;
  MachineBasicBlock::iterator StartIt;
  MachineBasicBlock::iterator EndIt;
  unsigned NumInstructions;
  bool IsLoad;
};

class AArch64SVELoadStoreClustering : public MachineFunctionPass {
public:
  static char ID;

  AArch64SVELoadStoreClustering() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return "AArch64 SVE Load/Store Clustering";
  }

private:
  static bool isSupportedAccess(const MachineInstr &MI) {
    return MI.getOpcode() == AArch64::LDR_ZXI ||
           MI.getOpcode() == AArch64::STR_ZXI;
  }

  static void collectClusters(MachineBasicBlock &MBB,
                              SmallVectorImpl<SVELoadStoreCluster> &Clusters);
};

char AArch64SVELoadStoreClustering::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS(AArch64SVELoadStoreClustering, DEBUG_TYPE,
                "AArch64 SVE Load/Store Clustering", false, false)

void AArch64SVELoadStoreClustering::collectClusters(
    MachineBasicBlock &MBB, SmallVectorImpl<SVELoadStoreCluster> &Clusters) {
  MachineBasicBlock::iterator It = MBB.begin();
  while (It != MBB.end()) {
    if (!isSupportedAccess(*It)) {
      ++It;
      continue;
    }

    unsigned Opcode = It->getOpcode();
    Register Base = It->getOperand(1).getReg();
    MachineBasicBlock::iterator StartIt = It;
    MachineBasicBlock::iterator EndIt = It;
    int64_t LowOffset = It->getOperand(2).getImm();
    int64_t HighOffset = LowOffset;
    SmallDenseSet<int64_t, 8> Offsets;
    SmallVector<std::pair<int64_t, Register>, 4> OffsetAndRegs;
    unsigned NumInstructions = 0;
    bool HasDuplicateOffset = false;

    do {
      int64_t Offset = It->getOperand(2).getImm();
      if (!Offsets.insert(Offset).second) {
        HasDuplicateOffset = true;
        break;
      }
      OffsetAndRegs.push_back({Offset, It->getOperand(0).getReg()});
      LowOffset = std::min(LowOffset, Offset);
      HighOffset = std::max(HighOffset, Offset);
      EndIt = It;
      ++NumInstructions;
      ++It;
    } while (It != MBB.end() && It->getOpcode() == Opcode &&
             It->getOperand(1).getReg() == Base);

    if (HasDuplicateOffset || NumInstructions < 2 ||
        HighOffset - LowOffset != NumInstructions - 1)
      continue;

    SmallVector<Register, 4> OffsetToReg(NumInstructions);
    for (auto [Offset, Reg] : OffsetAndRegs)
      OffsetToReg[Offset - LowOffset] = Reg;

    Clusters.push_back({Base, LowOffset, std::move(OffsetToReg), StartIt, EndIt,
                        NumInstructions, Opcode == AArch64::LDR_ZXI});
  }
}

bool AArch64SVELoadStoreClustering::runOnMachineFunction(MachineFunction &MF) {
  SmallVector<SVELoadStoreCluster, 4> Clusters;
  for (MachineBasicBlock &MBB : MF)
    collectClusters(MBB, Clusters);

  LLVM_DEBUG({
    for (SVELoadStoreCluster &Cluster : Clusters) {
      MachineBasicBlock &MBB = *Cluster.StartIt->getParent();
      dbgs() << "AArch64 SVE " << (Cluster.IsLoad ? "load" : "store")
             << " cluster in bb." << MBB.getNumber();
      if (MBB.hasName())
        dbgs() << '.' << MBB.getName();
      dbgs() << ": base "
             << printReg(Cluster.Base, MF.getSubtarget().getRegisterInfo())
             << ", low offset " << Cluster.LowOffset << ", "
             << Cluster.NumInstructions << " instructions\n";
      dbgs() << "  offset-to-reg:";
      for (unsigned Offset = 0; Offset != Cluster.OffsetToReg.size(); ++Offset)
        dbgs() << ' ' << Offset << '='
               << printReg(Cluster.OffsetToReg[Offset],
                           MF.getSubtarget().getRegisterInfo());
      dbgs() << '\n';
      dbgs() << "  start: " << *Cluster.StartIt;
      dbgs() << "  end:   " << *Cluster.EndIt;
    }
  });

  return false;
}

FunctionPass *llvm::createAArch64SVELoadStoreClusteringPass() {
  return new AArch64SVELoadStoreClustering();
}
