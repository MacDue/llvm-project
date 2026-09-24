//===- AArch64SVELoadStoreClustering.cpp - Cluster SVE loads and stores ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
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
  MachineOperand Base;
  int64_t LowOffset;
  SmallVector<MachineInstr *, 4> OffsetToMI;
  MachineBasicBlock::iterator FirstIt;
  MachineBasicBlock::iterator LastIt;
  unsigned NumInstructions;
  bool IsLoad;
};

struct MultiVectorOpcodes {
  unsigned Load2;
  unsigned Store2;
  unsigned Load4;
  unsigned Store4;
};

struct RewriteGroup {
  unsigned StartIndex;
  unsigned NumVectors;
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

  static bool isClusterableAccess(const MachineInstr &MI) {
    if (!isSupportedAccess(MI) || MI.isBundled())
      return false;

    const MachineOperand &Base = MI.getOperand(1);
    if ((!Base.isReg() && !Base.isFI()) || (Base.isReg() && Base.getSubReg()))
      return false;

    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isReg() && (MO.getReg().isPhysical() || MO.isUndef()))
        return false;
    }

    for (MachineMemOperand *MMO : MI.memoperands()) {
      if (MMO->isVolatile() || MMO->isAtomic())
        return false;
    }

    return true;
  }

  static void collectClusters(MachineBasicBlock &MBB,
                              const TargetRegisterInfo &TRI,
                              SmallVectorImpl<SVELoadStoreCluster> &Clusters);
  static bool rewriteClusters(MachineFunction &MF,
                              SmallVectorImpl<SVELoadStoreCluster> &Clusters);
};

char AArch64SVELoadStoreClustering::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS(AArch64SVELoadStoreClustering, DEBUG_TYPE,
                "AArch64 SVE Load/Store Clustering", false, false)

void AArch64SVELoadStoreClustering::collectClusters(
    MachineBasicBlock &MBB, const TargetRegisterInfo &TRI,
    SmallVectorImpl<SVELoadStoreCluster> &Clusters) {
  MachineBasicBlock::iterator It = MBB.begin();
  while (It != MBB.end()) {
    if (!isClusterableAccess(*It)) {
      ++It;
      continue;
    }

    unsigned Opcode = It->getOpcode();
    MachineOperand Base = It->getOperand(1);
    Base.clearParent();
    MachineBasicBlock::iterator FirstIt = It;
    MachineBasicBlock::iterator LastIt = It;
    int64_t LowOffset = It->getOperand(2).getImm();
    int64_t HighOffset = LowOffset;
    SmallDenseSet<int64_t, 8> Offsets;
    SmallVector<std::pair<int64_t, MachineInstr *>, 4> OffsetAndMIs;
    unsigned NumInstructions = 0;
    bool HasDuplicateOffset = false;

    while (It != MBB.end()) {
      if (!isClusterableAccess(*It) || It->getOpcode() != Opcode ||
          !It->getOperand(1).isIdenticalTo(Base)) {
        bool SawStore = true;
        if (!It->isDebugInstr() && (It->modifiesRegister(AArch64::VG, &TRI) ||
                                    !It->isSafeToMove(SawStore)))
          break;
        ++It;
        continue;
      }

      int64_t Offset = It->getOperand(2).getImm();
      if (!Offsets.insert(Offset).second) {
        HasDuplicateOffset = true;
        break;
      }
      OffsetAndMIs.push_back({Offset, &*It});
      LowOffset = std::min(LowOffset, Offset);
      HighOffset = std::max(HighOffset, Offset);
      LastIt = It;
      ++NumInstructions;
      ++It;
    }

    if (HasDuplicateOffset || NumInstructions < 2 ||
        HighOffset - LowOffset != NumInstructions - 1)
      continue;

    SmallVector<MachineInstr *, 4> OffsetToMI(NumInstructions);
    for (auto [Offset, MI] : OffsetAndMIs)
      OffsetToMI[Offset - LowOffset] = MI;

    Clusters.push_back({Base, LowOffset, std::move(OffsetToMI), FirstIt, LastIt,
                        NumInstructions, Opcode == AArch64::LDR_ZXI});
  }
}

static bool getMultiVectorOpcodes(unsigned ElementSize,
                                  MultiVectorOpcodes &Opcodes) {
  switch (ElementSize) {
  case 8:
    Opcodes = {AArch64::LD1B_2Z_IMM_PSEUDO, AArch64::ST1B_2Z_IMM_PSEUDO,
               AArch64::LD1B_4Z_IMM_PSEUDO, AArch64::ST1B_4Z_IMM_PSEUDO};
    return true;
  case 16:
    Opcodes = {AArch64::LD1H_2Z_IMM_PSEUDO, AArch64::ST1H_2Z_IMM_PSEUDO,
               AArch64::LD1H_4Z_IMM_PSEUDO, AArch64::ST1H_4Z_IMM_PSEUDO};
    return true;
  case 32:
    Opcodes = {AArch64::LD1W_2Z_IMM_PSEUDO, AArch64::ST1W_2Z_IMM_PSEUDO,
               AArch64::LD1W_4Z_IMM_PSEUDO, AArch64::ST1W_4Z_IMM_PSEUDO};
    return true;
  case 64:
    Opcodes = {AArch64::LD1D_2Z_IMM_PSEUDO, AArch64::ST1D_2Z_IMM_PSEUDO,
               AArch64::LD1D_4Z_IMM_PSEUDO, AArch64::ST1D_4Z_IMM_PSEUDO};
    return true;
  default:
    return false;
  }
}

static unsigned getSubRegIndex(unsigned Index) {
  switch (Index) {
  case 0:
    return AArch64::zsub0;
  case 1:
    return AArch64::zsub1;
  case 2:
    return AArch64::zsub2;
  case 3:
    return AArch64::zsub3;
  default:
    llvm_unreachable("Unexpected multi-vector subregister index");
  }
}

static const TargetRegisterClass &getTupleRegClass(bool Is4Vector,
                                                   bool AllowStrided) {
  if (Is4Vector)
    return AllowStrided ? AArch64::ZPR4StridedOrContiguousRegClass
                        : AArch64::ZPR4Mul4RegClass;
  return AllowStrided ? AArch64::ZPR2StridedOrContiguousRegClass
                      : AArch64::ZPR2Mul2RegClass;
}

static Register getOrCreatePTrue(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator Before,
                                 MachineRegisterInfo &MRI,
                                 const AArch64InstrInfo &TII) {
  MachineBasicBlock::iterator It = Before;
  while (It != MBB.begin()) {
    --It;
    if (It->modifiesRegister(AArch64::VG, &TII.getRegisterInfo()))
      break;
    if (It->getOpcode() != AArch64::PTRUE_C_B ||
        !It->getOperand(0).getReg().isVirtual())
      continue;

    Register Pred = It->getOperand(0).getReg();
    It->getOperand(0).setIsDead(false);
    MRI.clearKillFlags(Pred);
    return Pred;
  }

  Register Pred = MRI.createVirtualRegister(&AArch64::PNR_p8to15RegClass);
  BuildMI(MBB, Before, Before->getDebugLoc(), TII.get(AArch64::PTRUE_C_B),
          Pred);
  return Pred;
}

static bool getElementSize(SVELoadStoreCluster &Cluster,
                           unsigned &ElementSize) {
  ElementSize = 0;
  for (MachineInstr *MI : Cluster.OffsetToMI) {
    for (MachineMemOperand *MMO : MI->memoperands()) {
      if (!MMO->getMemoryType().isValid())
        return false;

      unsigned Size = MMO->getMemoryType().getScalarSizeInBits();
      // Type legalization represents full-register LDR/STR accesses using an
      // opaque scalable i128 element. Fall back to D for these on little
      // endian, as for accesses without a memory operand.
      if (Size == 128)
        continue;
      if (Size != 8 && Size != 16 && Size != 32 && Size != 64)
        return false;
      if (ElementSize && ElementSize != Size)
        return false;
      ElementSize = Size;
    }
  }

  if (ElementSize)
    return true;
  ElementSize = 64;
  return true;
}

bool AArch64SVELoadStoreClustering::rewriteClusters(
    MachineFunction &MF, SmallVectorImpl<SVELoadStoreCluster> &Clusters) {
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const AArch64InstrInfo &TII =
      *MF.getSubtarget<AArch64Subtarget>().getInstrInfo();
  bool Changed = false;

  for (SVELoadStoreCluster &Cluster : Clusters) {
    if (Cluster.LowOffset != 0)
      continue;

    SmallVector<RewriteGroup, 4> Groups;
    unsigned StartIndex = 0;
    while (Cluster.NumInstructions - StartIndex >= 2) {
      unsigned Remaining = Cluster.NumInstructions - StartIndex;
      unsigned NumVectors = Remaining >= 4 ? 4 : 2;
      // The maximum immediate offset for a multi-vector instruction is 7.
      if (StartIndex / NumVectors > 7)
        break;
      Groups.push_back({StartIndex, NumVectors});
      StartIndex += NumVectors;
    }
    if (Groups.empty())
      continue;

    unsigned ElementSize;
    if (!getElementSize(Cluster, ElementSize))
      continue;

    MultiVectorOpcodes Opcodes;
    if (!getMultiVectorOpcodes(ElementSize, Opcodes))
      continue;

    MachineBasicBlock &MBB = *Cluster.FirstIt->getParent();
    MachineBasicBlock::iterator InsertIt =
        Cluster.IsLoad ? Cluster.FirstIt : Cluster.LastIt;
    Register Pred = getOrCreatePTrue(MBB, Cluster.FirstIt, MRI, TII);
    DebugLoc DL = InsertIt->getDebugLoc();

    for (RewriteGroup &Group : Groups) {
      bool Is4Vector = Group.NumVectors == 4;
      unsigned Opcode = Cluster.IsLoad
                            ? (Is4Vector ? Opcodes.Load4 : Opcodes.Load2)
                            : (Is4Vector ? Opcodes.Store4 : Opcodes.Store2);
      Register Tuple = MRI.createVirtualRegister(&getTupleRegClass(
          Is4Vector, MF.getSubtarget<AArch64Subtarget>().hasSME2()));
      SmallVector<MachineMemOperand *, 4> MemRefs;
      for (unsigned I = 0; I != Group.NumVectors; ++I) {
        MachineInstr *MI = Cluster.OffsetToMI[Group.StartIndex + I];
        MemRefs.append(MI->memoperands_begin(), MI->memoperands_end());
      }

      if (Cluster.IsLoad) {
        BuildMI(MBB, InsertIt, DL, TII.get(Opcode), Tuple)
            .addReg(Pred)
            .add(Cluster.Base)
            .addImm(Group.StartIndex / Group.NumVectors)
            .setMemRefs(MemRefs);
        for (unsigned I = 0; I != Group.NumVectors; ++I) {
          MachineOperand &Dest =
              Cluster.OffsetToMI[Group.StartIndex + I]->getOperand(0);
          BuildMI(MBB, InsertIt, DL, TII.get(TargetOpcode::COPY))
              .add(Dest)
              .addReg(Tuple, RegState{}, getSubRegIndex(I));
        }
        continue;
      }

      MachineInstrBuilder RegSequence = BuildMI(
          MBB, InsertIt, DL, TII.get(TargetOpcode::REG_SEQUENCE), Tuple);
      for (unsigned I = 0; I != Group.NumVectors; ++I) {
        MachineOperand &Value =
            Cluster.OffsetToMI[Group.StartIndex + I]->getOperand(0);
        MRI.clearKillFlags(Value.getReg());
        RegSequence.addReg(Value.getReg(), RegState{}, Value.getSubReg())
            .addImm(getSubRegIndex(I));
      }
      BuildMI(MBB, InsertIt, DL, TII.get(Opcode))
          .addReg(Tuple)
          .addReg(Pred)
          .add(Cluster.Base)
          .addImm(Group.StartIndex / Group.NumVectors)
          .setMemRefs(MemRefs);
    }

    for (RewriteGroup &Group : Groups) {
      for (unsigned I = 0; I != Group.NumVectors; ++I)
        Cluster.OffsetToMI[Group.StartIndex + I]->eraseFromParent();
    }
    Changed = true;
  }

  return Changed;
}

bool AArch64SVELoadStoreClustering::runOnMachineFunction(MachineFunction &MF) {
  assert(MF.getRegInfo().isSSA() && "Expected Machine SSA form");

  if (!MF.getSubtarget<AArch64Subtarget>().isLittleEndian() ||
      !MF.getSubtarget<AArch64Subtarget>().enableSubRegLiveness() ||
      (!MF.getSubtarget<AArch64Subtarget>().hasSVE2p1() &&
       !(MF.getSubtarget<AArch64Subtarget>().hasSME2() &&
         MF.getSubtarget<AArch64Subtarget>().isStreaming())))
    return false;

  SmallVector<SVELoadStoreCluster, 4> Clusters;
  for (MachineBasicBlock &MBB : MF)
    collectClusters(MBB, *MF.getSubtarget().getRegisterInfo(), Clusters);

  LLVM_DEBUG({
    for (SVELoadStoreCluster &Cluster : Clusters) {
      MachineBasicBlock &MBB = *Cluster.FirstIt->getParent();
      dbgs() << "AArch64 SVE " << (Cluster.IsLoad ? "load" : "store")
             << " cluster in bb." << MBB.getNumber();
      if (MBB.hasName())
        dbgs() << '.' << MBB.getName();
      dbgs() << ": base ";
      if (Cluster.Base.isReg())
        dbgs() << printReg(Cluster.Base.getReg(),
                           MF.getSubtarget().getRegisterInfo());
      else
        Cluster.Base.print(dbgs(), MF.getSubtarget().getRegisterInfo());
      dbgs() << ", low offset " << Cluster.LowOffset << ", "
             << Cluster.NumInstructions << " instructions\n";
      dbgs() << "  offset-to-reg:";
      for (unsigned Offset = 0; Offset != Cluster.OffsetToMI.size(); ++Offset)
        dbgs() << ' ' << Offset << '='
               << printReg(Cluster.OffsetToMI[Offset]->getOperand(0).getReg(),
                           MF.getSubtarget().getRegisterInfo());
      dbgs() << '\n';
      dbgs() << "  first: " << *Cluster.FirstIt;
      dbgs() << "  last:  " << *Cluster.LastIt;
    }
  });

  return rewriteClusters(MF, Clusters);
}

FunctionPass *llvm::createAArch64SVELoadStoreClusteringPass() {
  return new AArch64SVELoadStoreClustering();
}
