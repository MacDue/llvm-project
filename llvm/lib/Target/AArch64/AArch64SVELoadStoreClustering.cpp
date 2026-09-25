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
  MachineOperand *Base;
  MachineOperand *Index;
  int64_t LowOffset;
  AArch64::ElementSizeType AccessSize;
  SmallVector<MachineInstr *, 4> OffsetToMI;
  MachineBasicBlock::iterator FirstIt;
  MachineBasicBlock::iterator LastIt;
  unsigned NumInstructions;
  bool IsLoad;
};

struct MultiVectorOpcodes {
  unsigned Load2Reg;
  unsigned Store2Reg;
  unsigned Load4Reg;
  unsigned Store4Reg;
  unsigned Load2Imm;
  unsigned Store2Imm;
  unsigned Load4Imm;
  unsigned Store4Imm;
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
    switch (MI.getOpcode()) {
    case AArch64::LDR_ZXI:
    case AArch64::STR_ZXI:
    case AArch64::LD1B:
    case AArch64::LD1H:
    case AArch64::LD1W:
    case AArch64::LD1D:
    case AArch64::ST1B:
    case AArch64::ST1H:
    case AArch64::ST1W:
    case AArch64::ST1D:
      return true;
    default:
      return false;
    }
  }

  static bool getAccessSize(const MachineInstr &MI,
                            AArch64::ElementSizeType &AccessSize);
  static bool isAllTruePredicate(const MachineInstr &MI);
  static void extractAddressing(MachineInstr &MI,
                                AArch64::ElementSizeType AccessSize,
                                MachineOperand *&Base, MachineOperand *&Index,
                                int64_t &Offset);

  static bool isClusterableAccess(MachineInstr &MI) {
    if (!isSupportedAccess(MI) || MI.isBundled())
      return false;

    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isReg() && (MO.getReg().isPhysical() || MO.isUndef()))
        return false;
    }

    for (MachineMemOperand *MMO : MI.memoperands()) {
      if (MMO->isVolatile() || MMO->isAtomic())
        return false;
    }

    return isAllTruePredicate(MI);
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

bool AArch64SVELoadStoreClustering::getAccessSize(
    const MachineInstr &MI, AArch64::ElementSizeType &AccessSize) {
  const AArch64InstrInfo &TII =
      *MI.getMF()->getSubtarget<AArch64Subtarget>().getInstrInfo();
  AccessSize = static_cast<AArch64::ElementSizeType>(
      TII.getElementSizeForOpcode(MI.getOpcode()));
  if (AccessSize != AArch64::ElementSizeNone)
    return true;

  for (MachineMemOperand *MMO : MI.memoperands()) {
    if (!MMO->getMemoryType().isValid())
      return false;

    AArch64::ElementSizeType MMOAccessSize;
    switch (MMO->getMemoryType().getScalarSizeInBits()) {
    case 8:
      MMOAccessSize = AArch64::ElementSizeB;
      break;
    case 16:
      MMOAccessSize = AArch64::ElementSizeH;
      break;
    case 32:
      MMOAccessSize = AArch64::ElementSizeS;
      break;
    case 64:
      MMOAccessSize = AArch64::ElementSizeD;
      break;
    default:
      return false;
    }
    if (AccessSize != AArch64::ElementSizeNone &&
        AccessSize != MMOAccessSize)
      return false;
    AccessSize = MMOAccessSize;
  }
  return AccessSize != AArch64::ElementSizeNone;
}

bool AArch64SVELoadStoreClustering::isAllTruePredicate(const MachineInstr &MI) {
  if (MI.getOpcode() == AArch64::LDR_ZXI || MI.getOpcode() == AArch64::STR_ZXI)
    return true;

  Register Pred = MI.getOperand(1).getReg();
  MachineInstr *Def = MI.getMF()->getRegInfo().getVRegDef(Pred);
  if (!isPTrueOpcode(Def->getOpcode()) || Def->getOperand(1).getImm() != 31)
    return false;

  const AArch64InstrInfo &TII =
      *MI.getMF()->getSubtarget<AArch64Subtarget>().getInstrInfo();
  return TII.getElementSizeForOpcode(Def->getOpcode()) ==
         TII.getElementSizeForOpcode(MI.getOpcode());
}

static bool isAddWithShift(const MachineInstr &MI, unsigned Shift) {
  if (MI.getOpcode() == AArch64::ADDXrr)
    return Shift == 0;
  return MI.getOpcode() == AArch64::ADDXrs &&
         MI.getOperand(3).getImm() == Shift;
}

void AArch64SVELoadStoreClustering::extractAddressing(
    MachineInstr &MI, AArch64::ElementSizeType AccessSize,
    MachineOperand *&Base, MachineOperand *&Index, int64_t &Offset) {
  Index = nullptr;
  Offset = 0;
  if (MI.getOpcode() != AArch64::LDR_ZXI &&
      MI.getOpcode() != AArch64::STR_ZXI) {
    Base = &MI.getOperand(2);
    Index = &MI.getOperand(3);
    return;
  }

  Base = &MI.getOperand(1);
  Offset = MI.getOperand(2).getImm();
  if (!Base->isReg())
    return;

  MachineInstr *Def = MI.getMF()->getRegInfo().getVRegDef(Base->getReg());
  if (AccessSize == AArch64::ElementSizeNone)
    return;

  unsigned Shift = AccessSize - AArch64::ElementSizeB;
  if (!isAddWithShift(*Def, Shift))
    return;

  MachineOperand *AddBase = &Def->getOperand(1);
  MachineOperand *AddIndex = &Def->getOperand(2);
  if (AddBase->getReg().isPhysical() || AddBase->isUndef() ||
      AddIndex->getReg().isPhysical() || AddIndex->isUndef())
    return;
  Base = AddBase;
  Index = AddIndex;
}

static bool hasSameAddressOperand(const MachineOperand *LHS,
                                  const MachineOperand *RHS) {
  if (!LHS || !RHS)
    return LHS == RHS;
  if (LHS->isReg() && RHS->isReg())
    return LHS->getReg() == RHS->getReg();
  if (LHS->isFI() && RHS->isFI())
    return LHS->getIndex() == RHS->getIndex();
  return false;
}

void AArch64SVELoadStoreClustering::collectClusters(
    MachineBasicBlock &MBB, const TargetRegisterInfo &TRI,
    SmallVectorImpl<SVELoadStoreCluster> &Clusters) {
  MachineBasicBlock::iterator It = MBB.begin();
  while (It != MBB.end()) {
    if (!isClusterableAccess(*It)) {
      ++It;
      continue;
    }

    MachineOperand *Base = nullptr;
    MachineOperand *Index = nullptr;
    AArch64::ElementSizeType AccessSize = AArch64::ElementSizeNone;
    bool IsLoad = It->mayLoad();
    MachineBasicBlock::iterator FirstIt = It;
    MachineBasicBlock::iterator LastIt = It;
    int64_t LowOffset = 0;
    int64_t HighOffset = 0;
    SmallDenseSet<int64_t, 8> Offsets;
    SmallVector<std::pair<int64_t, MachineInstr *>, 4> OffsetAndMIs;
    unsigned NumInstructions = 0;
    bool HasDuplicateOffset = false;

    while (It != MBB.end()) {
      MachineOperand *CandidateBase;
      MachineOperand *CandidateIndex;
      int64_t Offset;
      AArch64::ElementSizeType CandidateAccessSize;
      bool Matches = isClusterableAccess(*It) && It->mayLoad() == IsLoad;
      if (Matches) {
        if (!getAccessSize(*It, CandidateAccessSize))
          CandidateAccessSize = AArch64::ElementSizeNone;
        extractAddressing(*It, CandidateAccessSize, CandidateBase,
                          CandidateIndex, Offset);
        Matches = !Base || (CandidateAccessSize == AccessSize &&
                            hasSameAddressOperand(CandidateBase, Base) &&
                            hasSameAddressOperand(CandidateIndex, Index));
      }
      if (!Matches) {
        bool SawStore = true;
        if (!It->isDebugInstr() && (It->modifiesRegister(AArch64::VG, &TRI) ||
                                    !It->isSafeToMove(SawStore)))
          break;
        ++It;
        continue;
      }

      if (!Base) {
        Base = CandidateBase;
        Index = CandidateIndex;
        AccessSize = CandidateAccessSize;
        LowOffset = Offset;
        HighOffset = Offset;
      }

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

    Clusters.push_back({Base, Index, LowOffset, AccessSize,
                        std::move(OffsetToMI), FirstIt, LastIt, NumInstructions,
                        IsLoad});
  }
}

static bool getMultiVectorOpcodes(AArch64::ElementSizeType AccessSize,
                                  MultiVectorOpcodes &Opcodes) {
  switch (AccessSize) {
  case AArch64::ElementSizeB:
    Opcodes = {AArch64::LD1B_2Z_PSEUDO,     AArch64::ST1B_2Z_PSEUDO,
               AArch64::LD1B_4Z_PSEUDO,     AArch64::ST1B_4Z_PSEUDO,
               AArch64::LD1B_2Z_IMM_PSEUDO, AArch64::ST1B_2Z_IMM_PSEUDO,
               AArch64::LD1B_4Z_IMM_PSEUDO, AArch64::ST1B_4Z_IMM_PSEUDO};
    return true;
  case AArch64::ElementSizeH:
    Opcodes = {AArch64::LD1H_2Z_PSEUDO,     AArch64::ST1H_2Z_PSEUDO,
               AArch64::LD1H_4Z_PSEUDO,     AArch64::ST1H_4Z_PSEUDO,
               AArch64::LD1H_2Z_IMM_PSEUDO, AArch64::ST1H_2Z_IMM_PSEUDO,
               AArch64::LD1H_4Z_IMM_PSEUDO, AArch64::ST1H_4Z_IMM_PSEUDO};
    return true;
  case AArch64::ElementSizeS:
    Opcodes = {AArch64::LD1W_2Z_PSEUDO,     AArch64::ST1W_2Z_PSEUDO,
               AArch64::LD1W_4Z_PSEUDO,     AArch64::ST1W_4Z_PSEUDO,
               AArch64::LD1W_2Z_IMM_PSEUDO, AArch64::ST1W_2Z_IMM_PSEUDO,
               AArch64::LD1W_4Z_IMM_PSEUDO, AArch64::ST1W_4Z_IMM_PSEUDO};
    return true;
  case AArch64::ElementSizeD:
    Opcodes = {AArch64::LD1D_2Z_PSEUDO,     AArch64::ST1D_2Z_PSEUDO,
               AArch64::LD1D_4Z_PSEUDO,     AArch64::ST1D_4Z_PSEUDO,
               AArch64::LD1D_2Z_IMM_PSEUDO, AArch64::ST1D_2Z_IMM_PSEUDO,
               AArch64::LD1D_4Z_IMM_PSEUDO, AArch64::ST1D_4Z_IMM_PSEUDO};
    return true;
  default:
    return false;
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

static Register getOrCreateAddress(const SVELoadStoreCluster &Cluster,
                                   MachineRegisterInfo &MRI,
                                   const AArch64InstrInfo &TII) {
  MachineBasicBlock &MBB = *Cluster.FirstIt->getParent();
  unsigned Shift = Cluster.AccessSize - AArch64::ElementSizeB;
  for (MachineBasicBlock::iterator It = MBB.begin(); It != Cluster.FirstIt;
       ++It) {
    if (!isAddWithShift(*It, Shift) ||
        !It->getOperand(0).getReg().isVirtual() ||
        It->getOperand(1).isUndef() || It->getOperand(2).isUndef() ||
        It->getOperand(1).getReg() != Cluster.Base->getReg() ||
        It->getOperand(2).getReg() != Cluster.Index->getReg())
      continue;
    Register Address = It->getOperand(0).getReg();
    It->getOperand(0).setIsDead(false);
    MRI.clearKillFlags(Address);
    return Address;
  }

  Register Address = MRI.createVirtualRegister(&AArch64::GPR64commonRegClass);
  MachineInstrBuilder Add =
      BuildMI(MBB, Cluster.FirstIt, Cluster.FirstIt->getDebugLoc(),
              TII.get(Shift ? AArch64::ADDXrs : AArch64::ADDXrr), Address)
          .addReg(Cluster.Base->getReg())
          .addReg(Cluster.Index->getReg());
  if (Shift)
    Add.addImm(Shift);
  return Address;
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

    MultiVectorOpcodes Opcodes;
    if (!getMultiVectorOpcodes(Cluster.AccessSize, Opcodes))
      continue;

    MachineBasicBlock &MBB = *Cluster.FirstIt->getParent();
    if (Cluster.Base->isReg())
      MRI.clearKillFlags(Cluster.Base->getReg());
    if (Cluster.Index)
      MRI.clearKillFlags(Cluster.Index->getReg());
    Register Address;
    if (Cluster.Index && Groups.size() > 1)
      Address = getOrCreateAddress(Cluster, MRI, TII);
    MachineBasicBlock::iterator InsertIt =
        Cluster.IsLoad ? Cluster.FirstIt : Cluster.LastIt;
    Register Pred = getOrCreatePTrue(MBB, Cluster.FirstIt, MRI, TII);
    DebugLoc DL = InsertIt->getDebugLoc();

    for (RewriteGroup &Group : Groups) {
      bool Is4Vector = Group.NumVectors == 4;
      bool UseRegisterOffset = Cluster.Index && Group.StartIndex == 0;
      unsigned Opcode;
      if (Cluster.IsLoad)
        Opcode = UseRegisterOffset
                     ? (Is4Vector ? Opcodes.Load4Reg : Opcodes.Load2Reg)
                     : (Is4Vector ? Opcodes.Load4Imm : Opcodes.Load2Imm);
      else
        Opcode = UseRegisterOffset
                     ? (Is4Vector ? Opcodes.Store4Reg : Opcodes.Store2Reg)
                     : (Is4Vector ? Opcodes.Store4Imm : Opcodes.Store2Imm);
      Register Tuple = MRI.createVirtualRegister(&getTupleRegClass(
          Is4Vector, MF.getSubtarget<AArch64Subtarget>().hasSME2()));
      SmallVector<MachineMemOperand *, 4> MemRefs;
      for (unsigned I = 0; I != Group.NumVectors; ++I) {
        MachineInstr *MI = Cluster.OffsetToMI[Group.StartIndex + I];
        MemRefs.append(MI->memoperands_begin(), MI->memoperands_end());
      }

      if (Cluster.IsLoad) {
        MachineInstrBuilder Load =
            BuildMI(MBB, InsertIt, DL, TII.get(Opcode), Tuple).addReg(Pred);
        if (UseRegisterOffset)
          Load.add(*Cluster.Base).add(*Cluster.Index);
        else if (Cluster.Index)
          Load.addReg(Address).addImm(Group.StartIndex / Group.NumVectors);
        else
          Load.add(*Cluster.Base).addImm(Group.StartIndex / Group.NumVectors);
        Load.setMemRefs(MemRefs);
        for (unsigned I = 0; I != Group.NumVectors; ++I) {
          MachineOperand &Dest =
              Cluster.OffsetToMI[Group.StartIndex + I]->getOperand(0);
          BuildMI(MBB, InsertIt, DL, TII.get(TargetOpcode::COPY))
              .add(Dest)
              .addReg(Tuple, RegState{}, AArch64::zsub0 + I);
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
            .addImm(AArch64::zsub0 + I);
      }
      MachineInstrBuilder Store = BuildMI(MBB, InsertIt, DL, TII.get(Opcode))
                                      .addReg(Tuple)
                                      .addReg(Pred);
      if (UseRegisterOffset)
        Store.add(*Cluster.Base).add(*Cluster.Index);
      else if (Cluster.Index)
        Store.addReg(Address).addImm(Group.StartIndex / Group.NumVectors);
      else
        Store.add(*Cluster.Base).addImm(Group.StartIndex / Group.NumVectors);
      Store.setMemRefs(MemRefs);
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
      if (Cluster.Base->isReg())
        dbgs() << printReg(Cluster.Base->getReg(),
                           MF.getSubtarget().getRegisterInfo());
      else
        Cluster.Base->print(dbgs(), MF.getSubtarget().getRegisterInfo());
      if (Cluster.Index)
        dbgs() << ", index "
               << printReg(Cluster.Index->getReg(),
                           MF.getSubtarget().getRegisterInfo());
      dbgs() << ", low offset " << Cluster.LowOffset << ", "
             << Cluster.NumInstructions << " instructions\n";
      dbgs() << "  offset-to-mi:\n";
      for (unsigned Offset = 0; Offset != Cluster.OffsetToMI.size(); ++Offset)
        dbgs() << "    " << Offset << ": " << *Cluster.OffsetToMI[Offset];
    }
  });

  return rewriteClusters(MF, Clusters);
}

FunctionPass *llvm::createAArch64SVELoadStoreClusteringPass() {
  return new AArch64SVELoadStoreClustering();
}
