//===- MachineSMEABIPass.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64InstrInfo.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64Subtarget.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/EdgeBundles.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/LiveRegUnits.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-machine-sme-abi"

static cl::opt<int>
    LoopEdgeWeight("aarch64-sme-abi-loop-edge-weight", cl::ReallyHidden,
                   cl::init(10),
                   cl::desc("Edge weight for basic blocks witin loops (used "
                            "for placing ZA saves/restores)"));

namespace {

enum ZAState {
  ANY = 0,
  ACTIVE,
  LOCAL_SAVED,
  CALLER_DORMANT,
  OFF,
  NUM_ZA_STATE
};

struct TPIDR2State {
  int FrameIndex = -1;
};

StringRef getZAStateString(ZAState State) {
  switch (State) {
  case ZAState::ANY:
    return "ANY";
  case ZAState::LOCAL_SAVED:
    return "LOCAL_SAVED";
  case ZAState::ACTIVE:
    return "ACTIVE";
  case ZAState::OFF:
    return "OFF";
  case ZAState::CALLER_DORMANT:
    return "CALLER_DORMANT";
  default:
    return "???";
  }
}

static bool isZARegOp(const TargetRegisterInfo &TRI, const MachineOperand &MO) {
  if (!MO.isReg() || !MO.getReg().isPhysical())
    return false;
  return any_of(TRI.subregs_inclusive(MO.getReg()), [](const MCPhysReg &SR) {
    return AArch64::MPR128RegClass.contains(SR);
  });
}

static ZAState getInstNeededZAState(const TargetRegisterInfo &TRI,
                                    MachineInstr &MI, bool ZALiveAtReturn) {
  if (MI.getOpcode() == AArch64::InOutZAUsePseudo)
    return ZAState::ACTIVE;

  if (MI.getOpcode() == AArch64::RequiresZASavePseudo)
    return ZAState::LOCAL_SAVED;

  if (MI.isReturn())
    return ZALiveAtReturn ? ZAState::ACTIVE : ZAState::OFF;

  for (auto &MO : MI.operands()) {
    if (isZARegOp(TRI, MO))
      return ZAState::ACTIVE;
  }

  return ZAState::ANY;
}

struct MachineSMEABI : public MachineFunctionPass {
  inline static char ID = 0;

  MachineSMEABI() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "Machine SME ABI pass"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<EdgeBundlesWrapperLegacy>();
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addPreservedID(MachineLoopInfoID);
    AU.addPreservedID(MachineDominatorsID);
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  void collectNeededZAStates(MachineFunction &MF, SMEAttrs);
  void pickBundleZAStates(MachineFunction &MF);
  void insertStateChanges(MachineFunction &MF);

  void emitRestoreLazySave(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI);
  void emitSetupLazySave(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI);

  MachineBasicBlock *emitCommitLazySave(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MBBI);

  void emitAllocateLazySaveBuffer(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator MBBI);

  void emitStateChange(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                       ZAState From, ZAState To);

  TPIDR2State getTPIDR2Block(MachineFunction &MF);

private:
  struct InstInfo {
    ZAState NeededState{ZAState::ANY};
    MachineBasicBlock::iterator InsertPt;
  };

  struct BlockInfo {
    ZAState FixedEntryState{ZAState::ANY};
    SmallVector<InstInfo> Insts;
  };

  SmallVector<BlockInfo> Blocks;
  SmallVector<ZAState> BundleStates;
  std::optional<TPIDR2State> TPIDR2Block;

  EdgeBundles *Bundles = nullptr;
  MachineLoopInfo *MLI = nullptr;
};

void MachineSMEABI::collectNeededZAStates(MachineFunction &MF,
                                          SMEAttrs SMEFnAttrs) {
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  assert(SMEFnAttrs.hasZAState() && "Expected function to have ZA state!");

  Blocks.resize(MF.getNumBlockIDs());
  for (MachineBasicBlock &MBB : MF) {
    BlockInfo &Block = Blocks[MBB.getNumber()];
    if (&MBB == &MF.front()) {
      // Entry block:
      Block.FixedEntryState = SMEFnAttrs.hasPrivateZAInterface()
                                  ? ZAState::CALLER_DORMANT
                                  : ZAState::ACTIVE;
    } else if (MBB.isEHPad()) {
      // EH entry block:
      Block.FixedEntryState = ZAState::LOCAL_SAVED;
    }

    for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ++I) {
      ZAState NeededState = getInstNeededZAState(
          TRI, *I, /*ZALiveAtReturn=*/SMEFnAttrs.hasSharedZAInterface());
      if (NeededState != ZAState::ANY)
        Block.Insts.push_back({NeededState, I});
    }
  }
}

void MachineSMEABI::pickBundleZAStates(MachineFunction &MF) {
  BundleStates.resize(Bundles->getNumBundles());
  for (unsigned I = 0, E = Bundles->getNumBundles(); I != E; ++I) {
    LLVM_DEBUG(dbgs() << "Picking ZA state for edge bundle: " << I << '\n');

    // Attempt to pick a ZA state for this bundle that minimizes state
    // transitions. Edges within loops are given a higher weight as we assume
    // they will be executed more than once.
    int EdgeStateCounts[ZAState::NUM_ZA_STATE] = {0};
    for (unsigned BlockID : Bundles->getBlocks(I)) {
      LLVM_DEBUG(dbgs() << "- bb." << BlockID);

      BlockInfo &Block = Blocks[BlockID];
      if (Block.Insts.empty()) {
        LLVM_DEBUG(dbgs() << " (no state preference)\n");
        continue;
      }
      bool IsLoop = MLI->getLoopFor(MF.getBlockNumbered(BlockID));
      bool InEdge = Bundles->getBundle(BlockID, /*Out=*/false) == I;
      bool OutEdge = Bundles->getBundle(BlockID, /*Out=*/true) == I;
      int EdgeWeight = IsLoop ? LoopEdgeWeight : 1;
      if (IsLoop)
        LLVM_DEBUG(dbgs() << " IsLoop");

      LLVM_DEBUG(dbgs() << " (EdgeWeight: " << EdgeWeight << ')');
      if (InEdge) {
        ZAState DesiredIncomingState = Block.Insts.front().NeededState;
        EdgeStateCounts[DesiredIncomingState] += EdgeWeight;
        LLVM_DEBUG(dbgs() << " DesiredIncomingState: "
                          << getZAStateString(DesiredIncomingState));
      }
      if (OutEdge) {
        ZAState DesiredOutgoingState = Block.Insts.front().NeededState;
        EdgeStateCounts[DesiredOutgoingState] += EdgeWeight;
        LLVM_DEBUG(dbgs() << " DesiredOutgoingState: "
                          << getZAStateString(DesiredOutgoingState));
      }
      LLVM_DEBUG(dbgs() << '\n');
    }

    ZAState BundleState =
        ZAState(max_element(EdgeStateCounts) - EdgeStateCounts);

    // Force ZA to be active in bundles that don't have a preferred state.
    // TODO: Something better here (to avoid extra mode switches).
    if (BundleState == ZAState::ANY)
      BundleState = ZAState::ACTIVE;

    LLVM_DEBUG({
      dbgs() << "Chosen ZA state: " << getZAStateString(BundleState) << '\n'
             << "Edge counts:";
      for (auto [State, Count] : enumerate(EdgeStateCounts))
        dbgs() << " " << getZAStateString(ZAState(State)) << ": " << Count;
      dbgs() << "\n\n";
    });

    BundleStates[I] = BundleState;
  }
}

void MachineSMEABI::insertStateChanges(MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF) {
    BlockInfo &Block = Blocks[MBB.getNumber()];
    ZAState InState =
        BundleStates[Bundles->getBundle(MBB.getNumber(), /*Out=*/false)];
    ZAState OutState =
        BundleStates[Bundles->getBundle(MBB.getNumber(), /*Out=true*/ true)];

    ZAState CurrentState = Block.FixedEntryState;
    if (CurrentState == ZAState::ANY)
      CurrentState = InState;

    for (auto &Inst : Block.Insts) {
      if (CurrentState != Inst.NeededState)
        emitStateChange(MBB, Inst.InsertPt, CurrentState, Inst.NeededState);
      CurrentState = Inst.NeededState;
    }

    if (MBB.succ_empty())
      continue;

    if (CurrentState != OutState)
      emitStateChange(MBB, MBB.getFirstInstrTerminator(), CurrentState,
                      OutState);
  }
}

TPIDR2State MachineSMEABI::getTPIDR2Block(MachineFunction &MF) {
  if (TPIDR2Block)
    return *TPIDR2Block;
  MachineFrameInfo &MFI = MF.getFrameInfo();
  TPIDR2Block = TPIDR2State{MFI.CreateStackObject(16, Align(16), false)};
  return *TPIDR2Block;
}

static DebugLoc getDebugLoc(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI) {
  if (MBBI != MBB.end())
    return MBBI->getDebugLoc();
  return DebugLoc();
}

void MachineSMEABI::emitSetupLazySave(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MBBI) {
  MachineFunction &MF = *MBB.getParent();
  auto &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  DebugLoc DL = getDebugLoc(MBB, MBBI);

  // Get pointer to TPIDR2 block.
  Register TPIDR2 = MRI.createVirtualRegister(&AArch64::GPR64spRegClass);
  Register TPIDR2Ptr = MRI.createVirtualRegister(&AArch64::GPR64RegClass);
  BuildMI(MBB, MBBI, DL, TII.get(AArch64::ADDXri), TPIDR2)
    .addFrameIndex(getTPIDR2Block(MF).FrameIndex)
    .addImm(0)
    .addImm(0);
 BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::COPY), TPIDR2Ptr)
        .addReg(TPIDR2);
  // Set TPIDR2_EL0 to point to TPIDR2 block.
  BuildMI(MBB, MBBI, DL, TII.get(AArch64::MSR))
      .addImm(AArch64SysReg::TPIDR2_EL0)
      .addReg(TPIDR2Ptr);
}

void MachineSMEABI::emitRestoreLazySave(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MBBI) {
  MachineFunction &MF = *MBB.getParent();
  auto &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  const AArch64RegisterInfo &TRI = *Subtarget.getRegisterInfo();
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  DebugLoc DL = getDebugLoc(MBB, MBBI);
  Register TPIDR2EL0 = MRI.createVirtualRegister(&AArch64::GPR64RegClass);
  Register TPIDR2 = MRI.createVirtualRegister(&AArch64::GPR64spRegClass);
  Register Flags = MRI.createVirtualRegister(&AArch64::GPR64RegClass);

        // BuildMI(MBB, MBBI, DL, TII.get(AArch64::MRS))
        //     .addReg(Flags, RegState::Define)
        //     .addImm(AArch64SysReg::NZCV)
        //     .addReg(AArch64::NZCV, RegState::Implicit);

  // Enable ZA.
  BuildMI(MBB, MBBI, DL, TII.get(AArch64::MSRpstatesvcrImm1))
      .addImm(AArch64SVCR::SVCRZA)
      .addImm(1);
  // Get current TPIDR2_EL0.
  BuildMI(MBB, MBBI, DL, TII.get(AArch64::MRS))
      .addReg(TPIDR2EL0, RegState::Define)
      .addImm(AArch64SysReg::TPIDR2_EL0);
  // Get pointer to TPIDR2 block.
  BuildMI(MBB, MBBI, DL, TII.get(AArch64::ADDXri), TPIDR2)
    .addFrameIndex(getTPIDR2Block(MF).FrameIndex)
    .addImm(0)
    .addImm(0);
  // (Conditionally) restore ZA state.
  BuildMI(MBB, MBBI, DL, TII.get(AArch64::RestoreZAPseudo))
      .addReg(TPIDR2EL0)
      .addReg(TPIDR2)
      .addExternalSymbol("__arm_tpidr2_restore")
      .addRegMask(TRI.SMEABISupportRoutinesCallPreservedMaskFromX0());
  // Zero TPIDR2_EL0.
  BuildMI(MBB, MBBI, DL, TII.get(AArch64::MSR))
      .addImm(AArch64SysReg::TPIDR2_EL0)
      .addReg(AArch64::XZR);

// BuildMI(MBB, MBBI, DL, TII.get(AArch64::MSR))
//                                 .addImm(AArch64SysReg::NZCV)
//                                 .addReg(Flags)
//                                 .addReg(AArch64::NZCV, RegState::ImplicitDefine);
}

void MachineSMEABI::emitAllocateLazySaveBuffer(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) {

  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  auto &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // TODO This function grows the stack with a subtraction, which doesn't work
  // on Windows. Some refactoring to share the functionality in
  // LowerWindowsDYNAMIC_STACKALLOC will be required once the Windows ABI
  // supports SME
  assert(!Subtarget.isTargetWindows() &&
         "Lazy ZA save is not yet supported on Windows");

  DebugLoc DL = getDebugLoc(MBB, MBBI);
  Register SP = MRI.createVirtualRegister(&AArch64::GPR64RegClass);
  Register SVL = MRI.createVirtualRegister(&AArch64::GPR64RegClass);
  Register Buffer = MRI.createVirtualRegister(&AArch64::GPR64RegClass);

  // 1. Allocate the lazy save buffer.
  {
    // Calculate SVL.
    BuildMI(MBB, MBBI, DL, TII.get(AArch64::RDSVLI_XI), SVL).addImm(1);
    // Get original stack pointer.
    BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::COPY), SP).addReg(AArch64::SP);
    // Allocate a lazy-save buffer object of the size given, normally SVL * SVL
    BuildMI(MBB, MBBI, DL, TII.get(AArch64::MSUBXrrr), Buffer)
        .addReg(SVL)
        .addReg(SVL)
        .addReg(SP);
    BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::COPY), AArch64::SP)
        .addReg(Buffer);
    // We have just allocated a variable sized object, tell this to PEI.
    MFI.CreateVariableSizedObject(Align(16), nullptr);
  }

  // 2. Setup the TPIDR2 block.
  {
    Register TPIDInitSaveSlicesReg = SVL;
    if (!Subtarget.isLittleEndian()) {
      Register TmpReg =
          MF.getRegInfo().createVirtualRegister(&AArch64::GPR64RegClass);
      // For big-endian targets move "num_za_save_slices" to the top two bytes.
      BuildMI(MBB, MBBI, DL, TII.get(AArch64::UBFMXri), TmpReg)
          .addReg(TPIDInitSaveSlicesReg)
          .addImm(16)
          .addImm(15);
      TPIDInitSaveSlicesReg = TmpReg;
    }
    // Get pointer to TPIDR2 block.
    Register TPIDR2 = MRI.createVirtualRegister(&AArch64::GPR64spRegClass);
    BuildMI(MBB, MBBI, DL, TII.get(AArch64::ADDXri), TPIDR2)
      .addFrameIndex(getTPIDR2Block(MF).FrameIndex)
      .addImm(0)
      .addImm(0);
    // Store buffer pointer and num_za_save_slices.
    // Bytes 10-15 are implicitly zeroed.
    BuildMI(MBB, MBBI, DL, TII.get(AArch64::STPXi))
        .addReg(Buffer)
        .addReg(TPIDInitSaveSlicesReg)
        .addReg(TPIDR2)
        .addImm(0);
  }
}

void MachineSMEABI::emitStateChange(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator InsertPt,
                                    ZAState From, ZAState To) {

  // ZA not used.
  if (From == ZAState::ANY || To == ZAState::ANY)
    return;

  if (From == ZAState::ACTIVE && To == ZAState::LOCAL_SAVED)
    emitSetupLazySave(MBB, InsertPt);
  else if (From == ZAState::LOCAL_SAVED && To == ZAState::ACTIVE)
    emitRestoreLazySave(MBB, InsertPt);
  else
    assert(false && "not done");
}

} // end anonymous namespace

INITIALIZE_PASS(MachineSMEABI, "aarch64-machine-sme-abi", "Machine SME ABI",
                false, false)

bool MachineSMEABI::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  if (!MF.getSubtarget<AArch64Subtarget>().hasSME())
    return false;

  auto *AFI = MF.getInfo<AArch64FunctionInfo>();
  SMEAttrs SMEFnAttrs = AFI->getSMEFnAttrs();

  if (!SMEFnAttrs.hasZAState())
    return false;

  assert(MF.getRegInfo().isSSA() && "Expected to be run on SSA form!");

  Blocks.clear();
  Bundles = &getAnalysis<EdgeBundlesWrapperLegacy>().getEdgeBundles();
  MLI = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();

  collectNeededZAStates(MF, SMEFnAttrs);
  pickBundleZAStates(MF);
  insertStateChanges(MF);

  // Allocate lazy save buffer (if needed).
  if (TPIDR2Block.has_value()) {
    MachineBasicBlock &EntryBlock = MF.front();
    emitAllocateLazySaveBuffer(EntryBlock, EntryBlock.getFirstNonPHI());
  }

  return true;
}

FunctionPass *llvm::createMachineSMEABIPass() { return new MachineSMEABI(); }
