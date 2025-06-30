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
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

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
  ACTIVE,         // 1
  LOCAL_SAVED,    // 2
  CALLER_DORMANT, // 3
  OFF,            // 4
  NUM_ZA_STATE
};

static bool isLegalEdgeBundleZAState(ZAState State) {
  switch (State) {
  case ZAState::ACTIVE:
    return true;
  case ZAState::LOCAL_SAVED:
    return true;
  default:
    return false;
  }
}
struct TPIDR2State {
  int FrameIndex = -1;
};

StringRef getZAStateString(ZAState State) {
#define MAKE_CASE(V)                                                           \
  case V:                                                                      \
    return #V;
  switch (State) {
    MAKE_CASE(ZAState::ANY)
    MAKE_CASE(ZAState::ACTIVE)
    MAKE_CASE(ZAState::LOCAL_SAVED)
    MAKE_CASE(ZAState::CALLER_DORMANT)
    MAKE_CASE(ZAState::OFF)
  default:
    llvm_unreachable("Unexpected ZAState");
  }
#undef MAKE_CASE
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
  if (MI.getOpcode() == AArch64::ADJCALLSTACKDOWN) {
    MachineBasicBlock::iterator MBBI(MI);
    // Note: The marker occurs after the ADJCALLSTACKDOWN (though we need to
    // insert any state changes before the ADJCALLSTACKDOWN, not after).
    auto MarkerNode = std::next(MBBI);
    auto &MBB = *MI.getParent();
    if (MarkerNode == MBB.end())
      return ZAState::ANY;
    if (MarkerNode->getOpcode() == AArch64::InOutZAUsePseudo)
      return ZAState::ACTIVE;
    if (MarkerNode->getOpcode() == AArch64::RequiresZASavePseudo)
      return ZAState::LOCAL_SAVED;
    return ZAState::ANY;
  }

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
                           MachineBasicBlock::iterator MBBI, bool NZCVLive);
  void emitSetupLazySave(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI);

  void emitNewZAPrologue(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI);

  void emitAllocateLazySaveBuffer(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator MBBI);

  void emitStateChange(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                       ZAState From, ZAState To, bool NZCVLive);

  void emitZAOff(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                 bool ClearTPIDR2);

  TPIDR2State getTPIDR2Block(MachineFunction &MF);

private:
  struct InstInfo {
    ZAState NeededState{ZAState::ANY};
    MachineBasicBlock::iterator InsertPt;
    bool NZCVLive = false;
  };

  struct BlockInfo {
    ZAState FixedEntryState{ZAState::ANY};
    SmallVector<InstInfo> Insts;
    bool NZCVLiveAtExit = false;
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
  assert((SMEFnAttrs.hasZT0State() || SMEFnAttrs.hasZAState()) &&
         "Expected function to have ZA/ZT0 state!");

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

    LiveRegUnits LiveRegs(TRI);
    LiveRegs.addLiveOuts(MBB);

    Block.NZCVLiveAtExit = !LiveRegs.available(AArch64::NZCV);
    auto FirstTerminatorInsertPt = MBB.getFirstTerminator();
    for (MachineInstr &MI : reverse(MBB)) {
      LiveRegs.stepBackward(MI);
      ZAState NeededState = getInstNeededZAState(
          TRI, MI, /*ZALiveAtReturn=*/SMEFnAttrs.hasSharedZAInterface());
      MachineBasicBlock::iterator InsertPt(MI);
      // TODO: Do something to avoid state changes where NZCV is live.
      bool NZCVLive = !LiveRegs.available(AArch64::NZCV);
      if (InsertPt == FirstTerminatorInsertPt)
        Block.NZCVLiveAtExit = NZCVLive;
      if (NeededState != ZAState::ANY)
        Block.Insts.push_back({NeededState, InsertPt, NZCVLive});
    }

    // Reverse vector (as we had to iterate backwards for liveness).
    std::reverse(Block.Insts.begin(), Block.Insts.end());
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
      ZAState DesiredIncomingState = Block.Insts.front().NeededState;
      if (InEdge && isLegalEdgeBundleZAState(DesiredIncomingState)) {
        EdgeStateCounts[DesiredIncomingState] += EdgeWeight;
        LLVM_DEBUG(dbgs() << " DesiredIncomingState: "
                          << getZAStateString(DesiredIncomingState));
      }
      ZAState DesiredOutgoingState = Block.Insts.front().NeededState;
      if (OutEdge && isLegalEdgeBundleZAState(DesiredOutgoingState)) {
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
        BundleStates[Bundles->getBundle(MBB.getNumber(), /*Out=*/true)];

    ZAState CurrentState = Block.FixedEntryState;
    if (CurrentState == ZAState::ANY)
      CurrentState = InState;

    for (auto &Inst : Block.Insts) {
      if (CurrentState != Inst.NeededState)
        emitStateChange(MBB, Inst.InsertPt, CurrentState, Inst.NeededState,
                        Inst.NZCVLive);
      CurrentState = Inst.NeededState;
    }

    if (MBB.succ_empty())
      continue;

    if (CurrentState != OutState)
      emitStateChange(MBB, MBB.getFirstTerminator(), CurrentState, OutState,
                      Block.NZCVLiveAtExit);
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
  BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::COPY), TPIDR2Ptr).addReg(TPIDR2);
  // Set TPIDR2_EL0 to point to TPIDR2 block.
  BuildMI(MBB, MBBI, DL, TII.get(AArch64::MSR))
      .addImm(AArch64SysReg::TPIDR2_EL0)
      .addReg(TPIDR2Ptr);
}

void MachineSMEABI::emitRestoreLazySave(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator MBBI,
                                        bool NZCVLive) {
  MachineFunction &MF = *MBB.getParent();
  auto &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  const AArch64RegisterInfo &TRI = *Subtarget.getRegisterInfo();
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  DebugLoc DL = getDebugLoc(MBB, MBBI);
  Register TPIDR2EL0 = MRI.createVirtualRegister(&AArch64::GPR64RegClass);
  Register StatusFlags = MRI.createVirtualRegister(&AArch64::GPR64RegClass);
  Register TPIDR2 = AArch64::X0;

  // TODO: Emit these within the restore MBB to prevent unnecessary saves.
  if (NZCVLive)
    BuildMI(MBB, MBBI, DL, TII.get(AArch64::MRS))
        .addReg(StatusFlags, RegState::Define)
        .addImm(AArch64SysReg::NZCV)
        .addReg(AArch64::NZCV, RegState::Implicit);

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

  if (NZCVLive)
    BuildMI(MBB, MBBI, DL, TII.get(AArch64::MSR))
        .addImm(AArch64SysReg::NZCV)
        .addReg(StatusFlags)
        .addReg(AArch64::NZCV, RegState::ImplicitDefine);
}

void MachineSMEABI::emitZAOff(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              bool ClearTPIDR2) {
  MachineFunction &MF = *MBB.getParent();
  auto &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  DebugLoc DL = getDebugLoc(MBB, MBBI);

  // Clear TPIDR2.
  if (ClearTPIDR2)
    DebugLoc DL = getDebugLoc(MBB, MBBI);
  BuildMI(MBB, MBBI, DL, TII.get(AArch64::MSR))
      .addImm(AArch64SysReg::TPIDR2_EL0)
      .addReg(AArch64::XZR);

  // Disable ZA.
  BuildMI(MBB, MBBI, DL, TII.get(AArch64::MSRpstatesvcrImm1))
      .addImm(AArch64SVCR::SVCRZA)
      .addImm(0);
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

static void emitZeroZA(const TargetInstrInfo &TII, DebugLoc DL,
                       MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                       unsigned Mask) {
  MachineInstrBuilder MIB =
      BuildMI(MBB, MBBI, DL, TII.get(AArch64::ZERO_M)).addImm(Mask);
  for (unsigned I = 0; I < 8; I++) {
    if (Mask & (1 << I))
      MIB.addDef(AArch64::ZAD0 + I, RegState::ImplicitDefine);
  }
}

void MachineSMEABI::emitNewZAPrologue(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MBBI) {
  MachineFunction &MF = *MBB.getParent();
  auto &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  const AArch64RegisterInfo &TRI = *Subtarget.getRegisterInfo();
  DebugLoc DL = getDebugLoc(MBB, MBBI);

  // Get current TPIDR2_EL0.
  Register TPIDR2EL0 = MRI.createVirtualRegister(&AArch64::GPR64RegClass);
  BuildMI(MBB, MBBI, DL, TII.get(AArch64::MRS))
      .addReg(TPIDR2EL0, RegState::Define)
      .addImm(AArch64SysReg::TPIDR2_EL0);
  // If TPIDR2_EL0 is non-zero, commit the lazy save.
  BuildMI(MBB, MBBI, DL, TII.get(AArch64::CommitZAPseudo))
      .addReg(TPIDR2EL0)
      .addExternalSymbol("__arm_tpidr2_save")
      .addRegMask(TRI.SMEABISupportRoutinesCallPreservedMaskFromX0());
  // Clear TPIDR2_EL0.
  BuildMI(MBB, MBBI, DL, TII.get(AArch64::MSR))
      .addImm(AArch64SysReg::TPIDR2_EL0)
      .addReg(AArch64::XZR);
  // Enable ZA (as ZA could have previously been in the OFF state).
  BuildMI(MBB, MBBI, DL, TII.get(AArch64::MSRpstatesvcrImm1))
      .addImm(AArch64SVCR::SVCRZA)
      .addImm(1);
  // Zero ZA (all tiles).
  emitZeroZA(TII, DL, MBB, MBBI, /*Mask=*/0b11111111);
}

void MachineSMEABI::emitStateChange(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator InsertPt,
                                    ZAState From, ZAState To, bool NZCVLive) {

  // ZA not used.
  if (From == ZAState::ANY || To == ZAState::ANY)
    return;

  // TODO: Avoid setting up the save buffer if there's no transition to
  // LOCAL_SAVED.
  if (From == ZAState::CALLER_DORMANT) {
    // Note: CALLER_DORMANT -> OFF would only occur for a single BB function
    // that does not use ZA.
    if (To != ZAState::OFF) {
      assert(&MBB == &MBB.getParent()->front() &&
             "CALLER_DORMANT state only valid in entry block");
      emitNewZAPrologue(MBB, MBB.getFirstNonPHI());
    }
    // Note: "emitNewZAPrologue" zeros ZA, so we may need to setup a lazy save
    // if "To" os "ZAState::LOCAL_SAVED". If may be possible to improve this
    // case by changing the placement of the zero instruction.
  }

  if ((From == ZAState::CALLER_DORMANT || From == ZAState::ACTIVE) &&
      To == ZAState::LOCAL_SAVED)
    emitSetupLazySave(MBB, InsertPt);
  else if (From == ZAState::LOCAL_SAVED && To == ZAState::ACTIVE)
    emitRestoreLazySave(MBB, InsertPt, NZCVLive);
  else if (To == ZAState::OFF)
    emitZAOff(MBB, InsertPt, /*ClearTPIDR2=*/From == ZAState::LOCAL_SAVED);
  else
    assert(false && "Unimplemented state transition");
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
  if (!SMEFnAttrs.hasZAState() && !SMEFnAttrs.hasZT0State())
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
