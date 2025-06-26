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
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-machine-sme-abi"

namespace {

enum ZAState { ANY = 0, LOCAL_SAVED, ACTIVE, OFF, NUM_ZA_STATE };

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
  default:
    return "???";
  }
}

static ZAState getInstNeededZAState(const TargetRegisterInfo &TRI,
                                    MachineInstr &MI) {
  if (MI.getOpcode() == AArch64::InOutZAUsePseudo)
    return ZAState::ACTIVE;

  if (MI.getOpcode() == AArch64::RequiresZASavePseudo)
    return ZAState::LOCAL_SAVED;

  if (MI.isReturn())
    return ZAState::ACTIVE; // Assume inout ZA

  for (auto &MO : MI.operands()) {
    if (!MO.isReg() || !MO.getReg().isPhysical())
      continue;
    bool UsesZA =
        any_of(TRI.subregs_inclusive(MO.getReg()), [](const MCPhysReg &SR) {
          return AArch64::MPR128RegClass.contains(SR);
        });
    if (UsesZA) {
      llvm::dbgs() << "Found ZA Inst\n";
      return ZAState::ACTIVE;
    }
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
    AU.addPreservedID(MachineLoopInfoID);
    AU.addPreservedID(MachineDominatorsID);
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  void collectNeededZAStates(MachineFunction &MF);
  void pickBundleZAStates();
  void insertStateChanges(MachineFunction &MF);

  void handleStateChange(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI, ZAState From,
                         ZAState To);

private:
  struct InstInfo {
    ZAState NeededState{ZAState::ANY};
    MachineBasicBlock::iterator InsertPt;
  };

  struct BlockInfo {
    bool FixedEntryState = false;
    ZAState NeededEntryState{ZAState::ANY};
    SmallVector<InstInfo> Insts;
  };

  SmallVector<BlockInfo> Blocks;
  SmallVector<ZAState> BundleStates;
  EdgeBundles *Bundles = nullptr;
};

void MachineSMEABI::collectNeededZAStates(MachineFunction &MF) {
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  Blocks.resize(MF.getNumBlockIDs());
  for (MachineBasicBlock &MBB : MF) {
    BlockInfo &Block = Blocks[MBB.getNumber()];
    if (&MBB == &MF.front()) {
      Block.NeededEntryState = ZAState::ACTIVE; // assume inout entry
      Block.FixedEntryState = true;
    } else if (MBB.isEHPad()) {
      Block.NeededEntryState = ZAState::LOCAL_SAVED;
      Block.FixedEntryState = true;
    }

    for (MachineBasicBlock::iterator I = MBB.begin(); I != MBB.end(); ++I) {
      ZAState NeededState = getInstNeededZAState(TRI, *I);
      if (NeededState != ZAState::ANY)
        Block.Insts.push_back({NeededState, I});
    }

    if (Block.Insts.size() > 0) {
      if (Block.NeededEntryState == ZAState::ANY)
        Block.NeededEntryState = Block.Insts.front().NeededState;
    }
  }
}

void MachineSMEABI::pickBundleZAStates() {
  BundleStates.resize(Bundles->getNumBundles());
  for (unsigned I = 0, E = Bundles->getNumBundles(); I != E; ++I) {
    int StateCounts[ZAState::NUM_ZA_STATE] = {0};
    for (unsigned ID : Bundles->getBlocks(I)) {
      BlockInfo &Block = Blocks[ID];
      if (Block.NeededEntryState != ZAState::ANY)
        StateCounts[Block.NeededEntryState]++;
    }
    ZAState BundleState = ZAState(max_element(StateCounts) - StateCounts);

    // TODO: Propagate.
    if (BundleState == ZAState::ANY)
      BundleState =
          ZAState::ACTIVE; // Force ZA active in basic blocks that don't care

    BundleStates[I] = BundleState;

    llvm::dbgs() << "Bundle state: " << I << " is "
                 << getZAStateString(BundleState) << '\n';
    int S = 0;
    for (auto C : StateCounts)
      llvm::dbgs() << getZAStateString(ZAState(S++)) << " " << C << " ";
    llvm::dbgs() << '\n';
  }
}

void MachineSMEABI::insertStateChanges(MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF) {
    BlockInfo &Block = Blocks[MBB.getNumber()];
    ZAState InState =
        BundleStates[Bundles->getBundle(MBB.getNumber(), /*Out=*/false)];
    ZAState OutState =
        BundleStates[Bundles->getBundle(MBB.getNumber(), /*Out=true*/ true)];

    if (Block.NeededEntryState == ZAState::ANY)
      Block.NeededEntryState =
          InState; // This block should be in InState due to preds;

    if (!Block.FixedEntryState && InState != Block.NeededEntryState)
      handleStateChange(MBB, MBB.getFirstNonPHI(), InState,
                        Block.NeededEntryState);

    ZAState CurrentState = Block.NeededEntryState;

    for (auto &Inst : Block.Insts) {
      if (CurrentState != Inst.NeededState)
        handleStateChange(MBB, Inst.InsertPt, CurrentState, Inst.NeededState);
      CurrentState = Inst.NeededState;
    }

    if (CurrentState != OutState)
      handleStateChange(MBB, MBB.getLastNonDebugInstr(), CurrentState,
                        OutState);
  }
}

// LOCAL_SAVED,
// ACTIVE,

void MachineSMEABI::handleStateChange(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator InsertPt,
                                      ZAState From, ZAState To) {

  // ZA not used.
  if (From == ZAState::ANY || To == ZAState::ANY)
    return;

  const TargetInstrInfo *TII = MBB.getParent()->getSubtarget().getInstrInfo();

  if (From == ZAState::ACTIVE && To == ZAState::LOCAL_SAVED)
    BuildMI(MBB, InsertPt, DebugLoc(),
            TII->get(AArch64::SetupLazySaveZAPseudo));
  else if (From == ZAState::LOCAL_SAVED && To == ZAState::ACTIVE)
    BuildMI(MBB, InsertPt, DebugLoc(),
            TII->get(AArch64::RestoreLazySaveZAPseudo));
  else
    assert(false && "not done");
}

} // end anonymous namespace

INITIALIZE_PASS(MachineSMEABI, "aarch64-sme-machine-abi", "Machine SME ABI",
                false, false)

bool MachineSMEABI::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  if (!MF.getSubtarget<AArch64Subtarget>().hasSME())
    return false;

  assert(MF.getRegInfo().isSSA() && "Expected to be run on SSA form!");

  Blocks.clear();
  Bundles = &getAnalysis<EdgeBundlesWrapperLegacy>().getEdgeBundles();

  collectNeededZAStates(MF);
  pickBundleZAStates();
  insertStateChanges(MF);

  return true;
}

FunctionPass *llvm::createMachineSMEABIPass() { return new MachineSMEABI(); }
