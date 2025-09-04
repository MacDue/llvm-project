//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64SVEFrameLowering.h"
#include "AArch64FrameLowering.h"

namespace llvm {

struct SVEEpiloguePartitions {
  MachineBasicBlock::iterator RestoreBegin, RestoreEnd;
};

struct CommonInfo {
  const MachineFrameInfo &MFI;
  const AArch64FunctionInfo &AFI;
  const TargetInstrInfo *TII;
};

static CommonInfo getCommonInfo(const MachineBasicBlock &MBB) {
  auto *MF = MBB.getParent();
  return {/*MFI=*/MF->getFrameInfo(),
          /*AFI=*/*MF->getInfo<AArch64FunctionInfo>(),
          /*TII=*/MF->getSubtarget().getInstrInfo()};
}

static SVEEpiloguePartitions
partitionSVEEpilogue(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                     StackOffset SVECalleeSavesSize,
                     StackOffset SVELocalsSize) {
  MachineBasicBlock::iterator RestoreBegin = MBBI, RestoreEnd = MBBI;
  if (SVECalleeSavesSize) {
    RestoreBegin = std::prev(RestoreEnd);
    while (RestoreBegin != MBB.begin() &&
           AArch64FrameLowering::isSVECalleeSave(std::prev(RestoreBegin)))
      --RestoreBegin;

    assert(AArch64FrameLowering::isSVECalleeSave(RestoreBegin) &&
           AArch64FrameLowering::isSVECalleeSave(std::prev(RestoreEnd)) &&
           "Unexpected instruction");
  }

  return {RestoreBegin, RestoreEnd};
}

StackOffset AArch64BaseSVEFrameLowering::getFrameIndexReferenceFromSP(
    const AArch64FunctionInfo &AFI, int ObjectOffset,
    FrameObjectType Type) const {
  if (Type == FrameObjectType::SVE)
    return StackOffset::get(-int64_t(AFI.getCalleeSavedStackSize()),
                            ObjectOffset);

  StackOffset ScalableOffset = {};
  if (Type == FrameObjectType::Default)
    ScalableOffset = -getSVEStackSize(AFI);

  return StackOffset::getFixed(ObjectOffset) + ScalableOffset;
}

StackOffset AArch64BaseSVEFrameLowering::determineScalableOffsetToNonSVEObject(
    const AArch64FunctionInfo &AFI, bool UseFP, FrameObjectType Type) const {
  bool IsFixedOrIsCSR =
      Type == FrameObjectType::Fixed || Type == FrameObjectType::CSR;
  if (UseFP && !IsFixedOrIsCSR)
    return -getSVEStackSize(AFI);
  if (!UseFP && IsFixedOrIsCSR)
    return getSVEStackSize(AFI);
  return StackOffset{};
}

OffsetPair AArch64BaseSVEFrameLowering::resolveSVEObjectOffset(
    const AArch64FunctionInfo &AFI, const MachineFrameInfo &MFI,
    int ObjectOffset) const {
  StackOffset FPOffset = StackOffset::get(
      -AFI.getCalleeSaveBaseToFrameRecordOffset(), ObjectOffset);
  StackOffset SPOffset =
      getSVEStackSize(AFI) +
      StackOffset::get(MFI.getStackSize() - AFI.getCalleeSavedStackSize(),
                       ObjectOffset);
  return {FPOffset, SPOffset};
}

MachineBasicBlock::iterator AArch64BaseSVEFrameLowering::allocateSVECalleeSaves(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI, unsigned *,
    int64_t &NumBytes, AArch64FrameLowering::FrameFlags &Flags,
    StackOffset CFAOffset, bool FollowupAllocs) const {
  auto [MFI, AFI, TII] = getCommonInfo(MBB);

  MachineBasicBlock::iterator CalleeSavesBegin = MBBI;
  assert(AArch64FrameLowering::isSVECalleeSave(CalleeSavesBegin) &&
         "Unexpected instruction");

  // Allocate space for the callee saves (if any).
  AArch64FrameLowering::allocateStackSpace(
      MBB, CalleeSavesBegin, 0, getSVECalleeSavesSize(AFI), Flags.NeedsWinCFI,
      &Flags.HasWinCFI, Flags.EmitAsyncCFI && !Flags.HasFP, CFAOffset,
      FollowupAllocs);

  while (AArch64FrameLowering::isSVECalleeSave(MBBI) &&
         MBBI != MBB.getFirstTerminator())
    ++MBBI;

  return MBBI;
}

MachineBasicBlock::iterator AArch64BaseSVEFrameLowering::deallocateSVEStack(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator FirstGPRRestoreI,
    int64_t &NumBytes, AArch64FrameLowering::FrameFlags &Flags,
    StackOffset CFAOffset, DebugLoc DL) const {
  auto [MFI, AFI, TII] = getCommonInfo(MBB);
  StackOffset SVECalleeSavesSize = getSVECalleeSavesSize(AFI);
  StackOffset SVELocalsSize = getSVELocalsSize(AFI);
  StackOffset SVEStackSize = SVECalleeSavesSize + SVELocalsSize;
  bool EmitCFAOffset = Flags.EmitCFI && !Flags.HasFP;
  auto [RestoreBegin, RestoreEnd] = partitionSVEEpilogue(
      MBB, FirstGPRRestoreI, SVECalleeSavesSize, SVELocalsSize);
  // If we have stack realignment or variable-sized objects we must use the
  // FP to restore SVE callee saves (as there is an unknown amount of
  // data/padding between the SP and SVE CS area).
  Register BaseForSVEDealloc =
      (AFI.isStackRealigned() || MFI.hasVarSizedObjects()) ? AArch64::FP
                                                           : AArch64::SP;
  if (SVECalleeSavesSize && BaseForSVEDealloc == AArch64::FP) {
    Register CalleeSaveBase = AArch64::FP;
    if (int64_t CalleeSaveBaseOffset =
            AFI.getCalleeSaveBaseToFrameRecordOffset()) {
      // If we have have an non-zero offset to the non-SVE CS base we need to
      // compute the base address by subtracting the offest in a temporary
      // register first (to avoid briefly deallocating the SVE CS).
      CalleeSaveBase = MBB.getParent()->getRegInfo().createVirtualRegister(
          &AArch64::GPR64RegClass);
      emitFrameOffset(MBB, RestoreBegin, DL, CalleeSaveBase, AArch64::FP,
                      StackOffset::getFixed(-CalleeSaveBaseOffset), TII,
                      MachineInstr::FrameDestroy);
    }
    // The code below will deallocate the stack space space by moving the
    // SP to the start of the SVE callee-save area.
    emitFrameOffset(MBB, RestoreBegin, DL, AArch64::SP, CalleeSaveBase,
                    -SVECalleeSavesSize, TII, MachineInstr::FrameDestroy);
  } else if (BaseForSVEDealloc == AArch64::SP) {
    if (SVECalleeSavesSize) {
      // Deallocate the non-SVE locals first before we can deallocate (and
      // restore callee saves) from the SVE area.
      emitFrameOffset(MBB, RestoreBegin, DL, AArch64::SP, AArch64::SP,
                      StackOffset::getFixed(NumBytes), TII,
                      MachineInstr::FrameDestroy, false, Flags.NeedsWinCFI,
                      &Flags.HasWinCFI, EmitCFAOffset,
                      SVEStackSize + CFAOffset);
      CFAOffset -= StackOffset::getFixed(NumBytes);
      NumBytes = 0;
    }

    emitFrameOffset(MBB, RestoreBegin, DL, AArch64::SP, AArch64::SP,
                    SVELocalsSize, TII, MachineInstr::FrameDestroy, false,
                    Flags.NeedsWinCFI, &Flags.HasWinCFI, EmitCFAOffset,
                    SVEStackSize + CFAOffset);

    emitFrameOffset(MBB, RestoreEnd, DL, AArch64::SP, AArch64::SP,
                    SVECalleeSavesSize, TII, MachineInstr::FrameDestroy, false,
                    Flags.NeedsWinCFI, &Flags.HasWinCFI, EmitCFAOffset,
                    SVECalleeSavesSize + CFAOffset);
  }

  return RestoreEnd;
}

StackOffset AArch64WindowsSVEFrameLowering::getFrameIndexReferenceFromSP(
    const AArch64FunctionInfo &AFI, int ObjectOffset,
    FrameObjectType Type) const {
  StackOffset SVECalleeSavesSize = getSVECalleeSavesSize(AFI);
  if (Type == FrameObjectType::SVE) {
    if (-ObjectOffset <= SVECalleeSavesSize.getScalable())
      return StackOffset::getScalable(ObjectOffset);
    return StackOffset::get(-int64_t(AFI.getCalleeSavedStackSize()),
                            ObjectOffset);
  }

  StackOffset ScalableOffset = {};
  if (Type == FrameObjectType::Default)
    ScalableOffset = -getSVEStackSize(AFI);
  else if (Type == FrameObjectType::CSR)
    ScalableOffset = -SVECalleeSavesSize;

  return StackOffset::getFixed(ObjectOffset) + ScalableOffset;
}

StackOffset
AArch64WindowsSVEFrameLowering::determineScalableOffsetToNonSVEObject(
    const AArch64FunctionInfo &AFI, bool UseFP, FrameObjectType Type) const {
  // In this stack layout, the FP is in between the callee saves and other
  // SVE allocations.
  if (UseFP) {
    if (Type == FrameObjectType::Fixed)
      return getSVECalleeSavesSize(AFI);
    else if (Type != FrameObjectType::CSR)
      return -getSVELocalsSize(AFI);
  } else {
    if (Type == FrameObjectType::Fixed)
      return getSVEStackSize(AFI);
    else if (Type == FrameObjectType::CSR)
      return getSVELocalsSize(AFI);
  }
  return StackOffset{};
}

OffsetPair AArch64WindowsSVEFrameLowering::resolveSVEObjectOffset(
    const AArch64FunctionInfo &AFI, const MachineFrameInfo &MFI,
    int ObjectOffset) const {
  auto [FPOffset, SPOffset] =
      AArch64BaseSVEFrameLowering::resolveSVEObjectOffset(AFI, MFI,
                                                          ObjectOffset);
  FPOffset += getSVECalleeSavesSize(AFI);
  if (-ObjectOffset <= getSVECalleeSavesSize(AFI).getScalable()) {
    FPOffset += StackOffset::getFixed(AFI.getCalleeSavedStackSize());
    SPOffset += StackOffset::getFixed(AFI.getCalleeSavedStackSize());
  }
  return {FPOffset, SPOffset};
}

MachineBasicBlock::iterator
AArch64WindowsSVEFrameLowering::allocateSVECalleeSaves(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    unsigned *ProloguePushBytes, int64_t &NumBytes,
    AArch64FrameLowering::FrameFlags &Flags, StackOffset CFAOffset,
    bool FollowupAllocs) const {
  assert(ProloguePushBytes &&
         "PrologueSaveBytes should be non-null before the frame record");
  auto [MFI, AFI, TII] = getCommonInfo(MBB);
  unsigned FixedObject = *ProloguePushBytes - AFI.getCalleeSavedStackSize();
  // If we're doing SVE saves first, we need to immediately allocate space
  // for fixed objects, then space for the SVE callee saves.
  //
  // Windows unwind requires that the scalable size is a multiple of 16;
  // that's handled when the callee-saved size is computed.
  auto SaveSize =
      getSVECalleeSavesSize(AFI) + StackOffset::getFixed(FixedObject);
  AArch64FrameLowering::allocateStackSpace(
      MBB, MBBI, 0, SaveSize, Flags.NeedsWinCFI, &Flags.HasWinCFI,
      /*EmitCFI=*/Flags.EmitAsyncCFI && !Flags.HasFP, CFAOffset,
      /*FollowupAllocs=*/FollowupAllocs);
  *ProloguePushBytes -= FixedObject;
  NumBytes -= FixedObject;

  // Now allocate space for the GPR callee saves.
  while (MBBI != MBB.end() && AArch64FrameLowering::isSVECalleeSave(MBBI))
    ++MBBI;

  return MBBI;
}

MachineBasicBlock::iterator AArch64WindowsSVEFrameLowering::deallocateSVEStack(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator FirstGPRRestoreI,
    int64_t &NumBytes, AArch64FrameLowering::FrameFlags &Flags,
    StackOffset CFAOffset, DebugLoc DL) const {
  auto [MFI, AFI, TII] = getCommonInfo(MBB);
  StackOffset SVECalleeSavesSize = getSVECalleeSavesSize(AFI);
  StackOffset SVELocalsSize = getSVELocalsSize(AFI);
  auto [RestoreBegin, RestoreEnd] = partitionSVEEpilogue(
      MBB, SVECalleeSavesSize ? MBB.getFirstTerminator() : FirstGPRRestoreI,
      SVECalleeSavesSize, SVELocalsSize);

  // If the callee-save area is before FP, restoring the FP implicitly
  // deallocates non-callee-save SVE allocations.  Otherwise, deallocate
  // them explicitly.
  if (!AFI.isStackRealigned() && !MFI.hasVarSizedObjects()) {
    emitFrameOffset(MBB, FirstGPRRestoreI, DL, AArch64::SP, AArch64::SP,
                    SVELocalsSize, TII, MachineInstr::FrameDestroy, false,
                    Flags.NeedsWinCFI, &Flags.HasWinCFI);
  }

  // Deallocate callee-save SVE registers.
  emitFrameOffset(MBB, RestoreEnd, DL, AArch64::SP, AArch64::SP,
                  SVECalleeSavesSize, TII, MachineInstr::FrameDestroy, false,
                  Flags.NeedsWinCFI, &Flags.HasWinCFI);

  return RestoreEnd;
}

} // namespace llvm
