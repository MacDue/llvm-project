//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains declarations of the SVE frame lowering classes, which
/// implement hooks needed to lower SVE stack frames.
///
//===----------------------------------------------------------------------===//

#include "AArch64FrameLowering.h"
#include "AArch64MachineFunctionInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"

namespace llvm {

enum class SVEFramePosition {
  AboveFrameRecord,
  BelowFrameRecord,
};

struct OffsetPair {
  StackOffset FPOffset;
  StackOffset SPOffset;
};

class TargetInstrInfo;
class MachineFrameInfo;
class AArch64FunctionInfo;

/// The base implementation of the SVE frame lowering hooks. This implementation
/// is used on all targets other than Windows. This class (and implementations)
/// derived from it is expected to be a stateless collection of methods.
class AArch64BaseSVEFrameLowering {
protected:
  AArch64BaseSVEFrameLowering() = default;

public:
  /// Returns the singleton instance of `Arch64BaseSVEFrameLowering`.
  static AArch64BaseSVEFrameLowering &the() {
    static AArch64BaseSVEFrameLowering Impl;
    return Impl;
  }

  StackOffset getSVEStackSize(const AArch64FunctionInfo &AFI) const {
    return StackOffset::getScalable(AFI.getStackSizeSVE());
  }

  StackOffset getSVECalleeSavesSize(const AArch64FunctionInfo &AFI) const {
    return StackOffset::getScalable(AFI.getSVECalleeSavedStackSize());
  }

  StackOffset getSVELocalsSize(const AArch64FunctionInfo &AFI) const {
    return getSVEStackSize(AFI) - getSVECalleeSavesSize(AFI);
  }

  /// Returns the position of the SVE locals relative to the frame record.
  /// Note: Currently only "BelowFrameRecord" is supported.
  virtual SVEFramePosition getSVELocalsFramePosition() const {
    return SVEFramePosition::BelowFrameRecord;
  }

  /// Returns the position of the SVE locals relative to the frame record.
  virtual SVEFramePosition getSVECalleeSavesFramePosition() const {
    return SVEFramePosition::BelowFrameRecord;
  }

  /// Resolves "ObjectOffset" to a `StackOffset` from the stack pointer for a
  /// a given object type.
  virtual StackOffset
  getFrameIndexReferenceFromSP(const AArch64FunctionInfo &AFI, int ObjectOffset,
                               FrameObjectType) const;

  /// Returns the scalable portion of an offset to a non-SVE object (i.e. all
  /// types other than FrameObjectType::SVE). If \p UseFP is true, the offset
  /// should be determined using the FP as the base, otherwise the stack pointer
  /// should be used.
  virtual StackOffset
  determineScalableOffsetToNonSVEObject(const AArch64FunctionInfo &AFI,
                                        bool UseFP, FrameObjectType) const;

  /// Resolves the offset to an SVE stack object (from both the frame pointer
  /// and stack pointer).
  virtual OffsetPair resolveSVEObjectOffset(const AArch64FunctionInfo &AFI,
                                            const MachineFrameInfo &MFI,
                                            int ObjectOffset) const;

  /// Inserts the necessary MIs to allocate the SVE callee-saves. Returns an
  /// iterator pointing to the first instruction after the SVE callee-saves.
  ///
  /// If `SVEFramePosition == AboveFrameRecord` \p ProloguePushBytes is a
  /// pointer to the (non-SVE) prologue bytes yet to be allocated (FixedObject +
  /// non-SVE-CSRs). This hook can decrement \p ProloguePushBytes if it handles
  /// the allocation.
  ///
  /// Note: SVE locals are allocated at a fixed position in AArch64FrameLowering
  /// and are not handled by this hook.
  virtual MachineBasicBlock::iterator
  allocateSVECalleeSaves(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI,
                         unsigned *ProloguePushBytes, int64_t &NumBytes,
                         AArch64FrameLowering::FrameFlags &Flags,
                         StackOffset CFAOffset, bool FollowupAllocs) const;

  /// Inserts the necessary MIs to deallocate the SVE area (both callee-saves
  /// and locals). The input iterator can be assumed to point at the first GPR
  /// restore. Returns an iterator pointing to the end of the SVE callee-save
  /// restores.
  virtual MachineBasicBlock::iterator
  deallocateSVEStack(MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator FirstGPRRestoreI,
                     int64_t &NumBytes, AArch64FrameLowering::FrameFlags &Flags,
                     StackOffset CFAOffset, DebugLoc DL) const;

  virtual ~AArch64BaseSVEFrameLowering() = default;
};

/// The Windows implementation of the SVE frame lowering. The Windows lowering
/// places the SVE callee-saves above the frame record (to allow SEH to describe
/// the locations of SVE callee-saves).
class AArch64WindowsSVEFrameLowering : public AArch64BaseSVEFrameLowering {
  AArch64WindowsSVEFrameLowering() = default;

public:
  /// Returns the singleton instance of `AArch64WindowsSVEFrameLowering`.
  static AArch64WindowsSVEFrameLowering &the() {
    static AArch64WindowsSVEFrameLowering Impl;
    return Impl;
  }

  virtual SVEFramePosition getSVECalleeSavesFramePosition() const override {
    // Windows unwind can't represent the required stack adjustments if we have
    // both SVE callee-saves and dynamic stack allocations, and the frame
    // pointer is before the SVE spills.  The allocation of the frame pointer
    // must be the last instruction in the prologue so the unwinder can restore
    // the stack pointer correctly. (And there isn't any unwind opcode for
    // `addvl sp, x29, -17`.)
    //
    // Because of this, we do spills in the opposite order on Windows: first
    // SVE, then GPRs. The main side-effect of this is that it makes accessing
    // parameters passed on the stack more expensive.
    //
    // We could consider rearranging the spills for simpler cases.
    return SVEFramePosition::AboveFrameRecord;
  }

  virtual StackOffset
  getFrameIndexReferenceFromSP(const AArch64FunctionInfo &AFI, int ObjectOffset,
                               FrameObjectType) const override;

  virtual StackOffset
  determineScalableOffsetToNonSVEObject(const AArch64FunctionInfo &AFI,
                                        bool UseFP,
                                        FrameObjectType) const override;

  virtual OffsetPair resolveSVEObjectOffset(const AArch64FunctionInfo &AFI,
                                            const MachineFrameInfo &MFI,
                                            int ObjectOffset) const override;

  virtual MachineBasicBlock::iterator allocateSVECalleeSaves(
      MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
      unsigned *ProloguePushBytes, int64_t &NumBytes,
      AArch64FrameLowering::FrameFlags &Flags, StackOffset CFAOffset,
      bool FollowupAllocs) const override;

  virtual MachineBasicBlock::iterator
  deallocateSVEStack(MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator FirstGPRRestoreI,
                     int64_t &NumBytes, AArch64FrameLowering::FrameFlags &Flags,
                     StackOffset CFAOffset, DebugLoc DL) const override;
};

} // namespace llvm
