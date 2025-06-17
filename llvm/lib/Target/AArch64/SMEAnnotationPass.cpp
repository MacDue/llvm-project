

#include "AArch64.h"
#include "Utils/AArch64SMEAttributes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

using namespace llvm;
#define DEBUG_TYPE "aarch64-sme-annotation"

namespace {

static constexpr StringLiteral SME_ANNOTATED_ATTR = "aarch64_sme_annotated";

struct SMEAnnotationContext {
  SMEAnnotationContext(Function &F)
      : M(F.getParent()), F(&F), Builder(F.getContext()), FnAttrs(F) {
    ZaType = TargetExtType::get(F.getContext(), "aarch64.za.generation");
  }

  Module *M = nullptr;
  Function *F = nullptr;
  IRBuilder<> Builder;
  SMEAttrs FnAttrs;
  Type *ZaType = nullptr;
  AllocaInst *ZaAlloca = nullptr;

  template <typename... Args>
  CallInst *CreateIntr(Intrinsic::ID IID, Args... CallArgs) {
    Function *Intr = Intrinsic::getOrInsertDeclaration(M, IID);
    return Builder.CreateCall(Intr->getFunctionType(), Intr,
                              ArrayRef<Value *>{CallArgs...});
  }

  CallInst *CreateGetCurrentZAStateIntr() {
    return CreateIntr(Intrinsic::aarch64_sme_current_za_state);
  }
  CallInst *CreateMarkUpdateZAStateIntr(Value *PrevZaState) {
    return CreateIntr(Intrinsic::aarch64_sme_mark_update_za_state, PrevZaState);
  }
  CallInst *CreateMarkUseZAStateIntr(Value *ZaState) {
    return CreateIntr(Intrinsic::aarch64_sme_mark_use_za_state, ZaState);
  }
  CallInst *CreateZAClobberIntr() {
    return CreateIntr(Intrinsic::aarch64_sme_clobber_za_state);
  }
};

static void SetupZAEntryAndExits(SMEAnnotationContext &Ctx) {
  BasicBlock *EntryBlock = &Ctx.F->getEntryBlock();
  Ctx.Builder.SetInsertPoint(&EntryBlock->front());
  Ctx.ZaAlloca = Ctx.Builder.CreateAlloca(Ctx.ZaType, nullptr, "za.state");

  // Only the alloca needs to be created for new ZA functions.
  if (!Ctx.FnAttrs.hasSharedZAInterface())
    return;

  // Use current ZA value on entry (since for shared ZA it is livein).
  auto *CurrentZAValue = Ctx.CreateGetCurrentZAStateIntr();
  Ctx.Builder.CreateStore(CurrentZAValue, Ctx.ZaAlloca);
  for (auto Block = Ctx.F->begin(), E = Ctx.F->end(); Block != E; ++Block) {
    auto *Return = dyn_cast<ReturnInst>(Block->getTerminator());
    if (!Return)
      continue;
    // Mark the current ZA state as used at exits (since for shared ZA it is
    // liveout).
    Ctx.Builder.SetInsertPoint(Return);
    Value *ExitZaState =
        Ctx.Builder.CreateLoad(Ctx.ZaType, Ctx.ZaAlloca, "za.exit.state");
    Ctx.CreateMarkUseZAStateIntr(ExitZaState);
  }
}

/// Does this intrinsic update ZA state? E.g. Load a tile slice.
/// TODO: Somehow table-generate this?
static bool isZAUpdate(Intrinsic::ID IID) {
  switch (IID) {
  default:
    return false;
  // Tile loads:
  case Intrinsic::aarch64_sme_ld1b_horiz:
  case Intrinsic::aarch64_sme_ld1b_vert:
  case Intrinsic::aarch64_sme_ld1h_horiz:
  case Intrinsic::aarch64_sme_ld1h_vert:
  case Intrinsic::aarch64_sme_ld1w_horiz:
  case Intrinsic::aarch64_sme_ld1w_vert:
  case Intrinsic::aarch64_sme_ld1d_horiz:
  case Intrinsic::aarch64_sme_ld1d_vert:
  case Intrinsic::aarch64_sme_ld1q_horiz:
  case Intrinsic::aarch64_sme_ld1q_vert:
  // ZA fill:
  case Intrinsic::aarch64_sme_ldr:
  // Vector to tile:
  case Intrinsic::aarch64_sme_write_horiz:
  case Intrinsic::aarch64_sme_write_vert:
  case Intrinsic::aarch64_sme_writeq_horiz:
  case Intrinsic::aarch64_sme_writeq_vert:
  // Zero:
  case Intrinsic::aarch64_sme_zero:
  // MOPA:
  case Intrinsic::aarch64_sme_mopa:
  case Intrinsic::aarch64_sme_mops:
  case Intrinsic::aarch64_sme_mopa_wide:
  case Intrinsic::aarch64_sme_mops_wide:
  case Intrinsic::aarch64_sme_smopa_wide:
  case Intrinsic::aarch64_sme_smops_wide:
  case Intrinsic::aarch64_sme_umopa_wide:
  case Intrinsic::aarch64_sme_umops_wide:
  case Intrinsic::aarch64_sme_usmopa_wide:
  case Intrinsic::aarch64_sme_usmops_wide:
  case Intrinsic::aarch64_sme_sumopa_wide:
  case Intrinsic::aarch64_sme_sumops_wide:
    return true;
    // TODO: Finish list...
  }
}

/// Does this intrinsic use/read ZA state? E.g. Store a tile slice.
/// TODO: Somehow table-generate this?
static bool isZAUse(Intrinsic::ID IID) {
  switch (IID) {
  default:
    return false;
    // Tile stores:
  case Intrinsic::aarch64_sme_st1b_horiz:
  case Intrinsic::aarch64_sme_st1b_vert:
  case Intrinsic::aarch64_sme_st1h_horiz:
  case Intrinsic::aarch64_sme_st1h_vert:
  case Intrinsic::aarch64_sme_st1w_horiz:
  case Intrinsic::aarch64_sme_st1w_vert:
  case Intrinsic::aarch64_sme_st1d_horiz:
  case Intrinsic::aarch64_sme_st1d_vert:
  case Intrinsic::aarch64_sme_st1q_horiz:
  case Intrinsic::aarch64_sme_st1q_vert:
    // ZA spill:
  case Intrinsic::aarch64_sme_str:
  // Tile to vector:
  case Intrinsic::aarch64_sme_read_horiz:
  case Intrinsic::aarch64_sme_read_vert:
  case Intrinsic::aarch64_sme_readq_horiz:
  case Intrinsic::aarch64_sme_readq_vert:
  case Intrinsic::aarch64_sme_readz_horiz_x2:
  case Intrinsic::aarch64_sme_readz_vert_x2:
  case Intrinsic::aarch64_sme_readz_horiz_x4:
  case Intrinsic::aarch64_sme_readz_vert_x4:
  case Intrinsic::aarch64_sme_readz_horiz:
  case Intrinsic::aarch64_sme_readz_vert:
  case Intrinsic::aarch64_sme_readz_q_horiz:
  case Intrinsic::aarch64_sme_readz_q_vert:
  case Intrinsic::aarch64_sme_readz_x2:
  case Intrinsic::aarch64_sme_readz_x4:
    return true;
    // TODO: Finish list...
  }
}

enum class ZAStateUsage {
  None,
  Clobber,
  Call,
  Update,
  Use,
};

static ZAStateUsage getZAStateUsage(Instruction *Inst) {
  if (auto *Intr = dyn_cast<IntrinsicInst>(Inst)) {
    Intrinsic::ID IID = Intr->getIntrinsicID();
    if (isZAUpdate(IID))
      return ZAStateUsage::Update;
    if (isZAUse(IID))
      return ZAStateUsage::Use;
    return ZAStateUsage::None;
  }
  auto *CallInst = dyn_cast<CallBase>(Inst);
  if (!CallInst)
    return ZAStateUsage::None;
  SMECallAttrs CallAttrs(*CallInst);
  if (CallAttrs.clobbersZAState())
    return ZAStateUsage::Clobber;
  return ZAStateUsage::Call;
}

static void insertSMEAnnotations(SMEAnnotationContext &Ctx) {
  Ctx.F->addFnAttr(SME_ANNOTATED_ATTR);

  if (!Ctx.FnAttrs.hasZAState())
    return;

  SetupZAEntryAndExits(Ctx);

  for (auto Block = Ctx.F->begin(), E = Ctx.F->end(); Block != E; ++Block) {
    for (Instruction &I : make_early_inc_range(
             make_range(Block->getFirstNonPHIIt(), Block->end()))) {
      Ctx.Builder.SetInsertPoint(&I);
      auto Usage = getZAStateUsage(&I);
      switch (Usage) {
      case ZAStateUsage::Clobber:
        Ctx.CreateZAClobberIntr();
        break;
      case ZAStateUsage::Use:
      case ZAStateUsage::Update:
      case ZAStateUsage::Call: {
        Value *ZaState =
            Ctx.Builder.CreateLoad(Ctx.ZaType, Ctx.ZaAlloca, "za.state");
        if (Usage == ZAStateUsage::Update) {
          Value *NewZaState = Ctx.CreateMarkUpdateZAStateIntr(ZaState);
          Ctx.Builder.SetInsertPoint(I.getNextNode());
          Ctx.Builder.CreateStore(NewZaState, Ctx.ZaAlloca);
        } else {
          Ctx.CreateMarkUseZAStateIntr(ZaState);
          if (Usage == ZAStateUsage::Call) {
            Ctx.Builder.SetInsertPoint(I.getNextNode());
            Value *NewZaState = Ctx.CreateGetCurrentZAStateIntr();
            Ctx.Builder.CreateStore(NewZaState, Ctx.ZaAlloca);
          }
        }
        break;
      }
      default:
        break;
      }
    }
  }

  return;
}

} // namespace

PreservedAnalyses SMEAnnotationPass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  if (F.isDeclaration() || F.hasFnAttribute(SME_ANNOTATED_ATTR))
    return PreservedAnalyses::all();

  SMEAnnotationContext Ctx(F);
  insertSMEAnnotations(Ctx);

  // TODO: Do we need to mark anything as preserved?
  return PreservedAnalyses::none();
}
