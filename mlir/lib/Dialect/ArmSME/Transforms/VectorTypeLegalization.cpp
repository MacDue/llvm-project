#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "arm-sme-vector-type-legalization"

namespace mlir::arm_sme {
#define GEN_PASS_DEF_VECTORTYPELEGALIZATION
#include "mlir/Dialect/ArmSME/Transforms/Passes.h.inc"
} // namespace mlir::arm_sme

using namespace mlir;
using namespace mlir::arm_sme;

namespace {

struct VectorTypeLegalizationPass
    : public arm_sme::impl::VectorTypeLegalizationBase<
          VectorTypeLegalizationPass> {
  void runOnOperation() override {}
};

} // namespace

std::unique_ptr<Pass> mlir::arm_sme::createVectorTypeLegalizationPass() {
  return std::make_unique<VectorTypeLegalizationPass>();
}
