//===- ValueBoundsOpInterfaceImpl.cpp - Impl. of ValueBoundsOpInterface ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Vector/IR/ValueBoundsOpInterfaceImpl.h"

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"

using namespace mlir;

namespace mlir::vector {
namespace {

struct VectorScaleOpInterface
    : public ValueBoundsOpInterface::ExternalModel<VectorScaleOpInterface,
                                                   VectorScaleOp> {
  void populateBoundsForIndexValue(Operation *op, Value value,
                                   ValueBoundsConstraintSet &cstr) const {
    auto vscaleOp = cast<VectorScaleOp>(op);
    assert(value == vscaleOp.getResult() && "invalid value");
    cstr.bound(value) >= 1;
    cstr.bound(value) <= 16;
  }
};

} // namespace
} // namespace mlir::vector

void mlir::vector::registerValueBoundsOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, vector::VectorDialect *dialect) {
    vector::VectorScaleOp::attachInterface<vector::VectorScaleOpInterface>(
        *ctx);
  });
}
