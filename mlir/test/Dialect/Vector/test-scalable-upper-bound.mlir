// RUN: mlir-opt %s -test-affine-reify-value-bounds -cse -verify-diagnostics \
// RUN:     -split-input-file | FileCheck %s

#fixedDim0Map = affine_map<(d0)[s0] -> (-d0 + 32400, s0)>
#fixedDim1Map = affine_map<(d0)[s0] -> (-d0 + 16, s0)>

// Here the upper bound for min_i is 4 x vscale, as we know 4 x vscale is
// always less than 32400. The bound for min_j is 16 as at vscale > 4,
// 4 x vscale will be > 16, so the value will be clamped at 16.

// CHECK-LABEL: @fixed_size_loop_nest
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[VSCALE:.*]] = vector.vscale
// CHECK-DAG: %[[C4_VSCALE:.*]] = arith.muli %[[VSCALE]], %[[C4]] : index
// CHECK: "test.some_use"(%[[C4_VSCALE]], %[[C16]]) : (index, index) -> ()
func.func @fixed_size_loop_nest() {
  %c16 = arith.constant 16 : index
  %c32400 = arith.constant 32400 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %vscale, %c4 : index
  scf.for %i = %c0 to %c32400 step %c4_vscale {
    %min_i = affine.min #fixedDim0Map(%i)[%c4_vscale]
    scf.for %j = %c0 to %c16 step %c4_vscale {
      %min_j = affine.min #fixedDim1Map(%j)[%c4_vscale]
      %bound_i = "test.reify_scalable_bound"(%min_i) {type = "UB"} : (index) -> index
      %bound_j = "test.reify_scalable_bound"(%min_j) {type = "UB"} : (index) -> index
      "test.some_use"(%bound_i, %bound_j) : (index, index) -> ()
    }
  }
  return
}

// -----

#dynamicDim0Map = affine_map<(d0, d1)[s0] -> (-d0 + d1, s0)>
#dynamicDim1Map = affine_map<(d0, d1)[s0] -> (-d0 + d1, s0)>

// Here upper bounds for both min_i and min_j are both 4 x vscale, as we know
// that is always the largest value they could take. As if `dim < 4 x vscale`
// then 4 x vscale is an overestimate, and if `dim > 4 x vscale` then the min
// will be clamped to 4 x vscale.

// CHECK-LABEL: @dynamic_size_loop_nest
//   CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
//   CHECK-DAG:   %[[VSCALE:.*]] = vector.vscale
//   CHECK-DAG:   %[[C4_VSCALE:.*]] = arith.muli  %[[VSCALE]], %[[C4]] : index
//       CHECK:   "test.some_use"(%[[C4_VSCALE]], %[[C4_VSCALE]]) : (index, index) -> ()
func.func @dynamic_size_loop_nest(%dim0: index, %dim1: index) {
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %vscale = vector.vscale
  %c4_vscale = arith.muli %vscale, %c4 : index
  scf.for %i = %c0 to %dim0 step %c4_vscale {
    %min_i = affine.min #dynamicDim0Map(%i)[%c4_vscale, %dim0]
    scf.for %j = %c0 to %dim1 step %c4_vscale {
      %min_j = affine.min #dynamicDim1Map(%j)[%c4_vscale, %dim1]
      %bound_i = "test.reify_scalable_bound"(%min_i) {type = "UB"} : (index) -> index
      %bound_j = "test.reify_scalable_bound"(%min_j) {type = "UB"} : (index) -> index
      "test.some_use"(%bound_i, %bound_j) : (index, index) -> ()
    }
  }
  return
}

// -----

// We don't know how to express a upper bound of the form `vscale + constant`
// so this conservatively returns 24 (the bound when vscale is 16).

// CHECK-LABEL: @add_to_vscale
//       CHECK:   %[[C24:.*]] = arith.constant 24 : index
//       CHECK:   "test.some_use"(%[[C24]]) : (index) -> ()
func.func @add_to_vscale() {
  %vscale = vector.vscale
  %c8 = arith.constant 8 : index
  %vscale_plus_c8 = arith.addi %vscale, %c8 : index
  %bound = "test.reify_scalable_bound"(%vscale_plus_c8) {type = "UB"} : (index) -> index
  "test.some_use"(%bound) : (index) -> ()
  return
}

// -----

// Here we know vscale is always 2 so we get a constant upper bound.

// CHECK-LABEL: @vscale_fixed_size
//       CHECK:   %[[C2:.*]] = arith.constant 2 : index
//       CHECK:   "test.some_use"(%[[C2]]) : (index) -> ()
func.func @vscale_fixed_size() {
  %vscale = vector.vscale
  %bound = "test.reify_scalable_bound"(%vscale) {type = "UB", vscale_min = 2, vscale_max = 2} : (index) -> index
  "test.some_use"(%bound) : (index) -> ()
  return
}
