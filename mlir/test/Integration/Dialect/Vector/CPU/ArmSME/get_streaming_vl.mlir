// RUN: mlir-opt %s -convert-arm-sme-to-llvm -test-lower-to-llvm | \
// RUN: %mcr_aarch64_cmd -march=aarch64 -mattr=+sve,+sme \
// RUN:  -e entry -entry-point-result=void \
// RUN:  -shared-libs=%mlir_runner_utils,%mlir_c_runner_utils,%arm_sme_abi_shlib | \
// RUN: FileCheck %s

func.func @get_streaming_vl_non_streaming() {
  %svl_b = arm_sme.streaming_vl <bytes>
  %svl_h = arm_sme.streaming_vl <half_words>
  %svl_w = arm_sme.streaming_vl <words>
  %svl_d = arm_sme.streaming_vl <double_words>

  // CHECK-LABEL: get_streaming_vl_non_streaming


  vector.print str "get_streaming_vl_non_streaming:"
  vector.print %svl_b : index
  vector.print %svl_h : index
  vector.print %svl_w : index
  vector.print %svl_d : index
  return
}

func.func @get_streaming_vl_streaming() attributes { arm_locally_streaming } {
  %svl_b = arm_sme.streaming_vl <bytes>
  %svl_h = arm_sme.streaming_vl <half_words>
  %svl_w = arm_sme.streaming_vl <words>
  %svl_d = arm_sme.streaming_vl <double_words>
  vector.print str "get_streaming_vl_streaming:"
  vector.print %svl_b : index
  vector.print %svl_h : index
  vector.print %svl_w : index
  vector.print %svl_d : index
  return
}

func.func @entry() {
  func.call @get_streaming_vl_non_streaming() : () -> ()
  func.call @get_streaming_vl_streaming() : () -> ()
  return
}
