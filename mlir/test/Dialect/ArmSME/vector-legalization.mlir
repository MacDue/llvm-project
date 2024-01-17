// RUN: mlir-opt %s -arm-sme-vector-legalization -cse -canonicalize

func.func @outerproduct_f32_scalable_8x8_no_acc(%lhs: vector<[8]xf32>, %rhs: vector<[8]xf32>) -> vector<[8]x[8]xf32>
{
  %0 = vector.outerproduct %lhs, %rhs : vector<[8]xf32>, vector<[8]xf32>
  return %0 : vector<[8]x[8]xf32>
}

func.func @outerproduct_f32_scalable_8x8_acc(%lhs: vector<[8]xf32>, %rhs: vector<[8]xf32>, %acc: vector<[8]x[8]xf32>) -> vector<[8]x[8]xf32>
{
  %0 = vector.outerproduct %lhs, %rhs, %acc : vector<[8]xf32>, vector<[8]xf32>
  return %0 : vector<[8]x[8]xf32>
}

func.func @outerproduct_f32_masked_scalable_8x8(%lhs: vector<[8]xf32>, %rhs: vector<[8]xf32>, %lhs_dim: index, %rhs_dim: index) -> vector<[8]x[8]xf32>
{

  %mask = vector.create_mask %lhs_dim, %rhs_dim : vector<[8]x[8]xi1>
  %0 = vector.mask %mask { vector.outerproduct %lhs, %rhs : vector<[8]xf32>, vector<[8]xf32> } : vector<[8]x[8]xi1> -> vector<[8]x[8]xf32>
  return %0 : vector<[8]x[8]xf32>
}

func.func @outerproduct_f32_masked_scalable_8x8_acc(%lhs: vector<[8]xf32>, %rhs: vector<[8]xf32>, %acc: vector<[8]x[8]xf32>, %lhs_dim: index, %rhs_dim: index) -> vector<[8]x[8]xf32>
{
  %mask = vector.create_mask %lhs_dim, %rhs_dim : vector<[8]x[8]xi1>
  %0 = vector.mask %mask { vector.outerproduct %lhs, %rhs, %acc : vector<[8]xf32>, vector<[8]xf32> } : vector<[8]x[8]xi1> -> vector<[8]x[8]xf32>
  return %0 : vector<[8]x[8]xf32>
}

// This demonstrates a rectangular tiling that uses all f64 accumulators.
func.func @outerproduct_f64_scalable_8x4_no_acc(%lhs: vector<[8]xf64>, %rhs: vector<[4]xf64>) -> vector<[8]x[4]xf64>
{
  %0 = vector.outerproduct %lhs, %rhs : vector<[8]xf64>, vector<[4]xf64>
  return %0 : vector<[8]x[4]xf64>
}

func.func @transfer_read_f32_scalable_8x8(%src: memref<?x?xf32>) -> vector<[8]x[8]xf32>
{
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %0 = vector.transfer_read %src[%c0, %c0], %pad {in_bounds = [true, true]} : memref<?x?xf32>, vector<[8]x[8]xf32>
  return %0 : vector<[8]x[8]xf32>
}

func.func @transfer_read_f32_scalable_8x8_masked(%src: memref<?x?xf32>, %dim0: index, %dim1: index) -> vector<[8]x[8]xf32>
{
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : f32
  %mask = vector.create_mask %dim0, %dim0 : vector<[8]x[8]xi1>
  %0 = vector.transfer_read %src[%c0, %c0], %pad, %mask {in_bounds = [true, true]} : memref<?x?xf32>, vector<[8]x[8]xf32>
  return %0 : vector<[8]x[8]xf32>
}


func.func @transfer_write_f32_scalable_8x8(%dest: memref<?x?xf32>, %vec: vector<[8]x[8]xf32>)
{
  %c0 = arith.constant 0 : index
  vector.transfer_write %vec, %dest[%c0, %c0] {in_bounds = [true, true]} : vector<[8]x[8]xf32>, memref<?x?xf32>
  return
}

func.func @transfer_write_f32_scalable_8x8_masked(%dest: memref<?x?xf32>, %vec: vector<[8]x[8]xf32>, %dim0: index, %dim1: index)
{
  %c0 = arith.constant 0 : index
  %mask = vector.create_mask %dim0, %dim0 : vector<[8]x[8]xi1>
  vector.transfer_write %vec, %dest[%c0, %c0], %mask {in_bounds = [true, true]} : vector<[8]x[8]xf32>, memref<?x?xf32>
  return
}

// func.func @
