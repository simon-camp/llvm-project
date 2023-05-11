// RUN: mlir-opt -pass-pipeline="builtin.module(func.func(convert-arith-to-emitc))" %s -split-input-file | FileCheck %s

// CHECK-LABEL: @ops
func.func @ops(f32, f32, i32, i32, f64) -> (f32, i32) {
^bb0(%arg0: f32, %arg1: f32, %arg2: i32, %arg3: i32, %arg4: f64):
// CHECK: = emitc.sub %arg0, %arg1 : (f32, f32) -> f32
  %0 = arith.subf %arg0, %arg1: f32
// CHECK: = emitc.sub %arg2, %arg3 : (i32, i32) -> i32
  %1 = arith.subi %arg2, %arg3: i32
// CHECK: = emitc.div %arg0, %arg1 : (f32, f32) -> f32
  %8 = arith.divf %arg0, %arg1 : f32
// CHECK: = "emitc.constant"() <{value = 7.900000e-01 : f64}> : () -> f64
  %15 = arith.constant 7.9e-01 : f64
  return %0, %1 : f32, i32
}

// CHECK-LABEL: @index_cast
func.func @index_cast(%arg0: index, %arg1: i32) {
// CHECK: = emitc.cast %arg0 : index to i32
  %0 = arith.index_cast %arg0: index to i32
// CHECK-NEXT: = emitc.cast %arg1 : i32 to index
  %1 = arith.index_cast %arg1: i32 to index
  return
}
