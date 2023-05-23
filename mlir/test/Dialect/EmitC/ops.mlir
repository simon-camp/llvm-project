// RUN: mlir-opt %s | mlir-opt | FileCheck %s
// RUN: mlir-opt %s -canonicalize | FileCheck %s

// CHECK: emitc.include <"test.h">
// CHECK: emitc.include "test.h"
emitc.include <"test.h">
emitc.include "test.h"

// CHECK-LABEL: func @f(%{{.*}}: i32, %{{.*}}: !emitc.opaque<"int32_t">) {
func.func @f(%arg0: i32, %f: !emitc.opaque<"int32_t">) {
  %1 = "emitc.call"() {callee = "blah"} : () -> i64
  emitc.call "foo" (%1) {args = [
    0 : index, dense<[0, 1]> : tensor<2xi32>, 0 : index
  ]} : (i64) -> ()
  return
}

func.func @cast(%arg0: i32) {
  %1 = emitc.cast %arg0: i32 to f32
  return
}

func.func @c() {
  %1 = "emitc.constant"(){value = 42 : i32} : () -> i32
  return
}

func.func @a(%arg0: i32, %arg1: i32) {
  %1 = "emitc.apply"(%arg0) {applicableOperator = "&"} : (i32) -> !emitc.ptr<i32>
  %2 = emitc.apply "&"(%arg1) : (i32) -> !emitc.ptr<i32>
  return
}

func.func @add_int(%arg0: i32, %arg1: i32) {
  %1 = "emitc.add" (%arg0, %arg1) : (i32, i32) -> i32
  return
}

func.func @add_pointer(%arg0: !emitc.ptr<f32>, %arg1: i32, %arg2: !emitc.opaque<"unsigned int">) {
  %1 = "emitc.add" (%arg0, %arg1) : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
  %2 = "emitc.add" (%arg0, %arg2) : (!emitc.ptr<f32>, !emitc.opaque<"unsigned int">) -> !emitc.ptr<f32>
  return
}

func.func @sub_int(%arg0: i32, %arg1: i32) {
  %1 = "emitc.sub" (%arg0, %arg1) : (i32, i32) -> i32
  return
}

func.func @sub_pointer(%arg0: !emitc.ptr<f32>, %arg1: i32, %arg2: !emitc.opaque<"unsigned int">, %arg3: !emitc.ptr<f32>) {
  %1 = "emitc.sub" (%arg0, %arg1) : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
  %2 = "emitc.sub" (%arg0, %arg2) : (!emitc.ptr<f32>, !emitc.opaque<"unsigned int">) -> !emitc.ptr<f32>
  %3 = "emitc.sub" (%arg0, %arg3) : (!emitc.ptr<f32>, !emitc.ptr<f32>) -> !emitc.opaque<"ptrdiff_t">
  %4 = "emitc.sub" (%arg0, %arg3) : (!emitc.ptr<f32>, !emitc.ptr<f32>) -> i32
  return
}

func.func @mul_int(%arg0: i32, %arg1: i32) {
  %1 = "emitc.mul" (%arg0, %arg1) : (i32, i32) -> i32
  return
}

func.func @mul_float(%arg0: f64, %arg1: f64) {
  %1 = "emitc.mul" (%arg0, %arg1) : (f64, f64) -> f64
  return
}

func.func @div_int(%arg0: i32, %arg1: i32) {
  %1 = "emitc.div" (%arg0, %arg1) : (i32, i32) -> i32
  return
}

func.func @div_float(%arg0: f64, %arg1: f64) {
  %1 = "emitc.div" (%arg0, %arg1) : (f64, f64) -> f64
  return
}

func.func @arrays() {
  %0 = "emitc.constant"(){value = #emitc.opaque<"{0}">} : () -> !emitc.array<3, i32>
  %1 = "emitc.constant"(){value = 1 : index} : () -> index
  %2 = emitc.subscript.read %0, %1 : (!emitc.array<3, i32>, index) -> i32
  %3 = emitc.add %2, %2 : (i32, i32) -> i32
  emitc.subscript.write %3, %0, %1 : i32, !emitc.array<3, i32>, index
  return
}

emitc.struct.def !emitc.struct<"string_view", #emitc.member<"data" : !emitc.ptr<i8>>, #emitc.member<"size" : index>>
func.func @structs(%arg0 : !emitc.struct<"string_view", #emitc.member<"data" : !emitc.ptr<i8>>, #emitc.member<"size" : index>>) {
  %0 = emitc.struct.member.read %arg0 <"size" : none> : (!emitc.struct<"string_view", #emitc.member<"data" : !emitc.ptr<i8>>, #emitc.member<"size" : index>>) -> index
  emitc.struct.member.write %0 %arg0 <"size" : none> : index, !emitc.struct<"string_view", <"data" : !emitc.ptr<i8>>, <"size" : index>>
  return
}

!struct_type_a = !emitc.struct<"struct_array", #emitc.member<"array" : !emitc.array<3, i8>>, #emitc.member<"pointer" : !emitc.ptr<i8>>>
emitc.struct.def !struct_type_a
func.func @struct_array_member(%arg0 : !struct_type_a, %arg1 : !emitc.array<3, i8>) {
  %0 = emitc.struct.member.read %arg0 <"array" : none> : (!struct_type_a) -> !emitc.ptr<i8>
  emitc.struct.member.write %arg1 %arg0 <"pointer" : none> :  !emitc.array<3, i8>, !struct_type_a
  return
}
