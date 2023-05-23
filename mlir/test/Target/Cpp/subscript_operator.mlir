// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s

func.func @subscript_read(%arg0: !emitc.ptr<i8>, %arg1: index) {
  %1 = "emitc.subscript.read" (%arg0, %arg1) : (!emitc.ptr<i8>, index) -> i8
  return
}
// CHECK-LABEL: void subscript_read(
// CHECK-SAME: int8_t* [[V0:[^ ]*]], size_t [[V1:[^ ]*]]) {
// CHECK-NEXT: int8_t [[V2:[^ ]*]] = [[V0]][[[V1]]];

func.func @subscript_write(%arg0 : i8, %arg1: !emitc.ptr<i8>, %arg2: index) {
  "emitc.subscript.write" (%arg0, %arg1, %arg2) : (i8, !emitc.ptr<i8>, index) -> ()
  return
}
// CHECK-LABEL: void subscript_write(
// CHECK-SAME: int8_t [[V0:[^ ]*]], int8_t* [[V1:[^ ]*]], size_t [[V2:[^ ]*]]) {
// CHECK-NEXT: [[V1]][[[V2]]] = [[V0]];

