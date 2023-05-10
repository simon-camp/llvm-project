// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s

func.func @eq(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: f32) {
  %1 = "emitc.eq" (%arg0, %arg1) : (i32, i32) -> i1
  %2 = "emitc.eq" (%arg2, %arg3) : (f32, f32) -> i1
  return
}
// CHECK-LABEL: void eq
// CHECK-NEXT:  bool [[V4:[^ ]*]] = [[V0:[^ ]*]] == [[V1:[^ ]*]]
// CHECK-NEXT:  bool [[V5:[^ ]*]] = [[V2:[^ ]*]] == [[V3:[^ ]*]]

func.func @ge(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: f32) {
  %1 = "emitc.ge" (%arg0, %arg1) : (i32, i32) -> i1
  %2 = "emitc.ge" (%arg2, %arg3) : (f32, f32) -> i1
  return
}
// CHECK-LABEL: void ge
// CHECK-NEXT:  bool [[V4:[^ ]*]] = [[V0:[^ ]*]] >= [[V1:[^ ]*]]
// CHECK-NEXT:  bool [[V5:[^ ]*]] = [[V2:[^ ]*]] >= [[V3:[^ ]*]]

func.func @gt(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: f32) {
  %1 = "emitc.gt" (%arg0, %arg1) : (i32, i32) -> i1
  %2 = "emitc.gt" (%arg2, %arg3) : (f32, f32) -> i1
  return
}
// CHECK-LABEL: void gt
// CHECK-NEXT:  bool [[V4:[^ ]*]] = [[V0:[^ ]*]] > [[V1:[^ ]*]]
// CHECK-NEXT:  bool [[V5:[^ ]*]] = [[V2:[^ ]*]] > [[V3:[^ ]*]]

func.func @le(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: f32) {
  %1 = "emitc.le" (%arg0, %arg1) : (i32, i32) -> i1
  %2 = "emitc.le" (%arg2, %arg3) : (f32, f32) -> i1
  return
}
// CHECK-LABEL: void le
// CHECK-NEXT:  bool [[V4:[^ ]*]] = [[V0:[^ ]*]] <= [[V1:[^ ]*]]
// CHECK-NEXT:  bool [[V5:[^ ]*]] = [[V2:[^ ]*]] <= [[V3:[^ ]*]]

func.func @lt(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: f32) {
  %1 = "emitc.lt" (%arg0, %arg1) : (i32, i32) -> i1
  %2 = "emitc.lt" (%arg2, %arg3) : (f32, f32) -> i1
  return
}
// CHECK-LABEL: void lt
// CHECK-NEXT:  bool [[V4:[^ ]*]] = [[V0:[^ ]*]] < [[V1:[^ ]*]]
// CHECK-NEXT:  bool [[V5:[^ ]*]] = [[V2:[^ ]*]] < [[V3:[^ ]*]]

func.func @ne(%arg0: i32, %arg1: i32, %arg2: f32, %arg3: f32) {
  %1 = "emitc.ne" (%arg0, %arg1) : (i32, i32) -> i1
  %2 = "emitc.ne" (%arg2, %arg3) : (f32, f32) -> i1
  return
}
// CHECK-LABEL: void ne
// CHECK-NEXT:  bool [[V4:[^ ]*]] = [[V0:[^ ]*]] != [[V1:[^ ]*]]
// CHECK-NEXT:  bool [[V5:[^ ]*]] = [[V2:[^ ]*]] != [[V3:[^ ]*]]
