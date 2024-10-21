// DEFINE: %{pipeline} = "builtin.module(\
// DEFINE:   func.func(\
// DEFINE:     tosa-to-linalg-named,\
// DEFINE:     tosa-to-linalg,\
// DEFINE:     linalg-generalize-named-ops,\
// DEFINE:     tosa-to-arith,\
// DEFINE:     tosa-to-tensor\
// DEFINE:   ),\
// DEFINE:   one-shot-bufferize{\
// DEFINE:     allow-unknown-ops\
// DEFINE:     bufferize-function-boundaries\
// DEFINE:     function-boundary-type-conversion=identity-layout-map\
// DEFINE:     buffer-alignment=0\
// DEFINE:   },\
// DEFINE:   buffer-results-to-out-params{\
// DEFINE:     hoist-static-allocs=true\
// DEFINE:   },\
// DEFINE:   func.func(\
// DEFINE:     promote-buffers-to-stack{\
// DEFINE:       max-alloc-size-in-bytes=1024\
// DEFINE:       max-rank-of-allocated-memref=1\
// DEFINE:     },\
// DEFINE:     convert-linalg-to-loops,\
// DEFINE:     convert-arith-to-emitc,\
// DEFINE:     convert-memref-to-emitc\
// DEFINE:   ),\
// DEFINE:   convert-func-to-emitc,\
// DEFINE;   reconcile-unrealized-casts,\
// DEFINE:   canonicalize\
// DEFINE: )"

// RUN: mlir-opt --pass-pipeline=%{pipeline} %s | FileCheck %s
// RUN: mlir-opt --pass-pipeline=%{pipeline} %s | mlir-translate --mlir-to-cpp

// -----

// CHECK-NOT: tosa
// CHECK-NOT: linalg
// CHECK-NOT: memref
// CHECK-NOT: arith

module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<f32> {ml_program.identifier = "serve_b:0", tf_saved_model.index_path = ["b"]}, %arg1: tensor<f32> {ml_program.identifier = "serve_a:0", tf_saved_model.index_path = ["a"]}) -> (tensor<f32> {ml_program.identifier = "PartitionedCall:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serve"]} {
    %0 = tosa.add %arg1, %arg0 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %1 = tosa.mul %arg1, %arg0 {shift = 0 : i8} : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = tosa.add %1, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %3 = tosa.add %2, %1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %4 = tosa.add %arg1, %3 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %5 = tosa.mul %1, %arg1 {shift = 0 : i8} : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %6 = tosa.sub %5, %4 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %7 = tosa.reciprocal %5 : (tensor<f32>) -> tensor<f32>
    %8 = tosa.mul %6, %7 {shift = 0 : i8} : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %9 = tosa.sub %1, %arg0 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %10 = tosa.add %0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %11 = tosa.add %arg0, %10 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %12 = tosa.add %9, %11 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %13 = tosa.add %1, %12 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %14 = tosa.add %8, %13 : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %14 : tensor<f32>
  }
}
