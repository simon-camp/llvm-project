// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @illegal_array_type_1() {
    // expected-error @+1 {{multidimensional arrays are not supported}}
    %1 = "emitc.constant"(){value = #emitc.opaque<"{{0.0f, 1.0f}, {2.0f, 3.0f}}">} : () -> !emitc.array<2, !emitc.array<2, f32>>
}

// -----

func.func @illegal_array_type_2() {
    // expected-error @+1 {{elements of type pointer are not supported}}
    %1 = "emitc.constant"(){value = #emitc.opaque<"{NULL}">} : () -> !emitc.array<1, !emitc.ptr<i32>>
}

// -----

func.func @illegal_opaque_type_1() {
    // expected-error @+1 {{expected non empty string in !emitc.opaque type}}
    %1 = "emitc.variable"(){value = "42" : !emitc.opaque<"">} : () -> !emitc.opaque<"mytype">
}

// -----

func.func @illegal_opaque_type_2() {
    // expected-error @+1 {{pointer not allowed as outer type with !emitc.opaque, use !emitc.ptr instead}}
    %1 = "emitc.variable"(){value = "nullptr" : !emitc.opaque<"int32_t*">} : () -> !emitc.opaque<"int32_t*">
}

// -----

func.func @illegal_pointer_type_1() {
    // expected-error @+1 {{pointers to arrays are not supported}}
    %1 = "emitc.constant"(){value = #emitc.opaque<"NULL">} : () -> !emitc.ptr<!emitc.array<1, i32>>
}
