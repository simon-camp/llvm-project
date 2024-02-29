// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @illegal_array_type_1() {
    // expected-error @+1 {{arrays of pointers are not supported yet}}
    %1 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.array<!emitc.ptr<i32>, 1>
}

// -----

func.func @illegal_array_type_2() {
    // expected-error @+1 {{nested arrays are not supported yet}}
    %1 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.array<!emitc.array<i32, 1>, 1>
}

// -----

func.func @illegal_opaque_type_1() {
    // expected-error @+1 {{expected non empty string in !emitc.opaque type}}
    %1 = "emitc.variable"() {value = "42" : !emitc.opaque<"">} : () -> !emitc.opaque<"mytype">
}

// -----

func.func @illegal_opaque_type_2() {
    // expected-error @+1 {{pointer not allowed as outer type with !emitc.opaque, use !emitc.ptr instead}}
    %1 = "emitc.variable"() {value = "nullptr" : !emitc.opaque<"int32_t*">} : () -> !emitc.opaque<"int32_t*">
}

// -----

func.func @illegal_pointer_type_1() {
    // expected-error @+1 {{pointers to arrays are not supported yet}}
    %1 = "emitc.variable"() {value = #emitc.opaque<"">} : () -> !emitc.ptr<!emitc.array<i32, 1>>
}
