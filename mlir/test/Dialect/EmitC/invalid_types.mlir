// RUN: mlir-opt %s -split-input-file -verify-diagnostics

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

// expected-error @+1 {{duplicate member `a`}}
!illegal_struct_type_1 = !emitc.struct<"duplicate", #emitc.member<"a" : i32>, #emitc.member<"a" : f32>>

// -----

// expected-error @+1 {{nested struct types are not supported}}
!illegal_struct_type_2 = !emitc.struct<"nested", #emitc.member<"a" : !emitc.struct<"inner", #emitc.member<"value" : i32>>>>

// -----

// expected-error @+1 {{invalid identifier for struct type}}
!illegal_struct_type_3 = !emitc.struct<"invaild name", #emitc.member<"a" : i32>>

// -----

// expected-error @+1 {{invalid identifier for struct type}}
!illegal_struct_type_4 = !emitc.struct<"0_struct", #emitc.member<"a" : i32>>

// -----

// expected-error @+1 {{invalid identifier for struct type}}
!illegal_struct_type_5 = !emitc.struct<"", #emitc.member<"a" : i32>>