// RUN: mlir-opt %s -split-input-file -verify-diagnostics

func.func @const_attribute_return_type_1() {
    // expected-error @+1 {{'emitc.constant' op requires attribute's type ('i64') to match op's return type ('i32')}}
    %c0 = "emitc.constant"(){value = 42: i64} : () -> i32
    return
}

// -----

func.func @const_attribute_return_type_2() {
    // expected-error @+1 {{'emitc.constant' op requires attribute's type ('!emitc.opaque<"char">') to match op's return type ('!emitc.opaque<"mychar">')}}
    %c0 = "emitc.constant"(){value = "CHAR_MIN" : !emitc.opaque<"char">} : () -> !emitc.opaque<"mychar">
    return
}

// -----

func.func @empty_constant() {
    // expected-error @+1 {{'emitc.constant' op value must not be empty}}
    %c0 = "emitc.constant"(){value = ""} : () -> i32
    return
}

// -----

func.func @index_args_out_of_range_1() {
    // expected-error @+1 {{'emitc.call' op index argument is out of range}}
    emitc.call "test" () {args = [0 : index]} : () -> ()
    return
}

// -----

func.func @index_args_out_of_range_2(%arg : i32) {
    // expected-error @+1 {{'emitc.call' op index argument is out of range}}
    emitc.call "test" (%arg, %arg) {args = [2 : index]} : (i32, i32) -> ()
    return
}

// -----

func.func @empty_callee() {
    // expected-error @+1 {{'emitc.call' op callee must not be empty}}
    emitc.call "" () : () -> ()
    return
}

// -----

func.func @nonetype_arg(%arg : i32) {
    // expected-error @+1 {{'emitc.call' op array argument has no type}}
    emitc.call "nonetype_arg"(%arg) {args = [0 : index, [0, 1, 2]]} : (i32) -> i32
    return
}

// -----

func.func @array_template_arg(%arg : i32) {
    // expected-error @+1 {{'emitc.call' op template argument has invalid type}}
    emitc.call "nonetype_template_arg"(%arg) {template_args = [[0, 1, 2]]} : (i32) -> i32
    return
}

// -----

func.func @dense_template_argument(%arg : i32) {
    // expected-error @+1 {{'emitc.call' op template argument has invalid type}}
    emitc.call "dense_template_argument"(%arg) {template_args = [dense<[1.0, 1.0]> : tensor<2xf32>]} : (i32) -> i32
    return
}

// -----

func.func @empty_operator(%arg : i32) {
    // expected-error @+1 {{'emitc.apply' op applicable operator must not be empty}}
    %2 = emitc.apply ""(%arg) : (i32) -> !emitc.ptr<i32>
    return
}

// -----

func.func @illegal_operator(%arg : i32) {
    // expected-error @+1 {{'emitc.apply' op applicable operator is illegal}}
    %2 = emitc.apply "+"(%arg) : (i32) -> !emitc.ptr<i32>
    return
}

// -----

func.func @illegal_operand() {
    %1 = "emitc.constant"(){value = 42: i32} : () -> i32
    // expected-error @+1 {{'emitc.apply' op cannot apply to constant}}
    %2 = emitc.apply "&"(%1) : (i32) -> !emitc.ptr<i32>
    return
}

// -----

func.func @var_attribute_return_type_1() {
    // expected-error @+1 {{'emitc.variable' op requires attribute's type ('i64') to match op's return type ('i32')}}
    %c0 = "emitc.variable"(){value = 42: i64} : () -> i32
    return
}

// -----

func.func @var_attribute_return_type_2() {
    // expected-error @+1 {{'emitc.variable' op requires attribute's type ('!emitc.ptr<i64>') to match op's return type ('!emitc.ptr<i32>')}}
    %c0 = "emitc.variable"(){value = "nullptr" : !emitc.ptr<i64>} : () -> !emitc.ptr<i32>
    return
}

// -----

func.func @cast_tensor(%arg : tensor<f32>) {
    // expected-error @+1 {{'emitc.cast' op operand type 'tensor<f32>' and result type 'tensor<f32>' are cast incompatible}}
    %1 = emitc.cast %arg: tensor<f32> to tensor<f32>
    return
}

// -----

func.func @add_two_pointers(%arg0: !emitc.ptr<f32>, %arg1: !emitc.ptr<f32>) {
    // expected-error @+1 {{'emitc.add' op requires that at most one operand is a pointer}}
    %1 = "emitc.add" (%arg0, %arg1) : (!emitc.ptr<f32>, !emitc.ptr<f32>) -> !emitc.ptr<f32>
    return
}

// -----

func.func @add_pointer_float(%arg0: !emitc.ptr<f32>, %arg1: f32) {
    // expected-error @+1 {{'emitc.add' op requires that one operand is an integer or of opaque type if the other is a pointer}}
    %1 = "emitc.add" (%arg0, %arg1) : (!emitc.ptr<f32>, f32) -> !emitc.ptr<f32>
    return
}

// -----

func.func @add_float_pointer(%arg0: f32, %arg1: !emitc.ptr<f32>) {
    // expected-error @+1 {{'emitc.add' op requires that one operand is an integer or of opaque type if the other is a pointer}}
    %1 = "emitc.add" (%arg0, %arg1) : (f32, !emitc.ptr<f32>) -> !emitc.ptr<f32>
    return
}

// -----

func.func @sub_int_pointer(%arg0: i32, %arg1: !emitc.ptr<f32>) {
    // expected-error @+1 {{'emitc.sub' op rhs can only be a pointer if lhs is a pointer}}
    %1 = "emitc.sub" (%arg0, %arg1) : (i32, !emitc.ptr<f32>) -> !emitc.ptr<f32>
    return
}


// -----

func.func @sub_pointer_float(%arg0: !emitc.ptr<f32>, %arg1: f32) {
    // expected-error @+1 {{'emitc.sub' op requires that rhs is an integer, pointer or of opaque type if lhs is a pointer}}
    %1 = "emitc.sub" (%arg0, %arg1) : (!emitc.ptr<f32>, f32) -> !emitc.ptr<f32>
    return
}

// -----

func.func @sub_pointer_pointer(%arg0: !emitc.ptr<f32>, %arg1: !emitc.ptr<f32>) {
    // expected-error @+1 {{'emitc.sub' op requires that the result is an integer or of opaque type if lhs and rhs are pointers}}
    %1 = "emitc.sub" (%arg0, %arg1) : (!emitc.ptr<f32>, !emitc.ptr<f32>) -> !emitc.ptr<f32>
    return
}

// -----

func.func @struct_read_missing_member(%arg0 : !emitc.struct<"string_view", #emitc.member<"data" : !emitc.ptr<i8>>, #emitc.member<"size" : index>>) {
  // expected-error @+1 {{'emitc.struct.member.read' op no member named 'length' in 'struct string_view'}}
  %0 = emitc.struct.member.read %arg0 <"length" : none> : (!emitc.struct<"string_view", #emitc.member<"data" : !emitc.ptr<i8>>, #emitc.member<"size" : index>>) -> index
  return
}

// -----

func.func @struct_read_invlaid_type(%arg0 : !emitc.struct<"string_view", #emitc.member<"data" : !emitc.ptr<i8>>, #emitc.member<"size" : index>>) {
  // expected-error @+1 {{'emitc.struct.member.read' op member named 'size' of type 'index' is incompatible with result of type 'f32'}}
  %0 = emitc.struct.member.read %arg0 <"size" : none> : (!emitc.struct<"string_view", #emitc.member<"data" : !emitc.ptr<i8>>, #emitc.member<"size" : index>>) -> f32
  return
}

// -----

func.func @struct_write_missing_member(%arg0 : !emitc.struct<"string_view", #emitc.member<"data" : !emitc.ptr<i8>>, #emitc.member<"size" : index>>, %arg1 : index) {
  // expected-error @+1 {{'emitc.struct.member.write' op no member named 'length' in 'struct string_view'}}
  emitc.struct.member.write %arg1 %arg0 <"length" : none> : index, !emitc.struct<"string_view", <"data" : !emitc.ptr<i8>>, <"size" : index>>
  return
}

// -----

func.func @struct_write_invalid_type(%arg0 : !emitc.struct<"string_view", #emitc.member<"data" : !emitc.ptr<i8>>, #emitc.member<"size" : index>>, %arg1 : f32) {
  // expected-error @+1 {{'emitc.struct.member.write' op value of type 'f32' is incompatible with member named 'size' of type 'index'}}
  emitc.struct.member.write %arg1 %arg0 <"size" : none> : f32, !emitc.struct<"string_view", <"data" : !emitc.ptr<i8>>, <"size" : index>>
  return
}

// -----

func.func @struct_write_array_type(%arg0 : !emitc.struct<"fixed_str", #emitc.member<"data" : !emitc.array<32, i8>>>, %arg1 : !emitc.array<32, i8>) {
  // expected-error @+1 {{'emitc.struct.member.write' op cannot assign a member of array type}}
  emitc.struct.member.write %arg1 %arg0 <"data" : none> : !emitc.array<32, i8>, !emitc.struct<"fixed_str", #emitc.member<"data" : !emitc.array<32, i8>>>
  return
}
