//===- EmitC.td - EmitC operations--------------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the MLIR EmitC operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_EMITC_IR_EMITC
#define MLIR_DIALECT_EMITC_IR_EMITC

include "mlir/Dialect/EmitC/IR/EmitCAttributes.td"
include "mlir/Dialect/EmitC/IR/EmitCTypes.td"

include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

//===----------------------------------------------------------------------===//
// EmitC op definitions
//===----------------------------------------------------------------------===//

// Base class for EmitC dialect ops.
class EmitC_Op<string mnemonic, list<Trait> traits = []>
    : Op<EmitC_Dialect, mnemonic, traits>;

// Base class for binary operations.
class EmitC_BinaryOp<string mnemonic, list<Trait> traits = []> :
    EmitC_Op<mnemonic, traits> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

  let hasVerifier = 1;
}

def EmitC_AddOp : EmitC_BinaryOp<"add", [Commutative]> {
  let summary = "Addition operation";
  let description = [{
    With the `add` operation the arithmetic operator + (addition) can
    be applied.

    Example:

    ```mlir
    // Custom form of the addition operation.
    %0 = emitc.add %arg0, %arg1 : (i32, i32) -> i32
    %1 = emitc.add %arg2, %arg3 : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>

    // Generic form of the same operations.
    %0 = "emitc.add" (%arg0, %arg1) : (i32, i32) -> i32
    %1 = "emitc.add" (%arg2, %arg3) : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
    ```
  }];
}

def EmitC_ApplyOp : EmitC_Op<"apply", []> {
  let summary = "Apply operation";
  let description = [{
    With the `apply` operation the operators & (address of) and * (contents of)
    can be applied to a single operand.

    Example:

    ```mlir
    // Custom form of applying the & operator.
    %0 = emitc.apply "&"(%arg0) : (i32) -> !emitc.ptr<i32>

    // Generic form of the same operation.
    %0 = "emitc.apply"(%arg0) {applicableOperator = "&"}
        : (i32) -> !emitc.ptr<i32>

    ```
  }];
  let arguments = (ins
    Arg<StrAttr, "the operator to apply">:$applicableOperator,
    AnyType:$operand
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $applicableOperator `(` $operand `)` attr-dict `:` functional-type($operand, results)
  }];
  let hasVerifier = 1;
}

def EmitC_CallOp : EmitC_Op<"call", []> {
  let summary = "Call operation";
  let description = [{
    The `call` operation represents a C++ function call. The call allows
    specifying order of operands and attributes in the call as follows:

    - integer value of index type refers to an operand;
    - attribute which will get lowered to constant value in call;

    Example:

    ```mlir
    // Custom form defining a call to `foo()`.
    %0 = emitc.call "foo" () : () -> i32

    // Generic form of the same operation.
    %0 = "emitc.call"() {callee = "foo"} : () -> i32
    ```
  }];
  let arguments = (ins
    Arg<StrAttr, "the C++ function to call">:$callee,
    Arg<OptionalAttr<ArrayAttr>, "the order of operands and further attributes">:$args,
    Arg<OptionalAttr<ArrayAttr>, "template arguments">:$template_args,
    Variadic<AnyType>:$operands
  );
  let results = (outs Variadic<AnyType>);
  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
  let hasVerifier = 1;
}

def EmitC_CastOp : EmitC_Op<"cast", [
    DeclareOpInterfaceMethods<CastOpInterface>,
    SameOperandsAndResultShape
  ]> {
  let summary = "Cast operation";
  let description = [{
    The `cast` operation performs an explicit type conversion and is emitted
    as a C-style cast expression. It can be applied to integer, float, index
    and EmitC types.

    Example:

    ```mlir
    // Cast from `int32_t` to `float`
    %0 = emitc.cast %arg0: i32 to f32

    // Cast from `void` to `int32_t` pointer
    %1 = emitc.cast %arg1 :
        !emitc.ptr<!emitc.opaque<"void">> to !emitc.ptr<i32>
    ```
  }];

  let arguments = (ins AnyType:$source);
  let results = (outs AnyType:$dest);
  let assemblyFormat = "$source attr-dict `:` type($source) `to` type($dest)";
}

def EmitC_ConstantOp : EmitC_Op<"constant", [ConstantLike]> {
  let summary = "Constant operation";
  let description = [{
    The `constant` operation produces an SSA value equal to some constant
    specified by an attribute. This can be used to form simple integer and
    floating point constants, as well as more exotic things like tensor
    constants. The `constant` operation also supports the EmitC opaque
    attribute and the EmitC opaque type. Since folding is supported,
    it should not be used with pointers.

    Example:

    ```mlir
    // Integer constant
    %0 = "emitc.constant"(){value = 42 : i32} : () -> i32

    // Constant emitted as `char = CHAR_MIN;`
    %1 = "emitc.constant"()
        {value = #emitc.opaque<"CHAR_MIN"> : !emitc.opaque<"char">}
        : () -> !emitc.opaque<"char">
    ```
  }];

  let arguments = (ins EmitC_OpaqueOrTypedAttr:$value);
  let results = (outs AnyType);

  let hasFolder = 1;
  let hasVerifier = 1;
}

def EmitC_DivOp : EmitC_BinaryOp<"div", []> {
  let summary = "Division operation";
  let description = [{
    With the `div` operation the arithmetic operator / (division) can
    be applied.

    Example:

    ```mlir
    // Custom form of the division operation.
    %0 = emitc.div %arg0, %arg1 : (i32, i32) -> i32
    %0 = emitc.div %arg2, %arg3 : (f64, f64) -> f64

    // Generic form of the same operations.
    %0 = "emitc.div" (%arg0, %arg1) : (i32, i32) -> i32
    %0 = "emitc.div" (%arg2, %arg3) : (f64, f64) -> f64
    ```
  }];
}

def EmitC_EqOp : EmitC_BinaryOp<"eq", []> {
  let summary = "Equal-to operation";
  let description = [{
    With the `eq` operation the arithmetic operator == (equal to) can
    be applied.

    Example:

    ```mlir
    // Custom form of the equal-to operation.
    %0 = emitc.eq %arg0, %arg1 : (i32, i32) -> i1
    %0 = emitc.eq %arg2, %arg3 : (f64, f64) -> i1

    // Generic form of the same operations.
    %0 = "emitc.eq" (%arg0, %arg1) : (i32, i32) -> i1
    %0 = "emitc.eq" (%arg2, %arg3) : (f64, f64) -> i1
    ```
  }];
}

def EmitC_GeOp : EmitC_BinaryOp<"ge", []> {
  let summary = "Greater-than-or-equal-to operation";
  let description = [{
    With the `ge` operation the arithmetic operator >= (greater than or equal to) can
    be applied.

    Example:

    ```mlir
    // Custom form of the greater-than-or-equal-to operation.
    %0 = emitc.ge %arg0, %arg1 : (i32, i32) -> i1
    %0 = emitc.ge %arg2, %arg3 : (f64, f64) -> i1

    // Generic form of the same operations.
    %0 = "emitc.ge" (%arg0, %arg1) : (i32, i32) -> i1
    %0 = "emitc.ge" (%arg2, %arg3) : (f64, f64) -> i1
    ```
  }];
}

def EmitC_GtOp : EmitC_BinaryOp<"gt", []> {
  let summary = "Greater-than operation";
  let description = [{
    With the `gt` operation the arithmetic operator > (greater than) can
    be applied.

    Example:

    ```mlir
    // Custom form of the greater-than operation.
    %0 = emitc.gt %arg0, %arg1 : (i32, i32) -> i1
    %0 = emitc.gt %arg2, %arg3 : (f64, f64) -> i1

    // Generic form of the same operations.
    %0 = "emitc.gt" (%arg0, %arg1) : (i32, i32) -> i1
    %0 = "emitc.gt" (%arg2, %arg3) : (f64, f64) -> i1
    ```
  }];
}

def EmitC_IncludeOp
    : EmitC_Op<"include", [HasParent<"ModuleOp">]> {
  let summary = "Include operation";
  let description = [{
    The `include` operation allows to define a source file inclusion via the
    `#include` directive.

    Example:

    ```mlir
    // Custom form defining the inclusion of `<myheader>`.
    emitc.include <"myheader.h">

    // Generic form of the same operation.
    "emitc.include" (){include = "myheader.h", is_standard_include} : () -> ()

    // Custom form defining the inclusion of `"myheader"`.
    emitc.include "myheader.h"

    // Generic form of the same operation.
    "emitc.include" (){include = "myheader.h"} : () -> ()
    ```
  }];
  let arguments = (ins
    Arg<StrAttr, "source file to include">:$include,
    UnitAttr:$is_standard_include
  );
  let hasCustomAssemblyFormat = 1;
}

def EmitC_LeOp : EmitC_BinaryOp<"le", []> {
  let summary = "Less-than-or-equal-to operation";
  let description = [{
    With the `le` operation the arithmetic operator <= (less than or equal to) can
    be applied.

    Example:

    ```mlir
    // Custom form of the less-than-or-equal-to operation.
    %0 = emitc.le %arg0, %arg1 : (i32, i32) -> i1
    %0 = emitc.le %arg2, %arg3 : (f64, f64) -> i1

    // Generic form of the same operations.
    %0 = "emitc.le" (%arg0, %arg1) : (i32, i32) -> i1
    %0 = "emitc.le" (%arg2, %arg3) : (f64, f64) -> i1
    ```
  }];
}

def EmitC_LtOp : EmitC_BinaryOp<"lt", []> {
  let summary = "Less-than operation";
  let description = [{
    With the `lt` operation the arithmetic operator < (less than) can
    be applied.

    Example:

    ```mlir
    // Custom form of the less-than operation.
    %0 = emitc.lt %arg0, %arg1 : (i32, i32) -> i1
    %0 = emitc.lt %arg2, %arg3 : (f64, f64) -> i1

    // Generic form of the same operations.
    %0 = "emitc.lt" (%arg0, %arg1) : (i32, i32) -> i1
    %0 = "emitc.lt" (%arg2, %arg3) : (f64, f64) -> i1
    ```
  }];
}

def EmitC_MulOp : EmitC_BinaryOp<"mul", [Commutative]> {
  let summary = "Multiplication operation";
  let description = [{
    With the `,ul` operation the arithmetic operator * (Multiplication) can
    be applied.

    Example:

    ```mlir
    // Custom form of the multiplication operation.
    %0 = emitc.mul %arg0, %arg1 : (i32, i32) -> i32
    %0 = emitc.mul %arg2, %arg3 : (f64, f64) -> f64

    // Generic form of the same operations.
    %0 = "emitc.mul" (%arg0, %arg1) : (i32, i32) -> i32
    %0 = "emitc.mul" (%arg2, %arg3) : (f64, f64) -> f64
    ```
  }];
}

def EmitC_NeOp : EmitC_BinaryOp<"ne", []> {
  let summary = "Not-equal-to operation";
  let description = [{
    With the `ne` operation the arithmetic operator != (not equal to) can
    be applied.

    Example:

    ```mlir
    // Custom form of the not-equal-to operation.
    %0 = emitc.ne %arg0, %arg1 : (i32, i32) -> i1
    %0 = emitc.ne %arg2, %arg3 : (f64, f64) -> i1

    // Generic form of the same operations.
    %0 = "emitc.ne" (%arg0, %arg1) : (i32, i32) -> i1
    %0 = "emitc.ne" (%arg2, %arg3) : (f64, f64) -> i1
    ```
  }];
}

def EmitC_SubOp : EmitC_BinaryOp<"sub", []> {
  let summary = "Subtraction operation";
  let description = [{
    With the `sub` operation the arithmetic operator - (subtraction) can
    be applied.

    Example:

    ```mlir
    // Custom form of the substraction operation.
    %0 = emitc.sub %arg0, %arg1 : (i32, i32) -> i32
    %1 = emitc.sub %arg2, %arg3 : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
    %2 = emitc.sub %arg4, %arg5 : (!emitc.ptr<i32>, !emitc.ptr<i32>)
        -> !emitc.opaque<"ptrdiff_t">

    // Generic form of the same operations.
    %0 = "emitc.sub" (%arg0, %arg1) : (i32, i32) -> i32
    %1 = "emitc.sub" (%arg2, %arg3) : (!emitc.ptr<f32>, i32) -> !emitc.ptr<f32>
    %2 = "emitc.sub" (%arg4, %arg5) : (!emitc.ptr<i32>, !emitc.ptr<i32>)
        -> !emitc.opaque<"ptrdiff_t">
    ```
  }];
}

def SubscriptReadOp : EmitC_Op<"subscript.read", []> {
  let summary = "subcript operation";
  let description = [{
    With the `subcript` operation the operator[] can
    be applied to read a value from the operand at the specified offset.    ```
  }];
  let arguments = (ins
    AnyType:$operand,
    AnyType:$index
  );
  let results = (outs AnyType);

  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";

  let hasVerifier = 1;
}

def EmitC_SubscriptWriteOp : EmitC_Op<"subscript.write", []> {
  let summary = "subcript operation";
  let description = [{
    With the `subcript` operation the operator[] can
    be applied to write a value to the operand at the specified offset.    ```
  }];
  let arguments = (ins
    AnyType:$value,
    AnyType:$operand,
    AnyType:$index
  );

  let assemblyFormat = "operands attr-dict `:` type(operands)";

  let hasVerifier = 1;
}

def EmitC_StructDefOp : EmitC_Op<"struct.def", [HasParent<"ModuleOp">]> {
  let arguments = (ins TypeAttrOf<EmitC_StructType>:$type);

  let assemblyFormat = "$type attr-dict";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    bool hasMember(StringRef member);
  }];
}

def EmitC_StructMemberReadOp : EmitC_Op<"struct.member.read"> {
  let arguments = (ins
    EmitC_StructType:$struct_operand,
    Arg<EmitC_MemberAttr, "the struct member to read">:$member
  );
  let results = (outs AnyType:$result);

  let assemblyFormat = "$struct_operand $member `:` functional-type(operands, results) attr-dict";

  let hasVerifier = 1;
}

def EmitC_StructMemberWriteOp : EmitC_Op<"struct.member.write"> {
  let arguments = (ins
    AnyType:$value,
    EmitC_StructType:$struct_operand,
    Arg<EmitC_MemberAttr, "the struct member to write">:$member
  );

  let assemblyFormat = "$value $struct_operand $member `:` type(operands) attr-dict";

  let hasVerifier = 1;
}

def EmitC_VariableOp : EmitC_Op<"variable", []> {
  let summary = "Variable operation";
  let description = [{
    The `variable` operation produces an SSA value equal to some value
    specified by an attribute. This can be used to form simple integer and
    floating point variables, as well as more exotic things like tensor
    variables. The `variable` operation also supports the EmitC opaque
    attribute and the EmitC opaque type. If further supports the EmitC
    pointer type, whereas folding is not supported.
    The `variable` is emitted as a C/C++ local variable.

    Example:

    ```mlir
    // Integer variable
    %0 = "emitc.variable"(){value = 42 : i32} : () -> i32

    // Variable emitted as `int32_t* = NULL;`
    %1 = "emitc.variable"()
        {value = #emitc.opaque<"NULL"> : !emitc.opaque<"int32_t*">}
        : () -> !emitc.opaque<"int32_t*">
    ```

    Since folding is not supported, it can be used with pointers.
    As an example, it is valid to create pointers to `variable` operations
    by using `apply` operations and pass these to a `call` operation.
    ```mlir
    %0 = "emitc.variable"() {value = 0 : i32} : () -> i32
    %1 = "emitc.variable"() {value = 0 : i32} : () -> i32
    %2 = emitc.apply "&"(%0) : (i32) -> !emitc.ptr<i32>
    %3 = emitc.apply "&"(%1) : (i32) -> !emitc.ptr<i32>
    emitc.call "write"(%2, %3) : (!emitc.ptr<i32>, !emitc.ptr<i32>) -> ()
    ```
  }];

  let arguments = (ins EmitC_OpaqueOrTypedAttr:$value);
  let results = (outs AnyType);

  let hasVerifier = 1;
}

#endif // MLIR_DIALECT_EMITC_IR_EMITC
