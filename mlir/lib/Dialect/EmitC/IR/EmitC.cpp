//===- EmitC.cpp - EmitC Dialect ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::emitc;

#include "mlir/Dialect/EmitC/IR/EmitCDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// EmitCDialect
//===----------------------------------------------------------------------===//

void EmitCDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/EmitC/IR/EmitC.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "mlir/Dialect/EmitC/IR/EmitCTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mlir/Dialect/EmitC/IR/EmitCAttributes.cpp.inc"
      >();
}

/// Materialize a single constant operation from a given attribute value with
/// the desired resultant type.
Operation *EmitCDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, Type type,
                                             Location loc) {
  return builder.create<emitc::ConstantOp>(loc, type, value);
}

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

bool mlir::emitc::isValidIdentifier(llvm::StringRef identifier) {
  auto validFirstChar = [](char c) {
    return ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || c == '_';
  };
  auto validChar = [&](char c) {
    return validFirstChar(c) || ('0' <= c && c <= '9');
  };
  return identifier.size() > 0 && validFirstChar(identifier.front()) &&
         llvm::all_of(identifier, [&](char c) { return validChar(c); });
}

Type mlir::emitc::decayType(Type type) {
  if (auto aType = dyn_cast<mlir::emitc::ArrayType>(type))
    return mlir::emitc::PointerType::get(aType.getElementType());
  return type;
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

LogicalResult AddOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();

  if (lhsType.isa<emitc::PointerType>() && rhsType.isa<emitc::PointerType>())
    return emitOpError("requires that at most one operand is a pointer");

  if ((lhsType.isa<emitc::PointerType>() &&
       !rhsType.isa<IntegerType, emitc::OpaqueType>()) ||
      (rhsType.isa<emitc::PointerType>() &&
       !lhsType.isa<IntegerType, emitc::OpaqueType>()))
    return emitOpError("requires that one operand is an integer or of opaque "
                       "type if the other is a pointer");

  return success();
}

//===----------------------------------------------------------------------===//
// ApplyOp
//===----------------------------------------------------------------------===//

LogicalResult ApplyOp::verify() {
  StringRef applicableOperatorStr = getApplicableOperator();

  // Applicable operator must not be empty.
  if (applicableOperatorStr.empty())
    return emitOpError("applicable operator must not be empty");

  // Only `*` and `&` are supported.
  if (applicableOperatorStr != "&" && applicableOperatorStr != "*")
    return emitOpError("applicable operator is illegal");

  Operation *op = getOperand().getDefiningOp();
  if (op && dyn_cast<ConstantOp>(op))
    return emitOpError("cannot apply to constant");

  return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

bool CastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  Type input = inputs.front(), output = outputs.front();

  return ((input.isa<IntegerType, FloatType, IndexType, emitc::OpaqueType,
                     emitc::PointerType>()) &&
          (output.isa<IntegerType, FloatType, IndexType, emitc::OpaqueType,
                      emitc::PointerType>()));
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult emitc::CallOp::verify() {
  // Callee must not be empty.
  if (getCallee().empty())
    return emitOpError("callee must not be empty");

  if (std::optional<ArrayAttr> argsAttr = getArgs()) {
    for (Attribute arg : *argsAttr) {
      auto intAttr = arg.dyn_cast<IntegerAttr>();
      if (intAttr && intAttr.getType().isa<IndexType>()) {
        int64_t index = intAttr.getInt();
        // Args with elements of type index must be in range
        // [0..operands.size).
        if ((index < 0) || (index >= static_cast<int64_t>(getNumOperands())))
          return emitOpError("index argument is out of range");

        // Args with elements of type ArrayAttr must have a type.
      } else if (arg.isa<ArrayAttr>() /*&& arg.getType().isa<NoneType>()*/) {
        // FIXME: Array attributes never have types
        return emitOpError("array argument has no type");
      }
    }
  }

  if (std::optional<ArrayAttr> templateArgsAttr = getTemplateArgs()) {
    for (Attribute tArg : *templateArgsAttr) {
      if (!tArg.isa<TypeAttr, IntegerAttr, FloatAttr, emitc::OpaqueAttr>())
        return emitOpError("template argument has invalid type");
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

/// The constant op requires that the attribute's type matches the return type.
LogicalResult emitc::ConstantOp::verify() {
  if (getValueAttr().isa<emitc::OpaqueAttr>())
    return success();

  // Value must not be empty
  StringAttr strAttr = getValueAttr().dyn_cast<StringAttr>();
  if (strAttr && strAttr.getValue().empty())
    return emitOpError() << "value must not be empty";

  auto value = cast<TypedAttr>(getValueAttr());
  Type type = getType();
  if (!value.getType().isa<NoneType>() && type != value.getType())
    return emitOpError() << "requires attribute's type (" << value.getType()
                         << ") to match op's return type (" << type << ")";
  return success();
}

OpFoldResult emitc::ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

LogicalResult DivOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// EqOp
//===----------------------------------------------------------------------===//

LogicalResult EqOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// GeOp
//===----------------------------------------------------------------------===//

LogicalResult GeOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// GtOp
//===----------------------------------------------------------------------===//

LogicalResult GtOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// IncludeOp
//===----------------------------------------------------------------------===//

void IncludeOp::print(OpAsmPrinter &p) {
  bool standardInclude = getIsStandardInclude();

  p << " ";
  if (standardInclude)
    p << "<";
  p << "\"" << getInclude() << "\"";
  if (standardInclude)
    p << ">";
}

ParseResult IncludeOp::parse(OpAsmParser &parser, OperationState &result) {
  bool standardInclude = !parser.parseOptionalLess();

  StringAttr include;
  OptionalParseResult includeParseResult =
      parser.parseOptionalAttribute(include, "include", result.attributes);
  if (!includeParseResult.has_value())
    return parser.emitError(parser.getNameLoc()) << "expected string attribute";

  if (standardInclude && parser.parseOptionalGreater())
    return parser.emitError(parser.getNameLoc())
           << "expected trailing '>' for standard include";

  if (standardInclude)
    result.addAttribute("is_standard_include",
                        UnitAttr::get(parser.getContext()));

  return success();
}

//===----------------------------------------------------------------------===//
// LeOp
//===----------------------------------------------------------------------===//

LogicalResult LeOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// LtOp
//===----------------------------------------------------------------------===//

LogicalResult LtOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

LogicalResult MulOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// NeOp
//===----------------------------------------------------------------------===//

LogicalResult NeOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

LogicalResult SubOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();
  Type resultType = getResult().getType();

  if (rhsType.isa<emitc::PointerType>() && !lhsType.isa<emitc::PointerType>())
    return emitOpError("rhs can only be a pointer if lhs is a pointer");

  if (lhsType.isa<emitc::PointerType>() &&
      !rhsType.isa<IntegerType, emitc::OpaqueType, emitc::PointerType>())
    return emitOpError("requires that rhs is an integer, pointer or of opaque "
                       "type if lhs is a pointer");

  if (lhsType.isa<emitc::PointerType>() && rhsType.isa<emitc::PointerType>() &&
      !resultType.isa<IntegerType, emitc::OpaqueType>())
    return emitOpError("requires that the result is an integer or of opaque "
                       "type if lhs and rhs are pointers");

  return success();
}

//===----------------------------------------------------------------------===//
// StructDefOp
//===----------------------------------------------------------------------===//

LogicalResult StructDefOp::verify() { return success(); }

//===----------------------------------------------------------------------===//
// StructMemberReadOp
//===----------------------------------------------------------------------===//

LogicalResult StructMemberReadOp::verify() {
  StructType structType = getStructOperand().getType();
  StringRef memberName = getMember().getName();
  if (!structType.hasMember(memberName))
    return emitOpError() << "no member named '" << memberName << "' in 'struct "
                         << structType.getName() << "'";

  Type memberType = structType.getMember(memberName).getType();
  if (decayType(memberType) != getResult().getType()) {
    return emitOpError() << "member named '" << memberName << "' of type "
                         << memberType
                         << " is incompatible with result of type "
                         << getResult().getType();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// StructMemberWriteOp
//===----------------------------------------------------------------------===//

LogicalResult StructMemberWriteOp::verify() {
  StructType structType = getStructOperand().getType();
  StringRef memberName = getMember().getName();
  if (!structType.hasMember(memberName))
    return emitOpError() << "no member named '" << memberName << "' in 'struct "
                         << structType.getName() << "'";

  Type memberType = structType.getMember(memberName).getType();
  if (isa<ArrayType>(memberType))
    return emitOpError() << "cannot assign a member of array type";
  Type valueType = getValue().getType();
  if (decayType(valueType) != memberType) {
    return emitOpError() << "value of type " << valueType
                         << " is incompatible with member named '" << memberName
                         << "' of type " << memberType;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SubscriptReadOp
//===----------------------------------------------------------------------===//

LogicalResult SubscriptReadOp::verify() {
  if (!isa<ArrayType, PointerType, OpaqueType>(getOperand().getType()))
    return emitOpError()
           << "requires operand to be of array, pointer or opaque type";
  if (!isa<IndexType, IntegerType, OpaqueType>(getIndex().getType()))
    return emitOpError()
           << "requires index to be of integer, index or opaque type";
  // TODO: check that pointee/element type is compatible with result type
  return success();
}

//===----------------------------------------------------------------------===//
// SubscriptWriteOp
//===----------------------------------------------------------------------===//

LogicalResult SubscriptWriteOp::verify() {
  if (!isa<ArrayType, PointerType, OpaqueType>(getOperand().getType()))
    return emitOpError()
           << "requires operand to be of array, pointer or opaque type";
  if (!isa<IntegerType, IndexType, OpaqueType>(getIndex().getType()))
    return emitOpError()
           << "requires index to be of integer, index or opaque type";
  // TODO: check that pointee/element type is compatible with value type
  return success();
}

//===----------------------------------------------------------------------===//
// VariableOp
//===----------------------------------------------------------------------===//

/// The variable op requires that the attribute's type matches the return type.
LogicalResult emitc::VariableOp::verify() {
  if (getValueAttr().isa<emitc::OpaqueAttr>())
    return success();

  auto value = cast<TypedAttr>(getValueAttr());
  Type type = getType();
  if (!value.getType().isa<NoneType>() && type != value.getType())
    return emitOpError() << "requires attribute's type (" << value.getType()
                         << ") to match op's return type (" << type << ")";
  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "mlir/Dialect/EmitC/IR/EmitC.cpp.inc"

//===----------------------------------------------------------------------===//
// EmitC Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/EmitC/IR/EmitCAttributes.cpp.inc"

Attribute emitc::OpaqueAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return Attribute();
  std::string value;
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseOptionalString(&value)) {
    parser.emitError(loc) << "expected string";
    return Attribute();
  }
  if (parser.parseGreater())
    return Attribute();

  return get(parser.getContext(), value);
}

void emitc::OpaqueAttr::print(AsmPrinter &printer) const {
  printer << "<\"";
  llvm::printEscapedString(getValue(), printer.getStream());
  printer << "\">";
}

//===----------------------------------------------------------------------===//
// EmitC Types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "mlir/Dialect/EmitC/IR/EmitCTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// ArrayType
//===----------------------------------------------------------------------===//

LogicalResult ArrayType::verify(function_ref<InFlightDiagnostic()> emitError,
                                Type elementType,
                                std::optional<uint64_t> size) {
  if (elementType.isa<ArrayType>()) {
    return emitError() << "multidimensional arrays are not supported";
  }
  if (elementType.isa<PointerType>()) {
    return emitError() << "elements of type pointer are not supported";
  }
  return success();
}

bool ArrayType::hasUnknownSize() { return !getSize().has_value(); }

//===----------------------------------------------------------------------===//
// OpaqueType
//===----------------------------------------------------------------------===//

Type emitc::OpaqueType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();
  std::string value;
  SMLoc loc = parser.getCurrentLocation();
  if (parser.parseOptionalString(&value) || value.empty()) {
    parser.emitError(loc) << "expected non empty string in !emitc.opaque type";
    return Type();
  }
  if (value.back() == '*') {
    parser.emitError(loc) << "pointer not allowed as outer type with "
                             "!emitc.opaque, use !emitc.ptr instead";
    return Type();
  }
  if (parser.parseGreater())
    return Type();
  return get(parser.getContext(), value);
}

void emitc::OpaqueType::print(AsmPrinter &printer) const {
  printer << "<\"";
  llvm::printEscapedString(getValue(), printer.getStream());
  printer << "\">";
}

//===----------------------------------------------------------------------===//
// PointerType
//===----------------------------------------------------------------------===//

LogicalResult PointerType::verify(function_ref<InFlightDiagnostic()> emitError,
                                  Type pointee) {
  if (pointee.isa<ArrayType>()) {
    return emitError() << "pointers to arrays are not supported";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// StructType
//===----------------------------------------------------------------------===//

LogicalResult StructType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<MemberAttr> members, StringRef name) {
  if (!isValidIdentifier(name))
    return emitError() << "invalid identifier for struct type";
  SetVector<StringRef> structMembers;
  for (MemberAttr member : members) {
    if (member.getType().isa<StructType>())
      return emitError() << "nested struct types are not supported";
    if (!structMembers.insert(member.getName()))
      return emitError() << "duplicate member `" << member.getName() << "`";
  }
  if (members.size() == 1) {
    MemberAttr member = members[0];
    auto aType = member.getType().dyn_cast<ArrayType>();
    if (aType && aType.hasUnknownSize())
      return emitError() << "flexible array member '" << member.getName()
                         << "' not allowed in otherwise empty struct";
  }
  for (size_t i = 0; i < members.size() - 1; ++i) {
    MemberAttr member = members[i];
    auto aType = member.getType().dyn_cast<ArrayType>();
    if (aType && aType.hasUnknownSize())
      return emitError() << "flexible array member '" << member.getName()
                         << "' must be at the end of a struct";
  }
  return success();
}

bool StructType::hasMember(StringRef name) {
  return llvm::any_of(getMembers(), [&name](MemberAttr member) {
    return member.getName() == name;
  });
}

MemberAttr StructType::getMember(StringRef name) {
  assert(hasMember(name));
  for (auto member : getMembers()) {
    if (member.getName() == name)
      return member;
  }
  return nullptr;
}
