//===- TypeConversions.cpp - Convert signless types into C/C++ types ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/EmitC/Transforms/TypeConversions.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#include <optional>

#define DEBUG_TYPE "mlir-emitc-conversion"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

namespace {

Value materializeAsUnrealizedCast(OpBuilder &builder, Type resultType,
                                  ValueRange inputs, Location loc) {
  if (inputs.size() != 1)
    return Value();

  return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
      .getResult(0);
}

} // namespace

void mlir::populateEmitCSizeTTypeConversions(TypeConverter &converter) {
  converter.addConversion(
      [](IndexType type) { return emitc::SizeTType::get(type.getContext()); });

  converter.addSourceMaterialization(materializeAsUnrealizedCast);
  converter.addTargetMaterialization(materializeAsUnrealizedCast);
  converter.addArgumentMaterialization(materializeAsUnrealizedCast);
}

void mlir::populateMemRefToEmitCTypeConversions(
    TypeConverter &typeConverter, const EmitCConversionOptions &options) {
  typeConverter.addConversion(
      [&](MemRefType memRefType) -> std::optional<Type> {
        if (!options.memrefToArray) {
          LDBG("Memref type conversion not enabled\n");
          return {};
        }
        if (!memRefType.hasStaticShape()) {
          LDBG("Memref type must have static shape\n");
          return {};
        }
        if (!memRefType.getLayout().isIdentity()) {
          LDBG("Memref type must have identity layout map\n");
          return {};
        }
        if (llvm::any_of(memRefType.getShape(),
                         [](int64_t dim) { return dim == 0; })) {
          LDBG("Memref type must not have 0 dims\n");
          return {};
        }
        Type convertedElementType =
            typeConverter.convertType(memRefType.getElementType());
        if (!convertedElementType) {
          LDBG("Failed to convert element type\n");
          return {};
        }
        if (memRefType.getRank() == 0) {
          if (options.promote0dMemref)
            return emitc::ArrayType::get({1}, convertedElementType);
          LDBG("Rank 0 promotion not enabled\n");
          return {};
        }
        return emitc::ArrayType::get(memRefType.getShape(),
                                     convertedElementType);
      });
}

/// Get an unsigned integer or size data type corresponding to \p ty.
std::optional<Type> mlir::emitc::getUnsignedTypeFor(Type ty) {
  if (ty.isInteger())
    return IntegerType::get(ty.getContext(), ty.getIntOrFloatBitWidth(),
                            IntegerType::SignednessSemantics::Unsigned);
  if (isa<PtrDiffTType, SignedSizeTType>(ty))
    return SizeTType::get(ty.getContext());
  if (isa<SizeTType>(ty))
    return ty;
  return {};
}

/// Get a signed integer or size data type corresponding to \p ty that supports
/// arithmetic on negative values.
std::optional<Type> mlir::emitc::getSignedTypeFor(Type ty) {
  if (ty.isInteger())
    return IntegerType::get(ty.getContext(), ty.getIntOrFloatBitWidth(),
                            IntegerType::SignednessSemantics::Signed);
  if (isa<SizeTType, SignedSizeTType>(ty))
    return PtrDiffTType::get(ty.getContext());
  if (isa<PtrDiffTType>(ty))
    return ty;
  return {};
}
