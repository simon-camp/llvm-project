//===- TypeConversions.h - Convert signless types into C/C++ types -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_EMITC_TRANSFORMS_TYPECONVERSIONS_H
#define MLIR_DIALECT_EMITC_TRANSFORMS_TYPECONVERSIONS_H

#include <optional>

namespace mlir {
class TypeConverter;
class Type;

struct EmitCConversionOptions {
  /// Convert statically shaped memref types to emitc array types.
  bool memrefToArray = false;

  /// Promote 0-d memrefs to 1-d arrays of size 1. This option can only be used
  /// in conjunction with `memrefToArray`.
  bool promote0dMemref = false;
};

void populateEmitCSizeTTypeConversions(TypeConverter &converter);
void populateMemRefToEmitCTypeConversions(
    TypeConverter &typeConverter, const EmitCConversionOptions &options);

namespace emitc {
std::optional<Type> getUnsignedTypeFor(Type ty);
std::optional<Type> getSignedTypeFor(Type ty);
} // namespace emitc

} // namespace mlir

#endif // MLIR_DIALECT_EMITC_TRANSFORMS_TYPECONVERSIONS_H
