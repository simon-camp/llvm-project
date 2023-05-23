//===- MemRefToEmitC.h - MemRef to EmitC dialect conversion -------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_MEMREFTOEMITC_MEMREFTOEMITC_H
#define MLIR_CONVERSION_MEMREFTOEMITC_MEMREFTOEMITC_H

#include <memory>

namespace mlir {

class RewritePatternSet;
class Pass;
class TypeConverter;

#define GEN_PASS_DECL_MEMREFTOEMITCCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"

namespace memref {
void populateMemRefToEmitCConversionPatterns(RewritePatternSet &patterns,
                                             TypeConverter converter);
} // namespace memref
} // namespace mlir

#endif // MLIR_CONVERSION_MEMREFTOEMITC_MEMREFTOEMITC_H
