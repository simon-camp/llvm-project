//===- MemRefToEmitC.cpp - Arithmetic to EmitC dialect conversion ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
#define GEN_PASS_DEF_MEMREFTOEMITCCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

void defineStruct(OpBuilder builder, Operation *op, emitc::StructType type) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  builder.setInsertionPoint(funcOp);
  builder.create<emitc::StructDefOp>(op->getLoc(), type);
}

void legalizeForExport(ModuleOp module) {
  llvm::SetVector<emitc::StructType> structDefns;

  OpBuilder builder(module.getContext());
  module.walk([&](Operation *op) {
    if (auto structDefOp = dyn_cast<emitc::StructDefOp>(op))
      structDefns.insert(structDefOp.getType());

    for (Type type : op->getResultTypes()) {
      if (auto sType = dyn_cast<emitc::StructType>(type)) {
        if (!structDefns.contains(sType)) {
          defineStruct(builder, op, sType);
          structDefns.insert(sType);
        }
      }
    }
    for (Type type : op->getOperandTypes()) {
      if (auto sType = dyn_cast<emitc::StructType>(type)) {
        if (!structDefns.contains(sType)) {
          defineStruct(builder, op, sType);
          structDefns.insert(sType);
        }
      }
    }
  });
}

FailureOr<std::string> elementTypeToStr(Type type) {
  if (auto iType = dyn_cast<IntegerType>(type)) {
    return std::string("i") + std::to_string(iType.getWidth());
  }
  if (auto fType = dyn_cast<FloatType>(type)) {
    return std::string("f") + std::to_string(fType.getWidth());
  }
  return failure();
}

Type convertMemRefToEmitCType(MemRefType type) {
  if (!type.hasRank() || !isStrided(type))
    return nullptr;

  FailureOr<std::string> elementName = elementTypeToStr(type.getElementType());
  if (failed(elementName))
    return nullptr;

  MLIRContext *ctx = type.getContext();
  Twine name = Twine("memref_") + elementName.value() + "_" +
               std::to_string(type.getRank()) + "d";
  std::string name_str = name.str();
  Type ptrType = emitc::PointerType::get(ctx, type.getElementType());
  Type indexType = IndexType::get(ctx);
  Type indexArrayType = emitc::ArrayType::get(indexType, type.getRank());

  SmallVector<emitc::MemberAttr, 5> members;
  members.push_back(emitc::MemberAttr::get(ctx, ptrType, "allocated"));
  members.push_back(emitc::MemberAttr::get(ctx, ptrType, "aligned"));
  members.push_back(emitc::MemberAttr::get(ctx, indexType, "offset"));
  if (type.getRank() > 0) {
    members.push_back(emitc::MemberAttr::get(ctx, indexArrayType, "sizes"));
    members.push_back(emitc::MemberAttr::get(ctx, indexArrayType, "strides"));
  }
  return emitc::StructType::get(ctx, members, name_str);
}

Value buildIndexExpression(OpBuilder builder, Location loc, Value memrefStruct,
                           ValueRange indices, MemRefType type) {
  MLIRContext *ctx = builder.getContext();
  Type indexType = builder.getIndexType();
  Type indexPtrType = mlir::emitc::PointerType::get(indexType);
  emitc::MemberAttr offsetAttr =
      emitc::MemberAttr::get(ctx, indexType, "offset");
  emitc::MemberAttr stridesAttr =
      emitc::MemberAttr::get(ctx, indexType, "strides");

  Value offset = builder
                     .create<emitc::StructMemberReadOp>(
                         loc, indexType, memrefStruct, offsetAttr)
                     .getResult();

  if (type.getRank() > 0) {
    Value stridesPtr = builder
                           .create<emitc::StructMemberReadOp>(
                               loc, indexPtrType, memrefStruct, stridesAttr)
                           .getResult();
    for (int i = 0; i < type.getRank(); ++i) {
      Value dim = builder
                      .create<emitc::ConstantOp>(loc, indexType,
                                                 builder.getIndexAttr(i))
                      .getResult();
      Value stride =
          builder
              .create<emitc::SubscriptReadOp>(loc, indexType, stridesPtr, dim)
              .getResult();

      Value stridedOffset =
          builder.create<emitc::MulOp>(loc, indexType, stride, indices[i])
              .getResult();

      offset =
          builder.create<emitc::AddOp>(loc, indexType, offset, stridedOffset)
              .getResult();
    }
  }
  return offset;
}

FlatSymbolRefAttr buildFunctionSymbol(MemRefType type, StringRef opName) {
  if (!type.hasRank())
    return nullptr;

  FailureOr<std::string> elementName = elementTypeToStr(type.getElementType());
  if (failed(elementName))
    return nullptr;

  // Drop dialect prefix from name.
  opName.consume_front("memref.");
  Twine name = Twine("memref_") + opName + "_" + elementName.value() + "_" +
               std::to_string(type.getRank()) + "d";

  return SymbolRefAttr::get(type.getContext(), name.str());
}

/// Create a SymbolRef from the given memref type and operation name and create
/// a func.call to that symbol with the origianl operands of the given
/// operation. If the symbol lookup fails this additionally creates a new
/// func.func op and calls the buildCallback which is responsible for populating
/// the newly created function to implement the original ops behaviour.
template <typename Op>
void replaceOpWithFunctionCall(
    PatternRewriter &rewriter, TypeConverter *converter, Op op, MemRefType type,
    std::function<void(OpBuilder, Block *)> buildCallback) {
  FlatSymbolRefAttr callee =
      buildFunctionSymbol(type, op->getName().getStringRef());

  func::FuncOp func =
      SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, callee);
  if (!func) {
    OpBuilder::InsertionGuard guard(rewriter);
    auto parentFunc = op->template getParentOfType<func::FuncOp>();

    rewriter.setInsertionPoint(parentFunc);

    SmallVector<Type> operandTypes;
    for (auto t : op->getOperandTypes()) {
      operandTypes.push_back(converter->convertType(t));
    }

    SmallVector<Type> resultTypes;
    for (auto t : op->getResultTypes()) {
      resultTypes.push_back(converter->convertType(t));
    }

    FunctionType funcType =
        FunctionType::get(rewriter.getContext(), operandTypes, resultTypes);

    auto funcOp =
        rewriter.create<func::FuncOp>(op.getLoc(), callee.getValue(), funcType);
    Block *entryBlock = funcOp.addEntryBlock();
    rewriter.setInsertionPointToStart(entryBlock);
    buildCallback(rewriter, entryBlock);
  }

  rewriter.replaceOpWithNewOp<func::CallOp>(op, callee, op->getResultTypes(),
                                            op.getOperands());
}

struct AllocOpLowering : public OpConversionPattern<memref::AllocOp> {
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getAlignment().has_value())
      return rewriter.notifyMatchFailure(op, "alignment is not supported");

    MemRefType memRefType = cast<MemRefType>(op.getResult().getType());
    FlatSymbolRefAttr callee =
        buildFunctionSymbol(memRefType, op->getName().getStringRef());

    func::FuncOp func =
        SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(op, callee);
    if (!func) {
      OpBuilder::InsertionGuard guard(rewriter);
      auto parentFunc = op->template getParentOfType<func::FuncOp>();

      rewriter.setInsertionPoint(parentFunc);

      // We need one operand for the offset, and a size and stride operand for
      // every dimension.
      SmallVector<Type, 3> operandTypes(2 * memRefType.getRank() + 1,
                                        rewriter.getIndexType());

      FunctionType funcType = FunctionType::get(
          rewriter.getContext(), operandTypes, op->getResultTypes());

      auto funcOp = rewriter.create<func::FuncOp>(op.getLoc(),
                                                  callee.getValue(), funcType);
      Block *entryBlock = funcOp.addEntryBlock();
      rewriter.setInsertionPointToStart(entryBlock);
      if (failed(buildAllocFunction(rewriter, entryBlock, op)))
        return failure();
    }

    auto sizes = memRefType.getShape();
    auto [strides, offset] = getStridesAndOffset(memRefType);

    size_t operandIndex = 0;
    SmallVector<Value, 5> operands;
    auto push_operand = [&](int64_t value) {
      if (value == ShapedType::kDynamic) {
        operands.push_back(op.getOperands()[operandIndex++]);
      } else {
        Value staticValue =
            rewriter
                .create<emitc::ConstantOp>(op.getLoc(), rewriter.getIndexType(),
                                           rewriter.getIndexAttr(value))
                .getResult();
        operands.push_back(staticValue);
      }
    };
    push_operand(offset);
    for (auto size : sizes) {
      push_operand(size);
    }
    for (auto stride : strides) {
      push_operand(stride);
    }
    rewriter.replaceOpWithNewOp<func::CallOp>(op, callee, op->getResultTypes(),
                                              operands);
    return success();
  }

private:
  LogicalResult buildAllocFunction(OpBuilder builder, Block *entryBlock,
                                   memref::AllocOp op) const {
    MLIRContext *ctx = builder.getContext();
    Location loc = op.getLoc();

    MemRefType memRefType = cast<MemRefType>(op.getResult().getType());
    Type elementPtrType = emitc::PointerType::get(memRefType.getElementType());
    Type indexType = builder.getIndexType();
    emitc::MemberAttr allocated =
        emitc::MemberAttr::get(ctx, elementPtrType, "allocated");
    emitc::MemberAttr aligned =
        emitc::MemberAttr::get(ctx, elementPtrType, "aligned");
    emitc::MemberAttr offset = emitc::MemberAttr::get(ctx, indexType, "offset");

    // Create result struct
    Value resultValue =
        builder
            .create<emitc::VariableOp>(
                loc, convertMemRefToEmitCType(memRefType),
                emitc::OpaqueAttr::get(builder.getContext(), ""))
            .getResult();

    // Allocate memory with malloc
    SmallVector<Value, 1> sizes;
    // The size arguments range from 1 to 1 + rank
    for (int i = 1; i <= memRefType.getRank(); ++i) {
      sizes.push_back(entryBlock->getArgument(i));
    }
    Value numElements = buildSizeExpression(builder, loc, sizes);
    FailureOr<Value> data =
        buildMallocCall(builder, loc, memRefType, numElements);
    if (failed(data))
      return failure();

    // Set members
    builder.create<emitc::StructMemberWriteOp>(loc, data.value(), resultValue,
                                               allocated);
    builder.create<emitc::StructMemberWriteOp>(loc, data.value(), resultValue,
                                               aligned);
    builder.create<emitc::StructMemberWriteOp>(loc, entryBlock->getArgument(0),
                                               resultValue, offset);

    if (memRefType.getRank() > 0) {
      Type indexPtrType = emitc::PointerType::get(indexType);
      size_t operandIndex = 1;

      // Populate sizes array
      emitc::MemberAttr sizes =
          emitc::MemberAttr::get(ctx, indexPtrType, "sizes");
      Value sizesPtr = builder
                           .create<emitc::StructMemberReadOp>(
                               loc, indexPtrType, resultValue, sizes)
                           .getResult();
      for (int64_t i = 0; i < memRefType.getRank(); ++i) {
        Value index = builder
                          .create<emitc::ConstantOp>(loc, indexType,
                                                     builder.getIndexAttr(i))
                          .getResult();
        builder.create<emitc::SubscriptWriteOp>(
            loc, entryBlock->getArgument(operandIndex++), sizesPtr, index);
      }

      // Populate strides array
      emitc::MemberAttr strides =
          emitc::MemberAttr::get(ctx, indexPtrType, "strides");
      Value stridesPtr = builder
                             .create<emitc::StructMemberReadOp>(
                                 loc, indexPtrType, resultValue, strides)
                             .getResult();
      for (int64_t i = 0; i < memRefType.getRank(); ++i) {
        Value index = builder
                          .create<emitc::ConstantOp>(loc, indexType,
                                                     builder.getIndexAttr(i))
                          .getResult();
        builder.create<emitc::SubscriptWriteOp>(
            loc, entryBlock->getArgument(operandIndex++), stridesPtr, index);
      }
    }
    builder.create<func::ReturnOp>(loc, SmallVector<Value>{resultValue});

    return success();
  }

  Value buildSizeExpression(OpBuilder builder, Location loc,
                            ArrayRef<Value> sizes) const {
    Type indexType = builder.getIndexType();
    Value size = builder.create<emitc::ConstantOp>(loc, indexType,
                                                   builder.getIndexAttr(1));

    for (auto s : sizes) {
      size = builder.create<emitc::MulOp>(loc, indexType, size, s).getResult();
    }
    return size;
  }

  FailureOr<Value> buildMallocCall(OpBuilder builder, Location loc,
                                   MemRefType type, Value numElements) const {
    if (!type.getLayout().isIdentity())
      return failure();

    if (type.getElementTypeBitWidth() % 8 != 0)
      return failure();

    Type elementPtrType = emitc::PointerType::get(type.getElementType());

    Type indexType = builder.getIndexType();
    uint64_t bitWidth = type.getElementTypeBitWidth();
    Value byteWidth =
        builder
            .create<mlir::emitc::ConstantOp>(loc, indexType,
                                             builder.getIndexAttr(bitWidth / 8))
            .getResult();
    Value numBytes =
        builder
            .create<mlir::emitc::MulOp>(loc, indexType, numElements, byteWidth)
            .getResult();

    MLIRContext *ctx = builder.getContext();

    // TODO: malloc returns void* and needs to be explicitly cast in C++;
    return builder
        .create<mlir::emitc::CallOp>(
            /*location=*/loc,
            /*type=*/elementPtrType,
            /*callee=*/StringAttr::get(ctx, "malloc"),
            /*args=*/ArrayAttr::get(ctx, {builder.getIndexAttr(0)}),
            /*templateArgs=*/ArrayAttr{},
            /*operands=*/ArrayRef<Value>{numBytes})
        .getResult(0);
  }
};

struct DimOpLowering : public OpConversionPattern<memref::DimOp> {
  using OpConversionPattern<memref::DimOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    MemRefType memRefType = cast<MemRefType>(op.getSource().getType());
    replaceOpWithFunctionCall(
        rewriter, typeConverter, op, memRefType,
        [&op, &adaptor](OpBuilder builder, Block *entryBlock) {
          MLIRContext *ctx = op.getContext();
          Location loc = op.getLoc();

          Value memref = entryBlock->getArgument(0);
          Value index = entryBlock->getArgument(1);

          Type indexType = builder.getIndexType();
          Type indexPtrType = mlir::emitc::PointerType::get(indexType);
          emitc::MemberAttr sizes =
              emitc::MemberAttr::get(ctx, indexPtrType, "sizes");

          Value sizesPtr = builder
                               .create<emitc::StructMemberReadOp>(
                                   loc, indexPtrType, memref, sizes)
                               .getResult();
          Value dim = builder
                          .create<emitc::SubscriptReadOp>(loc, indexType,
                                                          sizesPtr, index)
                          .getResult();

          builder.create<func::ReturnOp>(loc, ValueRange({dim}));
        });
    return success();
  }
};

struct LoadOpLowering : public OpConversionPattern<memref::LoadOp> {
  using OpConversionPattern<memref::LoadOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    MemRefType memRefType = cast<MemRefType>(op.getMemRef().getType());
    replaceOpWithFunctionCall(
        rewriter, typeConverter, op, memRefType,
        [&op, &memRefType](OpBuilder builder, Block *entryBlock) {
          Location loc = op.getLoc();
          Type elementType = memRefType.getElementType();
          Type elementPtrType = mlir::emitc::PointerType::get(elementType);

          Value memrefStruct = entryBlock->getArgument(0);
          MutableArrayRef<BlockArgument> indices =
              entryBlock->getArguments().slice(1, memRefType.getRank());
          Value index = buildIndexExpression(builder, loc, memrefStruct,
                                             indices, memRefType);

          emitc::MemberAttr alignedAttr = emitc::MemberAttr::get(
              builder.getContext(), elementPtrType, "aligned");
          Value aligned =
              builder
                  .create<emitc::StructMemberReadOp>(loc, elementPtrType,
                                                     memrefStruct, alignedAttr)
                  .getResult();
          Value value = builder
                            .create<emitc::SubscriptReadOp>(loc, elementType,
                                                            aligned, index)
                            .getResult();
          builder.create<func::ReturnOp>(loc, ValueRange({value}));
        });
    return success();
  }
};

struct StoreOpLowering : public OpConversionPattern<memref::StoreOp> {
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    MemRefType memRefType = cast<MemRefType>(op.getMemRef().getType());
    replaceOpWithFunctionCall(
        rewriter, typeConverter, op, memRefType,
        [&op, &memRefType](OpBuilder builder, Block *entryBlock) {
          Location loc = op.getLoc();
          Type elementType = memRefType.getElementType();
          Type elementPtrType = mlir::emitc::PointerType::get(elementType);

          Value value = entryBlock->getArgument(0);
          Value memrefStruct = entryBlock->getArgument(1);
          MutableArrayRef<BlockArgument> indices =
              entryBlock->getArguments().slice(2, memRefType.getRank());
          Value index = buildIndexExpression(builder, loc, memrefStruct,
                                             indices, memRefType);

          emitc::MemberAttr alignedAttr = emitc::MemberAttr::get(
              builder.getContext(), elementPtrType, "aligned");
          Value aligned =
              builder
                  .create<emitc::StructMemberReadOp>(loc, elementPtrType,
                                                     memrefStruct, alignedAttr)
                  .getResult();
          builder.create<emitc::SubscriptWriteOp>(loc, value, aligned, index);
          builder.create<func::ReturnOp>(loc, ValueRange());
        });
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct MemRefToEmitCConversionPass
    : public impl::MemRefToEmitCConversionPassBase<
          MemRefToEmitCConversionPass> {
  using Base::Base;

  void runOnOperation() override {
    ConversionTarget target(getContext());

    TypeConverter typeConverter;

    typeConverter.addConversion([](Type type) -> Type { return type; });
    typeConverter.addConversion([&](FunctionType type) -> Type {
      SmallVector<Type> operandTypes;
      for (Type operandType : type.getInputs()) {
        operandTypes.push_back(typeConverter.convertType(operandType));
      }
      SmallVector<Type> resultTypes;
      for (Type resultType : type.getResults()) {
        resultTypes.push_back(typeConverter.convertType(resultType));
      }
      return FunctionType::get(type.getContext(), operandTypes, resultTypes);
    });
    typeConverter.addConversion([&](MemRefType type) -> Type {
      return convertMemRefToEmitCType(type);
    });

    target.addIllegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<emitc::EmitCDialect>();

    target.addDynamicallyLegalOp<func::FuncOp>(
        [&typeConverter](func::FuncOp op) {
          return typeConverter.isLegal(op.getFunctionType());
        });
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(
        [&typeConverter](Operation *op) { return typeConverter.isLegal(op); });

    RewritePatternSet patterns(&getContext());

    mlir::memref::populateMemRefToEmitCConversionPatterns(patterns,
                                                          typeConverter);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();

    // TODO: Move to transform
    legalizeForExport(getOperation());
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void mlir::memref::populateMemRefToEmitCConversionPatterns(
    RewritePatternSet &patterns, TypeConverter converter) {
  // Populate `func.*` conversion patterns.
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 converter);
  populateCallOpTypeConversionPattern(patterns, converter);
  populateReturnOpTypeConversionPattern(patterns, converter);
  // clang-format off
  patterns.add<
    AllocOpLowering,
    DimOpLowering,
    LoadOpLowering,
    StoreOpLowering
  >(converter, patterns.getContext());
  // clang-format on
}
