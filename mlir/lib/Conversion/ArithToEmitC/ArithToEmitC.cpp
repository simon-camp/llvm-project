//===- ArithToEmitC.cpp - Arithmetic to EmitC dialect conversion ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_ARITHTOEMITCCONVERSIONPASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

bool isScalarType(Type type) { return !isa<TensorType, VectorType>(type); }

Value castSignedness(OpBuilder builder, Location loc, Value operand,
                     IntegerType::IntegerType::SignednessSemantics signedness) {
  IntegerType type = operand.getType().cast<IntegerType>();
  if (type.getSignedness() == signedness) {
    return operand;
  }

  IntegerType integerType = builder.getIntegerType(type.getWidth(), signedness);
  return builder.create<emitc::CastOp>(loc, integerType, operand).getResult();
}

template <typename Op, typename EmitCOp>
struct BinaryOpLowering : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isScalarType(op.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "non scalar types are not supported");
    }
    if (auto fastMathInterface =
            dyn_cast<arith::ArithFastMathInterface>(op.getOperation())) {
      auto flags = fastMathInterface.getFastMathFlagsAttr().getValue();
      if (flags != arith::FastMathFlags::none) {
        return rewriter.notifyMatchFailure(op,
                                           "fast math flags are not supported");
      }
    }

    rewriter.replaceOpWithNewOp<EmitCOp>(op, op.getType(), op.getLhs(),
                                         op.getRhs());
    return success();
  }
};

/// TODO: Make sure that unsigned casts are performed correctly
template <typename Op>
struct CastOpLowering : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isScalarType(op.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "non scalar types are not supported");
    }

    rewriter.replaceOpWithNewOp<emitc::CastOp>(op, op.getType(),
                                               op.getOperand());
    return success();
  }
};

struct CmpIOpLowering : public OpConversionPattern<arith::CmpIOp> {
  using OpConversionPattern<arith::CmpIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::CmpIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isScalarType(op.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "non scalar types are not supported");
    }

    FailureOr<Value> lhsCast = castOperand(op, op.getLhs(), rewriter);
    FailureOr<Value> rhsCast = castOperand(op, op.getRhs(), rewriter);

    if (failed(lhsCast) || failed(rhsCast))
      return failure();
    Type type = op.getType();

    switch (op.getPredicate()) {
    case arith::CmpIPredicate::eq:
      rewriter.replaceOpWithNewOp<emitc::EqOp>(op, type, *lhsCast, *rhsCast);
      break;
    case arith::CmpIPredicate::ne:
      rewriter.replaceOpWithNewOp<emitc::NeOp>(op, type, *lhsCast, *rhsCast);
      break;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::ult:
      rewriter.replaceOpWithNewOp<emitc::LtOp>(op, type, *lhsCast, *rhsCast);
      break;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::ule:
      rewriter.replaceOpWithNewOp<emitc::LeOp>(op, type, *lhsCast, *rhsCast);
      break;
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::ugt:
      rewriter.replaceOpWithNewOp<emitc::GtOp>(op, type, *lhsCast, *rhsCast);
      break;
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::uge:
      rewriter.replaceOpWithNewOp<emitc::GeOp>(op, type, *lhsCast, *rhsCast);
      break;
    }

    return success();
  }

private:
  IntegerType::SignednessSemantics
  mapPredicateToSignedness(arith::CmpIOp op) const {
    switch (op.getPredicate()) {
    case arith::CmpIPredicate::eq:
    case arith::CmpIPredicate::ne:
      // We don't use the actual value in this case.
      return IntegerType::SignednessSemantics::Signless;
    case arith::CmpIPredicate::slt:
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::sgt:
    case arith::CmpIPredicate::sge:
      return IntegerType::SignednessSemantics::Signed;
    case arith::CmpIPredicate::ult:
    case arith::CmpIPredicate::ule:
    case arith::CmpIPredicate::ugt:
    case arith::CmpIPredicate::uge:
      return IntegerType::SignednessSemantics::Unsigned;
    }
    llvm_unreachable("Unexpected arith::CmpIPredicate");
  }

  FailureOr<Value> castOperand(arith::CmpIOp op, Value operand,
                               ConversionPatternRewriter &rewriter) const {
    arith::CmpIPredicate predicate = op.getPredicate();
    if (predicate == arith::CmpIPredicate::eq ||
        predicate == arith::CmpIPredicate::ne) {
      return operand;
    }

    if (isa<IndexType>(operand.getType()))
      return rewriter.notifyMatchFailure(
          op, "relational comparisons of index type are not supported");

    IntegerType::SignednessSemantics signedness = mapPredicateToSignedness(op);

    Location loc = op.getLoc();
    return castSignedness(rewriter, loc, operand, signedness);
  }
};

struct ConstOpLowering : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isScalarType(op.getType())) {
      return rewriter.notifyMatchFailure(op,
                                         "non scalar types are not supported");
    }

    rewriter.replaceOpWithNewOp<emitc::ConstantOp>(op, op.getType(),
                                                   op.getValue());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct ArithToEmitCConversionPass
    : public impl::ArithToEmitCConversionPassBase<ArithToEmitCConversionPass> {
  using Base::Base;

  void runOnOperation() override {
    ConversionTarget target(getContext());

    target.addLegalDialect<emitc::EmitCDialect>();

    RewritePatternSet patterns(&getContext());

    mlir::arith::populateArithToEmitCConversionPatterns(patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

void mlir::arith::populateArithToEmitCConversionPatterns(
    RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<
    BinaryOpLowering<arith::AddIOp, emitc::AddOp>,
    BinaryOpLowering<arith::AddFOp, emitc::AddOp>,
    BinaryOpLowering<arith::DivFOp, emitc::DivOp>,
    BinaryOpLowering<arith::MulIOp, emitc::MulOp>,
    BinaryOpLowering<arith::MulFOp, emitc::MulOp>,
    BinaryOpLowering<arith::SubIOp, emitc::SubOp>,
    BinaryOpLowering<arith::SubFOp, emitc::SubOp>,
    CastOpLowering<arith::IndexCastOp>,
    CmpIOpLowering,
    ConstOpLowering
  >(patterns.getContext());
  // clang-format on
}
