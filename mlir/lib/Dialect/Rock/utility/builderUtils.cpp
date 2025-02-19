//===- builderUtils.cpp - Rock utility functions ---------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------===//

#include "mlir/Dialect/Rock/utility/builderUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

using mlir::arith::ConstantOp;

namespace mlir {
namespace rock {
Value createConstantIntOp(OpBuilder &b, Location loc, Type type,
                          Type elementType, int64_t value) {
  APInt apValue(elementType.getIntOrFloatBitWidth(), value, true);
  auto constValue = b.getIntegerAttr(elementType, apValue);

  Value retValue;
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    retValue = b.create<ConstantOp>(
        loc, SplatElementsAttr::get(shapedType, constValue));
  } else {
    retValue = b.create<ConstantOp>(loc, type, constValue);
  }

  return retValue;
}

FailureOr<APInt> createAPInt(Type elemType, int64_t value) {
  auto bitWidth = elemType.getIntOrFloatBitWidth();
  bool isSigned = elemType.isSignedInteger();

  if (!isSigned && value < 0)
    return failure();

  APInt newValue(bitWidth, value, isSigned);
  APInt extended = isSigned ? newValue.sext(64) : newValue.zext(64);
  if (extended != APInt(64, value, true))
    return failure();

  return newValue;
}

std::pair<APFloat, llvm::detail::opStatus> createAPFloat(Type elemType,
                                                         float value) {
  auto semantics = static_cast<APFloat::Semantics>(-1);
  if (elemType.isF32()) {
    semantics = APFloat::S_IEEEsingle;
  } else if (elemType.isF16()) {
    semantics = APFloat::S_IEEEhalf;
  } else if (elemType.isBF16()) {
    semantics = APFloat::S_BFloat;
  } else if (elemType.isFloat8E4M3FNUZ()) {
    semantics = APFloat::S_Float8E4M3FNUZ;
  } else if (elemType.isFloat8E5M2FNUZ()) {
    semantics = APFloat::S_Float8E5M2FNUZ;
  } else if (elemType.isFloat8E4M3FN()) {
    semantics = APFloat::S_Float8E4M3FN;
  } else if (elemType.isFloat8E5M2()) {
    semantics = APFloat::S_Float8E5M2;
  } else {
    llvm_unreachable("Unexpected float semantics");
  }

  APFloat apValue(value);
  bool lostInfo = false;
  auto status = apValue.convert(APFloat::EnumToSemantics(semantics),
                                APFloat::rmNearestTiesToEven, &lostInfo);

  return std::make_pair(apValue, status);
}

Value createConstantFloatOp(OpBuilder &b, Location loc, Type type,
                            Type elemType, float value,
                            APFloat::opStatus expectedStatus) {
  std::pair<APFloat, llvm::detail::opStatus> floatRes =
      createAPFloat(elemType, value);
  APFloat apValue = floatRes.first;
  auto status = floatRes.second;
  assert(status == expectedStatus);
  Value retValue;

  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    Attribute constValue = b.getFloatAttr(elemType, apValue);
    assert(shapedType.getElementType() == elemType);
    retValue = b.create<ConstantOp>(
        loc, SplatElementsAttr::get(shapedType, constValue));
  } else {
    retValue =
        b.create<ConstantOp>(loc, type, b.getFloatAttr(elemType, apValue));
  }

  return retValue;
}

Value createZeroConstantOp(OpBuilder &b, Location loc, Type type) {
  Type elementType = getElementTypeOrSelf(type);
  if (elementType.isIntOrIndex()) {
    return createConstantIntOp(b, loc, type, elementType, 0);
  } else {
    return createConstantFloatOp(b, loc, type, elementType, 0.0);
  }
}

//===----------------------------------------------------------------------===//
// Utility function to emit type conversion ops.
//===----------------------------------------------------------------------===//
Value createTypeConversionOp(OpBuilder &b, Location loc, Value source,
                             Type destType) {
  // Convert from sourceType to destType if necessary.
  Value result = source;
  Type sourceType = source.getType();
  if (auto sourceVec = dyn_cast<VectorType>(sourceType)) {
    if (auto destVec = dyn_cast<VectorType>(destType)) {
      assert(sourceVec.getNumElements() == destVec.getNumElements() &&
             "source and destination have same length");
    } else {
      llvm_unreachable("Can't store vector sources to scalar destinations in "
                       "output writeback");
    }
  }
  Type sourceElemType = getElementTypeOrSelf(sourceType);
  Type destElemType = getElementTypeOrSelf(destType);
  unsigned sourceWidth = sourceElemType.getIntOrFloatBitWidth();
  unsigned destWidth = destElemType.getIntOrFloatBitWidth();
  if (sourceElemType != destElemType) {
    // All these ops act elementwise on vectors.
    if (isa<IntegerType>(sourceElemType) && isa<IntegerType>(destElemType)) {
      if (sourceWidth <= destWidth) {
        result = b.create<arith::ExtSIOp>(loc, destType, source);
      } else {
        result = b.create<arith::TruncIOp>(loc, destType, source);
      }
    } else if (isa<FloatType>(sourceElemType) && isa<FloatType>(destElemType)) {
      if (sourceWidth < destWidth) {
        result = b.create<arith::ExtFOp>(loc, destType, source);
      } else {
        result = b.create<arith::TruncFOp>(loc, destType, source);
      }
    } else {
      llvm_unreachable("Only float-to-float and int-to-int conversions "
                       "allowed");
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Utility function to create a linalg generic block to perform cast
// and copy to another memref.
//===----------------------------------------------------------------------===//
void createTypeConversionLaGeneric(PatternRewriter &rewriter, Location loc,
                                   Value src, Value dst) {
  MemRefType dstType = cast<MemRefType>(dst.getType());
  SmallVector<AffineMap, 2> indexingMaps{
      2, rewriter.getMultiDimIdentityMap(dstType.getRank())};
  SmallVector<utils::IteratorType> iteratorTypes(dstType.getRank(),
                                                 utils::IteratorType::parallel);
  rewriter.create<linalg::GenericOp>(
      loc, ValueRange(src), ValueRange(dst), indexingMaps, iteratorTypes,
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        Value cast = createTypeConversionOp(rewriter, loc, args[0],
                                            dstType.getElementType());
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, cast);
      });
}

Value createCollapseShapeOp(OpBuilder &b, Location loc, Value source) {
  auto ctx = b.getContext();
  auto sourceType = cast<ShapedType>(source.getType());
  assert(sourceType.hasStaticShape() &&
         "Only memrefs with static shapes are allowed");

  auto shape = sourceType.getShape();
  if (shape.size() == 1)
    return source;
  uint64_t collapsedDim = 1;
  SmallVector<AffineExpr, 2> exprs;
  for (uint32_t dim = 0; dim < shape.size(); ++dim) {
    collapsedDim *= shape[dim];
    exprs.push_back(getAffineDimExpr(dim, ctx));
  }

  SmallVector<int64_t, 1> collapsedShape;
  SmallVector<ReassociationExprs, 1> reassocs;
  collapsedShape.push_back(collapsedDim);
  reassocs.push_back(exprs);

  auto collapsedType =
      MemRefType::get(collapsedShape, sourceType.getElementType());
  Value result =
      b.create<memref::CollapseShapeOp>(loc, collapsedType, source, reassocs);
  return result;
}

int64_t getByteWidth(Type type) {
  if (auto vecType = dyn_cast<VectorType>(type))
    return (vecType.getElementTypeBitWidth() * vecType.getNumElements()) / 8;
  return type.getIntOrFloatBitWidth() / 8;
}

Type getFlattenedType(Type type) {
  if (auto mt = dyn_cast<MemRefType>(type)) {
    return MemRefType::get(mt.getNumElements(), mt.getElementType(), nullptr,
                           mt.getMemorySpace());
  }
  if (auto st = dyn_cast<ShapedType>(type))
    return st.cloneWith(st.getNumElements(), /*elementType=*/nullptr);
  return type;
}

Value getAsTensor(OpBuilder &builder, Location loc, mlir::Value value,
                  bool isWritable) {
  constexpr bool isRestrict{true};
  Value origTensor = builder.create<bufferization::ToTensorOp>(
      loc, value, isRestrict, isWritable);
  return origTensor;
}

Type vectorOfBoolShapedLike(Value v) {
  return VectorType::get(cast<ShapedType>(v.getType()).getShape(),
                         IntegerType::get(v.getContext(), 1));
}

} // namespace rock
} // namespace mlir
