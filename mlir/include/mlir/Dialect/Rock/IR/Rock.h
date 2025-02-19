//===- Rock.h - Rock MLIR Dialect ---------------------*- C++ -*-===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines Rock memref attributes and operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ROCKOPS_OPS_H_
#define MLIR_ROCKOPS_OPS_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DerivedAttributeOpInterface.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {
class PatternRewriter;
} // namespace mlir

//===----------------------------------------------------------------------===//
//  Rock Dialect
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/Rock/IR/RockOpsDialect.h.inc"
#include "mlir/Dialect/Rock/IR/RockTypes.h"

#include "mlir/Dialect/Rock/IR/ConvolutionDims.h"
#include "mlir/Dialect/Rock/IR/GemmSize.h"

namespace mlir {
namespace OpTrait {
namespace rock {
template <typename ConcreteType>
class FusionRoot : public TraitBase<ConcreteType, FusionRoot> {};
} // namespace rock
} // namespace OpTrait
} // namespace mlir

// Following ifdef could be used to change
// the attention operator to be a fused gemm-gemm
// kernel for debugging purposes. This will also
// adjust the test harness to verify the same as well
// #define ROCK_DEBUG_ATTENTION_REMOVE_SOFTMAX

namespace mlir {
namespace rock {
//===----------------------------------------------------------------------===//
// Utility method for creating an array attribute of n empty array attributes.
// We use this structure so transforms can be uniformly copied onto the final
// user(s) of the transformed value
//
// TODO(kdrewnia) See if this declaration should be elsewhere
//===----------------------------------------------------------------------===//
ArrayAttr noTransformsArray(Builder &b, size_t n);

ArrayAttr getIndexArrayAttr(Builder &b, ArrayRef<int64_t> values);

// maxWaves is a constant parameter that is applicable
// across all codegeneration done in rocMLIR. This will
// limit the maxWaves per workgroup to be 4.
constexpr int64_t maxWavesPerWG = 4;

// The largest workgroup size ("block size") that LLVM and the runtime
// support.
constexpr int64_t maxHardwareWorkgroupSize = 1024;

} // end namespace rock
} // end namespace mlir

#include "mlir/Dialect/Rock/IR/RockAccelTuningParamAttrInterface.h"
#include "mlir/Dialect/Rock/IR/RockTuningParamAttrInterface.h"
#define GET_ATTRDEF_CLASSES
#include "mlir/Dialect/Rock/IR/RockAttrDefs.h.inc"

#include "mlir/Dialect/Rock/IR/RockAcceptingViewOpInterface.h"
#include "mlir/Dialect/Rock/IR/RockConvInterface.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/RockWriterOpInterface.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/Rock/IR/RockOps.h.inc"

namespace mlir {
namespace rock {
TransformAttr getTransformAttrChecked(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    MLIRContext *context, TransformType type, ArrayRef<int64_t> params,
    ArrayRef<StringRef> upperNames, ArrayRef<uint32_t> upperDims,
    ArrayRef<StringRef> lowerNames, ArrayRef<uint32_t> lowerDims);

TransformMapAttr getTransformMapAttrChecked(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    MLIRContext *context, ArrayRef<TransformAttr> ops, AffineMapAttr map,
    DenseI64ArrayAttr upperBounds, DenseI64ArrayAttr lowerBounds);
} // namespace rock
} // namespace mlir
#endif // MLIR_ROCKOPS_OPS_H_
