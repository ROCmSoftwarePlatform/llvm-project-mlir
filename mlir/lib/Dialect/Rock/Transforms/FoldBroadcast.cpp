//===- FoldBroadcast.cpp - fold a broadcasted batch dim  ------===//
//
// Copyright 2024 Advanced Micro Devices.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/Dialect/Rock/IR/RockGemmWrapperInterface.h"
#include "mlir/Dialect/Rock/IR/TransformMapBuilder.h"
#include "mlir/Dialect/Rock/Passes.h"
#include "mlir/Dialect/Rock/utility/transformMapUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

namespace mlir {
namespace rock {
#define GEN_PASS_DEF_ROCKFOLDBROADCASTPASS
#include "mlir/Dialect/Rock/Passes.h.inc"
} // namespace rock
} // namespace mlir

#define DEBUG_TYPE "rock-fold-broadcast"

using namespace mlir;
using namespace mlir::rock;

namespace {

struct FoldBroadcast : public OpRewritePattern<rock::GemmOp> {
  using OpRewritePattern<rock::GemmOp>::OpRewritePattern;

  // Analze the stack to verify if the batch size is a broadcast
  bool isBatchDimFoldableInTheTransformStack(ArrayAttr views) const {
    // We start from the batch-size dimension (which is always the 0-th
    // dimension in gemm)
    DenseSet<uint32_t> workList{0};

    // Let's walk the transform stack backwards (from top to bottom)
    size_t level = 0;

    while (!workList.empty() && level < views.size()) {
      auto curMap = cast<TransformMapAttr>(views[level++]);
      auto ops = curMap.getOps();
      DenseSet<uint32_t> newWorkList;
      // Let's consider all the operations of the current transform map
      for (auto tr : ops) {
        for (auto [idx, upperDim] : llvm::enumerate(tr.getUpperDims())) {
          // If the current operation is transforming one of the dimensions
          // we are looking for, then dive into the operation to decide what to
          // do
          if (workList.contains(upperDim)) {
            switch (tr.getType()) {
            // If it is a single-length broadcast, don't do
            // anything, otherwise we need to be sure that the new
            // dimension is a single-length broadcast
            case rock::TransformType::Broadcast:
              if (tr.getParams()[idx] != 1)
                newWorkList.insert(tr.getLowerDims()[idx]);
              break;
            // AddDim and ConstDim are basically broadcasts. No
            // need to go further
            case rock::TransformType::AddDim:
            case rock::TransformType::ConstDim:
              break;
            // Follow the indices for the transformation that reroute them
            case rock::TransformType::PassThrough:
            case rock::TransformType::Slice:
            case rock::TransformType::Pad:
              newWorkList.insert(tr.getLowerDims()[idx]);
              break;
            // For a merge to be a valid broadcast
            // we need to ensure that all their (lower) dimensions
            // bigger than 1 lead to broadcasts
            case rock::TransformType::Merge:
              for (auto [length, dim] :
                   llvm::zip(tr.getParams(), tr.getLowerDims())) {
                if (length != 1)
                  newWorkList.insert(dim);
              }
              break;
            // For an umerge/embed, just follow the single dimension
            // down.
            case rock::TransformType::Unmerge:
            case rock::TransformType::Embed:
              newWorkList.insert(tr.getLowerDims().back());
              break;
            }
          }
        }
      }
      workList = newWorkList;
    }
    // If we want top/down and we determined that the batch dimension
    // led to a broadcast, then return true.
    return workList.empty();
  }

  // Determine if the first dimension of a view is a broadcast
  bool isBatchDimFoldable(PatternRewriter &rw, Value aView) const {
    auto [buffer, views, needs64BitIdxA] = untransform(rw, aView);

    // There are no views, hence no broadcast is possible
    if (views.empty())
      return false;

    auto trMap = cast<TransformMapAttr>(views[0]);

    // There is no batch, hence nothing that can be a broadcast
    if (trMap.getUpperBounds().size() != 3)
      return false;

    return isBatchDimFoldableInTheTransformStack(views);
  }

  // Merge the batch dimension into either M or N, i.e., transform (d0, d1, d2)
  // into (d0*d1, d2) or (d1, d0*d2)
  Value mergeBatch(PatternRewriter &rw, Location loc,
                   TypedValue<ShapedType> buffer, bool isTransposed) const {
    auto shape = buffer.getType().getShape();
    ArrayAttr mergeBatchAttr;
    if (isTransposed) {
      rock::TopDownTMBuilder mergeBatchBuilder(
          rw, {"d0", "gd1"}, {shape[1], shape[0] * shape[2]}, loc);
      mergeBatchBuilder.merge({"g", "d1"}, {0, 2}, "gd1", {shape[0], shape[2]});
      mergeBatchBuilder.passThrough({"d0"}, {1}, {"d0"});
      mergeBatchAttr = rw.getArrayAttr({mergeBatchBuilder.get()});
    } else {
      rock::TopDownTMBuilder mergeBatchBuilder(
          rw, {"gd0", "d1"}, {shape[0] * shape[1], shape[2]}, loc);
      mergeBatchBuilder.merge({"g", "d0"}, {0, 1}, "gd0", {shape[0], shape[1]});
      mergeBatchBuilder.passThrough({"d1"}, {2}, {"d1"});
      mergeBatchAttr = rw.getArrayAttr({mergeBatchBuilder.get()});
    }
    return rock::transform(rw, buffer, mergeBatchAttr);
  }

  // Select the 0th slice from a broadcast, de facto removing the broadcast
  // dimension
  Value unbroadcastBatch(PatternRewriter &rw, Location loc,
                         TypedValue<ShapedType> buffer) const {
    auto shape = buffer.getType().getShape();
    rock::TopDownTMBuilder unbroadcastBuilder(rw, {"d0", "d1"},
                                              {shape[1], shape[2]}, loc);
    unbroadcastBuilder.constDim({"g"}, 0, 0, shape[0]);
    unbroadcastBuilder.passThrough({"d0", "d1"}, {1, 2}, {"d0", "d1"});
    return rock::transform(rw, buffer,
                           rw.getArrayAttr({unbroadcastBuilder.get()}));
  }

  LogicalResult matchAndRewrite(rock::GemmOp op,
                                PatternRewriter &rw) const override {
    Location loc = op.getLoc();
    bool isABatchBroadcast = isBatchDimFoldable(rw, op.getA());
    bool isBBatchBroadcast = isBatchDimFoldable(rw, op.getB());

    if (!isABatchBroadcast && !isBBatchBroadcast)
      return failure();

    Value newA, newB, newC;
    if (isBBatchBroadcast && isABatchBroadcast) {
      // If both B and C are canonicalizable, simply
      // remove the broadcast from A,B and C
      newA = unbroadcastBatch(rw, loc, op.getA());
      newB = unbroadcastBatch(rw, loc, op.getB());
      newC = unbroadcastBatch(rw, loc, op.getC());
    } else if (isBBatchBroadcast) {
      newA = mergeBatch(rw, loc, op.getA(), op.getATransposed());
      newB = unbroadcastBatch(rw, loc, op.getB());
      newC = mergeBatch(rw, loc, op.getC(), op.getCTransposed());
    } else { // isABatchBroadcast
      // When the broadcasted batch is on A, matrix B and C need
      // to be considered as if they were transposed
      newA = unbroadcastBatch(rw, loc, op.getA());
      newB = mergeBatch(rw, loc, op.getB(), !op.getBTransposed());
      newC = mergeBatch(rw, loc, op.getC(), !op.getCTransposed());
    }

    // Create the new GemmOp
    auto gemm = rw.create<rock::GemmOp>(
        op.getLoc(), newC.getType(), newA, newB, newC, op.getATransposed(),
        op.getBTransposed(), op.getCTransposed(), op.getArch(),
        op.getNumCUAttr(), op.getFeatures(), op.getStoreMethod(),
        op.getDerivedBlockSizeAttr(), op.getGridSizeAttr(), op.getParamsAttr());

    // Convert optional attributes
    if (auto attr = (*op).template getAttrOfType<StringAttr>("perf_config"))
      gemm->setAttr("perf_config", attr);

    // Remove dummy transforms from the gemm output and use it to replace the
    // original op through all the IR
    Value result = rw.create<rock::TensorUntransformCastOp>(
        loc, cast<RankedTensorType>(op.getC().getType()), gemm.getResult(),
        gemm.getC());
    rw.replaceOp(op, result);

    return success();
  }
};

static Value insertSliceOfLastDim(PatternRewriter &rewriter, Location loc,
                                  Value source, Value dest, Value sliceIdx) {
  TensorType srcType = cast<TensorType>(source.getType());
  ArrayRef<int64_t> srcShape = srcType.getShape();
  int64_t mbMemRefTypeRank = srcType.getRank();
  IntegerAttr zero = rewriter.getIndexAttr(0);
  IntegerAttr one = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> offsets(mbMemRefTypeRank, zero);
  SmallVector<OpFoldResult> sizes(mbMemRefTypeRank, one);
  SmallVector<OpFoldResult> strides(mbMemRefTypeRank, one);
  // Offset is [0 ... 0, bufferIndex].
  offsets.back() = sliceIdx;
  for (int64_t i = 0, e = srcShape.size(); i != e; ++i)
    sizes[i] = rewriter.getIndexAttr(srcShape[i]);
  Value subview = rewriter.create<tensor::InsertSliceOp>(
      loc, source, dest, offsets, sizes, strides);
  return subview;
}

static Value createSliceOfLastDim(PatternRewriter &rewriter, Location loc,
                                  Value buffer, Value sliceIdx) {
  TensorType bufType = cast<TensorType>(buffer.getType());
  ArrayRef<int64_t> originalShape = bufType.getShape().drop_back(1);
  int64_t mbMemRefTypeRank = bufType.getRank();
  IntegerAttr zero = rewriter.getIndexAttr(0);
  IntegerAttr one = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> offsets(mbMemRefTypeRank, zero);
  SmallVector<OpFoldResult> sizes(mbMemRefTypeRank, one);
  SmallVector<OpFoldResult> strides(mbMemRefTypeRank, one);
  // Offset is [0 ... 0, bufferIndex].
  offsets.back() = sliceIdx;
  // Sizes is [original_size_0 ... original_size_n, 1].
  for (int64_t i = 0, e = originalShape.size(); i != e; ++i)
    sizes[i] = rewriter.getIndexAttr(originalShape[i]);
  // auto dstMemref =
  //     cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
  //         originalShape, bufType, offsets, sizes, strides));
  Value subview = rewriter.create<tensor::ExtractSliceOp>(loc, buffer, offsets,
                                                          sizes, strides);
  // subview.dump();
  // SmallVector<ReassociationIndices> reassocIndices = {{0}, {1, 2}};
  // Value collapsed =
  //     rewriter.create<tensor::CollapseShapeOp>(loc, subview, reassocIndices);
  return subview;
}

struct LoopGemmBroadcast : public OpRewritePattern<rock::GemmOp> {
  using OpRewritePattern<rock::GemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::GemmOp op,
                                PatternRewriter &rw) const override {

    if (!op->hasOneUse()) {
      llvm::errs() << "returning failure 0\n";
      return failure();
    }
    rock::TransformOp transform;
    linalg::GenericOp laGenericOp;
    SmallVector<Operation *> users(op->getUsers());
    while (!laGenericOp and !users.empty()) {
      Operation *currUser = users.pop_back_val();
      for (Operation *userUser : currUser->getUsers()) {
        if (dyn_cast<linalg::GenericOp>(userUser) &&
            dyn_cast<rock::TransformOp>(currUser)) {
          laGenericOp = dyn_cast<linalg::GenericOp>(userUser);
          transform = dyn_cast<rock::TransformOp>(currUser);
        } else {
          users.push_back(userUser);
        }
      }
    }
    Operation *transOp = transform;
    SmallVector<Operation *> transformChain;
    while (!dyn_cast<rock::GemmOp>(transOp)) {
      transformChain.push_back(transOp);
      transOp = dyn_cast<rock::TransformOp>(transOp).getInput().getDefiningOp();
    }
    transformChain.push_back(op);
    if (!transform or !laGenericOp) {
      llvm::errs() << "didn't find inputs\n";
      return failure();
    }
    rock::TransformMapAttr transMap = transform.getTransformAttr();
    // TODO(umang) add checks that it is broadcasting dimension of bound 1
    if (llvm::any_of(transMap.getOps(), [](rock::TransformAttr transformType) {
          return transformType.getType() != rock::TransformType::PassThrough and
                 transformType.getType() != rock::TransformType::Broadcast;
        })) {
      llvm::errs() << "returning failure: 1 \n";
      return failure();
    }
    auto outType = cast<ShapedType>(laGenericOp->getResult(0).getType());
    auto inpType = cast<ShapedType>(op->getResult(0).getType());
    int64_t broadcastingFactor = 1;
    if (outType.getShape() != inpType.getShape()) {
      broadcastingFactor = outType.getNumElements() / inpType.getNumElements();
    }
    auto loc = laGenericOp.getLoc();
    llvm::errs() << "before\n";
    op->getParentOp()->dump();
    auto zeroConstantOp = rw.create<arith::ConstantIndexOp>(loc, 0);
    Value nIterations =
        rw.create<mlir::arith::ConstantIndexOp>(loc, broadcastingFactor);
    Value step = rw.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto input1 = laGenericOp.getInputs().back();
    if (input1.getDefiningOp<tensor::ExtractSliceOp>()) {
      llvm::errs() << "returning failure: 2 \n";
      return failure();
    }
    tensor::EmptyOp outTensor = dyn_cast<tensor::EmptyOp>(
        (*laGenericOp.getOutputs().begin()).getDefiningOp());

    Value outputTensor = rw.create<tensor::EmptyOp>(
        loc, outTensor.getMixedSizes(), outTensor.getType().getElementType());
    auto loopOp = rw.create<scf::ForOp>(loc, zeroConstantOp, nIterations, step,
                                        ValueRange{outputTensor});
    {
      PatternRewriter::InsertionGuard guard(rw);
      rw.setInsertionPointToStart(loopOp.getBody());
      Value iv = loopOp.getInductionVar();
      linalg::GenericOp newGenOp;
      auto dims = cast<ShapedType>(input1.getType()).getShape();
      SmallVector<int64_t> outDims(dims);
      outDims[dims.size() - 1] = 1;
      Value slicedInput = createSliceOfLastDim(rw, loc, input1, iv);
      Value slicedOutput =
          createSliceOfLastDim(rw, loc, *loopOp.getInitArgs().begin(), iv);
      for (Operation *op : transformChain) {
        rw.moveOpAfter(op, slicedOutput.getDefiningOp());
      }
      Value origInput0 = transform.getInput();
      // Value input0 = origInput0;
      //  while (!input0.getDefiningOp<rock::GemmOp>()) {
      //    rw.moveOpAfter(input0.getDefiningOp(), slicedInput.getDefiningOp());
      //    input0 = input0.getDefiningOp<rock::TransformOp>().getInput();
      //  }
      //  rw.moveOpAfter(input0.getDefiningOp(), slicedInput.getDefiningOp());

      auto laIndexingMaps = laGenericOp.getIndexingMapsArray();
      SmallVector<AffineMap, 4> newAffineMaps;
      for (auto map : laIndexingMaps) {
        newAffineMaps.push_back(map);
      }
      linalg::GenericOp newlaGenericOp = rw.create<linalg::GenericOp>(
          loc, slicedOutput.getType(), ValueRange{origInput0, slicedInput},
          ValueRange{slicedOutput}, newAffineMaps,
          laGenericOp.getIteratorTypesArray());
      newlaGenericOp.getRegion().takeBody(laGenericOp->getRegion(0));
      Value insertedSlice =
          insertSliceOfLastDim(rw, loc, newlaGenericOp->getResults().front(),
                               loopOp.getInitArgs().front(), iv);
      rw.create<scf::YieldOp>(loc, insertedSlice);
    }
    llvm::errs() << "after\n";
    op->getParentOp()->dump();
    llvm::errs() << "erasing now\n";
    rw.replaceOp(laGenericOp, loopOp);
    llvm::errs() << "after erasing\n";
    op->getParentOp()->dump();
    return success();
  }
};

struct FoldTransformBroadcast : public OpRewritePattern<rock::TransformOp> {
  using OpRewritePattern<rock::TransformOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(rock::TransformOp op,
                                PatternRewriter &rw) const override {
    if (not dyn_cast<linalg::GenericOp>(*op->getUsers().begin())) {
      return failure();
    }
    rock::TransformMapAttr transMap = op.getTransformAttr();
    // TODO(umang) add checks that it is broadcasting dimension of bound 1
    if (llvm::any_of(transMap.getOps(), [](rock::TransformAttr transformType) {
          return transformType.getType() != rock::TransformType::PassThrough and
                 transformType.getType() != rock::TransformType::Broadcast;
        }))
      return failure();

    linalg::GenericOp laGenericOp =
        dyn_cast_or_null<linalg::GenericOp>(*op->getUsers().begin());
    if (!laGenericOp) {
      return failure();
    }
    SmallVector<Value, 4> newInputs;
    auto laIndexingMaps = laGenericOp.getIndexingMapsArray();
    SmallVector<AffineMap, 4> newAffineMaps;
    for (auto [idx, input] : llvm::enumerate(laGenericOp.getInputs())) {
      if (dyn_cast<rock::TransformOp>(input.getDefiningOp()) == op) {
        newAffineMaps.push_back(transMap.getMap().getAffineMap());
        newInputs.push_back(op.getInput());
      } else {
        newAffineMaps.push_back(laIndexingMaps[idx]);
        newInputs.push_back(input);
      }
    }
    newAffineMaps.push_back(laIndexingMaps.back());
    tensor::EmptyOp outTensor = dyn_cast<tensor::EmptyOp>(
        (*laGenericOp.getOutputs().begin()).getDefiningOp());

    Value outputTensor = rw.create<tensor::EmptyOp>(
        laGenericOp->getLoc(), outTensor.getMixedSizes(),
        outTensor.getType().getElementType());

    linalg::GenericOp newlaGenericOp = rw.create<linalg::GenericOp>(
        laGenericOp->getLoc(), outputTensor.getType(), ValueRange{newInputs},
        ValueRange{outputTensor}, newAffineMaps,
        laGenericOp.getIteratorTypesArray());
    newlaGenericOp.getRegion().takeBody(laGenericOp->getRegion(0));

    rw.replaceOp(laGenericOp, newlaGenericOp);
    rw.eraseOp(op);
    return success();
  }
};

struct RockFoldBroadcastPass
    : public rock::impl::RockFoldBroadcastPassBase<RockFoldBroadcastPass> {
  void runOnOperation() override;
};
} // end namespace

void RockFoldBroadcastPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto func = getOperation();
  if (!func->hasAttr("kernel")) {
    // disable for non-kernels
    return;
  }

  {
    // ConversionTarget target(*ctx);
    // target.addLegalDialect<arith::ArithDialect, rock::RockDialect,
    //                        memref::MemRefDialect, linalg::LinalgDialect,
    //                        scf::SCFDialect>();
    // target.addLegalOp<rock::GemmOp>();
    // RewritePatternSet patterns(ctx);
    // patterns.add<FoldBroadcast, LoopGemmBroadcast>(ctx);
    // if (failed(applyPartialConversion(getOperation(), target,
    //                                   std::move(patterns))))
    //   signalPassFailure();
    RewritePatternSet patterns(ctx);
    patterns.add<LoopGemmBroadcast>(ctx);
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
}
