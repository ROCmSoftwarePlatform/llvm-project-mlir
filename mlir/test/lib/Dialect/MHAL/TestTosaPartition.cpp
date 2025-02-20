//===- TosaTestPasses.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Test passes to exercise TOSA helper functions.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MHAL/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Utils/QuantUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::mhal;

namespace {

class TestTosaPartitionOptionsPass
    : public PassWrapper<TestTosaPartitionOptionsPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestTosaPartitionOptionsPass)

  StringRef getArgument() const final { return "test-tosa-partition-options"; }
  StringRef getDescription() const final {
    return "Tests the programmatic interface to --tosa-partition options.";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tosa::TosaDialect>();
  }

  TestTosaPartitionOptionsPass() = default;
  TestTosaPartitionOptionsPass(const TestTosaPartitionOptionsPass &) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    PassManager pm(module->getName(), mlir::PassManager::Nesting::Implicit);
    if (defaultCase) {
      pm.addPass(createTosaPartition());
    } else if (depthwiseOnly) {
      SmallVector<std::string> anchors = {"tosa.depthwise_conv2d"};
      TosaPartitionOptions options;
      options.anchorOps = anchors;
      pm.addPass(createTosaPartition(options));
    } else if (convOnly) {
      SmallVector<std::string> anchors = {"tosa.conv2d"};
      TosaPartitionOptions options;
      options.anchorOps = anchors;
      options.partitionTagOpt = "four";
      pm.addPass(createTosaPartition(options));
    } else if (attrOne) {
      // TODO: Once list options can have defaults, use that
      SmallVector<std::string> anchors = {"tosa.conv2d", "tosa.matmul",
                                          "tosa.depthwise_conv2d"};
      TosaPartitionOptions options;
      options.anchorOps = anchors;
      options.partitionTagOpt = "one";
      pm.addPass(createTosaPartition(options));
    } else if (nofrontArg) {
      SmallVector<std::string> anchors = {"tosa.depthwise_conv2d"};
      TosaPartitionOptions options;
      options.anchorOps = anchors;
      options.trailingOnly = true;
      options.partitionTagOpt = "three";
      pm.addPass(createTosaPartition(options));
    }

    if (failed(pm.run(module)))
      signalPassFailure();
  }

  Option<bool> defaultCase{*this, "default", llvm::cl::desc("Default.")};
  Option<bool> depthwiseOnly{*this, "depthwise-only",
                             llvm::cl::desc("Depthwise only.")};
  Option<bool> convOnly{*this, "conv-only", llvm::cl::desc("Only conv2d.")};
  Option<bool> attrOne{*this, "attr-one",
                       llvm::cl::desc("Attribute-name 'one'.")};
  Option<bool> nofrontArg{*this, "nofront-arg",
                          llvm::cl::desc("Nofront as arg.")};
};
} // namespace

namespace mlir {
namespace mhal {
void registerTestTosaPartitionOptionsPass() {
  PassRegistration<TestTosaPartitionOptionsPass>();
}
} // namespace mhal
} // namespace mlir
