// RUN: rocmlir-opt --tosa-to-rock %s -o -| FileCheck %s

module attributes {kernel.module, mhal.arch = "amdgcn-amd-amdhsa:gfx906"} {
// CHECK-LABEL: @test_basic
// CHECK-SAME: -> (tensor<2x10x1xf32> {mhal.read_access, rock.prefill = 0.000000e+00 : f32})
func.func @test_basic(%arg0: tensor<2x10x100xf32>) -> tensor<2x10x1xf32> attributes {kernel} {
  // CHECK: %[[outBuf:.*]] = bufferization.alloc_tensor() : tensor<2x10x1xf32>
  // CHECK: rock.reduce  sum %arg0 into %[[outBuf]] {{.*}} {axis = 2 : index, blockSize = 256 : i32, gridSize = 8 : i32} : tensor<2x10x100xf32> into tensor<2x10x1xf32> -> tensor<2x10x1xf32>
  %1 = "tosa.reduce_sum"(%arg0) {axis = 2 : i32} : (tensor<2x10x100xf32>) -> tensor<2x10x1xf32>
  return %1 : tensor<2x10x1xf32>
}

// CHECK-LABEL: @test_basic_f16
// CHECK-SAME: -> (tensor<2x10x1xf16> {mhal.read_access, rock.prefill = 0.000000e+00 : f16})
func.func @test_basic_f16(%arg0: tensor<2x10x100xf16>) -> tensor<2x10x1xf16> attributes {kernel} {
  // CHECK: %[[outBuf:.*]] = bufferization.alloc_tensor() : tensor<2x10x1xf16>
  // CHECK: rock.reduce  sum %arg0 into %[[outBuf]] {{.*}} {axis = 2 : index, blockSize = 256 : i32, gridSize = 8 : i32} : tensor<2x10x100xf16> into tensor<2x10x1xf16> -> tensor<2x10x1xf16>
  %1 = "tosa.reduce_sum"(%arg0) {axis = 2 : i32} : (tensor<2x10x100xf16>) -> tensor<2x10x1xf16>
  return %1 : tensor<2x10x1xf16>
}

// CHECK-LABEL: @test_middle_axis_reduction
// CHECK-SAME: -> (tensor<4x1x20xf32> {mhal.read_access, rock.prefill = 0.000000e+00 : f32})
func.func @test_middle_axis_reduction(%arg0: tensor<4x300x20xf32>) -> tensor<4x1x20xf32> attributes {kernel} {
  // CHECK: %[[outBuf:.*]] = bufferization.alloc_tensor() : tensor<4x1x20xf32>
  // CHECK: rock.reduce  sum %arg0 into %[[outBuf]] {{.*}} {axis = 1 : index, blockSize = 256 : i32, gridSize = 94 : i32} : tensor<4x300x20xf32> into tensor<4x1x20xf32> -> tensor<4x1x20xf32>
  %1 = "tosa.reduce_sum"(%arg0) {axis = 1 : i32} : (tensor<4x300x20xf32>) -> tensor<4x1x20xf32>
  return %1 : tensor<4x1x20xf32>
}

// CHECK-LABEL: @test_reduce_max
// CHECK-SAME: -> (tensor<2x10x1xf32> {mhal.read_access, rock.prefill = 0xFF800000 : f32})
func.func @test_reduce_max(%arg0: tensor<2x10x100xf32>) -> tensor<2x10x1xf32> attributes {kernel} {
  // CHECK: %[[outBuf:.*]] = bufferization.alloc_tensor() : tensor<2x10x1xf32>
  // CHECK: rock.reduce  max %arg0 into %[[outBuf]] {{.*}} {axis = 2 : index, blockSize = 256 : i32, gridSize = 8 : i32} : tensor<2x10x100xf32> into tensor<2x10x1xf32> -> tensor<2x10x1xf32>
  %1 = "tosa.reduce_max"(%arg0) {axis = 2 : i32} : (tensor<2x10x100xf32>) -> tensor<2x10x1xf32>
  return %1 : tensor<2x10x1xf32>
}

// CHECK-LABEL: @test_reduce_max_two_outputs
// CHECK-SAME: -> (tensor<2x10x1xf32> {mhal.read_access, rock.prefill = 0xFF800000 : f32}
// CHECK-SAME: , tensor<2x1x100xf32> {mhal.read_access, rock.prefill = 0xFF800000 : f32})
func.func @test_reduce_max_two_outputs(%arg0: tensor<2x10x100xf32>, %arg1: tensor<2x100x100xf32>) -> (tensor<2x10x1xf32>, tensor<2x1x100xf32>) attributes {kernel} {
  // CHECK: %[[outBuf:.*]] = bufferization.alloc_tensor() : tensor<2x10x1xf32>
  // CHECK: rock.reduce  max %arg0 into %[[outBuf]] {{.*}} {axis = 2 : index, blockSize = 256 : i32, gridSize = 8 : i32} : tensor<2x10x100xf32> into tensor<2x10x1xf32> -> tensor<2x10x1xf32>
  %1 = "tosa.reduce_max"(%arg0) {axis = 2 : i32} : (tensor<2x10x100xf32>) -> tensor<2x10x1xf32>
  // CHECK: %[[outBuf2:.*]] = bufferization.alloc_tensor() : tensor<2x1x100xf32>
  // CHECK: rock.reduce  max %arg1 into %[[outBuf2]] {{.*}} {axis = 1 : index, blockSize = 256 : i32, gridSize = 79 : i32} : tensor<2x100x100xf32> into tensor<2x1x100xf32> -> tensor<2x1x100xf32>
  %2 = "tosa.reduce_max"(%arg1) {axis = 1 : i32} : (tensor<2x100x100xf32>) -> tensor<2x1x100xf32>
  return %1, %2 : tensor<2x10x1xf32>, tensor<2x1x100xf32>
}

// CHECK-LABEL: @test_reduce_sum_two_outputs
// CHECK-SAME: -> (tensor<2x10x1xf32> {mhal.read_access, rock.prefill = 0.000000e+00 : f32}
// CHECK-SAME: , tensor<2x1x100xf32> {mhal.read_access, rock.prefill = 0.000000e+00 : f32})
func.func @test_reduce_sum_two_outputs(%arg0: tensor<2x10x100xf32>, %arg1: tensor<2x100x100xf32>) -> (tensor<2x10x1xf32>, tensor<2x1x100xf32>) attributes {kernel} {
  // CHECK: %[[outBuf:.*]] = bufferization.alloc_tensor() : tensor<2x10x1xf32>
  // CHECK: rock.reduce  sum %arg0 into %[[outBuf]] {{.*}} {axis = 2 : index, blockSize = 256 : i32, gridSize = 8 : i32} : tensor<2x10x100xf32> into tensor<2x10x1xf32> -> tensor<2x10x1xf32>
  %1 = "tosa.reduce_sum"(%arg0) {axis = 2 : i32} : (tensor<2x10x100xf32>) -> tensor<2x10x1xf32>
  // CHECK: %[[outBuf2:.*]] = bufferization.alloc_tensor() : tensor<2x1x100xf32>
  // CHECK: rock.reduce  sum %arg1 into %[[outBuf2]] {{.*}} {axis = 1 : index, blockSize = 256 : i32, gridSize = 79 : i32} : tensor<2x100x100xf32> into tensor<2x1x100xf32> -> tensor<2x1x100xf32>
  %2 = "tosa.reduce_sum"(%arg1) {axis = 1 : i32} : (tensor<2x100x100xf32>) -> tensor<2x1x100xf32>
  return %1, %2 : tensor<2x10x1xf32>, tensor<2x1x100xf32>
}

}
