// RUN: rocmlir-opt --rock-view-to-transform -rock-affix-params -rock-conv-to-gemm -rock-gemm-to-gridwise -rock-shuffle-gemm-for-reductions -rock-gridwise-gemm-to-blockwise -rock-linalg-align %s -mlir-print-local-scope | FileCheck %s

// CHECK: test_gemm_reduce_last_axis_fusion
func.func @test_gemm_reduce_last_axis_fusion(%arg0: memref<1x128x64xf32>, %arg1: memref<1x64x256xf32>, %arg2: memref<1x128x1xf32>) attributes {arch = "", kernel} {
  %0 = memref.alloc() : memref<1x128x256xf32>
  rock.gemm %0 = %arg0 * %arg1 features =  none storeMethod =  set {arch = ""} : memref<1x128x256xf32> = memref<1x128x64xf32> * memref<1x64x256xf32>
  // CHECK: rock.blockwise_broadcast_reduce  sum {{.*}} into %[[BLOCK_RED_OUT:[0-9]+]]

  // CHECK: %[[TR0:.+]] = rock.transform %arg2 by {{.*}}    : memref<1x128x1xf32> to memref<1x128x256xf32>
  // CHECK: %[[TR1:.+]] = rock.transform %[[TR0]] by {{.*}} : memref<1x128x256xf32> to memref<16x2x64x8x32xf32>
  // CHECK: %[[TR2:.+]] = rock.transform %[[TR1]] by {{.*}} : memref<16x2x64x8x32xf32> to memref<16x2x8x1x64x32xf32>
  // CHECK: %[[TR3:.+]] = rock.transform %[[TR2]] by {{.*}} : memref<16x2x8x1x64x32xf32> to memref<16x2x8x64x1xf32>
  // CHECK: %[[TR4:.+]] = rock.transform %[[TR3]] by {{.*}}<Pad{0, 31} ["dim1"] at [4] -> ["dim1"] at [4]>{{.*}} : memref<16x2x8x64x1xf32> to memref<16x2x8x64x32xf32>

  // CHECK: %[[TR5:.+]] = rock.transform %[[TR4]] by {{.*}} : memref<16x2x8x64x32xf32> to memref<16x2x8x64x32xf32>
  // CHECK: %[[TR6:.+]] = rock.transform %[[TR5]] by {{.*}} : memref<16x2x8x64x32xf32> to memref<16x2x8x64x32xf32>
  // CHECK: %[[TR7:.+]] = rock.transform %[[TR6]] by {{.*}} : memref<16x2x8x64x32xf32> to memref<16x2x8x64x32xf32>
  // CHECK: %[[TR8:.+]] = rock.transform %[[TR7]] by {{.*}} : memref<16x2x8x64x32xf32> to memref<16x2x8x2x2x4x4x4x2x2x2xf32>
  // CHECK: %[[TR9:.+]] = rock.transform %[[TR8]] by {{.*}} : memref<16x2x8x2x2x4x4x4x2x2x2xf32> to memref<16x2x8x64x32xf32>

  // CHECK: rock.threadwise_write_all {{.*}}%[[BLOCK_RED_OUT]] -> [](%[[TR9]]){{.*}} by  atomic_add : {{.*}}
  rock.reduce sum %0 into %arg2 features = mfma|dot|atomic_add {axis = 2 : index, blockSize = 256 : i32, gridSize = 1 : i32} : memref<1x128x256xf32> into memref<1x128x1xf32>
  return
}


// CHECK: test_gemm_reduce_middle_axis_fusion
func.func @test_gemm_reduce_middle_axis_fusion(%arg0: memref<1x128x64xf32>, %arg1: memref<1x64x256xf32>, %arg2: memref<1x1x256xf32>) attributes {arch = "", kernel} {
  %0 = memref.alloc() : memref<1x128x256xf32>
  rock.gemm %0 = %arg0 * %arg1 features =  none storeMethod =  set {arch = ""} : memref<1x128x256xf32> = memref<1x128x64xf32> * memref<1x64x256xf32>
  // CHECK: rock.blockwise_broadcast_reduce  sum {{.*}} into %[[BLOCK_RED_OUT:[0-9]+]]

  // CHECK: %[[TR0:.+]] = rock.transform %arg2 by {{.*}}    : memref<1x1x256xf32> to memref<1x128x256xf32>
  // CHECK: %[[TR1:.+]] = rock.transform %[[TR0]] by {{.*}} : memref<1x128x256xf32> to memref<16x2x64x8x32xf32>
  // CHECK: %[[TR2:.+]] = rock.transform %[[TR1]] by {{.*}} : memref<16x2x64x8x32xf32> to memref<16x2x8x1x64x32xf32>
  // CHECK: %[[TR3:.+]] = rock.transform %[[TR2]] by {{.*}} : memref<16x2x8x1x64x32xf32> to memref<16x2x8x1x32xf32>
  // CHECK: %[[TR4:.+]] = rock.transform %[[TR3]] by {{.*}}<Pad{0, 63} ["dim0"] at [3] -> ["dim0"] at [3]>{{.*}} : memref<16x2x8x1x32xf32> to memref<16x2x8x64x32xf32>

  // CHECK: %[[TR5:.+]] = rock.transform %[[TR4]] by {{.*}} : memref<16x2x8x64x32xf32> to memref<16x2x8x64x32xf32>
  // CHECK: %[[TR6:.+]] = rock.transform %[[TR5]] by {{.*}} : memref<16x2x8x64x32xf32> to memref<16x2x8x64x32xf32>
  // CHECK: %[[TR7:.+]] = rock.transform %[[TR6]] by {{.*}} : memref<16x2x8x64x32xf32> to memref<16x2x8x64x32xf32>
  // CHECK: %[[TR8:.+]] = rock.transform %[[TR7]] by {{.*}} : memref<16x2x8x64x32xf32> to memref<16x2x8x2x2x4x4x4x2x2x2xf32>
  // CHECK: %[[TR9:.+]] = rock.transform %[[TR8]] by {{.*}} : memref<16x2x8x2x2x4x4x4x2x2x2xf32> to memref<16x2x8x64x32xf32>

  // CHECK: rock.threadwise_write_all {{.*}}%[[BLOCK_RED_OUT]] -> [](%[[TR9]]){{.*}} by  atomic_add : {{.*}}
  rock.reduce sum %0 into %arg2 features = mfma|dot|atomic_add {axis = 1 : index, blockSize = 256 : i32, gridSize = 1 : i32} : memref<1x128x256xf32> into memref<1x1x256xf32>
  return
}

// CHECK: test_gemm_add_reduce_fusion
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @test_gemm_add_reduce_fusion(%arg0: memref<1x128x64xf32>, %arg1: memref<1x64x256xf32>, %arg2: memref<1x128x256xf32>, %arg3: memref<1x128x1xf32>) attributes {arch = "", kernel} {
  %0 = memref.alloc() : memref<1x128x256xf32>
  rock.gemm %0 = %arg0 * %arg1 features =  none storeMethod =  set {arch = ""} : memref<1x128x256xf32> = memref<1x128x64xf32> * memref<1x64x256xf32>
  %1 = memref.alloc() : memref<1x128x256xf32>
  linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%0, %arg2 : memref<1x128x256xf32>, memref<1x128x256xf32>) outs(%1 : memref<1x128x256xf32>) {
  ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):
    %4 = arith.addf %arg4, %arg5 : f32
    linalg.yield %4 : f32
  }
  // CHECK: rock.blockwise_broadcast_reduce  sum {{.*}} into %[[BLOCK_RED_OUT:[0-9]+]]

  // CHECK: %[[TR0:.+]] = rock.transform %arg3 by {{.*}}    : memref<1x128x1xf32> to memref<1x128x256xf32>
  // CHECK: %[[TR1:.+]] = rock.transform %[[TR0]] by {{.*}} : memref<1x128x256xf32> to memref<16x2x64x8x32xf32>
  // CHECK: %[[TR2:.+]] = rock.transform %[[TR1]] by {{.*}} : memref<16x2x64x8x32xf32> to memref<16x2x8x1x64x32xf32>
  // CHECK: %[[TR3:.+]] = rock.transform %[[TR2]] by {{.*}} : memref<16x2x8x1x64x32xf32> to memref<16x2x8x64x1xf32>
  // CHECK: %[[TR4:.+]] = rock.transform %[[TR3]] by {{.*}}<Pad{0, 31} ["dim1"] at [4] -> ["dim1"] at [4]>{{.*}} : memref<16x2x8x64x1xf32> to memref<16x2x8x64x32xf32>

  // CHECK: %[[TR5:.+]] = rock.transform %[[TR4]] by {{.*}} : memref<16x2x8x64x32xf32> to memref<16x2x8x64x32xf32>
  // CHECK: %[[TR6:.+]] = rock.transform %[[TR5]] by {{.*}} : memref<16x2x8x64x32xf32> to memref<16x2x8x64x32xf32>
  // CHECK: %[[TR7:.+]] = rock.transform %[[TR6]] by {{.*}} : memref<16x2x8x64x32xf32> to memref<16x2x8x64x32xf32>
  // CHECK: %[[TR8:.+]] = rock.transform %[[TR7]] by {{.*}} : memref<16x2x8x64x32xf32> to memref<16x2x8x2x2x4x4x4x2x2x2xf32>
  // CHECK: %[[TR9:.+]] = rock.transform %[[TR8]] by {{.*}} : memref<16x2x8x2x2x4x4x4x2x2x2xf32> to memref<16x2x8x64x32xf32>

  // CHECK: rock.threadwise_write_all {{.*}}%[[BLOCK_RED_OUT]] -> [](%[[TR9]]){{.*}} by  atomic_add : {{.*}}
  rock.reduce sum %1 into %arg3 features = mfma|dot|atomic_add {axis = 2 : index, blockSize = 256 : i32, gridSize = 1 : i32} : memref<1x128x256xf32> into memref<1x128x1xf32>
  return
}

// CHECK: test_gemm_reduce_max
func.func @test_gemm_reduce_max(%arg0: memref<1x128x64xf32>, %arg1: memref<1x64x256xf32>, %arg2: memref<1x128x1xf32>) attributes {arch = "", kernel} {
  %0 = memref.alloc() : memref<1x128x256xf32>
  rock.gemm %0 = %arg0 * %arg1 features =  none storeMethod =  set {arch = ""} : memref<1x128x256xf32> = memref<1x128x64xf32> * memref<1x64x256xf32>
  // CHECK: rock.blockwise_broadcast_reduce  max {{.*}} into %[[BLOCK_RED_OUT:[0-9]+]]

  // CHECK: %[[TR0:.+]] = rock.transform %arg2 by {{.*}}    : memref<1x128x1xf32> to memref<1x128x256xf32>
  // CHECK: %[[TR1:.+]] = rock.transform %[[TR0]] by {{.*}} : memref<1x128x256xf32> to memref<16x2x64x8x32xf32>
  // CHECK: %[[TR2:.+]] = rock.transform %[[TR1]] by {{.*}} : memref<16x2x64x8x32xf32> to memref<16x2x8x1x64x32xf32>
  // CHECK: %[[TR3:.+]] = rock.transform %[[TR2]] by {{.*}} : memref<16x2x8x1x64x32xf32> to memref<16x2x8x64x1xf32>
  // CHECK: %[[TR4:.+]] = rock.transform %[[TR3]] by {{.*}}<Pad{0, 31} ["dim1"] at [4] -> ["dim1"] at [4]>{{.*}} : memref<16x2x8x64x1xf32> to memref<16x2x8x64x32xf32>

  // CHECK: %[[TR5:.+]] = rock.transform %[[TR4]] by {{.*}} : memref<16x2x8x64x32xf32> to memref<16x2x8x64x32xf32>
  // CHECK: %[[TR6:.+]] = rock.transform %[[TR5]] by {{.*}} : memref<16x2x8x64x32xf32> to memref<16x2x8x64x32xf32>
  // CHECK: %[[TR7:.+]] = rock.transform %[[TR6]] by {{.*}} : memref<16x2x8x64x32xf32> to memref<16x2x8x64x32xf32>
  // CHECK: %[[TR8:.+]] = rock.transform %[[TR7]] by {{.*}} : memref<16x2x8x64x32xf32> to memref<16x2x8x2x2x4x4x4x2x2x2xf32>
  // CHECK: %[[TR9:.+]] = rock.transform %[[TR8]] by {{.*}} : memref<16x2x8x2x2x4x4x4x2x2x2xf32> to memref<16x2x8x64x32xf32>

  // CHECK: rock.threadwise_write_all {{.*}}%[[BLOCK_RED_OUT]] -> [](%[[TR9]]){{.*}} by  atomic_max : {{.*}}
  rock.reduce max %0 into %arg2 features = mfma|dot|atomic_add {axis = 2 : index, blockSize = 256 : i32, gridSize = 1 : i32} : memref<1x128x256xf32> into memref<1x128x1xf32>
  return
}
