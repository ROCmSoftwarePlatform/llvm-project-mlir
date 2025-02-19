// This test is checking for a reduction in highest dimension

// RUN: sed -e 's/##TOKEN_ARCH##/%arch/g; s/##TOKEN_FEATURES##/%features/g' %s | rocmlir-gen -ph -print-results -fut test_reduce -verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext --entry-point-result=void | FileCheck %s --check-prefix=CLONE
// CLONE: [1 1 1]
// CLONE-NEXT: Unranked Memref base

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
module {
  func.func private @init_output(%arg0: memref<1x30x10xf32> {mhal.write_access}) {
    %cst = arith.constant 0xff800000 : f32
    linalg.fill ins(%cst : f32) outs(%arg0 : memref<1x30x10xf32>)
    return
  }
  func.func private @test_reduce__part_1(%arg0: memref<20x30x10xf32> {mhal.read_access}, %arg1: memref<1x30x10xf32> {mhal.read_access, mhal.write_access}) {
    %0 = memref.collapse_shape %arg1 [[0, 1], [2]] : memref<1x30x10xf32> into memref<30x10xf32>
    linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["reduction", "parallel", "parallel"]} ins(%arg0 : memref<20x30x10xf32>) outs(%0 : memref<30x10xf32>) {
    ^bb0(%arg2: f32, %arg3: f32):
      %2 = arith.maximumf %arg2, %arg3 : f32
      linalg.yield %2 : f32
    }
    return
  }
  func.func @test_reduce(%arg0: memref<20x30x10xf32>, %arg1: memref<1x30x10xf32> {mhal.read_access, mhal.write_access}) attributes {arch = ""} {
    call @init_output (%arg1) : (memref<1x30x10xf32>) -> ()
    %token1 = mhal.launch @test_reduce__part_1 (%arg0, %arg1) : (memref<20x30x10xf32>, memref<1x30x10xf32>)
    mhal.await %token1 : !mhal.token
    return
  }
  module @__xmodule_ attributes {mhal.arch = "##TOKEN_ARCH##", mhal.module} {
    func.func private @test_reduce__part_1(%arg0: memref<20x30x10xf32> {mhal.read_access}, %arg1: memref<1x30x10xf32> {mhal.read_access, mhal.write_access, rock.prefill = 0xFF800000 : f32}) attributes {kernel, original_func = @test_reduce__part_1, grid_size = 2, block_size = 256} {
      rock.reduce max %arg0 into %arg1 features = ##TOKEN_FEATURES## {axis = 0 : index, blockSize = 256 : i32, gridSize = 2 : i32} : memref<20x30x10xf32> into memref<1x30x10xf32>
      return
    }
  }
}
