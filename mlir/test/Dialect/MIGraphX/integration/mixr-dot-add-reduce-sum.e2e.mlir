// RUN: rocmlir-gen -fut dot_add_reduce_sum --arch %arch --clone-harness %s | rocmlir-driver -kernel-pipeline=migraphx | rocmlir-driver -host-pipeline=migraphx,highlevel | rocmlir-gen -ph -rand 1 -rand_type float -fut dot_add_reduce_sum_wrapper --verifier clone - | rocmlir-driver -host-pipeline mhal -kernel-pipeline full | xmir-runner --shared-libs=%linalg_test_lib_dir/libmlir_rocm_runtime%shlibext,%conv_validation_wrapper_library_dir/libconv-validation-wrappers%shlibext,%linalg_test_lib_dir/libmlir_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_float16_utils%shlibext,%linalg_test_lib_dir/libmlir_c_runner_utils%shlibext,%linalg_test_lib_dir/libmlir_async_runtime%shlibext --entry-point-result=void | FileCheck %s
// ALLOW_RETRIES: 2
// CHECK: [1 1 1]
module {
func.func @dot_add_reduce_sum(%arg0: !migraphx.shaped<16x64x2xf32, 128x2x1>, %arg1: !migraphx.shaped<16x16xf32, 16x1>, %arg2: !migraphx.shaped<16x64xf32, 64x1>) -> !migraphx.shaped<16x64x2xf32, 128x2x1> {
    %0 = migraphx.dot %arg1, %arg2 : <16x16xf32, 16x1>, <16x64xf32, 64x1> -> <16x64xf32, 64x1>
    %1 = migraphx.broadcast %0 {axis = 0, out_lens = [16, 64, 2]} : <16x64xf32, 64x1> -> <16x64x2xf32, 64x1x0>
    %2 = migraphx.sigmoid %1 : <16x64x2xf32, 64x1x0> -> <16x64x2xf32, 64x1x0>
    %3 = migraphx.mul %2, %arg0 : <16x64x2xf32, 64x1x0>, <16x64x2xf32, 128x2x1> -> <16x64x2xf32, 128x2x1>
    return %3 : !migraphx.shaped<16x64x2xf32, 128x2x1>
  }
}
