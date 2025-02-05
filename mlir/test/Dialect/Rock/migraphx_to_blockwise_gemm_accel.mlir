// RUN: sed s/##TOKEN_ARCH##/%arch/g %s | rocmlir-driver -kernel-pipeline migraphx | rocmlir-driver -host-pipeline partition,highlevel -targets %arch | rocmlir-gen -rand none - | rocmlir-driver -arch %arch -c --mlir-print-ir-after=rock-gridwise-gemm-to-blockwise -o /dev/null 2>&1 | FileCheck %s

module {
  // CHECK: %[[TRANS0:.*]] = rock.transform %{{.*}} by <affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, (d0 * 4 + d5) * 32 + d7, d3 * 32 + d4 + d6)> by [<PassThrough ["g_block"] at [1] -> ["g"] at [0]>, <Unmerge{32, 4, 32} ["k_loop", "k_thread", "k_iter"] at [0, 5, 7] -> ["k"] at [1]>, <Unmerge{384, 32, 1} ["n_block", "n_thread", "n_iter"] at [3, 4, 6] -> ["n"] at [2]>, <AddDim{1} ["m_block"] at [2] -> [] at []>] bounds = [32, 1, 1, 384, 32, 4, 1, 32] -> [1, 4096, 12288]> : memref<1x4096x12288xf16> to memref<32x1x1x384x32x4x1x32xf16>
  // CHECK: %[[TRANS1:.*]] = rock.transform %[[TRANS0]] by <affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4 floordiv 4, d4 mod 4, 0, d5)> by [<PassThrough ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3] -> ["k_loop", "g_block", "m_block", "n_block"] at [0, 1, 2, 3]>, <Merge{32, 4} ["tid"] at [4] -> ["n_thread", "k_thread"] at [4, 5]>, <Merge{1, 32} ["iter"] at [5] -> ["n_iter", "k_iter"] at [6, 7]>] bounds = [32, 1, 1, 384, 128, 32] -> [32, 1, 1, 384, 32, 4, 1, 32]> : memref<32x1x1x384x32x4x1x32xf16> to memref<32x1x1x384x128x32xf16>
  // CHECK: rock.threadwise_read_into {forceUnroll, useIndexDiffs} [](%[[TRANS1]]) [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] -> %{{.*}} : memref<32x1x1x384x128x32xf16> -> memref<32xf16, #gpu.address_space<private>>, vector<32xi1>
  func.func @mlir_transpose_reshape_unpack_int4_unsqueeze_reshape_slice_slice_squeeze_squeeze_dequantizelinear_unsqueeze_transpose_dot(%arg0: !migraphx.shaped<1x1x4096xf16, 4096x4096x1>, %arg1: !migraphx.shaped<12288x2048xui8, 2048x1>, %arg2: !migraphx.shaped<384x32x32x2xf16, 2048x64x2x1>) -> !migraphx.shaped<1x1x12288xf16, 12288x12288x1> attributes {arch = "##TOKEN_ARCH##", kernel = "mixr"} {
    %0 = migraphx.transpose %arg2 {permutation = [0, 2, 1, 3]} : <384x32x32x2xf16, 2048x64x2x1> -> <384x32x32x2xf16, 2048x2x64x1>
    %1 = migraphx.reshape %0 {dims = [12288, 32, 2]} : <384x32x32x2xf16, 2048x2x64x1> -> <12288x32x2xf16, 64x2x1>
    %2 = migraphx.unpack %arg1 {axis = 1 : i64} : <12288x2048xui8, 2048x1> -> <12288x4096xui8, 4096x1>
    %3 = migraphx.reshape %1 {dims = [12288, 32, 1, 2]} : <12288x32x2xf16, 64x2x1> -> <12288x32x1x2xf16, 64x2x2x1>
    %4 = migraphx.multibroadcast %3 {out_dyn_dims = [], out_lens = [12288, 32, 128, 2]} : <12288x32x1x2xf16, 64x2x2x1> -> <12288x32x128x2xf16, 64x2x0x1>
    %5 = migraphx.reshape %4 {dims = [12288, 4096, 2]} : <12288x32x128x2xf16, 64x2x0x1> -> <12288x4096x2xf16, 8192x2x1>
    %6 = migraphx.slice %5 {axes = [2], ends = [1], starts = [0]} : <12288x4096x2xf16, 8192x2x1> -> <12288x4096x1xf16, 8192x2x1>
    %7 = migraphx.slice %5 {axes = [2], ends = [2], starts = [1]} : <12288x4096x2xf16, 8192x2x1> -> <12288x4096x1xf16, 8192x2x1>
    %8 = migraphx.reshape %6 {dims = [12288, 4096]} : <12288x4096x1xf16, 8192x2x1> -> <12288x4096xf16, 8192x2>
    %9 = migraphx.reshape %7 {dims = [12288, 4096]} : <12288x4096x1xf16, 8192x2x1> -> <12288x4096xf16, 8192x2>
    %10 = migraphx.dequantizelinear %2, %8, %9 : <12288x4096xui8, 4096x1>, <12288x4096xf16, 8192x2>, !migraphx.shaped<12288x4096xf16, 8192x2> -> <12288x4096xf16, 4096x1>
    %11 = migraphx.reshape %10 {dims = [1, 12288, 4096]} : <12288x4096xf16, 4096x1> -> <1x12288x4096xf16, 50331648x4096x1>
    %12 = migraphx.transpose %11 {permutation = [0, 2, 1]} : <1x12288x4096xf16, 50331648x4096x1> -> <1x4096x12288xf16, 50331648x1x4096>
    %13 = migraphx.dot %arg0, %12 {perf_config="v2:16,32,8,16,16,16,1,1,1"} : <1x1x4096xf16, 4096x4096x1>, <1x4096x12288xf16, 50331648x1x4096> -> <1x1x12288xf16, 12288x12288x1>
    return %13 : !migraphx.shaped<1x1x12288xf16, 12288x12288x1>
  }
}
