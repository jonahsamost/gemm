import argparse
import math
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

from gemm.tile_scheduler import PersistenceMode, CTATileOrdering
from gemm.cta_swizzle import create_swizzle_lut, create_hilbert_lut
from utils.benchmark import bench_and_report
from gemm.gemm_v8 import GemmSm90_v8 as GemmSm90
from utils.correctness import check_correctness

_compile_cache = {}
_lut_cache = {}
_GROUP_SIZE = 12


def get_compiled_fn(
    tile_shape, cluster_shape,
    use_tvm_ffi: bool = False, use_pdl: bool = False
):
    key = (tile_shape, cluster_shape, use_tvm_ffi, use_pdl)
    if key not in _compile_cache:
        m = cute.sym_int(divisibility=8)
        n = cute.sym_int(divisibility=8)
        k = cute.sym_int(divisibility=8)
        a_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16, (m, k), stride_order=(1, 0), assumed_align=128
        )
        b_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16, (n, k), stride_order=(1, 0), assumed_align=128
        )
        out_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16, (m, n), stride_order=(1, 0), assumed_align=128
        )
        sem_fake = make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=4)
        lut_fake = make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=4)

        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=use_tvm_ffi)
        compile_kwargs = {}
        if use_tvm_ffi:
            compile_kwargs['options'] = '--enable-tvm-ffi'
        gemm = GemmSm90(
            persistence_mode=PersistenceMode.DYNAMIC,
            tile_ordering=CTATileOrdering.ONLINE_SWIZZLE,
            tile_shape_mnk=tile_shape,
            cluster_shape_mnk=cluster_shape,
            cta_swizzle_width=_GROUP_SIZE,
            use_pdl=use_pdl,
        )
        fn = cute.compile(
            gemm,
            a_fake, b_fake, out_fake,
            sem_fake, lut_fake,
            stream,
            **compile_kwargs
        )
        _compile_cache[key] = fn
    return _compile_cache[key]


def get_compiled_fn_non_tvm_ffi(A, B, out, tile_shape, cluster_shape, use_pdl=False):
    key = (
        A.dtype,
        tuple(A.shape),
        tuple(A.stride()),
        tuple(B.shape),
        tuple(B.stride()),
        tuple(out.shape),
        tuple(out.stride()),
        tile_shape,
        cluster_shape,
        use_pdl,
    )

    if key not in _compile_cache:
        A_cute = cute.runtime.from_dlpack(A)
        B_cute = cute.runtime.from_dlpack(B)
        out_cute = cute.runtime.from_dlpack(out)

        sem_fake = make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=4)
        lut_fake = make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=4)

        stream = cute.runtime.make_fake_stream()

        gemm = GemmSm90(
            persistence_mode=PersistenceMode.DYNAMIC,
            tile_ordering=CTATileOrdering.ONLINE_SWIZZLE,
            tile_shape_mnk=tile_shape,
            cluster_shape_mnk=cluster_shape,
            cta_swizzle_width=_GROUP_SIZE,
            use_pdl=use_pdl,
        )

        fn = cute.compile(
            gemm,
            A_cute, B_cute, out_cute,
            sem_fake, lut_fake,
            stream,
        )
        _compile_cache[key] = fn

    return _compile_cache[key]


def call_gemm(fn, A, B, out, sem, lut_ptr):
    fn(A, B, out, sem.data_ptr(), lut_ptr)


def call_gemm_non_tvm_ffi(fn, A, B, out, sem, lut_ptr):
    A_cute = cute.runtime.from_dlpack(A)
    B_cute = cute.runtime.from_dlpack(B)
    out_cute = cute.runtime.from_dlpack(out)
    sem_ptr = make_ptr(cutlass.Int32, sem.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
    lut_ptr = make_ptr(cutlass.Int32, lut_ptr, cute.AddressSpace.gmem, assumed_align=4)

    fn(
        A_cute, B_cute, out_cute,
        sem_ptr, lut_ptr,
        torch.cuda.current_stream().cuda_stream,
    )


def run(
    M, N, K, P,
    tile_shape, cluster_shape,
    use_pdl: bool = False,
    use_tvm_ffi: bool = False,
    bench=True
):
    A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    C = torch.randn((P, N), device="cuda", dtype=torch.bfloat16)
    out1 = torch.empty((M, N), device=A.device, dtype=A.dtype)
    out2 = torch.empty((M, P), device=A.device, dtype=A.dtype)
    sem1 = torch.zeros(1, device="cuda", dtype=torch.int32)
    sem2 = torch.zeros(1, device="cuda", dtype=torch.int32)

    if use_tvm_ffi:
        fn = get_compiled_fn(
            tile_shape, cluster_shape, use_tvm_ffi=use_tvm_ffi, use_pdl=use_pdl
        )
        call_gemm(fn, A, B, out1, sem1, 0)
        call_gemm(fn, out1, C, out2, sem2, 0)
    else:
        fn1 = get_compiled_fn_non_tvm_ffi(A, B, out1, tile_shape, cluster_shape, use_pdl=use_pdl)
        fn2 = get_compiled_fn_non_tvm_ffi(out1, C, out2, tile_shape, cluster_shape, use_pdl=use_pdl)
        call_gemm_non_tvm_ffi(fn1, A, B, out1, sem1, 0)
        call_gemm_non_tvm_ffi(fn2, out1, C, out2, sem2, 0)        

    def custom_chain_tvm_ffi():
        sem1.zero_()
        sem2.zero_()
        call_gemm(fn, A, B, out1, sem1, 0)
        call_gemm(fn, out1, C, out2, sem2, 0)
    
    def custom_chain_no_tvm_ffi():
        sem1.zero_()
        sem2.zero_()
        call_gemm_non_tvm_ffi(fn1, A, B, out1, sem1, 0)
        call_gemm_non_tvm_ffi(fn2, out1, C, out2, sem2, 0)
    
    custom_chain = custom_chain_tvm_ffi if use_tvm_ffi else custom_chain_no_tvm_ffi

    def torch_chain():
        return (A @ B.T) @ C.T

    if not bench:
        return

    ref = torch_chain()
    check_correctness(out2, ref)

    flops = 2 * M * N * K + 2 * M * P * N
    bytes_total = (M * K + N * K + M * N + P * N + M * P) * 2
    bench_and_report("custom chain", custom_chain, flops, gbps_bytes=bytes_total)
    bench_and_report("torch chain", torch_chain, flops, gbps_bytes=bytes_total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=8192)
    parser.add_argument("-N", type=int, default=8192)
    parser.add_argument("-K", type=int, default=4096)
    parser.add_argument("-P", type=int, default=4096)
    parser.add_argument("--tvm-ffi", action="store_true")
    parser.add_argument("--use-pdl", action="store_true")
    parser.add_argument("--no-bench", action="store_true")
    args = parser.parse_args()

    run(
        args.M, args.N, args.K, args.P,
        tile_shape=(128, 256), cluster_shape=(2, 1, 1),
        use_pdl=args.use_pdl,
        use_tvm_ffi=args.tvm_ffi,
        bench=not args.no_bench
    )
