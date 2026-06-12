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

'''
Profile with:
ncu --set full python -m gemm.run --ordering hilbert --no-bench &> /tmp/output_ncu.log 2>&1

ncu --metrics regex:lts__,regex:dram__,regex:smsp__ --kernel-name regex:kernel_cutlass \
    python -m gemm.run --ordering hilbert --no-bench &> /tmp/output_ncu.log 2>&1
'''

_compile_cache = {}
_lut_cache = {}


def get_compiled_fn(tile_ordering, persistence_mode, tile_shape, cluster_shape, group_size):
    key = (tile_ordering, persistence_mode, tile_shape, cluster_shape, group_size)
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
        gemm = GemmSm90(
            persistence_mode=persistence_mode,
            tile_ordering=tile_ordering,
            tile_shape_mnk=tile_shape,
            cluster_shape_mnk=cluster_shape,
            cta_swizzle_width=group_size,
        )
        fn = cute.compile(
            gemm,
            a_fake, b_fake, out_fake,
            sem_fake, lut_fake,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _compile_cache[key] = fn
    return _compile_cache[key]


def get_lut_ptr(tile_ordering, m, n, tile_shape, group_size):
    if tile_ordering not in (CTATileOrdering.OFFLINE_SWIZZLE, CTATileOrdering.HILBERT):
        return 0
    ncluster_m = math.ceil(m / tile_shape[0])
    ncluster_n = math.ceil(n / tile_shape[1])
    cache_key = (tile_ordering, ncluster_m, ncluster_n, group_size)
    if cache_key not in _lut_cache:
        if tile_ordering == CTATileOrdering.OFFLINE_SWIZZLE:
            _lut_cache[cache_key] = create_swizzle_lut(ncluster_m, ncluster_n, group_size)
        else:
            _lut_cache[cache_key] = create_hilbert_lut(ncluster_m, ncluster_n)
    return _lut_cache[cache_key].data_ptr()


def run(
    M, N, K,
    tile_ordering, persistence_mode, tile_shape, cluster_shape, group_size, bench=True
):
    A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((M, N), device=A.device, dtype=A.dtype)
    sem = torch.zeros(1, device="cuda", dtype=torch.int32)

    fn = get_compiled_fn(tile_ordering, persistence_mode, tile_shape, cluster_shape, group_size)
    lut_ptr = get_lut_ptr(tile_ordering, M, N, tile_shape, group_size)

    fn(A, B, out, sem.data_ptr(), lut_ptr)

    if bench:
        ref = A @ B.T
        check_correctness(out, ref)

        flops = 2 * M * N * K
        bytes_total = (M * K + N * K + M * N) * 2
        t_custom = bench_and_report("custom", lambda: fn(A, B, out, sem.data_ptr(), lut_ptr),
                                    flops, gbps_bytes=bytes_total)
        t_cublas = bench_and_report("cuBLAS", lambda: torch.mm(A, B.T), flops, gbps_bytes=bytes_total)
        print(f"\ncuBLAS speedup over custom: {t_cublas / t_custom:.2f}x")


ORDERING_NAMES = {
    "naive": CTATileOrdering.NONE,
    "none": CTATileOrdering.NONE,
    "online_swizzle": CTATileOrdering.ONLINE_SWIZZLE,
    "offline_swizzle": CTATileOrdering.OFFLINE_SWIZZLE,
    "hilbert": CTATileOrdering.HILBERT,
}

PERSISTENCE_NAMES = {
    "none": PersistenceMode.NONE,
    "static": PersistenceMode.STATIC,
    "dynamic": PersistenceMode.DYNAMIC,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordering", choices=list(ORDERING_NAMES), default="online_swizzle")
    parser.add_argument("--persistence", choices=list(PERSISTENCE_NAMES), default="dynamic")
    parser.add_argument("--group-size", type=int, default=12)
    parser.add_argument("-M", type=int, default=8192)
    parser.add_argument("-N", type=int, default=8192)
    parser.add_argument("-K", type=int, default=8192)
    parser.add_argument("--no-bench", action="store_true")
    args = parser.parse_args()

    tile_ordering = ORDERING_NAMES[args.ordering]
    persistence_mode = PERSISTENCE_NAMES[args.persistence]

    print(f"ordering={args.ordering}  persistence={args.persistence}  "
          f"group_size={args.group_size}  M={args.M} N={args.N} K={args.K}")

    run(
        args.M, args.N, args.K,
        tile_ordering, persistence_mode,
        tile_shape=(128, 256), cluster_shape=(2, 1, 1),
        group_size=args.group_size,
        bench=not args.no_bench
    )
