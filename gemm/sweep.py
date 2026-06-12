import math
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr
from itertools import product

from gemm.tile_scheduler import PersistenceMode, CTATileOrdering
from gemm.cta_swizzle import create_swizzle_lut, create_hilbert_lut
from gemm.gemm_v8 import GemmSm90_v8 as GemmSm90
from utils.correctness import check_correctness
from utils.benchmark import bench_and_report

_compile_cache = {}
_lut_cache = {}

TILE_SHAPE = (128, 256)
CLUSTER_SHAPE = (2, 1, 1)

'''
python sweep.py &> /tmp/output.log 2>&1
'''


def get_compiled_fn(tile_ordering, persistence_mode, group_size=8):
    key = (tile_ordering, persistence_mode, group_size)
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
        tile_count_sem_fake = make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=4)
        tile_order_lut_fake = make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=4)
        gemm = GemmSm90(
            persistence_mode=persistence_mode,
            tile_ordering=tile_ordering,
            tile_shape_mnk=TILE_SHAPE,
            cluster_shape_mnk=CLUSTER_SHAPE,
            cta_swizzle_width=group_size
        )
        fn = cute.compile(
            gemm,
            a_fake, b_fake, out_fake,
            tile_count_sem_fake,
            tile_order_lut_fake,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _compile_cache[key] = fn
    return _compile_cache[key]


def get_lut_ptr(tile_ordering, m, n, group_size=8):
    if tile_ordering not in (CTATileOrdering.OFFLINE_SWIZZLE, CTATileOrdering.HILBERT):
        return 0
    ncluster_m = math.ceil(m / TILE_SHAPE[0])
    ncluster_n = math.ceil(n / TILE_SHAPE[1])
    cache_key = (tile_ordering, ncluster_m, ncluster_n, group_size)
    if cache_key not in _lut_cache:
        if tile_ordering == CTATileOrdering.OFFLINE_SWIZZLE:
            _lut_cache[cache_key] = create_swizzle_lut(ncluster_m, ncluster_n, group_size)
        else:
            _lut_cache[cache_key] = create_hilbert_lut(ncluster_m, ncluster_n)
    return _lut_cache[cache_key].data_ptr()


def run_config(M, N, K, tile_ordering, persistence_mode, group_size=8):
    A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16)
    B = torch.randn((N, K), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((M, N), device=A.device, dtype=A.dtype)
    sem = torch.zeros(1, device="cuda", dtype=torch.int32)

    fn = get_compiled_fn(tile_ordering, persistence_mode, group_size)
    lut_ptr = get_lut_ptr(tile_ordering, M, N, group_size)

    fn(A, B, out, sem.data_ptr(), lut_ptr)

    ref = A @ B.T
    check_correctness(out, ref)

    flops = 2 * M * N * K
    gs_str = f" gs={group_size}" if tile_ordering in (
        CTATileOrdering.ONLINE_SWIZZLE, CTATileOrdering.OFFLINE_SWIZZLE
    ) else ""
    label = f"{tile_ordering.name:>16s}{gs_str} / {persistence_mode.name:<8s}"
    t_custom = bench_and_report(label, lambda: fn(A, B, out, sem.data_ptr(), lut_ptr), flops)
    t_cublas = bench_and_report("cuBLAS", lambda: torch.mm(A, B.T), flops)
    print(f"\ncuBLAS speedup over custom: {t_cublas / t_custom:.2f}x")


if __name__ == "__main__":
    M = N = K = 8192

    persistence_modes = [
        # PersistenceMode.NONE,
        # PersistenceMode.STATIC,
        PersistenceMode.DYNAMIC,
    ]
    orderings = [
        CTATileOrdering.ONLINE_SWIZZLE,
        # CTATileOrdering.OFFLINE_SWIZZLE,
        # CTATileOrdering.NONE,
        # CTATileOrdering.HILBERT,
    ]
    group_sizes = [4, 6, 8, 12, 16, 20, 24, 28, 32]

    for persistence in persistence_modes:
        # Swizzle orderings (sweep group_size)
        for ordering in [CTATileOrdering.ONLINE_SWIZZLE, CTATileOrdering.OFFLINE_SWIZZLE]:
            if ordering not in orderings:
                continue
            for gs in group_sizes:
                print(f"\n=== {ordering.name} gs={gs} / {persistence.name} ===")
                try:
                    run_config(M, N, K, ordering, persistence, group_size=gs)
                except Exception as e:
                    print(f"  FAILED: {e}")

        # Non-swizzle orderings (group_size irrelevant)
        for ordering in [CTATileOrdering.NONE, CTATileOrdering.HILBERT]:
            if ordering not in orderings:
                continue
            print(f"\n=== {ordering.name} / {persistence.name} ===")
            try:
                run_config(M, N, K, ordering, persistence)
            except Exception as e:
                print(f"  FAILED: {e}")
    