import math
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

from tile_scheduler import PersistenceMode, CTATileOrdering
from cta_swizzle import create_swizzle_lut, create_hilbert_lut
from correctness import check_correctness
from benchmark import bench_and_report
# from gemm_v1 import GemmSm90_v1
# from gemm_v2 import GemmSm90_v2
# from gemm_v3 import GemmSm90_v3
# from gemm_v4 import GemmSm90_v4 as GemmSm90
# from gemm_v5 import GemmSm90_v5 as GemmSm90
# from gemm_v6 import GemmSm90_v6 as GemmSm90
# from gemm_v7 import GemmSm90_v7 as GemmSm90
from gemm_v8 import GemmSm90_v8 as GemmSm90

TILE_ORDERING = CTATileOrdering.HILBERT
PERSISTENCE_MODE = PersistenceMode.NONE
TILE_SHAPE_MNK = (128, 128)
CLUSTER_SHAPE_MNK = (1, 1, 1)
CTA_SWIZZLE_WIDTH = 8


@torch.library.custom_op("jonah::gemm_fn", mutates_args={"out", "tile_count_semaphore"})
def _gemm_fn(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    tile_count_semaphore: torch.Tensor,
) -> None:
    compile_key = ()
    if compile_key not in _gemm_fn.compile_cache:
        m = cute.sym_int(divisibility=8)
        n = cute.sym_int(divisibility=8)
        k = cute.sym_int(divisibility=8)
        a_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16, (m, k), stride_order=(1, 0), assumed_align=128
        )
        b_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16, (n, k), stride_order=(1, 0), assumed_align=128
        )
        tile_count_sem_fake = make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=4)
        tile_order_lut_fake = make_ptr(cutlass.Int32, 0, cute.AddressSpace.gmem, assumed_align=4)
        out_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16, (m, n), stride_order=(1, 0), assumed_align=128
        )
        fn = cute.compile(
            GemmSm90(
                persistence_mode=PERSISTENCE_MODE,
                tile_ordering=TILE_ORDERING,
                tile_shape_mnk=TILE_SHAPE_MNK,
                cluster_shape_mnk=CLUSTER_SHAPE_MNK,
            ),
            a_fake, b_fake, out_fake,
            tile_count_sem_fake,
            tile_order_lut_fake,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _gemm_fn.compile_cache[compile_key] = fn

    m, n = A.size(0), B.size(0)
    lut_ptr = _get_tile_order_lut(m, n)
    _gemm_fn.compile_cache[compile_key](
        A, B, out, tile_count_semaphore.data_ptr(), lut_ptr
    )


_gemm_fn.compile_cache = {}
_gemm_fn.lut_cache = {}


def _get_tile_order_lut(m, n):
    """Create or retrieve cached tile-order LUT; returns data_ptr (int)."""
    if TILE_ORDERING not in (CTATileOrdering.OFFLINE_SWIZZLE, CTATileOrdering.HILBERT):
        return 0
    ncluster_m = math.ceil(m / TILE_SHAPE_MNK[0])
    ncluster_n = math.ceil(n / TILE_SHAPE_MNK[1])
    cache_key = (TILE_ORDERING, ncluster_m, ncluster_n)
    if cache_key not in _gemm_fn.lut_cache:
        if TILE_ORDERING == CTATileOrdering.OFFLINE_SWIZZLE:
            lut = create_swizzle_lut(ncluster_m, ncluster_n, CTA_SWIZZLE_WIDTH)
        else:
            lut = create_hilbert_lut(ncluster_m, ncluster_n)
        _gemm_fn.lut_cache[cache_key] = lut
    return _gemm_fn.lut_cache[cache_key].data_ptr()


def gemm_fn(
    A: torch.Tensor, B: torch.Tensor
):
    tile_cnt_semaphore = torch.zeros(1, device=A.device, dtype=torch.int32)
    m = A.size(0)
    n = B.size(0)
    out = torch.empty((m, n), device=A.device, dtype=A.dtype)
    _gemm_fn(A, B, out, tile_cnt_semaphore)
    return out


M = N = K = 8192
A = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
B = torch.randn((N, K), device='cuda', dtype=torch.bfloat16)
out = gemm_fn(A, B)

ref = A @ B.T
check_correctness(out, ref)

flops = 2 * M * N * K
bytes_total = (M * K + N * K + M * N) * 2  # bf16 = 2 bytes
def fn_custom():
    gemm_fn(A, B)
t_custom = bench_and_report("custom", fn_custom, flops, gbps_bytes=bytes_total)
# Benchmark cuBLAS
def fn_cublas():
    torch.mm(A, B.T)
t_cublas = bench_and_report("cuBLAS", fn_cublas, flops, gbps_bytes=bytes_total)
print(f"\ncuBLAS speedup over custom: {t_cublas / t_custom:.2f}x")
