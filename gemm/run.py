import torch
import cutlass
import cutlass.cute as cute
from correctness import check_correctness

from benchmark import bench_and_report
# from gemm_v1 import GemmSm90_v1
# from gemm_v2 import GemmSm90_v2
# from gemm_v3 import GemmSm90_v3
# from gemm_v4 import GemmSm90_v4 as GemmSm90
# from gemm_v5 import GemmSm90_v5 as GemmSm90
from gemm_v6 import GemmSm90_v6 as GemmSm90


@torch.library.custom_op("jonah::gemm_fn", mutates_args={"out"})
def _gemm_fn(
    A: torch.Tensor, B: torch.Tensor, out: torch.Tensor
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
        out_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.BFloat16, (m, n), stride_order=(1, 0), assumed_align=128
        )
        fn = cute.compile(
            GemmSm90(tile_shape_mnk=(128, 256)),
            a_fake, b_fake, out_fake,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _gemm_fn.compile_cache[compile_key] = fn
    
    _gemm_fn.compile_cache[compile_key](
        A, B, out
    )


_gemm_fn.compile_cache = {}


def gemm_fn(
    A: torch.Tensor, B: torch.Tensor
):
    tile_cnt_semaphore = torch.zeros(1, device=A.device, dtype=torch.int32)
    m = A.size(0)
    n = B.size(0)
    out = torch.empty((m, n), device=A.device, dtype=A.dtype)
    _gemm_fn(A, B, out)
    return out


M = 4096
N = 4096
K = 8192
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