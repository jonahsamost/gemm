import time
from triton.testing import do_bench
import torch
import cutlass
import cutlass.cute as cute

from gemm import GemmSm90

def _bench_and_report(name, fn, flops, warmup=5, rep=30, gbps_bytes=0):
    time.sleep(0.5)
    t = do_bench(fn, warmup=warmup, rep=rep)
    tflops = flops / (t * 1e9)
    if gbps_bytes:
        gbps = gbps_bytes / (t * 1e6)
        print(f"{name}: {t:.3f} ms,  {tflops:7.1f} TFLOP/s,  {gbps:.0f} GB/s")
    else:
        print(f"{name}: {t:.3f} ms,  {tflops:7.1f} TFLOP/s")
    return t


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
            GemmSm90(),
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
assert torch.allclose(out, ref, rtol=5e-2, atol=4.0)


flops = 2 * M * N * K
bytes_total = (M * K + N * K + M * N) * 2  # bf16 = 2 bytes

def fn_custom():
    _gemm_fn(A, B, out)
t_custom = _bench_and_report("custom", fn_custom, flops, gbps_bytes=bytes_total)
# Benchmark cuBLAS
def fn_cublas():
    torch.mm(A, B.T)
t_cublas = _bench_and_report("cuBLAS", fn_cublas, flops, gbps_bytes=bytes_total)
print(f"\ncuBLAS speedup over custom: {t_custom / t_cublas:.2f}x")