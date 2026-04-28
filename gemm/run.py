import torch
import cutlass
import cutlass.cute as cute

from gemm import GemmSm90

@torch.library.custom_op("jonah::gemm_fn", mutates_args={"out"})
def _gemm_fn(
    A: torch.Tensor, B: torch.Tensor, out: torch.Tensor
) -> None:
    compile_key = ()
    if compile_key not in _gemm_fn.compile_cache:
        m = cute.sym_int()        
        n = cute.sym_int()
        k = cute.sym_int()
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
    n = B.size(1)
    out = torch.empty((m, n), device=A.device, dtype=A.dtype)
    _gemm_fn(A, B, out)
    return out


M = 128
N = 1024
K = 1024
A = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
B = torch.randn((N, K), device='cuda', dtype=torch.bfloat16)
out = gemm_fn(A, B)

