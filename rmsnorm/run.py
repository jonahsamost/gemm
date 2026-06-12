import argparse
import math
import torch

import cutlass
import cutlass.cute as cute

from rmsnorm.rmsnorm import RMSNorm
from utils.correctness import check_correctness

_dtype_dict = {
    torch.bfloat16: cutlass.BFloat16,
}
_compile_cache = {}

def get_compile_fn(
    x: torch.Tensor,
    use_tvm_ffi: bool
):
    key = (x.dtype, x.size(-1), use_tvm_ffi)
    if key not in _compile_cache:
        M, N = x.shape
        dtype = _dtype_dict[x.dtype]
        div = math.gcd(N, 128 // dtype.width)
        stride = (cute.sym_int64(divisibility=div), 1)
        align = div * dtype.width // 8
        x_fake = cute.runtime.make_fake_tensor(
            dtype, (cute.sym_int(), N), stride=stride, assumed_align=align
        )
        weight_fake = cute.runtime.make_fake_tensor(
            dtype, (N,), stride=(1,), assumed_align=align
        )
        out_fake = cute.runtime.make_fake_tensor(
            dtype, (cute.sym_int(), N), stride=stride, assumed_align=align
        )
        kernel = RMSNorm(N, dtype)
        fake_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=use_tvm_ffi)
        compile_kwargs = {}
        if use_tvm_ffi:
            compile_kwargs['options'] = "--enable-tvm-ffi"
        fn = cute.compile(
            kernel,
            x_fake, weight_fake, out_fake, cutlass.Float32(0),
            fake_stream,
            **compile_kwargs,
        )
        _compile_cache[key] = fn
    return _compile_cache[key]


def run(
    M, N,
    dtype=torch.bfloat16,
    eps=1.192e-7,
    use_tvm_ffi: bool = False
):
    X = torch.randn((M, N), device='cuda', dtype=dtype)
    weight = torch.randn((N,), device='cuda', dtype=dtype)
    out = torch.empty((M, N), device='cuda', dtype=dtype)

    fn = get_compile_fn(X, use_tvm_ffi=use_tvm_ffi)
    if use_tvm_ffi:
        fn(X, weight, out, eps)
    else:
        fn(X, weight, out, eps, torch.cuda.current_stream().cuda_stream)


    ref = torch.nn.functional.rms_norm(X, weight.shape, weight=weight, eps=eps)
    check_correctness(out, ref)


M = 512
N = 1024

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tvm-ffi", action="store_true")
    args = parser.parse_args()
    run(M, N, use_tvm_ffi=args.tvm_ffi)
