import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda


class RMSNorm:
    def __init__(self, N, dtype):
        self.N = N
        self.dtype = dtype
    
    def __call__(
        self,
        X: cute.Tensor,
        weight: cute.Tensor,
        out: cute.Tensor,
        eps: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        pass