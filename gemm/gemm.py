import torch
import cutlass
from cutlass import const_expr
import cutlass.cute as cute
import cuda.bindings.driver as cuda


class GemmSm90:
    def __init__(self):
        self.WARPS = 8
        self.THREADS_PER_WARP = 32
        self.threads_per_cta = self.WARPS * self.THREADS_PER_WARP
        self.tile_size = 64
        self.bits_per_copy = 128
        self.vals_per_thread = self.bits_per_copy // 16
        self.threads_per_row = self.tile_size // self.vals_per_thread
        self.cta_rows = self.threads_per_cta // self.threads_per_row
        self.ping_pong = True
    
    @cute.jit
    def __call__(
        self,
        A: cute.Tensor,
        B: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream
    ):
        thr_layout = cute.make_layout(
            (self.cta_rows, self.threads_per_row), stride=(self.threads_per_row, 1)
        )
        val_layout = cute.make_layout(
            (1, self.vals_per_thread), stride=(1, 1)
        )
        print(f'thread layout: {thr_layout}')
        atom_async = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            cutlass.BFloat16,
            num_bits_per_copy=const_expr(self.bits_per_copy)
        )
        tiled_copy = cute.make_tiled_copy_tv(atom_async, thr_layout, val_layout)
        tiler_mn = (self.tile_size, self.tile_size)
        mrows = cute.ceil_div(A.shape[0], self.tile_size)
        mcols = cute.ceil_div(B.shape[0], self.tile_size)
        self.kernel(
            A, B, out,
            tiled_copy, tiler_mn, atom_async
        ).launch(
            grid=(mrows, mcols, 1),
            block=(self.cta_rows, self.threads_per_row, 1)
        )
    
    @cute.jit
    def pred_inp(
        self,
        predA: cute.Tensor, predB: cute.Tensor,
        msize: cute.Int32, nsize: cute.Int32,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        m0_range = cute.cosize(predA)
        for r in cutlass.range_constexpr(m0_range):
            cur_row = tidy + r * self.cta_rows
            a_row_in = cute.elem_less(bidx * self.tile_size + cur_row, msize)
            b_row_in = cute.elem_less(bidy * self.tile_size + cur_row, nsize)
            predA[r] = a_row_in
            predB[r] = b_row_in

    @cute.kernel
    def kernel(
        self,
        A: cute.Tensor,
        B: cute.Tensor,
        out: cute.Tensor,
        tiled_copy: cute.TiledCopy,
        tiler_mn: cute.Shape,
        atom_async: cute.CopyAtom,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        thr_idx = tidy * const_expr(self.threads_per_cta) + tidx

        smem_alloc = cutlass.utils.SmemAllocator()
        smem_a = smem_alloc.allocate_tensor(
            cutlass.BFloat16,
            cute.make_layout((2, self.tile_size, self.tile_size), stride=(self.tile_size * self.tile_size, self.tile_size, 1)),
            byte_alignment=128
        )
        smem_b = smem_alloc.allocate_tensor(
            cutlass.BFloat16,
            cute.make_layout((2, self.tile_size, self.tile_size), stride=(self.tile_size * self.tile_size, self.tile_size, 1)),
            byte_alignment=128
        )
        msize = A.shape[0]
        nsize = B.shape[0]

        gA = cute.local_tile(A, tiler_mn, (bidx, 0))
        gB = cute.local_tile(B, tiler_mn, (bidy, 0))

        thr_slice = tiled_copy.get_slice(thr_idx)
        gA_thr_vals = thr_slice.partition_S(gA)
        gB_thr_vals = thr_slice.partition_S(gB)
        gA_smem = thr_slice.partition_D(smem_a[(0, None, None)])
        gB_smem = thr_slice.partition_D(smem_b[(0, None, None)])

        ### ((8, 1), 2, 1) -- 8 values per thread, 2 row iters, 1 col iters
        predA = cute.make_rmem_tensor_like(
            cute.make_layout((cute.size(gA_thr_vals, mode=[1]), 1), stride=(1, 1)),
            cutlass.Boolean
        )
        predB = cute.make_rmem_tensor_like(
            cute.make_layout((cute.size(gB_thr_vals, mode=[1]), 1), stride=(1, 1)),
            cutlass.Boolean
        )
        self.pred_inp(predA, predB, msize, nsize)

        print(f"ga smem: {gA_smem}")
        print(f"ga thr values: {gA_thr_vals}")
        print(f'pred: {predA}')
        cute.copy(atom_async, gA_thr_vals, gA_smem, pred=predA)

        gA_regs = cute.make_rmem_tensor_like(gA_thr_vals, cutlass.BFloat16)
        gB_regs = cute.make_rmem_tensor_like(gA_thr_vals, cutlass.BFloat16)

        # kiters = cute.ceil_div(A.shape[1], self.tile_size)
        # for i in range(kiters):

