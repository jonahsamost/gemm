from typing import Union
import torch
import cutlass
from cutlass import const_expr
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass._mlir.dialects.cute import ReductionOp
from cutlass.utils import LayoutEnum

from cta_swizzle import get_swizzle_block
from smem_utils import make_smem_layout, make_smem_layout_simple, partition_D_pos_ind, partition_S_pos_ind

'''
Implement CTA swizzling and smem swizzling
'''

class GemmSm90_v2:
    def __init__(self):
        self.buffer_align_bytes = 1024
        self.WARPS = 8
        self.THREADS_PER_WARP = 32
        self.threads_per_cta = self.WARPS * self.THREADS_PER_WARP
        self.tile_size = 64
        self.bits_per_copy = 128
        self.vals_per_thread = self.bits_per_copy // 16
        self.threads_per_row = self.tile_size // self.vals_per_thread
        self.cta_rows = self.threads_per_cta // self.threads_per_row
        self.ping_pong = True
        self.out_vals_per_thread = self.tile_size * self.tile_size // self.threads_per_cta
        self.out_threads_per_row = self.tile_size // self.out_vals_per_thread

        # swizzling
        self.cta_swizzle_width = 4
    
    @cute.jit
    def __call__(
        self,
        A: cute.Tensor,
        B: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream
    ):
        self.a_dtype = A._dtype
        self.a_layout = LayoutEnum.from_tensor(A)
        self.b_dtype = B._dtype
        self.b_layout = LayoutEnum.from_tensor(B)

        atom_async = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            cutlass.BFloat16,
            num_bits_per_copy=const_expr(self.bits_per_copy)
        )
        atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.BFloat16,
            num_bits_per_copy=const_expr(self.bits_per_copy)
        )
        # load gmem -> smem layout
        thr_layout = cute.make_layout((self.cta_rows, self.threads_per_row), stride=(self.threads_per_row, 1))
        val_layout = cute.make_layout((1, self.vals_per_thread), stride=(1, 1))
        tiled_copy = cute.make_tiled_copy_tv(atom_async, thr_layout, val_layout)
        tiler_mn = (self.tile_size, self.tile_size)

        # load smem -> reg layout
        s_thr_layout = cute.make_layout((1, 1), stride=(1, 1))
        s_val_layout = cute.make_layout((1, self.tile_size), stride=(self.tile_size, 1))
        s_tiled_copy = cute.make_tiled_copy_tv(atom_sync, s_thr_layout, s_val_layout)

        # reg -> gmem
        r_thr_layout = cute.make_layout((self.tile_size, self.out_threads_per_row), stride=(self.out_threads_per_row, 1))
        r_val_layout = cute.make_layout((1, self.out_vals_per_thread), stride=(self.out_vals_per_thread, 1))
        r_tiled_copy = cute.make_tiled_copy_tv(atom_sync, r_thr_layout, r_val_layout)

        self.setup_attributes()
        self.setup_shared_storage()

        mrows = cute.ceil_div(A.shape[0], self.tile_size)
        mcols = cute.ceil_div(B.shape[0], self.tile_size)
        self.kernel(
            A, B, out,
            tiled_copy, s_tiled_copy, r_tiled_copy,
            tiler_mn,
            atom_async, atom_sync,
        ).launch(
            grid=(mrows, mcols, 1),
            block=(self.cta_rows, self.threads_per_row, 1),
            stream=stream,
        )
    
    def make_smem_outer_layout(self, tile_size: int, stage: int | None = None) -> tuple:
        if stage is not None:
            return (stage, tile_size, tile_size), (tile_size * tile_size, tile_size, 1)
        else:
            return (tile_size, tile_size), (tile_size, 1)
    
    def setup_attributes(self):
        stage = 2 if self.ping_pong else None
        if stage is not None:
            self.smem_shape = (stage, self.tile_size, self.tile_size)
            self.smem_stride = (self.tile_size * self.tile_size, self.tile_size, 1)
        else:
            self.smem_shape = (self.tile_size, self.tile_size)
            self.smem_stride = (self.tile_size, 1)
        # cosize for compact layout = product of shape dims
        cosize = 1
        for s in self.smem_shape:
            cosize *= s
        self.smem_cosize = cosize

    def setup_shared_storage(self):
        @cute.struct
        class SharedStorage:
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, self.smem_cosize
                ],
                const_expr(self.buffer_align_bytes),
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, self.smem_cosize
                ],
                const_expr(self.buffer_align_bytes),
            ]
        self.shared_storage = SharedStorage
    
    @cute.jit
    def pred_inp(
        self,
        predA: cute.Tensor, predB: cute.Tensor,
        msize: cute.Int32, nsize: cute.Int32, ksize: cute.Int32,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        m0_range = cute.cosize(predA)
        # col = (tidx % const_expr(self.threads_per_row)) * const_expr(self.vals_per_thread)
        # k_in = col < ksize
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
        tiled_copy: cute.TiledCopy, s_tiled_copy: cute.TiledCopy, r_tiled_copy: cute.TiledCopy,
        tiler_mn: cute.Shape,
        atom_async: cute.CopyAtom,
        atom_sync: cute.CopyAtom,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, bidy = get_swizzle_block(self.cta_swizzle_width)
        thr_idx = tidy * const_expr(self.THREADS_PER_WARP) + tidx

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        smem_outer = cute.make_layout(self.smem_shape, stride=self.smem_stride)
        smem_swizzle = cute.make_swizzle(3, 4, 3)
        smem_a = storage.sA.get_tensor(smem_outer, smem_swizzle)
        smem_b = storage.sB.get_tensor(smem_outer, smem_swizzle)

        msize = A.shape[0]
        nsize = B.shape[0]
        ksize = A.shape[1]
        tidx_mrow = thr_idx // self.out_threads_per_row
        tidx_nrow = (thr_idx % self.out_threads_per_row) * const_expr(self.out_vals_per_thread)

        gA = cute.local_tile(A, tiler_mn, (bidx, 0))
        gB = cute.local_tile(B, tiler_mn, (bidy, 0))
        gOut = cute.local_tile(out, tiler_mn, (bidx, bidy))

        s_thr_slice = s_tiled_copy.get_slice(tidx_mrow)
        thr_slice = tiled_copy.get_slice(thr_idx)
        gA_thr_vals = thr_slice.partition_S(gA)
        gB_thr_vals = thr_slice.partition_S(gB)
        # gA_smem = thr_slice.partition_D(smem_a[(0, None, None)])
        # gB_smem = thr_slice.partition_D(smem_b[(0, None, None)])
        gA_smem = partition_D_pos_ind(thr_slice, smem_a[(0, None, None)], assumed_align=128)
        gB_smem = partition_D_pos_ind(thr_slice, smem_b[(0, None, None)], assumed_align=128)

        ### ((1, 1), 2, 1) -- 8 values per thread (but 128 bit), 2 row iters, 1 col iters
        pred_layout = cute.make_layout(
            (2, 1, cute.size(gA_thr_vals, mode=[1]), 1),
            stride=(1, 1, 1, 1)
        )
        predA = cute.make_rmem_tensor_like(pred_layout, cutlass.Boolean)
        predB = cute.make_rmem_tensor_like(pred_layout, cutlass.Boolean)
        self.pred_inp(predA[0, None, None, None], predB[0, None, None, None], msize, nsize, ksize)

        cute.copy(atom_async, gA_thr_vals, gA_smem, pred=predA[0, None, None, None])
        cute.copy(atom_async, gB_thr_vals, gB_smem, pred=predB[0, None, None, None])

        gA_regs = cute.make_rmem_tensor(
            cute.make_layout( (self.tile_size, 1, 1, 1), stride=(1, 1, 1, 1)),
            cutlass.BFloat16
        )
        gB_regs = cute.make_rmem_tensor_like(gA_regs, cutlass.BFloat16)
        acc_regs = cute.make_rmem_tensor_like(
            cute.make_layout( (self.out_vals_per_thread, 1, 1), stride=(1, 1, 1)), cutlass.Float32
        )
        acc_regs_bf16 = cute.make_rmem_tensor_like(acc_regs, cutlass.BFloat16)
        acc_regs.fill(0.0)

        kiters = cute.ceil_div(A.shape[1], self.tile_size)
        last_iter = kiters - 1
        ping = 0
        for k in range(kiters):
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            if k != last_iter:
                pong = ping ^ 1
                nk = k + 1
                gA_next = cute.local_tile(A, tiler_mn, (bidx, nk))
                gB_next = cute.local_tile(B, tiler_mn, (bidy, nk))

                gA_thr_vals_next = thr_slice.partition_S(gA_next)
                gB_thr_vals_next = thr_slice.partition_S(gB_next)
                # gA_smem_next = thr_slice.partition_D(smem_a[(pong, None, None)])
                # gB_smem_next = thr_slice.partition_D(smem_b[(pong, None, None)])
                gA_smem_next = partition_D_pos_ind(thr_slice, smem_a[(pong, None, None)], assumed_align=128)
                gB_smem_next = partition_D_pos_ind(thr_slice, smem_b[(pong, None, None)], assumed_align=128)
                # update next predicate
                self.pred_inp(
                    predA[pong, None, None, None], predB[pong, None, None, None],
                    msize, nsize, nk * const_expr(self.tile_size)
                )
                cute.copy(atom_async, gA_thr_vals_next, gA_smem_next, pred=predA[pong, None, None, None])
                cute.copy(atom_async, gB_thr_vals_next, gB_smem_next, pred=predB[pong, None, None, None])
            
            gA_regs.fill(0.0)
            gB_regs.fill(0.0)
            sA = cute.local_tile(smem_a[(ping, None, None)], (1, self.tile_size), (tidx_mrow, None))
            sA_vals = s_thr_slice.partition_S(sA)
            cute.copy(atom_sync, sA_vals, gA_regs)
            a_data = gA_regs.load()
            for i in cutlass.range_constexpr(self.out_vals_per_thread):
                sB = cute.local_tile(smem_b[(ping, None, None)], (1, self.tile_size), (tidx_nrow + i, None))
                sB_vals = s_thr_slice.partition_S(sB)
                cute.copy(atom_sync, sB_vals, gB_regs)
                b_data = gB_regs.load()
                dot = (a_data * b_data).to(cutlass.Float32)
                dot = dot.reduce(ReductionOp.ADD, init_val=0.0, reduction_profile=0)
                acc_regs[i] += dot

            ping ^= 1

        out_thr_slice = r_tiled_copy.get_slice(thr_idx) 
        out_pos = out_thr_slice.partition_D(gOut)
        acc_data = acc_regs.load().to(cutlass.BFloat16)
        acc_regs_bf16.store(acc_data)
        cute.copy(atom_sync, acc_regs_bf16, out_pos)


