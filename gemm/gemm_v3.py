from typing import Type, Tuple
from gemm.smem_utils import make_smem_layout
import torch
import cutlass
from cutlass import const_expr
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass._mlir.dialects.cute import ReductionOp
import cutlass.utils.hopper_helpers as sm90_helpers
from cutlass.utils import LayoutEnum


class GemmSm90_v1:
    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        tile_shape_mnk: Tuple[int, int] | Tuple[int, int, int],
    ):
        self.warpgroups = 1
        self.threads_per_cta = self.warpgroups * 128

        self.acc_dtype = acc_dtype
        self.cta_tile_shape_mnk = (
            tuple(tile_shape_mnk) if len(tile_shape_mnk) == 3 else (*tile_shape_mnk, 0)
        )
        tile_M, tile_N = self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]
        if tile_M not in [64, 128, 192, 256, 320]:
            raise ValueError("CTA tile shape M must be 64/128/192/256/320")
        if tile_M in [192, 320]:  # special case
            tile_N_max = 256 if tile_M == 192 else 160
            if not (tile_N % 32 == 0 and tile_N <= tile_N_max):
                raise ValueError(
                    f"If tile_m == {tile_M}, CTA tile shape N must be divisible by 32 and <= {tile_N_max}"
                )
        else:
            if not (
                (tile_N % 16 == 0 and tile_N <= 256) or (tile_N % 32 == 0 and tile_N <= 512)
            ):
                raise ValueError(
                    "CTA tile shape N must be divisible by 16 and <= 256, or divisible by 32 and <= 512"
                )

        if tile_M == 320:  # tile_M / 64 is not even so we have to split along N
            atom_layout_m, atom_layout_n = 1, 2
        elif tile_M == 192:
            if tile_N <= 128:
                atom_layout_m, atom_layout_n = 3, 1
            else:
                atom_layout_m, atom_layout_n = 1, 2
        else:
            atom_layout_m = (
                self.cta_tile_shape_mnk[0] // 64 if self.cta_tile_shape_mnk[0] < 256 else 2
            )
            atom_layout_n = 1
        assert atom_layout_m in [1, 2, 3] and atom_layout_n in [1, 2]
        self.atom_layout_mnk = (atom_layout_m, atom_layout_n, 1)
        print(f'atom layout: {self.atom_layout_mnk}')

    
    @cute.jit
    def __call__(
        self,
        A: cute.Tensor,
        B: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream
    ):
        self.a_dtype = A._dtype
        self.b_dtype = B._dtype
        self.a_layout = LayoutEnum.from_tensor(A)
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

        mrows = cute.ceil_div(A.shape[0], self.tile_size)
        mcols = cute.ceil_div(B.shape[0], self.tile_size)

        self._setup_shared_storage()
        self._setup_tiled_mma()
        self._setup_smem_layout()

        self.kernel(
            A, B, out,
            tiled_copy, s_tiled_copy, r_tiled_copy,
            tiler_mn,
            atom_async, atom_sync,
        ).launch(
            grid=(mrows, mcols, 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
        )
    
    def _setup_smem_layout(self):
        smem_capacity = cutlass.utils.get_smem_capacity_in_bytes(f"sm_0"),  # smem_capacity
        print(f'smem cap: {smem_capacity}')
        a_shape = cute.slice_(self.cta_tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(self.cta_tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * self.a_dtype.width // 8 + cute.size(b_shape) * self.b_dtype.width // 8
        )
        # TODO this seems wrong?
        ab_stage = smem_capacity // ab_bytes_per_stage
        print(f'ab stage: {ab_stage}')
        
        self.a_smem_layout = make_smem_layout(
            self.a_dtype, self.a_layout, self.cta_tile_shape_mnk, dim=0, ab_stage=ab_stage
        )
        self.b_smem_layout = make_smem_layout(
            self.a_dtype, self.a_layout, self.cta_tile_shape_mnk, dim=1, ab_stage=ab_stage
        )

    def _setup_shared_storage(self):
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
    
    def _setup_tiled_mma(self):
        tiler_mn= (64, self.cta_tile_shape_mnk[1] // self.atom_layout_mnk[1])
        print(f'tiler_mn: {tiler_mn}')
        self.tiled_mma = sm90_helpers.make_trivial_tiled_mma(
            self.a_dtype, self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=tiler_mn
        )
        if const_expr(self.atom_layout_mnk[1] > 1):
            atom_n = self.atom_layout_mnk[1]
            perm_n = cute.make_ordered_layout(
                (8, self.cta_tile_shape_mnk[1] // atom_n // 8, atom_n), order=(0, 2, 1)
            )
            self.tiled_mma = cute.make_tiled_mma(
                cute.make_mma_atom(self.tiled_mma.op),
                self.atom_layout_mnk,
                permutatin_mnk=(None, perm_n, None),
            )
        
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        tile_k = (
            self.cta_tile_shape_mnk[2] if self.cta_tile_shape_mnk[2] > 0 else mma_inst_shape_k * 4
        )

        assert tile_k > 0, "CTA tile K must be positive"
        assert tile_k % mma_inst_shape_k == 0, (
            f"CTA tile K ({tile_k}) must be divisible by MMA instruction K ({mma_inst_shape_k})"
        )
        self.cta_tile_shape_mnk = (
            self.cta_tile_shape_mnk[0],
            self.cta_tile_shape_mnk[1],
            tile_k,
        )

    
    @cute.jit
    def pred_inp(
        self,
        predA: cute.Tensor, predB: cute.Tensor,
        msize: cute.Int32, nsize: cute.Int32, ksize: cute.Int32,
    ):
        tidx, tidy, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        m0_range = cute.cosize(predA)
        col = (tidx % const_expr(self.threads_per_row)) * const_expr(self.vals_per_thread)
        k_in = col < ksize
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
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = storage.sA.get_tensor(self.a_smem_layout.outer, swizzle=self.a_smem_layout.inner)
        sB = storage.sB.get_tensor(self.b_smem_layout.outer, swizzle=self.b_smem_layout.inner)
        