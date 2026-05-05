import math
from typing import Type, Tuple, Optional
from smem_utils import make_smem_layout, make_epi_smem_layout
import torch
import cutlass
from cutlass.cute.nvgpu import warpgroup
from cutlass import const_expr
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass._mlir.dialects.cute import ReductionOp
import cutlass.utils.hopper_helpers as sm90_helpers
from cutlass.utils import LayoutEnum
from cutlass.cute.nvgpu.warp import StMatrix8x8x16bOp


class GemmSm90_v3:
    def __init__(
        self,
        tile_shape_mnk: Tuple[int, int] | Tuple[int, int, int] = (64, 128),
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
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

        self.threads_per_row = 8
        self.rows = self.threads_per_cta // 8

    
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
        self.d_dtype = out._dtype
        self.d_layout = LayoutEnum.from_tensor(out)

        atom_async = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            cutlass.BFloat16,
            num_bits_per_copy=128
        )
        atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.BFloat16,
            num_bits_per_copy=128
        )
        tiled_mma = self._setup_tiled_mma()
        self._setup_attributes()
        a_smem_layout, b_smem_layout, d_smem_layout = self._setup_smem_layout()
        self._setup_shared_storage(a_smem_layout, b_smem_layout, d_smem_layout)
        # load gmem -> smem layout
        gs_thr_layout = cute.make_layout((self.rows, self.threads_per_row), stride=(self.threads_per_row, 1))
        gs_val_layout = cute.make_layout((1, 8), stride=(1, 1))
        tiled_copy = cute.make_tiled_copy_tv(atom_async, gs_thr_layout, gs_val_layout)

        mrows = cute.ceil_div(A.shape[0], self.cta_tile_shape_mnk[0])
        mcols = cute.ceil_div(B.shape[0], self.cta_tile_shape_mnk[1])

        self.kernel(
            A, B, out,
            tiled_copy,
            atom_async, atom_sync,
            a_smem_layout, b_smem_layout, d_smem_layout,
            tiled_mma,
        ).launch(
            grid=(mrows, mcols, 1),
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        A: cute.Tensor,
        B: cute.Tensor,
        out: cute.Tensor,
        tiled_copy: cute.TiledCopy,
        atom_async: cute.CopyAtom,
        atom_sync: cute.CopyAtom,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        d_smem_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = storage.sA.get_tensor(a_smem_layout.outer, swizzle=a_smem_layout.inner)
        sB = storage.sB.get_tensor(b_smem_layout.outer, swizzle=b_smem_layout.inner)
        sD = storage.sD.get_tensor(d_smem_layout.outer, swizzle=d_smem_layout.inner)

        tile_mn = cute.select(self.cta_tile_shape_mnk, [0, 1])
        tile_mk = cute.select(self.cta_tile_shape_mnk, [0, 2])
        tile_nk = cute.select(self.cta_tile_shape_mnk, [1, 2])

        gA = cute.local_tile(A, tile_mk, (bidx, None))  # grab full stripe
        gB = cute.local_tile(B, tile_nk, (bidy, None))
        thr_copy = tiled_copy.get_slice(tidx)
        # partition gmem to smem
        gA_src = thr_copy.partition_S(gA)
        gB_src = thr_copy.partition_S(gB)
        gA_dst = thr_copy.partition_D(sA)
        gB_dst = thr_copy.partition_D(sB)

        # partition smem for mma
        thr_mma = tiled_mma.get_slice(tidx)
        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C(tile_mn), cutlass.Float32)
        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sA))
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sB))
        acc.fill(0.0)
        tiled_copy_r2s, tRS_rD, tRS_sD = self.epilog_smem_store_and_partition(
            tiled_mma, sD, self.epi_tile, tidx,
        )
        tRS_rAcc = self.epi_retile_acc(acc, tRS_rD, tiled_copy_r2s)

        # grab first k tile and place it into first smem stage buffer
        ping = 0
        cute.copy(atom_async, gA_src[None, None, None, 0], gA_dst[None, None, None, ping])
        cute.copy(atom_async, gB_src[None, None, None, 0], gB_dst[None, None, None, ping])

        kiters = cute.ceil_div(A.shape[1], self.cta_tile_shape_mnk[2])
        last_iter = kiters - 1
        for k in range(kiters):
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            if k != last_iter:
                pong = ping ^ 1
                nk = k + 1
                cute.copy(atom_async, gA_src[None, None, None, nk], gA_dst[None, None, None, pong])
                cute.copy(atom_async, gB_src[None, None, None, nk], gB_dst[None, None, None, pong])
            
            warpgroup.fence()            
            mma_atom = cute.make_mma_atom(tiled_mma.op)
            mma_atom.set(warpgroup.Field.ACCUMULATE, k != 0)
            for mma_k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
                cute.gemm(mma_atom, acc, tCrA[None, None, mma_k, ping], tCrB[None, None, mma_k, ping], acc)
                mma_atom.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()
            warpgroup.wait_group(0)
            ping ^= 1


        tcols = const_expr(self.epi_tile[1] // 8)
        trows = self.threads_per_cta // tcols
        sg_thr_layout = cute.make_layout((trows, tcols), stride=(tcols,1))
        sg_val_layout = cute.make_layout((1, 8), stride=(1, 1))
        tiled_copy_s2g = cute.make_tiled_copy_tv(atom_sync, sg_thr_layout, sg_val_layout)
        thr_s2g = tiled_copy_s2g.get_slice(tidx)
        gOut = cute.local_tile(out, tile_mn, (bidx, bidy))
        s2g_src = thr_s2g.partition_S(sD[None, None, 0])
        # split gOut into ((epiM, epiN), (num_epiM, num_epiN))
        gOut_div = cute.zipped_divide(gOut, self.epi_tile)

        epi_tile_shape = gOut_div.shape[1]
        epi_tile_layout = cute.make_ordered_layout(epi_tile_shape, order=(1,0))
        episize = cute.size(epi_tile_shape)

        for s in cutlass.range_constexpr(episize):
            epi_coord = epi_tile_layout.get_hier_coord(s)
            # cute.autovec_copy(tRS_rAcc[None, None, None, epi_coord], tRS_rD)
            data = tRS_rAcc[None, None, None, epi_coord].load()
            tRS_rD.store(data.to(self.d_dtype))
            cute.copy(tiled_copy_r2s, tRS_rD, tRS_sD[None, None, None, 0])
            cute.arch.sync_threads()
            gOut_subtile = cute.local_tile(gOut, self.epi_tile, epi_coord)
            s2g_dst = thr_s2g.partition_D(gOut_subtile)
            cute.copy(tiled_copy_s2g, s2g_src, s2g_dst)
            cute.arch.sync_threads()
    
    def _setup_attributes(self):
        self.epi_tile = self.compute_tile_shape(
            self.cta_tile_shape_mnk,
            self.atom_layout_mnk,
            self.d_dtype,
        )
        self.epi_tile_shape = cute.ceil_div(
            self.cta_tile_shape_mnk[:2], self.epi_tile
        )
    
    def _setup_smem_layout(self):
        # smem_capacity = cutlass.utils.get_smem_capacity_in_bytes(f"sm_0")  # smem_capacity
        # print(f'smem cap: {smem_capacity}')
        # a_shape = cute.slice_(self.cta_tile_shape_mnk, (None, 0, None))
        # b_shape = cute.slice_(self.cta_tile_shape_mnk, (0, None, None))
        # ab_bytes_per_stage = (
        #     cute.size(a_shape) * self.a_dtype.width // 8 + cute.size(b_shape) * self.b_dtype.width // 8
        # )
        # ab_stage = smem_capacity // ab_bytes_per_stage
        ab_stage = 2 # simple double buffering for now
        
        a_smem_layout = make_smem_layout(
            self.a_dtype, self.a_layout, self.cta_tile_shape_mnk, dim=0, ab_stage=ab_stage
        )
        b_smem_layout = make_smem_layout(
            self.b_dtype, self.b_layout, self.cta_tile_shape_mnk, dim=1, ab_stage=ab_stage
        )
        d_smem_layout = make_epi_smem_layout(
            self.d_dtype, self.d_layout, self.epi_tile, stage=1,
        )
        return (
            a_smem_layout, b_smem_layout, d_smem_layout
        )

    def _setup_shared_storage(
        self,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        d_smem_layout: cute.ComposedLayout
    ):
        @cute.struct
        class SharedStorage:
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(a_smem_layout)
                ],
                1024,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(b_smem_layout)
                ],
                1024,
            ]
            sD: cute.struct.Align[
                cute.struct.MemRange[
                    self.d_dtype, cute.cosize(d_smem_layout)
                ],
                1024,
            ]
        self.shared_storage = SharedStorage
    
    def _setup_tiled_mma(self):
        tiler_mn = (64, self.cta_tile_shape_mnk[1] // self.atom_layout_mnk[1])
        print(f'tiler_mn: {tiler_mn}')
        tiled_mma = sm90_helpers.make_trivial_tiled_mma(
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
            tiled_mma = cute.make_tiled_mma(
                cute.make_mma_atom(tiled_mma.op),
                self.atom_layout_mnk,
                permutation_mnk=(None, perm_n, None),
            )
        
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
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
        return tiled_mma
    
    def compute_tile_shape(
        self,
        cta_tile_shape_mnk: Tuple[int, int, int],
        atom_layout_mnk: Tuple[int, int, int],
        element_type: Optional[Type[cutlass.Numeric]] = None,
    ):
        if cta_tile_shape_mnk[0] % 128 == 0 and atom_layout_mnk[0] > 1:
            tile_m = math.gcd(128, cute.size(cta_tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(32, cute.size(cta_tile_shape_mnk, mode=[1]))
        elif cta_tile_shape_mnk[0] % 192 == 0 and atom_layout_mnk[0] > 1:
            tile_m = math.gcd(192, cute.size(cta_tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(32, cute.size(cta_tile_shape_mnk, mode=[1]))
        else:
            n_perf = 64 if element_type is not None and element_type.width == 8 else 32
            tile_m = math.gcd(64, cute.size(cta_tile_shape_mnk, mode=[0]))
            tile_n = math.gcd(n_perf, cute.size(cta_tile_shape_mnk, mode=[1]))
        return (tile_m, tile_n)

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
    
    @cute.jit
    def convert_layout_acc_frgA(self, acc_layout: cute.Layout) -> cute.Layout:
        div = 2 if const_expr(acc_layout.shape[0][2] % 2 == 0) else 1
        l = cute.logical_divide(
            acc_layout, ((None, None, div), None, None)
        )  # ((2, 2, (2, N / 16)), MMA_M, MMA_N)
        rA_mma_view = cute.make_layout(
            (
                (l.shape[0][0], l.shape[0][1], l.shape[0][2][0]),
                l.shape[1],
                (l.shape[0][2][1], l.shape[2]),
            ),
            stride=(
                (l.stride[0][0], l.stride[0][1], l.stride[0][2][0]),
                l.stride[1],
                (l.stride[0][2][1], l.stride[2]),
            ),
        )
        return rA_mma_view
    
    def epi_retile_acc(
        self, acc, tRS_rD, tiled_copy_r2s,
    ):
        acc_reshaped = cute.make_tensor(acc.iterator, self.convert_layout_acc_frgA(acc.layout))
        epi_acc_shape = (
            acc_reshaped.shape[0],
            *cute.ceil_div(acc_reshaped.shape[1:], self.epi_tile_shape)
        )
        acc_divide = cute.flat_divide(acc_reshaped, epi_acc_shape)
        print(f'acc_divide: {acc_divide}')
        assert cute.size(acc_divide, mode=[3]) == 1
        tRS_rAcc = cute.group_modes(
            acc_divide[None, None, None, 0, None, None], 3, 5
        )
        print(f'trs_racc: {tRS_rAcc}')
        return tiled_copy_r2s.retile(tRS_rAcc)
    
    def epilog_smem_store_and_partition(
        self,
        tiled_mma: cute.TiledMma,
        sD: cute.Tensor,
        epi_tile: cute.Tile,
        tidx: cutlass.Int32,
    ):
        copy_atom_C = cute.make_copy_atom(
            StMatrix8x8x16bOp(self.d_layout.is_m_major_c(), num_matrices=4),
            cutlass.BFloat16
        )
        tiled_copy_C = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
        copy_atom_r2s = sm90_helpers.sm90_get_smem_store_op(
            self.d_layout, elem_ty_d=self.d_dtype, elem_ty_acc=self.acc_dtype
        )
        tiled_copy_r2s = cute.make_tiled_copy_S(copy_atom_r2s, tiled_copy_C)
        thr_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sD = thr_r2s.partition_D(sD)
        tRS_rD = cute.make_rmem_tensor(
            thr_r2s.partition_S(cute.make_identity_tensor(epi_tile)).shape,
            self.d_dtype
        )
        return tiled_copy_r2s, tRS_rD, tRS_sD
