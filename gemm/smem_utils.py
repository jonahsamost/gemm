from typing import Type, Union, Optional, Tuple

import cutlass.utils.hopper_helpers as sm90_helpers
from cutlass.cutlass_dsl import Numeric, dsl_user_op
import cutlass.cute as cute
from cutlass.cute.nvgpu import warpgroup
from cutlass.utils import LayoutEnum
from cutlass import Int32, const_expr


@dsl_user_op
def make_smem_layout(
    dtype: Type[Numeric],
    layout: LayoutEnum,
    cta_tile_shape_mnk: Tuple[int, int, int],
    dim: int,
    ab_stage: int,
    *, loc=None, ip=None
) -> Union[cute.Layout, cute.ComposedLayout]:
    if dim == 0: # A tensor
        smem_shape = cute.slice_(cta_tile_shape_mnk, (None, 0, None))
    else: # B tensor
        smem_shape = cute.slice_(cta_tile_shape_mnk, (0, None, None))

    is_k_major = layout.sm90_mma_major_mode() == warpgroup.OperandMajorMode.K
    major_mode_size = cta_tile_shape_mnk[2 if is_k_major else dim]
    atom_kind = sm90_helpers.get_smem_layout_atom(
        layout, dtype, major_mode_size
    )
    smem_layout_atom = warpgroup.make_smem_layout_atom(
        atom_kind, dtype,
    )
    smem_layout = cute.tile_to_shape(
        smem_layout_atom,
        cute.prepend(smem_shape, ab_stage),
        order=(0, 1, 2) if is_k_major else (1, 0, 2),
    )
    return smem_layout


def swizzle_int(ptr_int: Int32, b: int, m: int, s: int) -> Int32:
    bit_mask = (1 << b) - 1
    yyy_msk = bit_mask << (m + s)
    return ptr_int ^ ((ptr_int & yyy_msk) >> s)


def swizzle_ptr(ptr: cute.Pointer):
    swiz = ptr.type.swizzle_type
    ptr_int = swizzle_int(ptr.toint(), swiz.num_bits, swiz.num_base, swiz.num_shift)
    return cute.make_ptr(ptr.dtype, ptr_int, ptr.memspace, assumed_align=ptr.alignment)


def as_pos_ind_swizzle_tensor(tensor: cute.Tensor) -> cute.Tensor:
    """
    Swizzle operates on byte addresses (B/M/S refer to bit positions within byte).
    The tensor layout may be another dtype (i.e. bf16).

    the new_layout refers to: 
        Given some logical coordinate, first compute the byte offset via
        the outer layout, then apply the XOR swizzle to that byte offset. 
        The returned tensor takes element coordinates in, and produces 
        swizzled element offsets out
    """
    outer = tensor.layout
    width = tensor.element_type.width
    swizzle_type = tensor.iterator.type.swizzle_type
    inner = cute.make_swizzle(
        swizzle_type.num_bits, swizzle_type.num_base, swizzle_type.num_shift
    )
    new_layout = cute.recast_layout(
        width,
        8,
        cute.make_composed_layout(
            inner, 0, cute.recast_layout(8, width, outer)
        )    
    )
    # remove the swizzle from the tensor because the swizzle is now in the layout
    return cute.make_tensor(
        cute.recast_ptr(tensor.iterator, dtype=tensor.element_type),
        new_layout
    )


def partition_S_pos_ind(
    thr_copy: cute.core.ThrCopy, tensor: cute.Tensor, assumed_align: int | None = None,
) -> cute.Tensor:
    return cute.make_tensor(
        swizzle_ptr(thr_copy.partition_S(tensor).iterator),
        thr_copy.partition_S(as_pos_ind_swizzle_tensor(tensor)).layout
    )

def partition_D_pos_ind(
    thr_copy: cute.core.ThrCopy, tensor: cute.Tensor, assumed_align: int | None = None,
) -> cute.Tensor:
    return cute.make_tensor(
        swizzle_ptr(thr_copy.partition_D(tensor).iterator),
        thr_copy.partition_D(as_pos_ind_swizzle_tensor(tensor)).layout
    )
