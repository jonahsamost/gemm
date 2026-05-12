from typing import Callable
import cutlass.cute as cute
from cutlass import const_expr, Int32
from cutlass._mlir.dialects import nvvm
from cutlass.cutlass_dsl import dsl_user_op, T
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline import PipelineState, PipelineUserType


@dsl_user_op
def tma_get_copy_fn(
    atom: cute.CopyAtom,
    cta_coord: cute.Coord,
    cta_layout: cute.Layout,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    g2s: bool = True,
    *,
    loc=None,
    ip=None,
    **kwargs,
) -> Callable:
    st, dt = (src_tensor, dst_tensor) if const_expr(g2s) else (dst_tensor, src_tensor)

    smem_group_rank = const_expr(cute.rank(dt) - 1)
    gmem_group_rank = const_expr(cute.rank(st) - 1)
    s, g = cpasync.tma_partition(
        atom,
        cta_coord,
        cta_layout,
        cute.group_modes(dt, 0, smem_group_rank), 
        cute.group_modes(st, 0, gmem_group_rank), 
        loc=loc,
        ip=ip,
    )

    src, dst = (g, s) if const_expr(g2s) else (s, g)

    @dsl_user_op
    def copy_tma(src_idx, dst_idx, *, loc=None, ip=None, **new_kwargs):
        cute.copy(
            atom, src[None, src_idx], dst[None, dst_idx], **new_kwargs, **kwargs, loc=loc, ip=ip
        )
    
    return copy_tma, src, dst


class PipelineStateWAdvance(PipelineState):
    @dsl_user_op
    def advance_iters(self, num_iterations: Int32, *, loc=None, ip=None):
        self._count += Int32(num_iterations)
        new_index = self._index + Int32(num_iterations)
        # How many times did we cross the stages boundary
        num_crossings = new_index // self.stages
        self._phase ^= num_crossings
        self._index = new_index % self.stages

    # This can be overridden by derived classes
    def __new_from_mlir_values__(self, values):
        return PipelineStateWAdvance(
            self.stages, Int32(values[0]), Int32(values[1]), Int32(values[2])
        )


def make_pipeline_state(type: PipelineUserType, stages: int):
    """
    Creates a pipeline state. Producers are assumed to start with an empty buffer and have a flipped phase bit of 1.
    """
    if type is PipelineUserType.Producer:
        return PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(1))
    elif type is PipelineUserType.Consumer:
        return PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(0))
    else:
        assert False, "Error: invalid PipelineUserType specified for make_pipeline_state."


@dsl_user_op
def atomic_inc_i32(
    a: int | Int32, gmem_ptr: cute.Pointer, *, loc=None, ip=None
):
    from cutlass import CUDA_VERSION

    # * NVVM call based on nvvm version
    if CUDA_VERSION.major == 12 and CUDA_VERSION.minor == 9:
        # Old API: requires explicit result type as first positional argument
        return nvvm.atomicrmw(
            res=T.i32(), op=nvvm.AtomicOpKind.INC, ptr=gmem_ptr.llvm_ptr, a=Int32(a).ir_value()
        )
    else:
        # New API: infers result type automatically
        return nvvm.atomicrmw(
            op=nvvm.AtomicOpKind.INC, ptr=gmem_ptr.llvm_ptr, a=Int32(a).ir_value()
        )
