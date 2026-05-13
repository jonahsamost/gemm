from ast import Tuple
from dataclasses import dataclass
from enum import IntEnum
from ssl import ALERT_DESCRIPTION_ILLEGAL_PARAMETER
from typing import Optional
import cutlass.cute as cute
import cutlass
from cutlass import Int32, const_expr, Boolean


from divmod import FastDivmod
from utils import PipelineStateWAdvance, atomic_inc_i32


class RasterOrder(IntEnum):
    AlongM = 0
    AlongN = 1


@dataclass
class TileSchedulerArgs:
    problem_shape_ntile_mnl: cute.Shape
    raster_order: RasterOrder
    group_size: Int32
    cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
    tile_count_semaphore: Optional[cute.Pointer]


class TileScheduler:
    @dataclass
    class Params:
        problem_shape_ncluster_mnl: cute.Shape
        raster_order: RasterOrder
        num_clusters_per_problem_fdd: FastDivmod
        num_groups_regular: Int32
        group_size_fdd: FastDivmod
        group_size_tail_fdd: FastDivmod
        num_clusters_in_group_fdd: FastDivmod
        tile_count_semaphore: Optional[cute.Pointer]
        cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
    
        @staticmethod
        @cute.jit
        def create(args: TileSchedulerArgs, *, loc=None, ip=None) -> "TileScheduler.Params":
            mn = cute.select(args.problem_shape_ntile_mnl, mode=[0, 1])
            ncluster_mn = (
                cute.ceil_div(mn[0], args.cluster_shape_mnk[0]),
                cute.ceil_div(mn[1], args.cluster_shape_mnk[1]),
            )
            ncluster_mnl = ncluster_mn + (1,)
            num_clusters_per_problem = cute.size(ncluster_mn)
            ncluster_fast = (
                ncluster_mn[0]
                if args.raster_order == RasterOrder.AlongM
                else ncluster_mn[1]
            )
            ncluster_slow = (
                ncluster_mn[1]
                if args.raster_order == RasterOrder.AlongM
                else ncluster_mn[0]
            )
            group_size = min(args.group_size, ncluster_fast)
            group_size_tail = ncluster_fast % group_size
            num_groups_regular = ncluster_fast // group_size
            num_clusters_in_group = group_size * ncluster_slow
            return TileScheduler.Params(
                ncluster_mnl,
                args.raster_order,
                FastDivmod(num_clusters_per_problem),
                num_groups_regular,
                FastDivmod(group_size),
                FastDivmod(group_size_tail),
                FastDivmod(num_clusters_in_group),
                args.tile_count_semaphore,
                args.cluster_shape_mnk,
            )

    def __init__(
        self,
        current_work_idx: Int32,
        num_tiles_executed: Int32,
        current_batch_idx: Int32,
        num_work_idx_before_cur_batch: Int32,
        sched_smem: Optional[cute.Tensor],
        scheduler_pipeline: Optional[cutlass.pipeline.PipelineAsync],
        pipeline_state: PipelineStateWAdvance,
        params: Params,
        *,
        loc=None,
        ip=None,
    ):
        self._current_work_idx = current_work_idx
        self.num_tiles_executed = num_tiles_executed
        self._current_batch_idx = current_batch_idx
        self._num_work_idx_before_cur_batch = num_work_idx_before_cur_batch
        self._sched_smem = sched_smem
        self._scheduler_pipeline = scheduler_pipeline
        self._pipeline_state = pipeline_state
        self.params = params
        self._loc = loc
        self._ip = ip
    
    @staticmethod
    def create_params(args: TileSchedulerArgs, *, loc=None, ip=None) -> Params:
        return TileScheduler.Params.create(args, loc=loc, ip=ip)
    
    @staticmethod
    def get_grid_shape(params: Params, max_active_clusters: Int32) -> Tuple[Int32, Int32, Int32]:
        num_ctas_in_problem = cute.size(
            params.problem_shape_ncluster_mnl
        ) * cute.size(params.cluster_shape_mnk)
        num_ctas_per_cluster = cute.size(params.cluster_shape_mnk)
        num_ctas_per_wave = max_active_clusters * num_ctas_per_cluster
        num_persistent_ctas = cutlass.min(num_ctas_in_problem, num_ctas_per_wave)
        num_persistent_clusters = num_persistent_ctas // num_ctas_per_cluster
        return (
            params.cluster_shape_mnk[0],
            params.cluster_shape_mnk[1],
            params.cluster_shape_mnk[2] * num_persistent_clusters,
        )
    
    @staticmethod
    @cute.jit
    def create(
        params: Params,
        sched_smem: cute.Tensor,
        sched_pipeline: cutlass.pipeline.PipelineAsync,
        *, loc=None, ip=None,
    ):
        current_work_idx = cute.arch.cluster_idx()[2]
        stages = const_expr(cute.size(sched_smem), mode=[1]) 
        return TileScheduler(
            current_work_idx,
            Int32(0),
            Int32(0),
            Int32(0),
            sched_smem,
            sched_pipeline,
            PipelineStateWAdvance(stages, Int32(0), Int32(0), Int32(0)),
            params,
            loc=loc,
            ip=ip
        )
    
    @cute.jit
    def _delinearize_work_idx(
        self,
        work_idx: Int32,
        bidz: Optional[Int32] = None,
        is_valid: Optional[Boolean] = None,
        block_zero_only: bool = False,
        *, loc=None, ip=None,
    ):
        params = self.params
        if const_expr(is_valid is None):
            is_valid = work_idx < cute.size(params.problem_shape_ncluster_mnl)
        pid_m, pid_n, batch_idx = Int32(0), Int32(0), Int32(0)
        if is_valid:
            bidz_, cluster_id_in_problem = divmod(work_idx, params.num_clusters_per_problem_fdd)
            if const_expr(bidz is not None):
                bidz_ = bidz
                cid_m, cid_n = self._swizzle_cta(cluster_id_in_problem, loc=loc, ip=ip)
                pid_m, pid_n = self._cluster_id_to_cta_id(
                    cid_m, cid_n, block_zero_only=block_zero_only, loc=loc, ip=ip
                )
                batch_idx = bidz_
        tile_coord_mnkl = (pid_m, pid_n, None, batch_idx)
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, is_valid)

    def _swizzle_cta(
        self, cluster_id_in_problem: Int32, *, loc=None, ip=None
    ):
        params = self.params
        group_id, id_in_group = divmod(cluster_id_in_problem, params.num_clusters_in_group_fdd)
        cid_fast_in_group, cid_slow = Int32(0), Int32(0)
        if group_id < params.num_groups_regular:
            cid_slow, cid_fast_in_group = divmod(id_in_group, params.group_size_fdd)
        else:
            cid_slow, cid_fast_in_group, divmod(id_in_group, params.group_size_tail_fdd)
        if group_id % 2 == 1:
            ncluster_slow = (
                params.problem_shape_ncluster_mnl[1]
                if params.raster_order == RasterOrder.AlongM
                else params.problem_shape_ncluster_mnl[0]
            )
            cid_slow = ncluster_slow - 1 - cid_slow
        cid_fast = group_id * params.group_size_fdd.divisor + cid_fast_in_group
        cid_m, cid_n = cid_fast, cid_slow
        if params.raster_order == RasterOrder.AlongN:
            cid_m, cid_n = cid_slow, cid_fast
        return cid_m, cid_n
    
    @cute.jit
    def _cluster_id_to_cta_id(
        self, cid_m: Int32, cid_n: Int32, *, block_zero_only: bool = False, loc=None, ip=None
    ):
        if const_expr(block_zero_only):
            bidx_in_cluster = (Int32(0), Int32(0))
        else:
            bidx_in_cluster = cute.arch.block_in_cluster_idx()
        pid_m = cid_m * self.params.cluster_shape_mnk[0] * bidx_in_cluster[0]
        pid_n = cid_n * self.params.cluster_shape_mnk[1] * bidx_in_cluster[1]
        return pid_m, pid_n
    
    def initial_work_tile_info(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        return self._delinearize_work_idx(self._current_work_idx, loc=loc, ip=ip)
    
    @cute.jit
    def _fetch_next_work_idx(
        self, *, loc=None, ip=None
    ) -> Int32 | Tuple[Int32, Int32, Boolean]:
        params = self.params
        next_work_linear_idx = Int32(0)
        num_persistent_clusters = cute.arch.cluster_dim()[2]
        if cute.arch.lane_idx() == 0:
            next_work_linear_idx = num_persistent_clusters + atomic_inc_i32(
                cute.size(params.problem_shape_ncluster_mnl) - 1,
                params.tile_count_semaphore
            )
        return cute.arch.shuffle_sync(next_work_linear_idx, 0)
    
    @cute.jit
    def write_work_tile_to_smem(
        self, work_tile: cutlass.utils.WorkTileInfo, *, loc=None, ip=None
    ):
        params = self.params
        pipeline_state_prod = PipelineStateWAdvance(
            self._pipeline_state.stages,
            self._pipeline_state.count,
            self._pipeline_state.index,
            self._pipeline_state.phase ^ 1,
        )
        self._scheduler_pipeline.producer_acquire(pipeline_state_prod)
        sched_data = [
            work_tile.tile_idx[0],
            work_tile.tile_idx[1],
            work_tile.tile_idx[2],
            Int32(work_tile.is_valid_tile),
        ]
        lane_idx = cute.arch.lane_idx()
        if lane_idx < cute.size(params.cluster_shape_mnk):
            pipeline_idx = self._pipeline_state.index
            if const_expr(cute.size(params.cluster_shape_mnk) == 1):
                for i in cutlass.range_constexpr(4):
                    self._sched_smem[i, pipeline_idx] = sched_data[i]
            else:
                ...
    
    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        params = self.params
        pid_m, pid_n, batch_idx, is_valid = Int32(0), Int32(0), Int32(0), Boolean(False)
        self._scheduler_pipeline.consumer_wait(self._pipeline_state)
        pid_m, pid_n, batch_idx, is_valid_i32 = [
            self._sched_smem[i, self._pipeline_state.index] for i in range(4)
        ]
        if const_expr(cute.size(params.cluster_shape_mnk) > 1):
            cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            self._scheduler_pipeline.consumer_release(self._pipeline_state)
        self._pipeline_state.advance()
        is_valid= Boolean(is_valid_i32)
        tile_coord_mnkl = (pid_m, pid_n, None, batch_idx)
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, Boolean(is_valid))
    
    @cute.jit
    def advance_to_next_work(
        self,
        is_scheduler_warp: bool | Boolean = False,
        *,
        advance_count: int = 1,
        loc=None, ip=None
    ):
        params = self.params
        self.num_tiles_executed += Int32(advance_count)
        if is_scheduler_warp:
            self._current_work_idx = self._fetch_next_work_idx(loc=loc, ip=ip)
            work_tile_info = self._delinearize_work_idx(
                self._current_work_idx, block_zero_only=True, loc=loc, ip=ip
            )
            self.write_work_tile_to_smem(work_tile_info, loc=loc, ip=ip)
