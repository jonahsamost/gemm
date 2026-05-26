from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple
import cutlass.cute as cute
import cutlass
from cutlass import Int32, const_expr, Boolean


from divmod import FastDivmod
from utils import PipelineStateWAdvance, atomic_inc_i32, store_shared_remote_x4


class RasterOrder(IntEnum):
    AlongM = 0
    AlongN = 1


class PersistenceMode(IntEnum):
    NONE = 0
    STATIC = 1
    DYNAMIC = 2


class CTATileOrdering(IntEnum):
    NONE = 0
    ONLINE_SWIZZLE = 1
    OFFLINE_SWIZZLE = 2
    HILBERT = 3


@dataclass
class TileSchedulerArgs:
    problem_shape_ntile_mnl: cute.Shape
    raster_order: RasterOrder
    group_size: Int32
    cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
    tile_count_semaphore: Optional[cute.Pointer]
    tile_order_lut: Optional[cute.Pointer] = None
    persistence_mode: cutlass.Constexpr[PersistenceMode] = PersistenceMode.DYNAMIC
    tile_ordering: cutlass.Constexpr[CTATileOrdering] = CTATileOrdering.ONLINE_SWIZZLE


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
        tile_order_lut: Optional[cute.Pointer]
        cluster_shape_mnk: cutlass.Constexpr[cute.Shape]
        persistence_mode: cutlass.Constexpr[PersistenceMode]
        tile_ordering: cutlass.Constexpr[CTATileOrdering]

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
            raster_order = (
                RasterOrder.AlongM
                if ncluster_mn[0] > ncluster_mn[1]
                else RasterOrder.AlongN
            )
            ncluster_fast = (
                ncluster_mn[0]
                if raster_order == RasterOrder.AlongM
                else ncluster_mn[1]
            )
            ncluster_slow = (
                ncluster_mn[1]
                if raster_order == RasterOrder.AlongM
                else ncluster_mn[0]
            )
            group_size = min(args.group_size, ncluster_fast)
            group_size_tail = ncluster_fast % group_size
            num_groups_regular = ncluster_fast // group_size
            num_clusters_in_group = group_size * ncluster_slow
            if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC):
                assert args.tile_count_semaphore is not None
            return TileScheduler.Params(
                ncluster_mnl,
                raster_order,
                FastDivmod(num_clusters_per_problem),
                num_groups_regular,
                FastDivmod(group_size),
                FastDivmod(group_size_tail if group_size_tail > 0 else 1),
                FastDivmod(num_clusters_in_group),
                args.tile_count_semaphore
                if const_expr(args.persistence_mode == PersistenceMode.DYNAMIC)
                else None,
                args.tile_order_lut
                if const_expr(args.tile_ordering in (
                    CTATileOrdering.OFFLINE_SWIZZLE, CTATileOrdering.HILBERT
                ))
                else None,
                args.cluster_shape_mnk,
                args.persistence_mode,
                args.tile_ordering,
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
        if const_expr(params.persistence_mode == PersistenceMode.NONE):
            return (
                params.cluster_shape_mnk[0] * cute.size(params.problem_shape_ncluster_mnl[:2]),
                params.cluster_shape_mnk[1],
                params.cluster_shape_mnk[2] * params.problem_shape_ncluster_mnl[2],
            )
        else:
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
        sched_smem: Optional[cute.Tensor] = None,
        sched_pipeline: Optional[cutlass.pipeline.PipelineAsync] = None,
        *, loc=None, ip=None,
    ):
        if const_expr(params.persistence_mode == PersistenceMode.NONE):
            current_work_idx = Int32(cute.arch.cluster_idx()[0])
        else:
            current_work_idx = Int32(cute.arch.cluster_idx()[2])
        stages = 0
        if const_expr(
            params.persistence_mode in [PersistenceMode.STATIC, PersistenceMode.DYNAMIC]
        ):
            assert sched_smem is not None
            assert sched_pipeline is not None
            stages = const_expr(cute.size(sched_smem, mode=[1]))
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
            if const_expr(params.persistence_mode == PersistenceMode.NONE):
                is_valid = self.num_tiles_executed == 0
            else:
                is_valid = work_idx < cute.size(params.problem_shape_ncluster_mnl)
        pid_m, pid_n, batch_idx = Int32(0), Int32(0), Int32(0)
        if is_valid:
            if const_expr(params.persistence_mode == PersistenceMode.NONE):
                cluster_id_in_problem = work_idx
                _, _, bidz_ = cute.arch.cluster_idx()
            else:
                bidz_, cluster_id_in_problem = divmod(
                    work_idx, params.num_clusters_per_problem_fdd
                )
            if const_expr(bidz is not None):
                bidz_ = bidz
            if const_expr(params.tile_ordering == CTATileOrdering.ONLINE_SWIZZLE):
                cid_m, cid_n = self._swizzle_cta(cluster_id_in_problem, loc=loc, ip=ip)
            elif const_expr(params.tile_ordering in (
                CTATileOrdering.OFFLINE_SWIZZLE, CTATileOrdering.HILBERT
            )):
                packed = params.tile_order_lut[cluster_id_in_problem]
                cid_m = packed >> 16
                cid_n = packed & 0xFFFF
            else:
                cid_m, cid_n = self._naive_raster(cluster_id_in_problem, loc=loc, ip=ip)
            pid_m, pid_n = self._cluster_id_to_cta_id(
                cid_m, cid_n, block_zero_only=block_zero_only, loc=loc, ip=ip
            )
            batch_idx = bidz_
        tile_coord_mnkl = (pid_m, pid_n, None, batch_idx)
        return cutlass.utils.WorkTileInfo(tile_coord_mnkl, is_valid)

    @cute.jit
    def _swizzle_cta(
        self, cluster_id_in_problem: Int32, *, loc=None, ip=None
    ):
        params = self.params
        group_id, id_in_group = divmod(cluster_id_in_problem, params.num_clusters_in_group_fdd)
        cid_fast_in_group, cid_slow = Int32(0), Int32(0)
        if group_id < params.num_groups_regular:
            cid_slow, cid_fast_in_group = divmod(id_in_group, params.group_size_fdd)
        else:
            cid_slow, cid_fast_in_group = divmod(id_in_group, params.group_size_tail_fdd)
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
    def _naive_raster(
        self, cluster_id_in_problem: Int32, *, loc=None, ip=None
    ) -> Tuple[Int32, Int32]:
        params = self.params
        ncluster_m = params.problem_shape_ncluster_mnl[0]
        ncluster_n = params.problem_shape_ncluster_mnl[1]
        ncluster_fast = (
            ncluster_m if params.raster_order == RasterOrder.AlongM else ncluster_n
        )
        cid_fast = cluster_id_in_problem % ncluster_fast
        cid_slow = cluster_id_in_problem // ncluster_fast
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
        pid_m = cid_m * self.params.cluster_shape_mnk[0] + bidx_in_cluster[0]
        pid_n = cid_n * self.params.cluster_shape_mnk[1] + bidx_in_cluster[1]
        return pid_m, pid_n
    
    def initial_work_tile_info(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        return self._delinearize_work_idx(self._current_work_idx, loc=loc, ip=ip)
    
    @cute.jit
    def _fetch_next_work_idx(
        self, *, loc=None, ip=None
    ) -> Int32 | Tuple[Int32, Int32, Boolean]:
        params = self.params
        num_persistent_clusters = cute.arch.cluster_dim()[2]
        if const_expr(params.persistence_mode == PersistenceMode.STATIC):
            return self._current_work_idx + num_persistent_clusters
        elif const_expr(params.persistence_mode == PersistenceMode.DYNAMIC):
            next_work_linear_idx = Int32(0)
            if cute.arch.lane_idx() == 0:
                next_work_linear_idx = num_persistent_clusters + atomic_inc_i32(
                    cute.size(params.problem_shape_ncluster_mnl) - 1,
                    params.tile_count_semaphore
                )
            return cute.arch.shuffle_sync(next_work_linear_idx, 0)
        else:
            return Int32(0)
    
    @cute.jit
    def write_work_tile_to_smem(
        self, work_tile: cutlass.utils.WorkTileInfo, *, loc=None, ip=None
    ):
        params = self.params
        if const_expr(params.persistence_mode == PersistenceMode.NONE):
            return
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
            work_tile.tile_idx[3],
            Int32(work_tile.is_valid_tile),
        ]
        lane_idx = cute.arch.lane_idx()
        if lane_idx < cute.size(params.cluster_shape_mnk):
            pipeline_idx = self._pipeline_state.index
            if const_expr(cute.size(params.cluster_shape_mnk) == 1):
                for i in cutlass.range_constexpr(4):
                    self._sched_smem[i, pipeline_idx] = sched_data[i]
                self._scheduler_pipeline.producer_commit(self._pipeline_state)
            else:
                peer_cta_rank_in_cluster = lane_idx
                bidx_in_cluster = peer_cta_rank_in_cluster % params.cluster_shape_mnk[0]
                bidy_in_cluster = (
                    peer_cta_rank_in_cluster // params.cluster_shape_mnk[0]
                ) % params.cluster_shape_mnk[1]
                mbar_ptr = self._scheduler_pipeline.producer_get_barrier(self._pipeline_state)
                cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, 16, peer_cta_rank_in_cluster)
                store_shared_remote_x4(
                    sched_data[0] + bidx_in_cluster,
                    sched_data[1] + bidy_in_cluster,
                    sched_data[2],
                    sched_data[3],
                    smem_ptr=self._sched_smem[None, pipeline_idx].iterator,
                    mbar_ptr=mbar_ptr,
                    peer_cta_rank_in_cluster=peer_cta_rank_in_cluster,
                )
                
    
    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> cutlass.utils.WorkTileInfo:
        params = self.params
        pid_m, pid_n, batch_idx, is_valid = Int32(0), Int32(0), Int32(0), Boolean(False)
        if const_expr(params.persistence_mode == PersistenceMode.NONE):
            pass
        else:
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
            is_valid = Boolean(is_valid_i32)
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
        if const_expr(self._pipeline_state is not None and advance_count > 1):
            self._pipeline_state.advance_iters(advance_count - 1)
        if const_expr(
            params.persistence_mode in [PersistenceMode.STATIC, PersistenceMode.DYNAMIC]
        ):
            if is_scheduler_warp:
                self._current_work_idx = self._fetch_next_work_idx(loc=loc, ip=ip)
                work_tile_info = self._delinearize_work_idx(
                    self._current_work_idx, block_zero_only=True, loc=loc, ip=ip
                )
                self.write_work_tile_to_smem(work_tile_info, loc=loc, ip=ip)

    def producer_tail(self):
        if const_expr(self._scheduler_pipeline is not None):
            pipeline_state_producer = PipelineStateWAdvance(
                self._pipeline_state.stages,
                self._pipeline_state.count,
                self._pipeline_state.index,
                self._pipeline_state.phase ^ 1,
            )
            self._scheduler_pipeline.producer_tail(pipeline_state_producer)
    
    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self._current_work_idx,
            self.num_tiles_executed,
            self._current_batch_idx,
            self._num_work_idx_before_cur_batch,
            self._sched_smem,
            self._scheduler_pipeline,
            self._pipeline_state,
            self.params,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self._current_work_idx,
                self.num_tiles_executed,
                self._current_batch_idx,
                self._num_work_idx_before_cur_batch,
                self._sched_smem,
                self._scheduler_pipeline,
                self._pipeline_state,
                self.params,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)

