from ast import Tuple
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
import cutlass.cute as cute
import cutlass
from cutlass import Int32


from divmod import FastDivmod


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

    def __init__(self):
        ...
    
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
    