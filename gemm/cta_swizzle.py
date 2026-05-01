from typing import Union
import cutlass.cute as cute
from cutlass import const_expr


@cute.jit
def get_swizzle_block(cta_swizzle_width: int) -> Union[int, int]:
    # cta swizzle (raster group + serpentine ordering)
    bidx, bidy, _ = cute.arch.block_idx()
    # cols==gdimy, rows==gdimx
    gdimx, gdimy, _ = cute.arch.grid_dim()
    linear_id = bidx * gdimy + bidy

    # raster order to pick the fast/slow dims
    raster_along_m = True if gdimy > gdimx else False
    # nfast = gdimx if raster_along_m else gdimy
    nslow = gdimy if raster_along_m else gdimx

    group_size = const_expr(cta_swizzle_width)
    num_clusters_in_grp = group_size * nslow

    group_id = linear_id // num_clusters_in_grp
    id_in_group = linear_id % num_clusters_in_grp

    cid_slow = id_in_group // group_size
    cid_fast = id_in_group % group_size

    if group_id % 2 == 1:
        cid_slow = nslow - 1 - cid_slow
    
    cid_fast = group_id * group_size + cid_fast
    return (cid_fast, cid_slow) if raster_along_m else (cid_slow, cid_fast)