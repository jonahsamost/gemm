from typing import List, Tuple, Union
import torch
import cutlass.cute as cute
from cutlass import const_expr


@cute.jit
def get_swizzle_block(cta_swizzle_width: int) -> Union[int, int]:
    bidx, bidy, _ = cute.arch.block_idx()
    gdimx, gdimy, _ = cute.arch.grid_dim()
    linear_id = bidx * gdimy + bidy

    raster_along_m = True if gdimy > gdimx else False
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


# ---------------------------------------------------------------------------
# Host-side look up table generation for offline tile orderings
# ---------------------------------------------------------------------------

def _swizzle_cta(cluster_id, ncluster_m, ncluster_n, group_size):
    """Pure-Python replica of TileScheduler._swizzle_cta."""
    raster_along_m = ncluster_m > ncluster_n
    ncluster_fast = ncluster_m if raster_along_m else ncluster_n
    ncluster_slow = ncluster_n if raster_along_m else ncluster_m
    group_size = min(group_size, ncluster_fast)
    group_size_tail = ncluster_fast % group_size
    num_groups_regular = ncluster_fast // group_size
    num_clusters_in_group = group_size * ncluster_slow

    group_id, id_in_group = divmod(cluster_id, num_clusters_in_group)
    if group_id < num_groups_regular:
        cid_slow, cid_fast_in_group = divmod(id_in_group, group_size)
    else:
        gs_tail = group_size_tail if group_size_tail > 0 else 1
        cid_slow, cid_fast_in_group = divmod(id_in_group, gs_tail)
    if group_id % 2 == 1:
        cid_slow = ncluster_slow - 1 - cid_slow
    cid_fast = group_id * group_size + cid_fast_in_group
    cid_m, cid_n = cid_fast, cid_slow
    if not raster_along_m:
        cid_m, cid_n = cid_slow, cid_fast
    return cid_m, cid_n


def _sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def _generate2d(x, y, ax, ay, bx, by):
    """Gilbert curve generator (BSD-2-Clause, Jakub Červený)."""
    w = abs(ax + ay)
    h = abs(bx + by)
    dax, day = _sgn(ax), _sgn(ay)
    dbx, dby = _sgn(bx), _sgn(by)
    if h == 1:
        for _ in range(w):
            yield (x, y)
            x, y = x + dax, y + day
        return
    if w == 1:
        for _ in range(h):
            yield (x, y)
            x, y = x + dbx, y + dby
        return
    ax2, ay2 = ax // 2, ay // 2
    bx2, by2 = bx // 2, by // 2
    w2 = abs(ax2 + ay2)
    h2 = abs(bx2 + by2)
    if 2 * w > 3 * h:
        if (w2 % 2) and (w > 2):
            ax2, ay2 = ax2 + dax, ay2 + day
        yield from _generate2d(x, y, ax2, ay2, bx, by)
        yield from _generate2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by)
    else:
        if (h2 % 2) and (h > 2):
            bx2, by2 = bx2 + dbx, by2 + dby
        yield from _generate2d(x, y, bx2, by2, ax2, ay2)
        yield from _generate2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
        yield from _generate2d(
            x + (ax - dax) + (bx2 - dbx), y + (ay - day) + (by2 - dby),
            -bx2, -by2, -(ax - ax2), -(ay - ay2),
        )


def _gilbert2d(width, height):
    if width >= height:
        yield from _generate2d(0, 0, width, 0, 0, height)
    else:
        yield from _generate2d(0, 0, 0, height, width, 0)


def _coords_to_lut(coords: List[Tuple[int, int]], device="cuda") -> torch.Tensor:
    packed = [(m << 16) | n for m, n in coords]
    return torch.tensor(packed, dtype=torch.int32, device=device)


def create_swizzle_lut(ncluster_m, ncluster_n, group_size, device="cuda") -> torch.Tensor:
    total = ncluster_m * ncluster_n
    coords = [_swizzle_cta(i, ncluster_m, ncluster_n, group_size) for i in range(total)]
    return _coords_to_lut(coords, device)


def create_hilbert_lut(ncluster_m, ncluster_n, device="cuda") -> torch.Tensor:
    coords = list(_gilbert2d(ncluster_m, ncluster_n))
    return _coords_to_lut(coords, device)