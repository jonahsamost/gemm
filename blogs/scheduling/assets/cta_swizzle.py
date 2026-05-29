import json
import math


def swizzle_cta(cluster_id, ncluster_m, ncluster_n, group_size, raster_along_m=True):
    ncluster_fast = ncluster_m if raster_along_m else ncluster_n
    ncluster_slow = ncluster_n if raster_along_m else ncluster_m
    group_size = min(group_size, ncluster_fast)
    group_size_tail = ncluster_fast % group_size
    num_groups_regular = ncluster_fast // group_size
    num_clusters_in_group = group_size * ncluster_slow

    group_id = cluster_id // num_clusters_in_group
    id_in_group = cluster_id % num_clusters_in_group
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


def make_id():
    import random, string
    return ''.join(random.choices(string.ascii_letters + string.digits, k=20))


def make_rect(x, y, w, h):
    return {
        "type": "rectangle", "id": make_id(), "x": x, "y": y,
        "width": w, "height": h, "angle": 0, "strokeColor": "#cccccc",
        "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0,
        "opacity": 100, "groupIds": [], "roundness": None,
        "boundElements": None, "updated": 1, "link": None,
        "locked": False, "version": 1, "versionNonce": 0,
        "isDeleted": False, "seed": 1,
    }


def make_text(x, y, text, font_size=14):
    return {
        "type": "text", "id": make_id(), "x": x, "y": y,
        "width": len(text) * font_size * 0.6, "height": font_size * 1.4,
        "angle": 0, "strokeColor": "#1e1e1e",
        "backgroundColor": "transparent", "fillStyle": "solid",
        "strokeWidth": 1, "strokeStyle": "solid", "roughness": 0,
        "opacity": 100, "groupIds": [], "roundness": None,
        "boundElements": None, "updated": 1, "link": None,
        "locked": False, "version": 1, "versionNonce": 0,
        "isDeleted": False, "seed": 1,
        "text": text, "fontSize": font_size, "fontFamily": 3,
        "textAlign": "center", "verticalAlign": "middle",
        "containerId": None, "originalText": text,
        "autoResize": True, "lineHeight": 1.2,
    }


PALETTE = [
    "#ffc9c9", "#b2f2bb", "#a5d8ff", "#ffec99", "#d0bfff", "#ffd8a8",
    "#99e9f2", "#eebefa", "#c3fae8", "#fcc2d7", "#d8f5a2", "#ffa8a8",
    "#8ce99a", "#74c0fc", "#ffe066", "#b197fc", "#ffc078", "#66d9e8",
    "#e599f7", "#96f2d7", "#f783ac", "#63e6be", "#91a7ff", "#ffa94d",
    "#e64980", "#20c997", "#4dabf7", "#fab005", "#be4bdb", "#fd7e14",
    "#12b886",
]


def make_line(points, color="#e03131"):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, min_y = min(xs), min(ys)
    local_points = [[p[0] - min_x, p[1] - min_y] for p in points]
    return {
        "type": "line", "id": make_id(), "x": min_x, "y": min_y,
        "width": max(xs) - min_x, "height": max(ys) - min_y,
        "angle": 0, "strokeColor": color, "backgroundColor": "transparent",
        "fillStyle": "solid", "strokeWidth": 2, "strokeStyle": "solid",
        "roughness": 0, "opacity": 100, "groupIds": [], "roundness": {"type": 2},
        "boundElements": None, "updated": 1, "link": None,
        "locked": False, "points": local_points, "version": 1,
        "versionNonce": 0, "isDeleted": False, "seed": 1,
        "lastCommittedPoint": None, "startBinding": None, "endBinding": None,
        "startArrowhead": None, "endArrowhead": None,
    }


def generate(ncluster_m, ncluster_n, group_size, cell=50, raster_along_m=True, wave_size=None):
    total = ncluster_m * ncluster_n
    coords = [swizzle_cta(i, ncluster_m, ncluster_n, group_size, raster_along_m) for i in range(total)]

    tile_to_wave = {}
    if wave_size:
        for i, (m, n) in enumerate(coords):
            tile_to_wave[(m, n)] = i // wave_size

    elements = []
    for m in range(ncluster_m):
        for n in range(ncluster_n):
            rect = make_rect(n * cell, m * cell, cell, cell)
            if wave_size:
                wave = tile_to_wave.get((m, n), 0)
                rect["backgroundColor"] = PALETTE[wave % len(PALETTE)]
            elements.append(rect)

    if not wave_size or total <= 256:
        points = [(n * cell + cell / 2, m * cell + cell / 2) for m, n in coords]
        elements.append(make_line(points))

    if wave_size:
        from collections import defaultdict
        wave_tiles = defaultdict(list)
        for i, (m, n) in enumerate(coords):
            wave_tiles[i // wave_size].append((m, n))
        for wave_id, tiles in wave_tiles.items():
            avg_m = sum(m for m, n in tiles) / len(tiles)
            avg_n = sum(n for m, n in tiles) / len(tiles)
            cx = avg_n * cell + cell / 2
            cy = avg_m * cell + cell / 2
            elements.append(make_text(cx, cy, str(wave_id), font_size=56))

    return {
        "type": "excalidraw", "version": 2, "source": "python-gen",
        "elements": elements,
        "appState": {"viewBackgroundColor": "#ffffff"}, "files": {},
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, default=12)
    parser.add_argument("-n", type=int, default=11)
    parser.add_argument("-g", "--group-size", type=int, default=6)
    parser.add_argument("--raster", choices=["M", "N"], default="M",
                        help="Raster along M (down) or N (across)")
    parser.add_argument("-w", "--wave-size", type=int, default=None,
                        help="Color tiles by wave (e.g. 132 for H100)")
    parser.add_argument("-o", "--output", default="cta_swizzle.excalidraw")
    args = parser.parse_args()

    scene = generate(args.m, args.n, args.group_size,
                     raster_along_m=(args.raster == "M"), wave_size=args.wave_size)
    with open(args.output, "w") as f:
        json.dump(scene, f, indent=2)
    print(f"Wrote {args.output}")