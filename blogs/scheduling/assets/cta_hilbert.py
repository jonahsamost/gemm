import json


def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def generate2d(x, y, ax, ay, bx, by):
    """Generalized Hilbert ('Gilbert') curve for arbitrary rectangles.
    Based on https://github.com/jakubcerveny/gilbert (BSD-2-Clause)."""
    w = abs(ax + ay)
    h = abs(bx + by)
    dax, day = sgn(ax), sgn(ay)
    dbx, dby = sgn(bx), sgn(by)

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
        yield from generate2d(x, y, ax2, ay2, bx, by)
        yield from generate2d(x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by)
    else:
        if (h2 % 2) and (h > 2):
            bx2, by2 = bx2 + dbx, by2 + dby
        yield from generate2d(x, y, bx2, by2, ax2, ay2)
        yield from generate2d(x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
        yield from generate2d(
            x + (ax - dax) + (bx2 - dbx), y + (ay - day) + (by2 - dby),
            -bx2, -by2, -(ax - ax2), -(ay - ay2),
        )


def gilbert2d(width, height):
    if width >= height:
        yield from generate2d(0, 0, width, 0, 0, height)
    else:
        yield from generate2d(0, 0, 0, height, width, 0)


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


PALETTE = [
    "#ffc9c9", "#b2f2bb", "#a5d8ff", "#ffec99", "#d0bfff", "#ffd8a8",
    "#99e9f2", "#eebefa", "#c3fae8", "#fcc2d7", "#d8f5a2", "#ffa8a8",
    "#8ce99a", "#74c0fc", "#ffe066", "#b197fc", "#ffc078", "#66d9e8",
    "#e599f7", "#96f2d7", "#f783ac", "#63e6be", "#91a7ff", "#ffa94d",
    "#e64980", "#20c997", "#4dabf7", "#fab005", "#be4bdb", "#fd7e14",
    "#12b886",
]


def generate(m, n, cell=50, wave_size=None):
    coords = list(gilbert2d(m, n))

    tile_to_wave = {}
    if wave_size:
        for i, (row, col) in enumerate(coords):
            tile_to_wave[(row, col)] = i // wave_size

    elements = []
    for row in range(m):
        for col in range(n):
            rect = make_rect(col * cell, row * cell, cell, cell)
            if wave_size:
                wave = tile_to_wave.get((row, col), 0)
                rect["backgroundColor"] = PALETTE[wave % len(PALETTE)]
            elements.append(rect)

    if not wave_size or m * n <= 256:
        points = [(col * cell + cell / 2, row * cell + cell / 2) for row, col in coords]
        elements.append(make_line(points))

    if wave_size:
        from collections import defaultdict
        wave_tiles = defaultdict(list)
        for i, (row, col) in enumerate(coords):
            wave_tiles[i // wave_size].append((row, col))
        for wave_id, tiles in wave_tiles.items():
            avg_row = sum(r for r, c in tiles) / len(tiles)
            avg_col = sum(c for r, c in tiles) / len(tiles)
            cx = avg_col * cell + cell / 2
            cy = avg_row * cell + cell / 2
            elements.append(make_text(cx, cy, str(wave_id), font_size=56))

    return {
        "type": "excalidraw", "version": 2, "source": "python-gen",
        "elements": elements,
        "appState": {"viewBackgroundColor": "#ffffff"}, "files": {},
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, default=8)
    parser.add_argument("-n", type=int, default=8)
    parser.add_argument("-w", "--wave-size", type=int, default=None,
                        help="Color tiles by wave (e.g. 132 for H100)")
    parser.add_argument("-o", "--output", default="cta_hilbert.excalidraw")
    args = parser.parse_args()

    scene = generate(args.m, args.n, wave_size=args.wave_size)
    with open(args.output, "w") as f:
        json.dump(scene, f, indent=2)
    print(f"Wrote {args.output}  ({args.m}x{args.n}, {args.m * args.n} tiles)")
