import numpy as np

def swizzle_int(addr, b, m, s):
    bit_mask = (1 << b) - 1
    yyy_msk = bit_mask << (m + s)
    return addr ^ ((addr & yyy_msk) >> s)

def addr_to_rc(addr, cols, elem_bytes):
    row_bytes = cols * elem_bytes
    r = addr // row_bytes
    c = (addr % row_bytes) // elem_bytes
    return r, c

ROWS, COLS = 8, 8
ELEM_BYTES = 4

before = np.array([[c for c in range(COLS)] for _ in range(ROWS)])

after = np.full((ROWS, COLS), -1)
for r in range(ROWS):
    for c in range(COLS):
        orig_addr = r * COLS * ELEM_BYTES + c * ELEM_BYTES
        new_addr = swizzle_int(orig_addr, b=3, m=2, s=3)
        nr, nc = addr_to_rc(new_addr, COLS, ELEM_BYTES)
        after[nr][nc] = before[r][c]

# ---- display ----
COLORS = [
    "\033[42m",   # 0: green
    "\033[43m",   # 1: yellow
    "\033[44m",   # 2: blue
    "\033[45m",   # 3: magenta
    "\033[46m",   # 4: cyan
    "\033[41m",   # 5: red
    "\033[47m",   # 6: white
    "\033[100m",  # 7: gray
]
RESET = "\033[0m"

def print_grid(grid, label):
    print(f"\n{label}")
    print("     " + "".join(f"  {c}  " for c in range(COLS)))
    for r in range(ROWS):
        row_str = f"  {r}  "
        for c in range(COLS):
            val = grid[r][c]
            row_str += f"{COLORS[val % len(COLORS)]} {val:2d} {RESET} "
        print(row_str)

print_grid(before, "BEFORE (column index = data value):")
print_grid(after, "AFTER swizzle_int(addr, b=3, m=2, s=3):")

r, c = 3, 0
orig_addr = r * COLS * ELEM_BYTES + c * ELEM_BYTES
new_addr = swizzle_int(orig_addr, b=3, m=2, s=3)
nr, nc = addr_to_rc(new_addr, COLS, ELEM_BYTES)
print(f"\nTrace: data at ({r},{c}) has addr {orig_addr} (0b{orig_addr:08b})")
print(f"  swizzled addr = {new_addr} (0b{new_addr:08b})")
print(f"  lands at ({nr},{nc})")
print(f"  value is still {before[r][c]}")