ROWS = 8
COLS = 8
PAD = 3
ELEM_BYTES = 4
BANK_WIDTH = 4
NUM_BANKS = 8  # 8 banks so colors map 1:1

padded_cols = COLS + PAD

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

def bank_of(row, col, stride):
    addr = row * stride * ELEM_BYTES + col * ELEM_BYTES
    return (addr // BANK_WIDTH) % NUM_BANKS

def print_bank_grid(label, stride, cols):
    print(f"\n{label}")
    print("     " + "".join(f"  {c}  " for c in range(cols)))
    for r in range(ROWS):
        row_str = f"  {r}  "
        for c in range(cols):
            b = bank_of(r, c, stride)
            row_str += f"{COLORS[b]} {b:2d} {RESET} "
        print(row_str)

print_bank_grid(
    f"Bank per cell WITHOUT padding ({ROWS}x{COLS}, stride={COLS}, {NUM_BANKS} banks):",
    stride=COLS, cols=COLS,
)
print_bank_grid(
    f"Bank per cell WITH padding (pad={PAD}, stride={padded_cols}, {NUM_BANKS} banks):",
    stride=padded_cols, cols=COLS,
)

for label, stride in [("WITHOUT", COLS), ("WITH", padded_cols)]:
    print(f"\nColumn-0 banks {label} padding (stride={stride}):")
    col0 = set()
    for r in range(ROWS):
        b = bank_of(r, 0, stride)
        col0.add(b)
        print(f"  row {r}: bank={b}")
    conflict = "no conflict" if len(col0) == ROWS else f"{ROWS // len(col0)}-way conflict"
    print(f"  -> {len(col0)} unique banks, {ROWS} rows = {conflict}")
