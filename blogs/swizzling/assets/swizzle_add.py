import numpy as np

ROWS, COLS = 8, 8
ELEM_BYTES = 4
B, M, S = 3, 2, 3  # 3 column bits, starting at bit 2; row bits start 3 bits above that

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"

COLORS = [
    "\033[30;42m",   # logical col 0
    "\033[30;43m",   # logical col 1
    "\033[37;44m",   # logical col 2
    "\033[37;45m",   # logical col 3
    "\033[30;46m",   # logical col 4
    "\033[37;41m",   # logical col 5
    "\033[30;47m",   # logical col 6
    "\033[37;100m",  # logical col 7
]


def rc_to_addr(r, c, cols, elem_bytes):
    return r * cols * elem_bytes + c * elem_bytes


def addr_to_rc(addr, cols, elem_bytes):
    row_bytes = cols * elem_bytes
    r = addr // row_bytes
    c = (addr % row_bytes) // elem_bytes
    return r, c


def swizzle_add_addr(addr, b, m, s):
    mask = (1 << b) - 1

    col = (addr >> m) & mask
    row = (addr >> (m + s)) & mask

    swizzled_col = (col + row) & mask  # modulo 2^b

    return (addr & ~(mask << m)) | (swizzled_col << m)


def unswizzle_add_addr(addr, b, m, s):
    mask = (1 << b) - 1

    phys_col = (addr >> m) & mask
    row = (addr >> (m + s)) & mask

    logical_col = (phys_col - row) & mask

    return (addr & ~(mask << m)) | (logical_col << m)


def bank_of_addr(addr):
    return (addr // ELEM_BYTES) % 32


def color_value(val):
    logical_col = val % 10
    return f"{COLORS[logical_col]} {val:2d} {RESET}"


def print_grid(grid, label):
    print(f"\n{BOLD}{label}{RESET}")
    print("      " + "".join(f"{c:^6d}" for c in range(COLS)))

    for r in range(ROWS):
        row = f"{r:3d}   "
        for c in range(COLS):
            row += color_value(grid[r][c]) + " "
        print(row)


def print_legend():
    print(f"\n{BOLD}Legend: colors track logical columns{RESET}")
    row = "      "
    for c in range(COLS):
        row += f"{COLORS[c]} col {c} {RESET} "
    print(row)


def bank_trace(logical_col, use_swizzle):
    entries = []

    for r in range(ROWS):
        logical_addr = rc_to_addr(r, logical_col, COLS, ELEM_BYTES)

        if use_swizzle:
            physical_addr = swizzle_add_addr(logical_addr, B, M, S)
        else:
            physical_addr = logical_addr

        pr, pc = addr_to_rc(physical_addr, COLS, ELEM_BYTES)
        bank = bank_of_addr(physical_addr)

        entries.append((r, logical_col, pr, pc, bank))

    return entries


def print_bank_trace(label, entries):
    counts = {}

    for _, _, _, _, bank in entries:
        counts[bank] = counts.get(bank, 0) + 1

    print(f"\n{BOLD}{label}{RESET}")

    for r, c, pr, pc, bank in entries:
        color = RED if counts[bank] > 1 else GREEN
        status = "conflict" if counts[bank] > 1 else "ok"

        print(
            f"  logical ({r},{c})"
            f" -> physical ({pr},{pc})"
            f" -> {color}bank {bank:2d} {status}{RESET}"
        )

    banks = [bank for _, _, _, _, bank in entries]
    conflict_count = len(banks) - len(set(banks))

    if conflict_count:
        print(f"banks touched: {banks}  {RED}CONFLICTS{RESET}")
    else:
        print(f"banks touched: {banks}  {GREEN}NO CONFLICTS{RESET}")


logical = np.array([[10 * r + c for c in range(COLS)] for r in range(ROWS)])

# Write logical data into physical SMEM using add-swizzle.
smem = np.full((ROWS, COLS), -1)

for r in range(ROWS):
    for c in range(COLS):
        logical_addr = rc_to_addr(r, c, COLS, ELEM_BYTES)
        physical_addr = swizzle_add_addr(logical_addr, B, M, S)

        pr, pc = addr_to_rc(physical_addr, COLS, ELEM_BYTES)
        smem[pr][pc] = logical[r][c]

# Read logical data back by applying the same logical -> physical swizzle.
read_back = np.full((ROWS, COLS), -1)

for r in range(ROWS):
    for c in range(COLS):
        logical_addr = rc_to_addr(r, c, COLS, ELEM_BYTES)
        physical_addr = swizzle_add_addr(logical_addr, B, M, S)

        pr, pc = addr_to_rc(physical_addr, COLS, ELEM_BYTES)
        read_back[r][c] = smem[pr][pc]

assert np.array_equal(logical, read_back)

print_legend()
print_grid(logical, "LOGICAL DATA")
print_grid(smem, "PHYSICAL SMEM AFTER ADD-SWIZZLE: physical_col = logical_col + row mod 8")
print_grid(read_back, "READ BACK THROUGH SAME SWIZZLE")

logical_col = 0

print_bank_trace(
    f"READING LOGICAL COLUMN {logical_col} WITHOUT SWIZZLE",
    bank_trace(logical_col, use_swizzle=False),
)

print_bank_trace(
    f"READING LOGICAL COLUMN {logical_col} WITH ADD-SWIZZLE",
    bank_trace(logical_col, use_swizzle=True),
)

print(f"\n{BOLD}One concrete value trace{RESET}")

r, c = 3, 0
logical_addr = rc_to_addr(r, c, COLS, ELEM_BYTES)
physical_addr = swizzle_add_addr(logical_addr, B, M, S)
pr, pc = addr_to_rc(physical_addr, COLS, ELEM_BYTES)

print(f"  logical value {logical[r][c]} starts at logical ({r},{c})")
print(f"  logical addr   = {logical_addr:3d} / 0b{logical_addr:08b}")
print(f"  physical addr  = {physical_addr:3d} / 0b{physical_addr:08b}")
print(f"  lands in SMEM  = physical ({pr},{pc})")
print(f"  bank touched   = {bank_of_addr(physical_addr)}")

unswizzled_addr = unswizzle_add_addr(physical_addr, B, M, S)
ur, uc = addr_to_rc(unswizzled_addr, COLS, ELEM_BYTES)

print(f"  inverse mapping says physical ({pr},{pc}) came from logical ({ur},{uc})")