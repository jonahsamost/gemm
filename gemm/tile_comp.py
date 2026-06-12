"""
Sweep CTA tile (M, N=128, K) and benchmark via run.run().

  cd gemm && python tile_comp.py

Tile M: 64, 128, ..., 512 (step 64). K tile = 16 * k_depth for k_depth in 4..16.
Problem sizes default to M=4096, N=128, K=8192.
"""

from tile_scheduler import CTATileOrdering, PersistenceMode
from gemm.run import run

# Problem dimensions (fixed across sweep)
M_PROBLEM = N_PROBLEM = K_PROBLEM = 8192

TILE_M_VALUES = range(64, 320 + 64, 64)
TILE_N = 128
K_DEPTHS = range(2, 17, 2)

ORDERING = CTATileOrdering.ONLINE_SWIZZLE
PERSISTENCE = PersistenceMode.DYNAMIC
CLUSTER_SHAPE = (1, 1, 1)
GROUP_SIZE = 8


def main():
    print(
        f"problem M={M_PROBLEM} N={N_PROBLEM} K={K_PROBLEM}  "
        f"ordering={ORDERING.name} persistence={PERSISTENCE.name}"
    )
    for tile_m in TILE_M_VALUES:
        for k_depth in K_DEPTHS:
            tile_k = 16 * k_depth
            tile_shape = (tile_m, TILE_N, tile_k)
            print(f"\n{'=' * 60}")
            print(f"tile_shape={tile_shape}  (k_depth={k_depth})")
            print("=" * 60)
            try:
                run(
                    M_PROBLEM,
                    N_PROBLEM,
                    K_PROBLEM,
                    ORDERING,
                    PERSISTENCE,
                    tile_shape=tile_shape,
                    cluster_shape=CLUSTER_SHAPE,
                    group_size=GROUP_SIZE,
                    bench=True,
                )
            except Exception as e:
                print(f"FAILED tile_shape={tile_shape}: {e}")


if __name__ == "__main__":
    main()
