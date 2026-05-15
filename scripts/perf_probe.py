from __future__ import annotations

import argparse
import time

from clifford_3plus2_d5.qca.bloch_rule import (
    bloch_path_a_projector_free_combined_layer,
)
from clifford_3plus2_d5.qca.rule_verdict import (
    bloch_floquet_operators,
    center_basis_of_algebra,
    generated_algebra_closure,
)


def _elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.3f}s"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run focused performance probes for active exact QCA kernels."
    )
    parser.add_argument(
        "--max-algebra-dim",
        type=int,
        default=16,
        help="Stop algebra closure once this many basis elements have been found.",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Also compute the center when algebra closure finishes below the cap.",
    )
    args = parser.parse_args()

    layer = bloch_path_a_projector_free_combined_layer()
    start = time.perf_counter()
    samples = bloch_floquet_operators((layer,), bloch_period=12)
    print(f"bloch_samples: {len(samples)} in {_elapsed(start)}")

    start = time.perf_counter()
    closure = generated_algebra_closure(
        samples,
        max_dimension=args.max_algebra_dim,
    )
    print(
        "generated_algebra: "
        f"dimension={len(closure.basis)} "
        f"closed={str(closure.closed).lower()} "
        f"time={_elapsed(start)}"
    )

    if args.center and closure.closed:
        start = time.perf_counter()
        center = center_basis_of_algebra(closure.basis)
        print(f"center: dimension={len(center)} time={_elapsed(start)}")
    elif args.center:
        print("center: skipped because algebra did not close below cap")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
