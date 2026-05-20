"""Run the Session 46 tiny spacetime-QCA simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from clifford_3plus2_d5.spacetime_qca.jax_runner import (
    SimulationRunConfig,
    run_simulation,
    save_simulation_result,
    simulation_summary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument("--step-size", type=float, default=0.0025)
    parser.add_argument("--sector", choices=("u1_y", "su2_l", "sm"), default="u1_y")
    parser.add_argument("--yukawa-mode", choices=("first_order", "unitary"), default="unitary")
    parser.add_argument("--output", type=Path, default=None, help="Optional .npz output path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = SimulationRunConfig(
        sector=args.sector,
        steps=args.steps,
        record_every=args.record_every,
        step_size=args.step_size,
        yukawa_mode=args.yukawa_mode,
    )
    result = run_simulation(config)
    summary = simulation_summary(result)

    if args.output is not None:
        npz_path, json_path = save_simulation_result(result, args.output)
        summary["output"] = str(npz_path)
        summary["metadata"] = str(json_path)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
