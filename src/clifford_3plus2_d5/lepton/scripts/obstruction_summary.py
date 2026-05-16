"""Print the consolidated lepton obstruction map as JSON."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from clifford_3plus2_d5.lepton.obstruction_map import (
    obstruction_entries,
    obstruction_summary,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entries",
        action="store_true",
        help="Print full obstruction entries in addition to the summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload: dict[str, object] = {"summary": obstruction_summary()}
    if args.entries:
        payload["entries"] = tuple(asdict(entry) for entry in obstruction_entries())
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
