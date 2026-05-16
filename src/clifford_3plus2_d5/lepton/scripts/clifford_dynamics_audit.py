"""Print the Session 14 rigid Clifford dynamics audit as JSON."""

from __future__ import annotations

import argparse
import json

from clifford_3plus2_d5.lepton.clifford_dynamics import (
    clifford_dynamics_audit_payload,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entries",
        action="store_true",
        help="include per-candidate audit rows",
    )
    args = parser.parse_args()
    print(json.dumps(clifford_dynamics_audit_payload(include_entries=args.entries), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
