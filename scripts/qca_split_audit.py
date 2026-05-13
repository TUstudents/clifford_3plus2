from __future__ import annotations

import argparse
import json
from pathlib import Path

from clifford_3plus2_d5.clifford_audit import (
    DEFAULT_QCA_DATA_PATH,
    audit_qca_split,
    audit_to_dict,
)
from clifford_3plus2_d5.status import BridgeVerdict


VERDICT_CHOICES: tuple[BridgeVerdict, ...] = (
    "structural_bridge",
    "conditional_bridge",
    "notation_only",
    "falsified",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether QCA data supplies the 3+2 split.")
    parser.add_argument("--check", action="store_true", help="Run as a reproducible check.")
    parser.add_argument("--data", type=Path, default=DEFAULT_QCA_DATA_PATH)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument(
        "--expect-verdict",
        choices=VERDICT_CHOICES,
        help="Exit nonzero unless the audit returns this verdict.",
    )
    args = parser.parse_args()

    audit = audit_qca_split(args.data)
    load_bearing = audit.verdict == "structural_bridge"

    if args.json:
        print(json.dumps(audit_to_dict(audit), indent=2, sort_keys=True))
    else:
        print("This audits the load-bearing QCA-to-3+2 bridge claim.")
        print("No qca_data.json, no bridge claim.")
        print(f"qca_split_audit_verdict: {audit.verdict}")
        print(
            "qca_supplies_structural_3plus2_split: "
            f"{str(audit.qca_supplies_structural_3plus2_split).lower()}"
        )
        print(
            "complex_structure_compatible_with_3plus2_split: "
            f"{str(audit.complex_structure_compatible_with_3plus2_split).lower()}"
        )
        print(f"load_bearing_qca_bridge: {str(load_bearing).lower()}")

    if args.expect_verdict is not None and audit.verdict != args.expect_verdict:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
