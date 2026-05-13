from __future__ import annotations

import argparse
from pathlib import Path

from clifford_3plus2_d5.clifford_audit import DEFAULT_QCA_DATA_PATH, audit_qca_split


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether QCA data supplies the 3+2 split.")
    parser.add_argument("--check", action="store_true", help="Run as a reproducible check.")
    parser.add_argument("--data", type=Path, default=DEFAULT_QCA_DATA_PATH)
    args = parser.parse_args()

    audit = audit_qca_split(args.data)
    load_bearing = audit.verdict == "structural_bridge"

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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
