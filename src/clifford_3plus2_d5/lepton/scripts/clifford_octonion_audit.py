"""Print the Cl(0,8) / octonion stabilizer audit as JSON."""

from __future__ import annotations

import json

from clifford_3plus2_d5.lepton.clifford_octonion import (
    clifford_octonion_audit_payload,
)


def main() -> int:
    print(json.dumps(clifford_octonion_audit_payload(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
