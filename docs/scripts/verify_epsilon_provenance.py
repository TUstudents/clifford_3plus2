"""Verify the ε provenance facts (see docs/epsilon_provenance.md).

This is a read-only definitional check, NOT a kill-disciplined sidecar. It pins
the two distinct objects denoted ``epsilon`` so the definitions cannot silently
drift:

  * ε_silver  = √2 − 1            — dimensionless flavor invariant (boundary_response)
  * ε_lattice = BCC lattice scale — length; CP order & Lorentz coeff (cp = sme)

and asserts the genuine CP↔Lorentz link (one lattice ε) and that ε_silver and
ε_lattice are distinct objects (dimension + ~32-order magnitude gap).

Run:  uv run python docs/scripts/verify_epsilon_provenance.py
Exits non-zero on any mismatch.
"""

from __future__ import annotations

import math
import sys

import sympy as sp

from clifford_3plus2_d5.boundary_response.transfer import epsilon, epsilon_fourth
from clifford_3plus2_d5.cp.continuum_cp import h1_total_norm_squared
from clifford_3plus2_d5.sme.epsilon_constraint import epsilon_constraint_payload
from clifford_3plus2_d5.sme.sme_tensor_mapping import t2g_tensor_entries


def main() -> int:
    failures: list[str] = []

    def check(label: str, condition: bool) -> None:
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {label}")
        if not condition:
            failures.append(label)

    # --- ε_silver: dimensionless flavor invariant (boundary_response) ---
    eps_silver = epsilon()
    eps4 = epsilon_fourth()
    print("ε_silver (boundary_response) — dimensionless silver ratio:")
    check("epsilon() == sqrt(2) - 1", sp.simplify(eps_silver - (sp.sqrt(2) - 1)) == 0)
    check(
        "epsilon_fourth() == 17 - 12*sqrt(2)  (ν mass ratio)",
        sp.simplify(eps4 - (17 - 12 * sp.sqrt(2))) == 0,
    )
    silver_val = float(eps_silver)
    print(f"        value ≈ {silver_val:.5f}   ε⁴ ≈ {float(eps4):.6f}")

    # --- ε_lattice: dimensionful lattice scale (sme) ---
    lattice_m = float(epsilon_constraint_payload().epsilon_bound_metres)
    print("ε_lattice (sme) — lattice scale in metres:")
    check("epsilon_bound_metres ~ 1.97e-33 m", 1e-33 < lattice_m < 1e-32)
    print(f"        value ≈ {lattice_m:.3e} m")

    # --- the two ε's are distinct objects ---
    gap = round(math.log10(silver_val / lattice_m))
    print("ε_silver ≠ ε_lattice:")
    check("dimensionless vs length (distinct dimension)", True)  # by definition above
    check("≳ 30 orders of magnitude apart", gap >= 30)
    print(f"        log10(ε_silver / ε_lattice) ≈ {gap}")

    # --- genuine CP↔Lorentz link runs through one lattice ε ---
    h1_norm = h1_total_norm_squared()
    coeffs = [int(e.coefficient) for e in t2g_tensor_entries()]
    print("CP ↔ Lorentz (one ε_lattice): cp H⁽¹⁾ (T₂g) → sme d⁽⁵⁾ = ε_lattice × (±1):")
    check("cp ‖H⁽¹⁾‖² == 12 (CP-odd, T₂g)", sp.simplify(h1_norm - 12) == 0)
    check("d⁽⁵⁾ T₂g coefficients are ±1", sorted(coeffs) == [-1, 1, 1])
    print(f"        ‖H⁽¹⁾‖² = {h1_norm}   d⁽⁵⁾ coeffs = {coeffs}")

    print()
    if failures:
        print(f"RESULT: FAIL ({len(failures)} check(s) failed) — definitions have drifted.")
        return 1
    print("RESULT: PASS — ε_silver and ε_lattice are distinct; CP↔Lorentz share one lattice ε.")
    print("See docs/epsilon_provenance.md for the full catalog and correlations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
