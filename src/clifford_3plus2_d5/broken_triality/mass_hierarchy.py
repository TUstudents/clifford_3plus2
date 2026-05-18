"""BT-2 kill test: mass hierarchy from the BT-1 Yukawa.

Construction: diagonalize the BT-1 Yukawa.  Compute ``lambda_max /
lambda_min`` where ``lambda`` ranges over the **non-zero** eigenvalues.
Zero eigenvalues are excluded because they correspond to "extra"
massless generations from rank deficits in the construction, not to
small but non-zero physical masses.

Pass condition: ``ratio >= 100``.  This is a generous threshold compared
to the observed SM ratios (~10^2 to 10^5 within a sector); it asks only
whether the structure is hierarchical at all.

Fail condition: ``ratio < 10`` (essentially flat) **or** fewer than two
non-zero eigenvalues (cannot define a hierarchy).

With the default ``v_* = Y'``, the BT-1 Yukawa has eigenvalues
``{5/7, 31/72, 0}``.  Excluding zero, the ratio is ``(5/7) / (31/72) =
360/217 ≈ 1.66``, which is far below the fail threshold of 10.  This
module formalizes that observation.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.broken_triality.yukawa_overlaps import (
    yukawa_nonzero_eigenvalues,
)

PASS_THRESHOLD = sp.Integer(100)
FAIL_THRESHOLD = sp.Integer(10)


def nonzero_eigenvalue_ratio() -> sp.Expr | None:
    """Return ``lambda_max / lambda_min`` over non-zero eigenvalues, or ``None``."""

    values = yukawa_nonzero_eigenvalues()
    if len(values) < 2:
        return None
    return sp.simplify(values[0] / values[-1])


def nonzero_eigenvalue_count() -> int:
    return len(yukawa_nonzero_eigenvalues())


def ratio_passes_pass_threshold() -> bool:
    ratio = nonzero_eigenvalue_ratio()
    if ratio is None:
        return False
    return sp.simplify(ratio - PASS_THRESHOLD) >= 0


def ratio_below_fail_threshold() -> bool:
    ratio = nonzero_eigenvalue_ratio()
    if ratio is None:
        return True
    return sp.simplify(ratio - FAIL_THRESHOLD) < 0


@dataclass(frozen=True)
class BT2Audit:
    nonzero_eigenvalue_count: int
    nonzero_eigenvalues: tuple[sp.Expr, ...]
    nonzero_ratio: sp.Expr | None
    nonzero_ratio_float: float | None
    pass_threshold: sp.Expr
    fail_threshold: sp.Expr
    passes: bool
    verdict: str
    interpretation: str


def bt2_audit_payload() -> BT2Audit:
    values = yukawa_nonzero_eigenvalues()
    count = len(values)
    ratio = nonzero_eigenvalue_ratio()
    ratio_float = float(ratio) if ratio is not None else None

    if count < 2:
        passes = False
        verdict = "BT-2 FAIL"
        interpretation = (
            f"Fewer than two non-zero eigenvalues ({count}); cannot define "
            f"a mass hierarchy.  Program closes at BT-2."
        )
    elif ratio_below_fail_threshold():
        passes = False
        verdict = "BT-2 FAIL"
        interpretation = (
            f"Non-zero eigenvalue ratio {ratio} approximately {ratio_float:.3f} "
            f"is below the fail threshold {FAIL_THRESHOLD}.  The triality "
            f"projection gives an essentially flat (O(1)) spectrum on the "
            f"non-zero generations.  Adding higher-order breaking could in "
            f"principle enhance the spread, but the pure-triality-projection "
            f"structure does not produce mass hierarchy.  Program closes "
            f"at BT-2."
        )
    elif ratio_passes_pass_threshold():
        passes = True
        verdict = "BT-2 PASS"
        interpretation = (
            f"Non-zero eigenvalue ratio {ratio} approximately {ratio_float:.3f} "
            f"is above the pass threshold {PASS_THRESHOLD}.  Mass hierarchy "
            f"is structurally available."
        )
    else:
        passes = False
        verdict = "BT-2 FAIL (marginal)"
        interpretation = (
            f"Non-zero eigenvalue ratio {ratio} approximately {ratio_float:.3f} "
            f"is between fail ({FAIL_THRESHOLD}) and pass ({PASS_THRESHOLD}) "
            f"thresholds.  Marginal failure; default disposition is to close "
            f"the program."
        )

    return BT2Audit(
        nonzero_eigenvalue_count=count,
        nonzero_eigenvalues=values,
        nonzero_ratio=ratio,
        nonzero_ratio_float=ratio_float,
        pass_threshold=PASS_THRESHOLD,
        fail_threshold=FAIL_THRESHOLD,
        passes=passes,
        verdict=verdict,
        interpretation=interpretation,
    )
