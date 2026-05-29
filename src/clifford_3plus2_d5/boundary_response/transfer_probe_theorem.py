"""V25 transfer-probe compatibility theorem.

Earlier gates evaluate the half-line Weyl function at ``z = 2 sqrt(2)``.
V25 sharpens that choice.  For the unit semi-infinite sterile chain, the
decaying Weyl transfer factor ``m`` and exterior resolvent probe ``z`` obey

    m + 1/m = z.

Therefore the residual transfer factor ``epsilon = sqrt(2) - 1`` uniquely
selects

    z = epsilon + epsilon^-1 = 2 sqrt(2).

This is a compatibility theorem, not a derivation of the unit-chain
normalization.  If the sterile hopping is scaled by ``t``, the compatible
probe scales to ``t (epsilon + epsilon^-1)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from clifford_3plus2_d5.boundary_response.explicit_hq import transfer_probe
from clifford_3plus2_d5.boundary_response.transfer import (
    epsilon,
    transfer_polynomial,
)
from clifford_3plus2_d5.boundary_response.weyl_sterile import (
    semi_infinite_weyl_function,
    weyl_fixed_point_residual,
)

REMAINING_DECLARED_INPUTS_AFTER_TRANSFER_PROBE = (
    "vacuum_framing",
    "unit_sterile_chain_normalization",
    "regular_boundary_fiber_or_max_entropy_prior",
)


def probe_from_transfer_factor(transfer_factor: sp.Expr) -> sp.Expr:
    """Return the exterior unit-chain probe compatible with ``m``."""

    selected = sp.sympify(transfer_factor)
    return sp.simplify(selected + 1 / selected)


def transfer_factor_from_probe(z_probe: sp.Expr) -> sp.Expr:
    """Return the decaying unit-chain transfer factor at ``z``."""

    return semi_infinite_weyl_function(z_probe)


def transfer_probe_uniqueness_residual(
    transfer_factor: sp.Expr,
    z_probe: sp.Expr,
) -> sp.Expr:
    """Return the residual of ``z = m + 1/m``."""

    return sp.simplify(z_probe - probe_from_transfer_factor(transfer_factor))


def reciprocal_transfer_branch(transfer_factor: sp.Expr) -> sp.Expr:
    """Return the non-decaying reciprocal Weyl root."""

    return sp.simplify(1 / sp.sympify(transfer_factor))


def reciprocal_branch_rejected(transfer_factor: sp.Expr) -> bool:
    """Return true when the reciprocal root is outside the decaying unit disk."""

    reciprocal = reciprocal_transfer_branch(transfer_factor)
    return bool(sp.N(reciprocal) > 1)


def scaled_probe_from_transfer_factor(
    transfer_factor: sp.Expr,
    hopping: sp.Expr,
) -> sp.Expr:
    """Return the compatible probe for a chain with hopping scale ``t``."""

    return sp.simplify(sp.sympify(hopping) * probe_from_transfer_factor(transfer_factor))


def scaled_probe_control_residual(
    transfer_factor: sp.Expr,
    hopping: sp.Expr,
) -> sp.Expr:
    """Return how far the scaled compatible probe is from the unit probe."""

    return sp.simplify(
        scaled_probe_from_transfer_factor(transfer_factor, hopping)
        - probe_from_transfer_factor(transfer_factor)
    )


@dataclass(frozen=True)
class TransferProbeTheoremAuditPayload:
    """Verdict payload for the V25 transfer-probe theorem."""

    final_verdict: str
    epsilon_value: sp.Expr
    derived_probe: sp.Expr
    existing_probe: sp.Expr
    recurrence_residual: sp.Expr
    weyl_value_at_probe: sp.Expr
    fixed_point_residual_at_probe: sp.Expr
    reciprocal_branch: sp.Expr
    reciprocal_branch_rejected: bool
    symbolic_uniqueness_residual: sp.Expr
    scaled_probe_control: sp.Expr
    unit_normalization_load_bearing: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def transfer_probe_theorem_audit_payload() -> TransferProbeTheoremAuditPayload:
    """Return the V25 transfer-probe compatibility verdict."""

    eps = epsilon()
    derived_probe = probe_from_transfer_factor(eps)
    existing_probe = transfer_probe()
    recurrence_residual = sp.simplify(transfer_polynomial(eps))
    weyl_value = transfer_factor_from_probe(derived_probe)
    fixed_point_residual = weyl_fixed_point_residual(derived_probe)
    reciprocal = reciprocal_transfer_branch(eps)
    reciprocal_rejected = reciprocal_branch_rejected(eps)

    z_symbol = sp.symbols("z")
    uniqueness_residual = transfer_probe_uniqueness_residual(eps, z_symbol)
    hopping = sp.symbols("t", positive=True)
    scaled_probe = scaled_probe_from_transfer_factor(eps, hopping)
    scaled_control = scaled_probe_control_residual(eps, hopping)
    unit_normalization_load_bearing = sp.simplify(
        scaled_control - (hopping - 1) * derived_probe
    ) == 0

    checks_pass = (
        recurrence_residual == 0
        and sp.simplify(derived_probe - existing_probe) == 0
        and sp.simplify(weyl_value - eps) == 0
        and fixed_point_residual == 0
        and reciprocal_rejected
        and sp.simplify(uniqueness_residual.subs(z_symbol, derived_probe)) == 0
        and sp.simplify(scaled_probe - hopping * existing_probe) == 0
        and unit_normalization_load_bearing
    )

    if checks_pass:
        final_verdict = "TRANSFER_PROBE_COMPATIBILITY_PASS"
        interpretation = (
            "The residual transfer factor epsilon uniquely selects the "
            "exterior unit-chain Weyl probe z = epsilon + epsilon^-1 = "
            "2 sqrt(2). At this probe, the decaying Weyl function returns "
            "epsilon exactly; the reciprocal branch is non-decaying and is "
            "rejected. A scaled hopping t moves the probe to t times this "
            "value, so the unit-chain normalization remains a named input."
        )
    else:
        final_verdict = "TRANSFER_PROBE_COMPATIBILITY_KILL"
        interpretation = (
            "The recurrence, probe equality, Weyl value, branch rejection, "
            "symbolic uniqueness, or scaled-chain control failed."
        )

    return TransferProbeTheoremAuditPayload(
        final_verdict=final_verdict,
        epsilon_value=eps,
        derived_probe=derived_probe,
        existing_probe=existing_probe,
        recurrence_residual=recurrence_residual,
        weyl_value_at_probe=weyl_value,
        fixed_point_residual_at_probe=fixed_point_residual,
        reciprocal_branch=reciprocal,
        reciprocal_branch_rejected=reciprocal_rejected,
        symbolic_uniqueness_residual=uniqueness_residual,
        scaled_probe_control=scaled_probe,
        unit_normalization_load_bearing=unit_normalization_load_bearing,
        remaining_declared_inputs=REMAINING_DECLARED_INPUTS_AFTER_TRANSFER_PROBE,
        interpretation=interpretation,
    )
