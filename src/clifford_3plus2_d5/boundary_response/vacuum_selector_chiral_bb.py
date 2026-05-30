"""V35 chiral BB filled-band selector-sign audit.

V34 shows that the oriented tetrahedral shell produces the ``A2u`` cubic
``C(h)=8xyz/sqrt(3)`` at cubic order.  V35 asks whether the actual
Bialynicki-Birula single-Weyl walk promotes the same helicity-odd cubic into
the real filled-quasienergy contribution.

Three diagnostics are kept separate:

1. The Floquet trace coefficient of the ``epsilon^3 kx ky kz`` term.  This is
   helicity-odd for a single Weyl walk and cancels for the Dirac pair.
2. The scalar real contribution extracted from the matrix-log effective
   Hamiltonian.  This parity-even polynomial trace probe is intentionally kept
   as a negative control: it cannot see the ``A2u`` selector.
3. The actual filled-band quasienergy, computed from Bloch eigenphases.  This
   probe sees the real helicity-locked angular ``A2u`` selector, while the
   vector/Dirac pair cancels it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Literal

import numpy as np
import sympy as sp

from clifford_3plus2_d5.boundary_response.vacuum_selector_condensation import (
    order_parameter_symbols,
    tetrahedral_cubic_polynomial,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_landau import (
    tetrahedral_rotation_group,
    transform_polynomial,
)
from clifford_3plus2_d5.boundary_response.vacuum_selector_schur_landau import (
    schur_shell_landau_audit_payload,
)
from clifford_3plus2_d5.spacetime_qca.bcc_weyl import (
    bcc_dirac_symbol,
    bcc_weyl_symbol,
    expected_weyl_hamiltonian,
)
from clifford_3plus2_d5.spacetime_qca.continuum import nth_order_in_epsilon
from clifford_3plus2_d5.spacetime_qca.dirac import block_diag

Sector = Literal["right", "left", "dirac"]
QuadraticCoefficientTuple = tuple[sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr, sp.Expr]
MomentumSample = tuple[float, float, float]

FILLED_BAND_TEST_EPSILON = 1.0e-3
FILLED_BAND_ZERO_TOLERANCE = 1.0e-9
FILLED_BAND_RATIO_TOLERANCE = 1.0e-6
FILLED_BAND_NONPOLYNOMIAL_GAP = 1.0e-2

REMAINING_DECLARED_INPUTS_AFTER_CHIRAL_BB_SELECTOR_SIGN: tuple[str, ...] = ()

A2U_SIGNED_ORBIT_SAMPLES: tuple[MomentumSample, ...] = (
    (1.0, 2.0, 3.0),
    (2.0, 1.0, 3.0),
    (3.0, 2.0, 1.0),
    (-1.0, 2.0, 3.0),
    (1.0, -2.0, 3.0),
    (1.0, 2.0, -3.0),
    (-1.0, -2.0, 3.0),
)

A2U_ZERO_LOCUS_SAMPLES: tuple[MomentumSample, ...] = (
    (1.0, 1.0, 0.0),
    (1.0, 0.0, 2.0),
    (0.0, 2.0, 3.0),
)

A2U_ANGULAR_CONTROL_SAMPLES: tuple[MomentumSample, ...] = (
    (1.0, 2.0, 3.0),
    (1.0, 2.0, 6.0),
    (1.0, 1.0, 1.0),
)


def _same_expr(left: sp.Expr, right: sp.Expr) -> bool:
    return sp.simplify(left - right) == 0


def _zero_matrix_like(matrix: sp.Matrix) -> sp.Matrix:
    return sp.zeros(matrix.rows, matrix.cols)


def _validate_sector(sector: str) -> Sector:
    if sector not in {"right", "left", "dirac"}:
        raise ValueError("sector must be 'right', 'left', or 'dirac'")
    return sector  # type: ignore[return-value]


def symbolic_momentum() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol, sp.Symbol]:
    """Return ``(epsilon,kx,ky,kz)`` with real momentum symbols."""

    epsilon = sp.symbols("epsilon", positive=True)
    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    return epsilon, kx, ky, kz


@cache
def bb_bloch_symbol(sector: Sector) -> sp.Matrix:
    """Return the exact BB Bloch symbol for a chiral sector or Dirac pair."""

    sector = _validate_sector(sector)
    epsilon, kx, ky, kz = symbolic_momentum()
    if sector == "dirac":
        return bcc_dirac_symbol(epsilon, kx, ky, kz)
    return bcc_weyl_symbol(epsilon, kx, ky, kz, helicity=sector)


@cache
def bb_bloch_order(sector: Sector, order: int) -> sp.Matrix:
    """Return the coefficient of ``epsilon**order`` in the BB Bloch symbol."""

    if order < 0:
        raise ValueError("order must be nonnegative")
    epsilon, _kx, _ky, _kz = symbolic_momentum()
    return nth_order_in_epsilon(bb_bloch_symbol(sector), epsilon, order)


def _xyz_coefficient(polynomial: sp.Expr) -> sp.Expr:
    _epsilon, kx, ky, kz = symbolic_momentum()
    return sp.simplify(sp.expand(polynomial).coeff(kx, 1).coeff(ky, 1).coeff(kz, 1))


def bb_trace_b3_xyz_coefficient(sector: Sector) -> sp.Expr:
    """Return the ``kx*ky*kz`` coefficient of ``Tr B_3``."""

    sector = _validate_sector(sector)
    return _xyz_coefficient(sp.trace(bb_bloch_order(sector, 3)))


def trace_b3_diagnostic_passes() -> bool:
    """Return true for the known helicity-odd Floquet-trace cubic diagnostic."""

    return (
        bb_trace_b3_xyz_coefficient("right") == -2
        and bb_trace_b3_xyz_coefficient("left") == 2
        and bb_trace_b3_xyz_coefficient("dirac") == 0
    )


@cache
def bb_log_coefficients(sector: Sector) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """Return ``(L1,L2,L3)`` for ``log(I + eps U1 + eps^2 U2 + ...)``."""

    sector = _validate_sector(sector)
    u1 = bb_bloch_order(sector, 1)
    u2 = bb_bloch_order(sector, 2)
    u3 = bb_bloch_order(sector, 3)
    l1 = u1
    l2 = (u2 - u1**2 / 2).applyfunc(sp.simplify)
    l3 = (u3 - (u1 * u2 + u2 * u1) / 2 + u1**3 / 3).applyfunc(sp.simplify)
    return l1, l2, l3


def bb_effective_hamiltonian_coefficient(sector: Sector, order: int) -> sp.Matrix:
    """Return ``H_order`` in ``H_eff = H0 + eps H1 + eps^2 H2 + ...``."""

    if order not in {0, 1, 2}:
        raise ValueError("implemented Hamiltonian coefficients are orders 0, 1, and 2")
    return (sp.I * bb_log_coefficients(sector)[order]).applyfunc(sp.simplify)


def leading_weyl_hamiltonian_matches(helicity: Literal["right", "left"]) -> bool:
    """Return true when ``H0`` is ``+/- sigma.k`` for a single Weyl block."""

    _epsilon, kx, ky, kz = symbolic_momentum()
    h0 = bb_effective_hamiltonian_coefficient(helicity, 0)
    expected = expected_weyl_hamiltonian(kx, ky, kz, helicity=helicity)
    return (h0 - expected).applyfunc(sp.simplify) == _zero_matrix_like(h0)


def dirac_leading_hamiltonian_matches() -> bool:
    """Return true when the Dirac pair has the block-diagonal Weyl ``H0``."""

    _epsilon, kx, ky, kz = symbolic_momentum()
    expected = block_diag(
        expected_weyl_hamiltonian(kx, ky, kz, helicity="right"),
        expected_weyl_hamiltonian(kx, ky, kz, helicity="left"),
    )
    h0 = bb_effective_hamiltonian_coefficient("dirac", 0)
    return (h0 - expected).applyfunc(sp.simplify) == _zero_matrix_like(h0)


def _scalar_hamiltonian_coefficient(sector: Sector, order: int) -> sp.Expr:
    matrix = bb_effective_hamiltonian_coefficient(sector, order)
    return sp.simplify(sp.trace(matrix) / matrix.rows)


def bb_scalar_h2_xyz_coefficient(sector: Sector) -> sp.Expr:
    """Return scalar ``kx*ky*kz`` coefficient of ``H2`` from ``Tr(H2)/dim``."""

    sector = _validate_sector(sector)
    return _xyz_coefficient(_scalar_hamiltonian_coefficient(sector, 2))


def _quadratic_coefficients(polynomial: sp.Expr) -> QuadraticCoefficientTuple:
    _epsilon, kx, ky, kz = symbolic_momentum()
    expanded = sp.expand(polynomial)
    return (
        sp.simplify(expanded.coeff(kx, 2).coeff(ky, 0).coeff(kz, 0)),
        sp.simplify(expanded.coeff(ky, 2).coeff(kx, 0).coeff(kz, 0)),
        sp.simplify(expanded.coeff(kz, 2).coeff(kx, 0).coeff(ky, 0)),
        sp.simplify(expanded.coeff(ky, 1).coeff(kz, 1).coeff(kx, 0)),
        sp.simplify(expanded.coeff(kz, 1).coeff(kx, 1).coeff(ky, 0)),
        sp.simplify(expanded.coeff(kx, 1).coeff(ky, 1).coeff(kz, 0)),
    )


def bb_scalar_h2_quadratic_coefficients(sector: Sector) -> QuadraticCoefficientTuple:
    """Return scalar degree-two coefficients of ``H1`` under the trace-energy rule."""

    sector = _validate_sector(sector)
    return _quadratic_coefficients(_scalar_hamiltonian_coefficient(sector, 1))


def scalar_h2_xyz_is_real(coefficient: sp.Expr) -> bool:
    """Return true when the coefficient is real under SymPy conjugation."""

    return _same_expr(coefficient, sp.conjugate(coefficient))


def scalar_filled_band_selector_sign_passes() -> bool:
    """Return true if the trace-energy scalar contains a helicity-odd cubic."""

    right = bb_scalar_h2_xyz_coefficient("right")
    left = bb_scalar_h2_xyz_coefficient("left")
    dirac = bb_scalar_h2_xyz_coefficient("dirac")
    return (
        right != 0
        and scalar_h2_xyz_is_real(right)
        and _same_expr(left, -right)
        and dirac == 0
    )


def scalar_quadratic_is_helicity_independent() -> bool:
    """Return true when scalar quadratic corrections match across sectors."""

    right = bb_scalar_h2_quadratic_coefficients("right")
    left = bb_scalar_h2_quadratic_coefficients("left")
    dirac = bb_scalar_h2_quadratic_coefficients("dirac")
    return right == left == dirac


def _numeric_bloch_symbol(
    sector: Sector,
    momentum: MomentumSample,
    *,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
) -> np.ndarray:
    """Return the BB Bloch symbol as a numeric complex matrix."""

    sector = _validate_sector(sector)
    epsilon, kx, ky, kz = symbolic_momentum()
    substituted = bb_bloch_symbol(sector).subs(
        {
            epsilon: sp.Float(epsilon_value),
            kx: sp.Float(momentum[0]),
            ky: sp.Float(momentum[1]),
            kz: sp.Float(momentum[2]),
        }
    )
    return np.array(substituted.evalf(30).tolist(), dtype=np.complex128)


def filled_band_quasienergies(
    sector: Sector,
    momentum: MomentumSample,
    *,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
) -> tuple[float, ...]:
    """Return sorted quasienergies ``E=-arg(lambda)/epsilon`` for a Bloch symbol."""

    symbol = _numeric_bloch_symbol(sector, momentum, epsilon_value=epsilon_value)
    eigenvalues = np.linalg.eigvals(symbol)
    energies = -np.angle(eigenvalues) / epsilon_value
    return tuple(float(value) for value in np.sort(energies))


def occupied_filled_band_energy(
    sector: Sector,
    momentum: MomentumSample,
    *,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
) -> float:
    """Return the filled negative-band energy for a Weyl block or Dirac pair."""

    energies = filled_band_quasienergies(
        sector,
        momentum,
        epsilon_value=epsilon_value,
    )
    occupied_count = len(energies) // 2
    return float(sum(energies[:occupied_count]))


def filled_band_parity_odd_energy(
    sector: Sector,
    momentum: MomentumSample,
    *,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
) -> float:
    """Return ``(E_occ(k)-E_occ(-k))/2`` for the filled band."""

    reversed_momentum: MomentumSample = (
        -momentum[0],
        -momentum[1],
        -momentum[2],
    )
    return (
        occupied_filled_band_energy(
            sector,
            momentum,
            epsilon_value=epsilon_value,
        )
        - occupied_filled_band_energy(
            sector,
            reversed_momentum,
            epsilon_value=epsilon_value,
        )
    ) / 2.0


def filled_band_selector_ratio(
    sector: Sector,
    momentum: MomentumSample,
    *,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
) -> float:
    """Return the normalized angular selector ratio ``E_odd/(eps*kx*ky*kz)``."""

    product = momentum[0] * momentum[1] * momentum[2]
    if product == 0:
        raise ValueError("selector ratio is undefined on the xyz=0 locus")
    return filled_band_parity_odd_energy(
        sector,
        momentum,
        epsilon_value=epsilon_value,
    ) / (epsilon_value * product)


def filled_band_helicity_relation_passes(
    *,
    momentum: MomentumSample = (1.0, 2.0, 3.0),
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    tolerance: float = FILLED_BAND_ZERO_TOLERANCE,
) -> bool:
    """Return true when right and left selectors are opposite."""

    right = filled_band_parity_odd_energy(
        "right",
        momentum,
        epsilon_value=epsilon_value,
    )
    left = filled_band_parity_odd_energy(
        "left",
        momentum,
        epsilon_value=epsilon_value,
    )
    return abs(right + left) <= tolerance and abs(right) > tolerance


def filled_band_dirac_cancels(
    *,
    momentum: MomentumSample = (1.0, 2.0, 3.0),
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    tolerance: float = FILLED_BAND_ZERO_TOLERANCE,
) -> bool:
    """Return true when the vector/Dirac pair cancels the selector."""

    dirac = filled_band_parity_odd_energy(
        "dirac",
        momentum,
        epsilon_value=epsilon_value,
    )
    return abs(dirac) <= tolerance


def filled_band_zero_locus_passes(
    *,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    tolerance: float = FILLED_BAND_ZERO_TOLERANCE,
) -> bool:
    """Return true when the selector vanishes on the ``xyz=0`` locus."""

    return all(
        abs(
            filled_band_parity_odd_energy(
                "right",
                sample,
                epsilon_value=epsilon_value,
            )
        )
        <= tolerance
        for sample in A2U_ZERO_LOCUS_SAMPLES
    )


def filled_band_signed_orbit_ratios(
    *,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
) -> tuple[float, ...]:
    """Return normalized right-Weyl ratios on a signed permutation orbit."""

    return tuple(
        filled_band_selector_ratio(
            "right",
            sample,
            epsilon_value=epsilon_value,
        )
        for sample in A2U_SIGNED_ORBIT_SAMPLES
    )


def filled_band_signed_orbit_is_a2u(
    *,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    tolerance: float = FILLED_BAND_RATIO_TOLERANCE,
) -> bool:
    """Return true when signed-permutation ratios agree, as for an ``A2u`` term."""

    ratios = filled_band_signed_orbit_ratios(epsilon_value=epsilon_value)
    reference = ratios[0]
    return all(abs(ratio - reference) <= tolerance for ratio in ratios[1:])


def filled_band_angular_ratios(
    *,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
) -> tuple[float, ...]:
    """Return ratios on inequivalent radii to diagnose the angular denominator."""

    return tuple(
        filled_band_selector_ratio(
            "right",
            sample,
            epsilon_value=epsilon_value,
        )
        for sample in A2U_ANGULAR_CONTROL_SAMPLES
    )


def filled_band_selector_is_nonpolynomial_angular(
    *,
    epsilon_value: float = FILLED_BAND_TEST_EPSILON,
    minimum_gap: float = FILLED_BAND_NONPOLYNOMIAL_GAP,
) -> bool:
    """Return true when ratios vary across radii, unlike a pure ``xyz`` monomial."""

    ratios = filled_band_angular_ratios(epsilon_value=epsilon_value)
    return max(ratios) - min(ratios) >= minimum_gap


def filled_band_selector_sign_passes() -> bool:
    """Return true when the filled-band energy carries the selector sign."""

    return (
        filled_band_helicity_relation_passes()
        and filled_band_dirac_cancels()
        and filled_band_zero_locus_passes()
        and filled_band_signed_orbit_is_a2u()
        and filled_band_selector_is_nonpolynomial_angular()
    )


def xyz_is_a2u_under_inversion_and_tetrahedral() -> bool:
    """Return true when ``xyz`` is inversion-odd and proper-tetrahedral invariant."""

    x, y, z = order_parameter_symbols()
    cubic = x * y * z
    inverted = cubic.subs({x: -x, y: -y, z: -z}, simultaneous=True)
    return (
        _same_expr(inverted, -cubic)
        and all(
            _same_expr(transform_polynomial(tetrahedral_cubic_polynomial(), rotation), tetrahedral_cubic_polynomial())
            for rotation in tetrahedral_rotation_group()
        )
        and _same_expr(tetrahedral_cubic_polynomial(), 8 * x * y * z / sp.sqrt(3))
    )


def v34_recovered() -> bool:
    """Return true when the V34 Schur-shell cubic-origin gate passes."""

    return (
        schur_shell_landau_audit_payload().final_verdict
        == "SCHUR_SHELL_TETRAHEDRAL_CUBIC_ORIGIN_PASS_SIGN_FREE"
    )


@dataclass(frozen=True)
class ChiralBBSelectorSignAuditPayload:
    """Verdict payload for the V35 chiral BB selector-sign gate."""

    final_verdict: str
    trace_b3_xyz_right: sp.Expr
    trace_b3_xyz_left: sp.Expr
    trace_b3_xyz_dirac: sp.Expr
    trace_b3_diagnostic_passes: bool
    leading_right_hamiltonian_matches: bool
    leading_left_hamiltonian_matches: bool
    leading_dirac_hamiltonian_matches: bool
    scalar_h2_xyz_right: sp.Expr
    scalar_h2_xyz_left: sp.Expr
    scalar_h2_xyz_dirac: sp.Expr
    scalar_xyz_real: bool
    scalar_selector_sign_passes: bool
    scalar_quadratic_helicity_independent: bool
    scalar_trace_probe_blind: bool
    filled_band_selector_right: float
    filled_band_selector_left: float
    filled_band_selector_dirac: float
    filled_band_right_selector_ratio: float
    filled_band_signed_orbit_ratios: tuple[float, ...]
    filled_band_angular_ratios: tuple[float, ...]
    filled_band_helicity_relation: bool
    filled_band_dirac_cancellation: bool
    filled_band_zero_locus: bool
    filled_band_a2u_signed_orbit: bool
    filled_band_nonpolynomial_angular: bool
    filled_band_selector_sign_passes: bool
    xyz_a2u_check: bool
    v34_recovered: bool
    remaining_declared_inputs: tuple[str, ...]
    interpretation: str


def chiral_bb_selector_sign_audit_payload() -> ChiralBBSelectorSignAuditPayload:
    """Return the V35 chiral BB selector-sign verdict."""

    trace_right = bb_trace_b3_xyz_coefficient("right")
    trace_left = bb_trace_b3_xyz_coefficient("left")
    trace_dirac = bb_trace_b3_xyz_coefficient("dirac")
    trace_pass = trace_b3_diagnostic_passes()
    leading_right = leading_weyl_hamiltonian_matches("right")
    leading_left = leading_weyl_hamiltonian_matches("left")
    leading_dirac = dirac_leading_hamiltonian_matches()
    scalar_right = bb_scalar_h2_xyz_coefficient("right")
    scalar_left = bb_scalar_h2_xyz_coefficient("left")
    scalar_dirac = bb_scalar_h2_xyz_coefficient("dirac")
    scalar_real = all(
        scalar_h2_xyz_is_real(value)
        for value in (scalar_right, scalar_left, scalar_dirac)
    )
    scalar_pass = scalar_filled_band_selector_sign_passes()
    quadratic_independent = scalar_quadratic_is_helicity_independent()
    scalar_blind = not scalar_pass and scalar_right == scalar_left == scalar_dirac == 0
    filled_right = filled_band_parity_odd_energy("right", (1.0, 2.0, 3.0))
    filled_left = filled_band_parity_odd_energy("left", (1.0, 2.0, 3.0))
    filled_dirac = filled_band_parity_odd_energy("dirac", (1.0, 2.0, 3.0))
    filled_right_ratio = filled_band_selector_ratio("right", (1.0, 2.0, 3.0))
    filled_orbit_ratios = filled_band_signed_orbit_ratios()
    filled_angular_ratios = filled_band_angular_ratios()
    filled_helicity = filled_band_helicity_relation_passes()
    filled_dirac_cancel = filled_band_dirac_cancels()
    filled_zero_locus = filled_band_zero_locus_passes()
    filled_a2u = filled_band_signed_orbit_is_a2u()
    filled_nonpolynomial = filled_band_selector_is_nonpolynomial_angular()
    filled_pass = filled_band_selector_sign_passes()
    a2u = xyz_is_a2u_under_inversion_and_tetrahedral()
    v34 = v34_recovered()

    prerequisites_pass = (
        trace_pass
        and leading_right
        and leading_left
        and leading_dirac
        and scalar_real
        and scalar_blind
        and quadratic_independent
        and a2u
        and v34
    )

    if prerequisites_pass and filled_pass:
        final_verdict = "CHIRAL_BB_FILLED_BAND_SELECTOR_SIGN_PASS"
        remaining_inputs: tuple[str, ...] = ()
        interpretation = (
            "The single-Weyl BB walk carries the helicity-odd A2u cubic in "
            "the real filled-band quasienergy.  The parity-even scalar trace "
            "probe is blind to this angular term, but the eigenphase energy "
            "shows opposite right/left selector signs and exact Dirac "
            "cancellation."
        )
    elif prerequisites_pass:
        final_verdict = "CHIRAL_BB_FILLED_BAND_SELECTOR_SIGN_KILL"
        remaining_inputs = REMAINING_DECLARED_INPUTS_AFTER_CHIRAL_BB_SELECTOR_SIGN
        interpretation = (
            "The BB Floquet trace contains the expected helicity-odd A2u "
            "cubic and Dirac cancellation, but the filled-band eigenphase "
            "checks did not produce the required real A2u selector pattern."
        )
    else:
        final_verdict = "CHIRAL_BB_SELECTOR_SIGN_AUDIT_KILL"
        remaining_inputs = REMAINING_DECLARED_INPUTS_AFTER_CHIRAL_BB_SELECTOR_SIGN
        interpretation = (
            "The BB trace diagnostic, leading Hamiltonian, scalar reality, "
            "quadratic control, A2u check, or V34 regression failed."
        )

    return ChiralBBSelectorSignAuditPayload(
        final_verdict=final_verdict,
        trace_b3_xyz_right=trace_right,
        trace_b3_xyz_left=trace_left,
        trace_b3_xyz_dirac=trace_dirac,
        trace_b3_diagnostic_passes=trace_pass,
        leading_right_hamiltonian_matches=leading_right,
        leading_left_hamiltonian_matches=leading_left,
        leading_dirac_hamiltonian_matches=leading_dirac,
        scalar_h2_xyz_right=scalar_right,
        scalar_h2_xyz_left=scalar_left,
        scalar_h2_xyz_dirac=scalar_dirac,
        scalar_xyz_real=scalar_real,
        scalar_selector_sign_passes=scalar_pass,
        scalar_quadratic_helicity_independent=quadratic_independent,
        scalar_trace_probe_blind=scalar_blind,
        filled_band_selector_right=filled_right,
        filled_band_selector_left=filled_left,
        filled_band_selector_dirac=filled_dirac,
        filled_band_right_selector_ratio=filled_right_ratio,
        filled_band_signed_orbit_ratios=filled_orbit_ratios,
        filled_band_angular_ratios=filled_angular_ratios,
        filled_band_helicity_relation=filled_helicity,
        filled_band_dirac_cancellation=filled_dirac_cancel,
        filled_band_zero_locus=filled_zero_locus,
        filled_band_a2u_signed_orbit=filled_a2u,
        filled_band_nonpolynomial_angular=filled_nonpolynomial,
        filled_band_selector_sign_passes=filled_pass,
        xyz_a2u_check=a2u,
        v34_recovered=v34,
        remaining_declared_inputs=remaining_inputs,
        interpretation=interpretation,
    )
