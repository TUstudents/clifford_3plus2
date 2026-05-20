"""Finite-spacing Lorentz/rotation recovery diagnostics for the BCC walk.

Session 20 proved the first-order ``alpha . k`` continuum precursor.  Session
42 asks the next question: where do finite-spacing rotational anisotropies
first appear in the free dispersion, and how does the BCC Dirac pair compare
with the naive hypercube control?

The BCC diagnostic compares the normalized trace of the exact Floquet symbol
against ``cos(epsilon * |k|)``.  This is deliberately narrower than a full
interacting Lorentz-recovery proof: it is a free-dispersion anisotropy audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import sympy as sp

from clifford_3plus2_d5.spacetime_qca.bcc_weyl import bcc_dirac_symbol, bcc_weyl_symbol


@dataclass(frozen=True)
class LorentzRecoveryAudit:
    """Compact Session 42 dispersion-recovery payload."""

    bcc_dirac_matches_continuum_through_order: int
    bcc_dirac_first_anisotropy_order: int
    bcc_dirac_directional_coefficients: dict[str, sp.Expr]
    hypercube_first_anisotropy_order: int
    hypercube_directional_coefficients: dict[str, sp.Expr]
    notes: tuple[str, ...]


def _series_through_order(expr: sp.Expr, epsilon: sp.Symbol, max_order: int) -> sp.Expr:
    if max_order < 0:
        raise ValueError(f"max_order must be non-negative, got {max_order}")
    return sp.series(expr, epsilon, 0, max_order + 1).removeO().expand()


def _momentum_squared(kx: sp.Expr, ky: sp.Expr, kz: sp.Expr) -> sp.Expr:
    return sp.simplify(kx**2 + ky**2 + kz**2)


def _directional_samples(magnitude: sp.Expr | None = None) -> dict[str, tuple[sp.Expr, sp.Expr, sp.Expr]]:
    q = sp.symbols("q") if magnitude is None else magnitude
    return {
        "axis": (q, 0, 0),
        "face_diagonal": (q / sp.sqrt(2), q / sp.sqrt(2), 0),
        "body_diagonal": (q / sp.sqrt(3), q / sp.sqrt(3), q / sp.sqrt(3)),
    }


@lru_cache(maxsize=32)
def bcc_weyl_trace_cosine(
    epsilon: sp.Symbol | sp.Expr,
    kx: sp.Expr,
    ky: sp.Expr,
    kz: sp.Expr,
    *,
    helicity: str = "right",
) -> sp.Expr:
    """Return ``tr(U_weyl(k)) / 2`` for the BCC Weyl Bloch symbol."""

    symbol = bcc_weyl_symbol(epsilon, kx, ky, kz, helicity=helicity)
    return sp.simplify(sp.trace(symbol) / symbol.rows)


@lru_cache(maxsize=16)
def bcc_dirac_trace_cosine(
    epsilon: sp.Symbol | sp.Expr,
    kx: sp.Expr,
    ky: sp.Expr,
    kz: sp.Expr,
) -> sp.Expr:
    """Return ``tr(U_dirac(k)) / 4`` for the BCC Dirac Bloch symbol."""

    symbol = bcc_dirac_symbol(epsilon, kx, ky, kz)
    return sp.simplify(sp.trace(symbol) / symbol.rows)


@lru_cache(maxsize=32)
def continuum_cosine_series(
    epsilon: sp.Symbol,
    kx: sp.Expr,
    ky: sp.Expr,
    kz: sp.Expr,
    *,
    max_order: int = 4,
) -> sp.Expr:
    """Return the small-spacing series for ``cos(epsilon * |k|)``."""

    return _series_through_order(
        sp.cos(epsilon * sp.sqrt(_momentum_squared(kx, ky, kz))),
        epsilon,
        max_order,
    )


@lru_cache(maxsize=32)
def bcc_weyl_cosine_residual_series(
    epsilon: sp.Symbol,
    kx: sp.Expr,
    ky: sp.Expr,
    kz: sp.Expr,
    *,
    helicity: str = "right",
    max_order: int = 4,
) -> sp.Expr:
    """Return BCC Weyl trace-cosine minus continuum cosine through ``max_order``."""

    actual = _series_through_order(
        bcc_weyl_trace_cosine(epsilon, kx, ky, kz, helicity=helicity),
        epsilon,
        max_order,
    )
    target = continuum_cosine_series(epsilon, kx, ky, kz, max_order=max_order)
    return sp.factor(sp.simplify(actual - target))


@lru_cache(maxsize=16)
def bcc_dirac_cosine_residual_series(
    epsilon: sp.Symbol,
    kx: sp.Expr,
    ky: sp.Expr,
    kz: sp.Expr,
    *,
    max_order: int = 4,
) -> sp.Expr:
    """Return BCC Dirac trace-cosine minus continuum cosine through ``max_order``."""

    actual = _series_through_order(
        bcc_dirac_trace_cosine(epsilon, kx, ky, kz),
        epsilon,
        max_order,
    )
    target = continuum_cosine_series(epsilon, kx, ky, kz, max_order=max_order)
    return sp.factor(sp.simplify(actual - target))


@lru_cache(maxsize=16)
def hypercube_energy_squared(
    epsilon: sp.Symbol | sp.Expr,
    kx: sp.Expr,
    ky: sp.Expr,
    kz: sp.Expr,
) -> sp.Expr:
    """Return ``sum_i sin(epsilon k_i)^2 / epsilon^2`` for the naive cube."""

    return sp.simplify(
        (
            sp.sin(epsilon * kx) ** 2
            + sp.sin(epsilon * ky) ** 2
            + sp.sin(epsilon * kz) ** 2
        )
        / epsilon**2
    )


@lru_cache(maxsize=16)
def hypercube_energy_squared_residual_series(
    epsilon: sp.Symbol,
    kx: sp.Expr,
    ky: sp.Expr,
    kz: sp.Expr,
    *,
    max_order: int = 4,
) -> sp.Expr:
    """Return naive cube ``E(k)^2 - |k|^2`` through ``max_order``."""

    actual = _series_through_order(
        hypercube_energy_squared(epsilon, kx, ky, kz),
        epsilon,
        max_order,
    )
    return sp.factor(sp.simplify(actual - _momentum_squared(kx, ky, kz)))


def first_nonzero_epsilon_order(
    expr: sp.Expr,
    epsilon: sp.Symbol,
    *,
    max_order: int = 8,
) -> int | None:
    """Return the first nonzero epsilon power in ``expr`` up to ``max_order``."""

    expanded = sp.expand(expr)
    for order in range(max_order + 1):
        if sp.simplify(expanded.coeff(epsilon, order)) != 0:
            return order
    return None


@lru_cache(maxsize=8)
def bcc_dirac_leading_anisotropy_coefficients(
    epsilon: sp.Symbol,
    *,
    magnitude: sp.Expr | None = None,
) -> dict[str, sp.Expr]:
    """Return leading ``epsilon**4`` BCC Dirac coefficients by direction."""

    kx, ky, kz = sp.symbols("kx ky kz")
    residual = bcc_dirac_cosine_residual_series(epsilon, kx, ky, kz)
    coefficient = sp.expand(residual).coeff(epsilon, 4)
    return {
        name: sp.simplify(coefficient.subs(dict(zip((kx, ky, kz), momentum, strict=True))))
        for name, momentum in _directional_samples(magnitude).items()
    }


@lru_cache(maxsize=8)
def hypercube_leading_anisotropy_coefficients(
    epsilon: sp.Symbol,
    *,
    magnitude: sp.Expr | None = None,
) -> dict[str, sp.Expr]:
    """Return leading ``epsilon**2`` naive-cube coefficients by direction."""

    kx, ky, kz = sp.symbols("kx ky kz")
    residual = hypercube_energy_squared_residual_series(epsilon, kx, ky, kz)
    coefficient = sp.expand(residual).coeff(epsilon, 2)
    return {
        name: sp.simplify(coefficient.subs(dict(zip((kx, ky, kz), momentum, strict=True))))
        for name, momentum in _directional_samples(magnitude).items()
    }


@lru_cache(maxsize=1)
def lorentz_recovery_audit_payload() -> LorentzRecoveryAudit:
    """Return the Session 42 finite-spacing dispersion audit payload."""

    epsilon, kx, ky, kz = sp.symbols("epsilon kx ky kz")
    bcc_residual = bcc_dirac_cosine_residual_series(epsilon, kx, ky, kz)
    cube_residual = hypercube_energy_squared_residual_series(epsilon, kx, ky, kz)
    bcc_order = first_nonzero_epsilon_order(bcc_residual, epsilon)
    cube_order = first_nonzero_epsilon_order(cube_residual, epsilon)
    if bcc_order is None or cube_order is None:
        raise RuntimeError("expected nonzero finite-spacing residuals")
    return LorentzRecoveryAudit(
        bcc_dirac_matches_continuum_through_order=bcc_order - 1,
        bcc_dirac_first_anisotropy_order=bcc_order,
        bcc_dirac_directional_coefficients=bcc_dirac_leading_anisotropy_coefficients(epsilon),
        hypercube_first_anisotropy_order=cube_order,
        hypercube_directional_coefficients=hypercube_leading_anisotropy_coefficients(epsilon),
        notes=(
            "BCC Dirac cancels the Weyl cubic anisotropy between helicities.",
            "Naive hypercube anisotropy appears at lower order and coexists with corner doublers.",
            "This is a free-dispersion audit, not a full interacting Lorentz-invariance proof.",
        ),
    )
