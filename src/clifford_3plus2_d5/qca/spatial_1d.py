"""Sidecar exact diagnostics for a 1D spatial transfer prototype."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import gcd, lcm

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.algebra.real_carrier import (
    clock_complex_structure,
    split_projectors_3_2,
)
from clifford_3plus2_d5.qca.gates import is_real_matrix
from clifford_3plus2_d5.qca.floquet_alpha import (
    floquet_alpha_operator,
    floquet_alpha_spectral_projectors,
)
from clifford_3plus2_d5.qca.floquet_alpha_noncommuting import (
    _compatible_pair_orientation_sign_choices,
    _qgenerated_algebra_basis,
    _qmatrix_from_sympy,
    _qmatrix_in_span,
    floquet_alpha_noncommuting_candidates,
    floquet_alpha_noncommuting_twist_operator,
    pair_orientation_j_operator,
)
from clifford_3plus2_d5.qca.rule_verdict import (
    center_basis_of_algebra,
    generated_algebra_basis,
    solve_central_idempotents,
)


@dataclass(frozen=True)
class SpatialTransferRule1D:
    name: str
    alpha_modes: tuple[int, ...]
    eta_modes: tuple[int, ...]
    alpha_winding: int
    eta_winding: int
    period: int

    @property
    def mode_count(self) -> int:
        return len(self.alpha_modes) + len(self.eta_modes)

    @property
    def dimension(self) -> int:
        return 2 * self.mode_count

    @property
    def locality_radius(self) -> int:
        return max(abs(self.alpha_winding), abs(self.eta_winding))


@dataclass(frozen=True)
class SpatialOrientationOrbit:
    alpha_sign: int
    eta_sign: int
    transport_allowed: bool


@dataclass(frozen=True)
class SpatialHoppingTerm:
    shift: int
    matrix: sp.Matrix


@dataclass(frozen=True)
class SpatialLocalQCALayer1D:
    name: str
    period: int
    dimension: int
    terms: tuple[SpatialHoppingTerm, ...]

    @property
    def locality_radius(self) -> int:
        return max((abs(term.shift) for term in self.terms), default=0)


@dataclass(frozen=True)
class Spatial1DAlphaCertificate:
    rule_name: str
    period: int
    alpha_winding: int
    eta_winding: int
    winding_gcd: int
    winding_lcm: int
    locality_radius: int
    sample_count: int
    transfer_unitary_on_samples: bool
    alpha_projector_rank: int
    eta_projector_rank: int
    coarse_6_4_band_split: bool
    orientation_choices_before_transport: int
    orientation_choices_after_transport: int
    orientation_orbits: tuple[SpatialOrientationOrbit, ...]
    sign_coupled_to_global_pm: bool
    strict_bridge_candidates: int
    route_label: str
    load_bearing_qca_bridge: bool = False


@dataclass(frozen=True)
class Spatial1DLocalHoppingCertificate:
    rule_name: str
    hopping_term_count: int
    hopping_shifts: tuple[int, ...]
    hopping_locality_radius: int
    mode_windings: tuple[int, ...]
    computed_alpha_winding: int | None
    computed_eta_winding: int | None
    computed_winding_gcd: int | None
    computed_winding_lcm: int | None
    reconstructs_transfer_on_samples: bool
    transfer_unitary_on_samples: bool
    coarse_6_4_band_split: bool
    orientation_choices_before_transport: int
    orientation_choices_after_transport: int
    orientation_orbits: tuple[SpatialOrientationOrbit, ...]
    sign_coupled_to_global_pm: bool
    route_label: str
    load_bearing_qca_bridge: bool = False


@dataclass(frozen=True)
class Spatial1DLocalQCACertificate:
    rule_name: str
    layer_name: str
    period: int
    dimension: int
    qca_term_count: int
    qca_shifts: tuple[int, ...]
    qca_locality_radius: int
    finite_radius: bool
    coefficient_matrices_real: bool
    laurent_orthogonal: bool
    symbol_reconstructs_transfer_on_samples: bool
    symbol_unitary_on_samples: bool
    coefficient_algebra_dimension: int
    coefficient_center_dimension: int
    central_idempotent_ranks: tuple[int, ...]
    lower_rank_central_idempotents: int
    coarse_6_4_center_pair: bool
    mode_windings: tuple[int, ...]
    computed_alpha_winding: int | None
    computed_eta_winding: int | None
    computed_winding_gcd: int | None
    computed_winding_lcm: int | None
    orientation_choices_before_transport: int
    orientation_choices_after_transport: int
    orientation_orbits: tuple[SpatialOrientationOrbit, ...]
    sign_coupled_to_global_pm: bool
    strict_bridge_candidates: int
    route_label: str
    load_bearing_qca_bridge: bool = False


@dataclass(frozen=True)
class Spatial1DUnseededCandidateCertificate:
    candidate_name: str
    layer_name: str
    qca_shifts: tuple[int, ...]
    qca_locality_radius: int
    finite_radius: bool
    coefficient_matrices_real: bool
    laurent_orthogonal: bool
    seeded_coefficient_guardrail_passed: bool
    seeded_coefficient_witnesses: tuple[str, ...]
    coefficient_algebra_dimension: int
    coefficient_center_dimension: int
    central_idempotent_ranks: tuple[int, ...]
    lower_rank_central_idempotents: int
    coarse_6_4_center_pair: bool
    mode_windings: tuple[int, ...]
    computed_alpha_winding: int | None
    computed_eta_winding: int | None
    computed_winding_gcd: int | None
    computed_winding_lcm: int | None
    orientation_choices_before_transport: int
    orientation_choices_after_transport: int
    sign_coupled_to_global_pm: bool
    strict_bridge_candidate: bool
    route_label: str
    load_bearing_qca_bridge: bool = False


@dataclass(frozen=True)
class Spatial1DUnseededSearchSummary:
    candidate_count: int
    unseeded_candidate_count: int
    seeded_guardrail_rejections: int
    laurent_orthogonal_candidates: int
    unseeded_coarse_6_4_center_candidates: int
    unseeded_sign_coupled_candidates: int
    unseeded_strict_bridge_candidates: int
    lower_rank_center_rejections: int
    route_label: str
    candidates: tuple[Spatial1DUnseededCandidateCertificate, ...]
    load_bearing_qca_bridge: bool = False


@dataclass(frozen=True)
class Spatial1DCombinedRouteCertificate:
    rule_name: str
    onsite_candidate_name: str
    layer_name: str
    period: int
    qca_term_count: int
    qca_shifts: tuple[int, ...]
    qca_locality_radius: int
    finite_radius: bool
    coefficient_matrices_real: bool
    laurent_orthogonal: bool
    symbol_unitary_on_samples: bool
    onsite_generated_algebra_dimension: int
    onsite_center_dimension: int
    onsite_central_idempotent_ranks: tuple[int, ...]
    onsite_lower_rank_central_idempotents: int
    onsite_compatible_j_count: int
    transported_compatible_j_count: int
    transported_j_commute_on_samples: bool
    sign_coupled_to_global_pm: bool
    coefficient_algebra_dimension: int
    coefficient_algebra_generates_alpha_eta_projectors: bool
    joint_rule_algebra_dimension: int
    joint_rule_generated_transported_j_count: int
    topological_pm_shape_candidate: bool
    strict_bridge_candidate: bool
    route_label: str
    load_bearing_qca_bridge: bool = False


def spatial_alpha_prototype() -> SpatialTransferRule1D:
    """Return the first root-of-unity 1D sidecar prototype.

    The prototype encodes the Floquet-alpha phases as spatial winding data over
    one period-12 Brillouin cycle: alpha advances by 4 units and eta by 3.
    It is a diagnostic transfer model, not a finite-depth QCA theorem.
    """

    return SpatialTransferRule1D(
        name="spatial_1d_alpha_period_12_winding_4_3",
        alpha_modes=(0, 1, 2),
        eta_modes=(3, 4),
        alpha_winding=4,
        eta_winding=3,
        period=12,
    )


def root_of_unity(period: int, sample: int) -> sp.Expr:
    return sp.exp(2 * sp.pi * sp.I * sp.Rational(sample, period))


def transfer_matrix_at_root(rule: SpatialTransferRule1D, sample: int) -> sp.Matrix:
    zeta = root_of_unity(rule.period, sample)
    entries: list[sp.Expr] = []
    for mode in range(rule.mode_count):
        winding = rule.alpha_winding if mode in rule.alpha_modes else rule.eta_winding
        entries.append(sp.simplify(zeta**winding))
    return sp.diag(*entries, *entries)


def local_hopping_terms(rule: SpatialTransferRule1D) -> tuple[SpatialHoppingTerm, ...]:
    """Return explicit finite-hop Laurent coefficients for the sidecar rule."""

    terms: dict[int, sp.Matrix] = {}
    for mode in range(rule.mode_count):
        shift = rule.alpha_winding if mode in rule.alpha_modes else rule.eta_winding
        matrix = terms.setdefault(shift, sp.zeros(rule.dimension))
        matrix[mode, mode] = 1
        matrix[mode + rule.mode_count, mode + rule.mode_count] = 1
    return tuple(
        SpatialHoppingTerm(shift=shift, matrix=terms[shift])
        for shift in sorted(terms)
    )


def spatial_alpha_local_qca_layer(
    rule: SpatialTransferRule1D | None = None,
) -> SpatialLocalQCALayer1D:
    """Return the explicit finite-radius real local QCA layer prototype."""

    rule = rule if rule is not None else spatial_alpha_prototype()
    return SpatialLocalQCALayer1D(
        name="spatial_1d_alpha_projector_shift_qca",
        period=rule.period,
        dimension=rule.dimension,
        terms=local_hopping_terms(rule),
    )


def spatial_1d_combined_local_qca_layer(
    rule: SpatialTransferRule1D | None = None,
) -> SpatialLocalQCALayer1D:
    """Return Route-1 on-site update composed with Route-2 winding hops."""

    rule = rule if rule is not None else spatial_alpha_prototype()
    candidate = floquet_alpha_noncommuting_candidates(pattern_index=0)[0]
    u1 = floquet_alpha_operator(candidate.pattern)
    u2 = floquet_alpha_noncommuting_twist_operator(candidate)
    u_local = sp.simplify(u2 * u1)
    p_alpha, p_eta = floquet_alpha_spectral_projectors(candidate.pattern)
    return SpatialLocalQCALayer1D(
        name="spatial_1d_route1_route2_combined_qca",
        period=rule.period,
        dimension=rule.dimension,
        terms=(
            SpatialHoppingTerm(shift=rule.eta_winding, matrix=sp.simplify(u_local * p_eta)),
            SpatialHoppingTerm(
                shift=rule.alpha_winding,
                matrix=sp.simplify(u_local * p_alpha),
            ),
        ),
    )


def _mode_permutation_matrix(permutation: tuple[int, ...]) -> sp.Matrix:
    mode_count = len(permutation)
    matrix = sp.zeros(2 * mode_count)
    for source, target in enumerate(permutation):
        matrix[target, source] = 1
        matrix[target + mode_count, source + mode_count] = 1
    return matrix


def spatial_1d_unseeded_candidate_layers(
    rule: SpatialTransferRule1D | None = None,
) -> tuple[SpatialLocalQCALayer1D, ...]:
    """Return the first conservative unseeded spatial candidate family.

    The first three candidates are block-blind finite-radius QCA layers. The
    final projector-shift layer is deliberately included as a seeded guardrail:
    the search should reject it even though its sign-coupling diagnostic works.
    """

    rule = rule if rule is not None else spatial_alpha_prototype()
    dimension = rule.dimension
    return (
        SpatialLocalQCALayer1D(
            name="unseeded_uniform_identity_shift",
            period=rule.period,
            dimension=dimension,
            terms=(SpatialHoppingTerm(shift=1, matrix=identity(dimension)),),
        ),
        SpatialLocalQCALayer1D(
            name="unseeded_uniform_clock_shift",
            period=rule.period,
            dimension=dimension,
            terms=(SpatialHoppingTerm(shift=1, matrix=clock_complex_structure()),),
        ),
        SpatialLocalQCALayer1D(
            name="unseeded_mode_5_cycle_shift",
            period=rule.period,
            dimension=dimension,
            terms=(
                SpatialHoppingTerm(
                    shift=1,
                    matrix=_mode_permutation_matrix((1, 2, 3, 4, 0)),
                ),
            ),
        ),
        spatial_alpha_local_qca_layer(rule),
    )


def transfer_matrix_from_hopping_at_root(
    terms: tuple[SpatialHoppingTerm, ...],
    *,
    period: int,
    sample: int,
) -> sp.Matrix:
    zeta = root_of_unity(period, sample)
    dimension = terms[0].matrix.rows if terms else 0
    transfer = sp.zeros(dimension)
    for term in terms:
        transfer += term.matrix * zeta**term.shift
    return transfer.applyfunc(sp.simplify)


def local_qca_symbol_at_root(layer: SpatialLocalQCALayer1D, sample: int) -> sp.Matrix:
    return transfer_matrix_from_hopping_at_root(
        layer.terms,
        period=layer.period,
        sample=sample,
    )


def alpha_eta_projectors(rule: SpatialTransferRule1D) -> tuple[sp.Matrix, sp.Matrix]:
    alpha_entries = [1 if mode in rule.alpha_modes else 0 for mode in range(rule.mode_count)]
    eta_entries = [1 if mode in rule.eta_modes else 0 for mode in range(rule.mode_count)]
    return (
        sp.diag(*alpha_entries, *alpha_entries),
        sp.diag(*eta_entries, *eta_entries),
    )


def transfer_is_unitary_on_samples(rule: SpatialTransferRule1D) -> bool:
    one = identity(rule.dimension)
    for sample in range(rule.period):
        transfer = transfer_matrix_at_root(rule, sample)
        if sp.simplify(transfer.conjugate().T * transfer - one) != sp.zeros(rule.dimension):
            return False
    return True


def local_hopping_reconstructs_transfer(rule: SpatialTransferRule1D) -> bool:
    terms = local_hopping_terms(rule)
    for sample in range(rule.period):
        expected = transfer_matrix_at_root(rule, sample)
        actual = transfer_matrix_from_hopping_at_root(
            terms,
            period=rule.period,
            sample=sample,
        )
        if sp.simplify(actual - expected) != sp.zeros(rule.dimension):
            return False
    return True


def _terms_by_shift(
    terms: tuple[SpatialHoppingTerm, ...],
) -> dict[int, sp.Matrix]:
    return {term.shift: term.matrix for term in terms}


def _laurent_deltas(terms: tuple[SpatialHoppingTerm, ...]) -> tuple[int, ...]:
    shifts = tuple(term.shift for term in terms)
    return tuple(sorted({left - right for left in shifts for right in shifts}))


def local_qca_laurent_orthogonal(layer: SpatialLocalQCALayer1D) -> bool:
    """Check exact Laurent coefficient identities for a real local QCA layer."""

    terms_by_shift = _terms_by_shift(layer.terms)
    deltas = _laurent_deltas(layer.terms)
    one = identity(layer.dimension)
    zero = sp.zeros(layer.dimension)
    for delta in deltas:
        right = sp.zeros(layer.dimension)
        left = sp.zeros(layer.dimension)
        for shift, matrix in terms_by_shift.items():
            shifted = terms_by_shift.get(shift + delta)
            if shifted is None:
                continue
            right += matrix.T * shifted
            left += matrix * shifted.T
        expected = one if delta == 0 else zero
        if sp.simplify(right - expected) != zero:
            return False
        if sp.simplify(left - expected) != zero:
            return False
    return True


def local_qca_symbol_reconstructs_transfer(
    layer: SpatialLocalQCALayer1D,
    rule: SpatialTransferRule1D,
) -> bool:
    for sample in range(rule.period):
        if local_qca_symbol_at_root(layer, sample) != transfer_matrix_at_root(rule, sample):
            return False
    return True


def local_qca_symbol_unitary_on_samples(layer: SpatialLocalQCALayer1D) -> bool:
    if local_qca_laurent_orthogonal(layer):
        return True
    one = identity(layer.dimension)
    for sample in range(layer.period):
        symbol = local_qca_symbol_at_root(layer, sample)
        if sp.simplify(symbol.conjugate().T * symbol - one) != sp.zeros(layer.dimension):
            return False
    return True


def mode_windings_from_terms(
    terms: tuple[SpatialHoppingTerm, ...],
    *,
    mode_count: int,
) -> tuple[int, ...]:
    windings = []
    for mode in range(mode_count):
        mode_shifts = tuple(
            term.shift
            for term in terms
            if term.matrix[mode, mode] == 1
            and term.matrix[mode + mode_count, mode + mode_count] == 1
        )
        if len(mode_shifts) != 1:
            return ()
        windings.append(mode_shifts[0])
    return tuple(windings)


def mode_windings_from_hopping(rule: SpatialTransferRule1D) -> tuple[int, ...]:
    return mode_windings_from_terms(
        local_hopping_terms(rule),
        mode_count=rule.mode_count,
    )


def _common_sector_winding(
    windings: tuple[int, ...],
    modes: tuple[int, ...],
) -> int | None:
    sector_windings = {windings[mode] for mode in modes}
    if len(sector_windings) != 1:
        return None
    return next(iter(sector_windings))


def spatial_orientation_orbits(rule: SpatialTransferRule1D) -> tuple[SpatialOrientationOrbit, ...]:
    """Return block-sign choices allowed by the shared spatial orientation.

    This sidecar tests the concrete Route-2 hypothesis: coprime alpha/eta
    windings over one period-12 cycle impose one shared orientation, so the
    independent block signs collapse from four choices to global +/-.
    """

    coprime_common_cycle = (
        gcd(abs(rule.alpha_winding), abs(rule.eta_winding)) == 1
        and lcm(abs(rule.alpha_winding), abs(rule.eta_winding)) == rule.period
    )
    orbits = []
    for alpha_sign, eta_sign in product((1, -1), repeat=2):
        allowed = alpha_sign == eta_sign if coprime_common_cycle else True
        orbits.append(
            SpatialOrientationOrbit(
                alpha_sign=alpha_sign,
                eta_sign=eta_sign,
                transport_allowed=allowed,
            )
        )
    return tuple(orbits)


def spatial_orientation_orbits_from_windings(
    *,
    alpha_winding: int | None,
    eta_winding: int | None,
    period: int,
) -> tuple[SpatialOrientationOrbit, ...]:
    if alpha_winding is None or eta_winding is None:
        coprime_common_cycle = False
    else:
        coprime_common_cycle = (
            gcd(abs(alpha_winding), abs(eta_winding)) == 1
            and lcm(abs(alpha_winding), abs(eta_winding)) == period
        )
    return tuple(
        SpatialOrientationOrbit(
            alpha_sign=alpha_sign,
            eta_sign=eta_sign,
            transport_allowed=(
                alpha_sign == eta_sign if coprime_common_cycle else True
            ),
        )
        for alpha_sign, eta_sign in product((1, -1), repeat=2)
    )


def spatial_1d_alpha_certificate(rule: SpatialTransferRule1D | None = None) -> Spatial1DAlphaCertificate:
    rule = rule if rule is not None else spatial_alpha_prototype()
    alpha_projector, eta_projector = alpha_eta_projectors(rule)
    orientation_orbits = spatial_orientation_orbits(rule)
    allowed_count = sum(orbit.transport_allowed for orbit in orientation_orbits)
    sign_coupled = allowed_count == 2 and all(
        orbit.alpha_sign == orbit.eta_sign
        for orbit in orientation_orbits
        if orbit.transport_allowed
    )
    coarse_split = alpha_projector.rank() == 6 and eta_projector.rank() == 4
    unitary = transfer_is_unitary_on_samples(rule)

    if not unitary:
        route_label = "spatial_not_unitary"
    elif not coarse_split:
        route_label = "spatial_no_coarse_6_4_band_split"
    elif sign_coupled:
        route_label = "spatial_signs_coupled_to_global_pm"
    else:
        route_label = "spatial_signs_uncoupled"

    return Spatial1DAlphaCertificate(
        rule_name=rule.name,
        period=rule.period,
        alpha_winding=rule.alpha_winding,
        eta_winding=rule.eta_winding,
        winding_gcd=gcd(abs(rule.alpha_winding), abs(rule.eta_winding)),
        winding_lcm=lcm(abs(rule.alpha_winding), abs(rule.eta_winding)),
        locality_radius=rule.locality_radius,
        sample_count=rule.period,
        transfer_unitary_on_samples=unitary,
        alpha_projector_rank=alpha_projector.rank(),
        eta_projector_rank=eta_projector.rank(),
        coarse_6_4_band_split=coarse_split,
        orientation_choices_before_transport=len(orientation_orbits),
        orientation_choices_after_transport=allowed_count,
        orientation_orbits=orientation_orbits,
        sign_coupled_to_global_pm=sign_coupled,
        strict_bridge_candidates=0,
        route_label=route_label,
    )


def spatial_1d_local_hopping_certificate(
    rule: SpatialTransferRule1D | None = None,
) -> Spatial1DLocalHoppingCertificate:
    rule = rule if rule is not None else spatial_alpha_prototype()
    terms = local_hopping_terms(rule)
    windings = mode_windings_from_hopping(rule)
    alpha_winding = _common_sector_winding(windings, rule.alpha_modes) if windings else None
    eta_winding = _common_sector_winding(windings, rule.eta_modes) if windings else None
    winding_gcd = (
        gcd(abs(alpha_winding), abs(eta_winding))
        if alpha_winding is not None and eta_winding is not None
        else None
    )
    winding_lcm = (
        lcm(abs(alpha_winding), abs(eta_winding))
        if alpha_winding is not None and eta_winding is not None
        else None
    )
    orientation_orbits = spatial_orientation_orbits_from_windings(
        alpha_winding=alpha_winding,
        eta_winding=eta_winding,
        period=rule.period,
    )
    allowed_count = sum(orbit.transport_allowed for orbit in orientation_orbits)
    sign_coupled = allowed_count == 2 and all(
        orbit.alpha_sign == orbit.eta_sign
        for orbit in orientation_orbits
        if orbit.transport_allowed
    )
    alpha_projector, eta_projector = alpha_eta_projectors(rule)
    coarse_split = alpha_projector.rank() == 6 and eta_projector.rank() == 4
    reconstructs = local_hopping_reconstructs_transfer(rule)
    unitary = transfer_is_unitary_on_samples(rule)

    if not reconstructs:
        route_label = "spatial_local_hopping_not_reconstructed"
    elif not unitary:
        route_label = "spatial_local_hopping_not_unitary"
    elif not coarse_split:
        route_label = "spatial_local_hopping_no_coarse_6_4_band_split"
    elif sign_coupled:
        route_label = "spatial_local_hopping_signs_coupled"
    else:
        route_label = "spatial_local_hopping_signs_uncoupled"

    return Spatial1DLocalHoppingCertificate(
        rule_name=rule.name,
        hopping_term_count=len(terms),
        hopping_shifts=tuple(term.shift for term in terms),
        hopping_locality_radius=max(abs(term.shift) for term in terms),
        mode_windings=windings,
        computed_alpha_winding=alpha_winding,
        computed_eta_winding=eta_winding,
        computed_winding_gcd=winding_gcd,
        computed_winding_lcm=winding_lcm,
        reconstructs_transfer_on_samples=reconstructs,
        transfer_unitary_on_samples=unitary,
        coarse_6_4_band_split=coarse_split,
        orientation_choices_before_transport=len(orientation_orbits),
        orientation_choices_after_transport=allowed_count,
        orientation_orbits=orientation_orbits,
        sign_coupled_to_global_pm=sign_coupled,
        route_label=route_label,
    )


def _coefficient_center_diagnostics(
    layer: SpatialLocalQCALayer1D,
) -> tuple[int, int, tuple[int, ...], int, bool]:
    coefficient_matrices = tuple(term.matrix for term in layer.terms)
    algebra = generated_algebra_basis(coefficient_matrices, dimension=layer.dimension)
    center = center_basis_of_algebra(algebra, dimension=layer.dimension)
    center_solved, idempotents = solve_central_idempotents(
        center,
        max_center_dimension=8,
        dimension=layer.dimension,
    )
    ranks = tuple(item.rank for item in idempotents) if center_solved else ()
    lower_rank_count = sum(rank not in (0, 4, 6, 10) for rank in ranks)
    coarse_pair = ranks == (0, 4, 6, 10)
    return len(algebra), len(center), ranks, lower_rank_count, coarse_pair


def _is_obvious_seed_projector(matrix: sp.Matrix) -> bool:
    if matrix == sp.zeros(matrix.rows) or matrix == identity(matrix.rows):
        return False
    if matrix * matrix != matrix:
        return False
    if matrix != matrix.T:
        return False
    if any(value not in (0, 1) for value in matrix):
        return False
    rank = matrix.rank()
    return 0 < rank < matrix.rows


def _seeded_coefficient_witnesses(
    layer: SpatialLocalQCALayer1D,
) -> tuple[str, ...]:
    p_alpha, p_eta = split_projectors_3_2()
    witnesses = []
    for term in layer.terms:
        if term.matrix == p_alpha:
            witnesses.append(f"shift_{term.shift}:P_alpha")
        elif term.matrix == p_eta:
            witnesses.append(f"shift_{term.shift}:P_eta")
        elif _is_obvious_seed_projector(term.matrix):
            witnesses.append(
                f"shift_{term.shift}:diagonal_projector_rank_{term.matrix.rank()}"
            )
    return tuple(witnesses)


def _matrix_in_qsqrt3_span(matrix: sp.Matrix, basis: tuple[tuple[tuple[sp.Rational, sp.Rational], ...], ...]) -> bool:
    return _qmatrix_in_span(_qmatrix_from_sympy(matrix), basis)


def _relative_block_flips(
    *,
    candidate_signs: tuple[int, ...],
    compatible_signs: tuple[int, ...],
    alpha_modes: tuple[int, ...],
    eta_modes: tuple[int, ...],
) -> tuple[int | None, int | None]:
    alpha_flips = {compatible_signs[mode] * candidate_signs[mode] for mode in alpha_modes}
    eta_flips = {compatible_signs[mode] * candidate_signs[mode] for mode in eta_modes}
    return (
        next(iter(alpha_flips)) if len(alpha_flips) == 1 else None,
        next(iter(eta_flips)) if len(eta_flips) == 1 else None,
    )


def _spatial_transport_allows_compatible_signs(
    *,
    candidate_signs: tuple[int, ...],
    compatible_signs: tuple[int, ...],
    alpha_modes: tuple[int, ...],
    eta_modes: tuple[int, ...],
) -> bool:
    alpha_flip, eta_flip = _relative_block_flips(
        candidate_signs=candidate_signs,
        compatible_signs=compatible_signs,
        alpha_modes=alpha_modes,
        eta_modes=eta_modes,
    )
    return alpha_flip is not None and eta_flip is not None and alpha_flip == eta_flip


def spatial_1d_combined_route_certificate(
    rule: SpatialTransferRule1D | None = None,
) -> Spatial1DCombinedRouteCertificate:
    rule = rule if rule is not None else spatial_alpha_prototype()
    candidate = floquet_alpha_noncommuting_candidates(pattern_index=0)[0]
    layer = spatial_1d_combined_local_qca_layer(rule)
    u1 = floquet_alpha_operator(candidate.pattern)
    u2 = floquet_alpha_noncommuting_twist_operator(candidate)
    p_alpha, p_eta = floquet_alpha_spectral_projectors(candidate.pattern)
    compatible_signs = _compatible_pair_orientation_sign_choices(candidate)
    transported_signs = tuple(
        signs
        for signs in compatible_signs
        if _spatial_transport_allows_compatible_signs(
            candidate_signs=candidate.orientation_signs,
            compatible_signs=signs,
            alpha_modes=candidate.pattern.alpha_modes,
            eta_modes=candidate.pattern.eta_modes,
        )
    )
    coefficient_qbasis_solved, coefficient_qbasis = _qgenerated_algebra_basis(
        tuple(_qmatrix_from_sympy(term.matrix) for term in layer.terms),
        max_dimension=100,
    )
    joint_qbasis_solved, joint_qbasis = _qgenerated_algebra_basis(
        tuple(_qmatrix_from_sympy(matrix) for matrix in (u1, u2, *(term.matrix for term in layer.terms))),
        max_dimension=100,
    )
    generated_transported_j_count = (
        sum(
            _matrix_in_qsqrt3_span(pair_orientation_j_operator(signs), joint_qbasis)
            for signs in transported_signs
        )
        if joint_qbasis_solved
        else 0
    )
    coefficient_generates_projectors = (
        coefficient_qbasis_solved
        and _matrix_in_qsqrt3_span(p_alpha, coefficient_qbasis)
        and _matrix_in_qsqrt3_span(p_eta, coefficient_qbasis)
    )
    onsite_algebra = generated_algebra_basis((u1, u2), dimension=rule.dimension)
    onsite_center = center_basis_of_algebra(onsite_algebra, dimension=rule.dimension)
    center_solved, onsite_idempotents = solve_central_idempotents(
        onsite_center,
        max_center_dimension=8,
        dimension=rule.dimension,
    )
    onsite_ranks = tuple(item.rank for item in onsite_idempotents) if center_solved else ()
    lower_rank_count = sum(rank not in (0, 4, 6, 10) for rank in onsite_ranks)
    transported_commute = True
    transported_matrices = tuple(pair_orientation_j_operator(signs) for signs in transported_signs)
    for sample in range(rule.period):
        symbol = local_qca_symbol_at_root(layer, sample)
        for matrix in transported_matrices:
            if sp.simplify(symbol * matrix - matrix * symbol) != sp.zeros(rule.dimension):
                transported_commute = False
    sign_coupled = len(transported_signs) == 2
    topological_shape = (
        layer.locality_radius < rule.period
        and local_qca_laurent_orthogonal(layer)
        and local_qca_symbol_unitary_on_samples(layer)
        and onsite_ranks == (0, 4, 6, 10)
        and lower_rank_count == 0
        and len(compatible_signs) == 4
        and sign_coupled
        and transported_commute
    )
    strict_bridge = topological_shape and generated_transported_j_count == 2

    if not topological_shape:
        route_label = "combined_route_shape_failed"
    elif strict_bridge:
        route_label = "combined_route_strict_bridge_candidate"
    else:
        route_label = "combined_route_signs_coupled_but_j_not_rule_generated"

    return Spatial1DCombinedRouteCertificate(
        rule_name=rule.name,
        onsite_candidate_name=candidate.name,
        layer_name=layer.name,
        period=rule.period,
        qca_term_count=len(layer.terms),
        qca_shifts=tuple(term.shift for term in layer.terms),
        qca_locality_radius=layer.locality_radius,
        finite_radius=bool(layer.terms) and layer.locality_radius < rule.period,
        coefficient_matrices_real=all(is_real_matrix(term.matrix) for term in layer.terms),
        laurent_orthogonal=local_qca_laurent_orthogonal(layer),
        symbol_unitary_on_samples=local_qca_symbol_unitary_on_samples(layer),
        onsite_generated_algebra_dimension=len(onsite_algebra),
        onsite_center_dimension=len(onsite_center),
        onsite_central_idempotent_ranks=onsite_ranks,
        onsite_lower_rank_central_idempotents=lower_rank_count,
        onsite_compatible_j_count=len(compatible_signs),
        transported_compatible_j_count=len(transported_signs),
        transported_j_commute_on_samples=transported_commute,
        sign_coupled_to_global_pm=sign_coupled,
        coefficient_algebra_dimension=len(coefficient_qbasis) if coefficient_qbasis_solved else 0,
        coefficient_algebra_generates_alpha_eta_projectors=coefficient_generates_projectors,
        joint_rule_algebra_dimension=len(joint_qbasis) if joint_qbasis_solved else 0,
        joint_rule_generated_transported_j_count=generated_transported_j_count,
        topological_pm_shape_candidate=topological_shape,
        strict_bridge_candidate=strict_bridge,
        route_label=route_label,
        load_bearing_qca_bridge=strict_bridge,
    )


def _spatial_unseeded_candidate_certificate(
    layer: SpatialLocalQCALayer1D,
    rule: SpatialTransferRule1D,
) -> Spatial1DUnseededCandidateCertificate:
    windings = mode_windings_from_terms(layer.terms, mode_count=rule.mode_count)
    alpha_winding = _common_sector_winding(windings, rule.alpha_modes) if windings else None
    eta_winding = _common_sector_winding(windings, rule.eta_modes) if windings else None
    winding_gcd = (
        gcd(abs(alpha_winding), abs(eta_winding))
        if alpha_winding is not None and eta_winding is not None
        else None
    )
    winding_lcm = (
        lcm(abs(alpha_winding), abs(eta_winding))
        if alpha_winding is not None and eta_winding is not None
        else None
    )
    orientation_orbits = spatial_orientation_orbits_from_windings(
        alpha_winding=alpha_winding,
        eta_winding=eta_winding,
        period=rule.period,
    )
    allowed_count = sum(orbit.transport_allowed for orbit in orientation_orbits)
    sign_coupled = allowed_count == 2 and all(
        orbit.alpha_sign == orbit.eta_sign
        for orbit in orientation_orbits
        if orbit.transport_allowed
    )
    (
        coefficient_algebra_dimension,
        coefficient_center_dimension,
        central_idempotent_ranks,
        lower_rank_central_idempotents,
        coarse_pair,
    ) = _coefficient_center_diagnostics(layer)
    finite_radius = bool(layer.terms) and layer.locality_radius < rule.period
    coefficient_matrices_real = all(is_real_matrix(term.matrix) for term in layer.terms)
    laurent_orthogonal = local_qca_laurent_orthogonal(layer)
    seeded_witnesses = _seeded_coefficient_witnesses(layer)
    seeded_guardrail_passed = not seeded_witnesses
    strict_bridge = (
        finite_radius
        and coefficient_matrices_real
        and laurent_orthogonal
        and seeded_guardrail_passed
        and coarse_pair
        and lower_rank_central_idempotents == 0
        and sign_coupled
    )

    if not finite_radius:
        route_label = "unseeded_spatial_not_finite_radius"
    elif not coefficient_matrices_real:
        route_label = "unseeded_spatial_not_real"
    elif not laurent_orthogonal:
        route_label = "unseeded_spatial_not_laurent_orthogonal"
    elif not seeded_guardrail_passed:
        route_label = "unseeded_spatial_seeded_coefficient_rejected"
    elif lower_rank_central_idempotents:
        route_label = "unseeded_spatial_lower_rank_center_rejected"
    elif not coarse_pair:
        route_label = "unseeded_spatial_no_coarse_6_4_center"
    elif not sign_coupled:
        route_label = "unseeded_spatial_no_sign_coupling"
    else:
        route_label = "unseeded_spatial_bridge_candidate"

    return Spatial1DUnseededCandidateCertificate(
        candidate_name=layer.name,
        layer_name=layer.name,
        qca_shifts=tuple(term.shift for term in layer.terms),
        qca_locality_radius=layer.locality_radius,
        finite_radius=finite_radius,
        coefficient_matrices_real=coefficient_matrices_real,
        laurent_orthogonal=laurent_orthogonal,
        seeded_coefficient_guardrail_passed=seeded_guardrail_passed,
        seeded_coefficient_witnesses=seeded_witnesses,
        coefficient_algebra_dimension=coefficient_algebra_dimension,
        coefficient_center_dimension=coefficient_center_dimension,
        central_idempotent_ranks=central_idempotent_ranks,
        lower_rank_central_idempotents=lower_rank_central_idempotents,
        coarse_6_4_center_pair=coarse_pair,
        mode_windings=windings,
        computed_alpha_winding=alpha_winding,
        computed_eta_winding=eta_winding,
        computed_winding_gcd=winding_gcd,
        computed_winding_lcm=winding_lcm,
        orientation_choices_before_transport=len(orientation_orbits),
        orientation_choices_after_transport=allowed_count,
        sign_coupled_to_global_pm=sign_coupled,
        strict_bridge_candidate=strict_bridge,
        route_label=route_label,
    )


def spatial_1d_unseeded_search_summary(
    rule: SpatialTransferRule1D | None = None,
) -> Spatial1DUnseededSearchSummary:
    rule = rule if rule is not None else spatial_alpha_prototype()
    candidates = tuple(
        _spatial_unseeded_candidate_certificate(layer, rule)
        for layer in spatial_1d_unseeded_candidate_layers(rule)
    )
    unseeded = tuple(
        candidate
        for candidate in candidates
        if candidate.seeded_coefficient_guardrail_passed
    )
    unseeded_strict_count = sum(candidate.strict_bridge_candidate for candidate in unseeded)
    route_label = (
        "unseeded_spatial_bridge_candidate_found"
        if unseeded_strict_count
        else "unseeded_spatial_no_bridge_candidates"
    )
    return Spatial1DUnseededSearchSummary(
        candidate_count=len(candidates),
        unseeded_candidate_count=len(unseeded),
        seeded_guardrail_rejections=sum(
            not candidate.seeded_coefficient_guardrail_passed
            for candidate in candidates
        ),
        laurent_orthogonal_candidates=sum(
            candidate.laurent_orthogonal for candidate in candidates
        ),
        unseeded_coarse_6_4_center_candidates=sum(
            candidate.coarse_6_4_center_pair for candidate in unseeded
        ),
        unseeded_sign_coupled_candidates=sum(
            candidate.sign_coupled_to_global_pm for candidate in unseeded
        ),
        unseeded_strict_bridge_candidates=unseeded_strict_count,
        lower_rank_center_rejections=sum(
            candidate.lower_rank_central_idempotents > 0 for candidate in unseeded
        ),
        route_label=route_label,
        candidates=candidates,
    )


def spatial_1d_local_qca_certificate(
    rule: SpatialTransferRule1D | None = None,
) -> Spatial1DLocalQCACertificate:
    rule = rule if rule is not None else spatial_alpha_prototype()
    layer = spatial_alpha_local_qca_layer(rule)
    windings = mode_windings_from_terms(layer.terms, mode_count=rule.mode_count)
    alpha_winding = _common_sector_winding(windings, rule.alpha_modes) if windings else None
    eta_winding = _common_sector_winding(windings, rule.eta_modes) if windings else None
    winding_gcd = (
        gcd(abs(alpha_winding), abs(eta_winding))
        if alpha_winding is not None and eta_winding is not None
        else None
    )
    winding_lcm = (
        lcm(abs(alpha_winding), abs(eta_winding))
        if alpha_winding is not None and eta_winding is not None
        else None
    )
    orientation_orbits = spatial_orientation_orbits_from_windings(
        alpha_winding=alpha_winding,
        eta_winding=eta_winding,
        period=rule.period,
    )
    allowed_count = sum(orbit.transport_allowed for orbit in orientation_orbits)
    sign_coupled = allowed_count == 2 and all(
        orbit.alpha_sign == orbit.eta_sign
        for orbit in orientation_orbits
        if orbit.transport_allowed
    )
    (
        coefficient_algebra_dimension,
        coefficient_center_dimension,
        central_idempotent_ranks,
        lower_rank_central_idempotents,
        coarse_pair,
    ) = _coefficient_center_diagnostics(layer)
    finite_radius = bool(layer.terms) and layer.locality_radius < rule.period
    coefficient_matrices_real = all(is_real_matrix(term.matrix) for term in layer.terms)
    laurent_orthogonal = local_qca_laurent_orthogonal(layer)
    reconstructs = local_qca_symbol_reconstructs_transfer(layer, rule)
    unitary = local_qca_symbol_unitary_on_samples(layer)

    if not finite_radius:
        route_label = "spatial_local_qca_not_finite_radius"
    elif not coefficient_matrices_real:
        route_label = "spatial_local_qca_not_real"
    elif not laurent_orthogonal:
        route_label = "spatial_local_qca_not_laurent_orthogonal"
    elif not reconstructs:
        route_label = "spatial_local_qca_symbol_mismatch"
    elif not coarse_pair or lower_rank_central_idempotents:
        route_label = "spatial_local_qca_center_locking_fail"
    elif sign_coupled:
        route_label = "spatial_local_qca_signs_coupled_not_load_bearing"
    else:
        route_label = "spatial_local_qca_signs_uncoupled"

    return Spatial1DLocalQCACertificate(
        rule_name=rule.name,
        layer_name=layer.name,
        period=rule.period,
        dimension=layer.dimension,
        qca_term_count=len(layer.terms),
        qca_shifts=tuple(term.shift for term in layer.terms),
        qca_locality_radius=layer.locality_radius,
        finite_radius=finite_radius,
        coefficient_matrices_real=coefficient_matrices_real,
        laurent_orthogonal=laurent_orthogonal,
        symbol_reconstructs_transfer_on_samples=reconstructs,
        symbol_unitary_on_samples=unitary,
        coefficient_algebra_dimension=coefficient_algebra_dimension,
        coefficient_center_dimension=coefficient_center_dimension,
        central_idempotent_ranks=central_idempotent_ranks,
        lower_rank_central_idempotents=lower_rank_central_idempotents,
        coarse_6_4_center_pair=coarse_pair,
        mode_windings=windings,
        computed_alpha_winding=alpha_winding,
        computed_eta_winding=eta_winding,
        computed_winding_gcd=winding_gcd,
        computed_winding_lcm=winding_lcm,
        orientation_choices_before_transport=len(orientation_orbits),
        orientation_choices_after_transport=allowed_count,
        orientation_orbits=orientation_orbits,
        sign_coupled_to_global_pm=sign_coupled,
        strict_bridge_candidates=0,
        route_label=route_label,
    )
