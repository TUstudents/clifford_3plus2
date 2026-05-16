"""Lab B R^6 structural and structural-wall regression candidates."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import product

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.lepton.carrier import (
    lab_b_complex_structure,
    lab_b_physical_wall_complex_structure,
    lab_b_physical_wall_site_projectors,
    lab_b_singlet_doublet_projectors,
)
from clifford_3plus2_d5.lepton.gauge import (
    su2_l_u1_y_generators_r6_for_singlet,
    su2_l_u1_y_generators_r6,
    transported_gauge_pairs,
)
from clifford_3plus2_d5.lepton.wall import (
    LocalityConstraint,
    LocalityModel,
    WallContext,
    solve_t_intertwiner_orthogonal,
    spectrum_matches,
)
from clifford_3plus2_d5.qca.rule_verdict import RuleLayerInput


LAB_B_DIMENSION = 6
LAB_B_MODE_COUNT = 3


@dataclass(frozen=True)
class LabBRegressionCandidate:
    name: str
    layers: tuple[RuleLayerInput, ...]
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class LabBStructuralWallCandidate(LabBRegressionCandidate):
    wall_context: WallContext | None = None


@dataclass(frozen=True)
class LabBDomainWallCandidate(LabBRegressionCandidate):
    wall_context: WallContext | None = None
    left_singlet_mode: int = 0
    right_singlet_mode: int = 1
    transition_tier: str = "no_solutions"
    transition_index: int = 0


@dataclass(frozen=True)
class LabBPhysicalDomainWallCandidate(LabBDomainWallCandidate):
    """Two-site R^12 domain-wall candidate with physical side projectors."""


def _same_matrix(left: sp.Matrix, right: sp.Matrix) -> bool:
    return (left - right).applyfunc(sp.simplify) == sp.zeros(left.rows, left.cols)


def _block_diag(*blocks: sp.Matrix) -> sp.Matrix:
    return sp.diag(*blocks)


def _angle_for_order(order: int) -> sp.Expr:
    if order <= 0:
        raise ValueError("angle order must be positive")
    return 2 * sp.pi * sp.Rational(1, order)


def _clock_rotation_block(angle: sp.Expr) -> sp.Matrix:
    cosine = sp.simplify(sp.cos(angle))
    sine = sp.simplify(sp.sin(angle))
    return sp.Matrix([[cosine, -sine], [sine, cosine]])


def lab_b_pair_rotation_matrix(orders: Sequence[int]) -> sp.Matrix:
    """Return a pair rotation on the three R^2 clock planes."""

    if len(orders) != LAB_B_MODE_COUNT:
        raise ValueError("Lab B pair rotation requires three mode orders")
    matrix = sp.zeros(LAB_B_DIMENSION)
    for mode, order in enumerate(orders):
        block = _clock_rotation_block(_angle_for_order(order))
        x_index = mode
        y_index = LAB_B_MODE_COUNT + mode
        matrix[x_index, x_index] = block[0, 0]
        matrix[x_index, y_index] = block[0, 1]
        matrix[y_index, x_index] = block[1, 0]
        matrix[y_index, y_index] = block[1, 1]
    return matrix.applyfunc(sp.simplify)


def lab_b_signed_twist_matrix(signs: Sequence[int]) -> sp.Matrix:
    if len(signs) != LAB_B_MODE_COUNT:
        raise ValueError("Lab B signed twist requires three signs")
    if any(sign not in (-1, 1) for sign in signs):
        raise ValueError("signed twist entries must be ±1")
    return sp.diag(*signs, *signs)


def lab_b_mode_pair_swap_matrix(left_mode: int, right_mode: int) -> sp.Matrix:
    if left_mode == right_mode:
        raise ValueError("mode swap requires distinct modes")
    if not (0 <= left_mode < LAB_B_MODE_COUNT and 0 <= right_mode < LAB_B_MODE_COUNT):
        raise ValueError("mode index out of range")
    matrix = sp.eye(LAB_B_DIMENSION)
    for offset in (0, LAB_B_MODE_COUNT):
        left = offset + left_mode
        right = offset + right_mode
        matrix[left, left] = 0
        matrix[right, right] = 0
        matrix[left, right] = 1
        matrix[right, left] = 1
    return matrix


def lab_b_doublet_swap_matrix() -> sp.Matrix:
    return lab_b_doublet_swap_matrix_for_singlet(0)


def lab_b_doublet_swap_matrix_for_singlet(singlet_mode: int) -> sp.Matrix:
    doublet_modes = tuple(mode for mode in range(LAB_B_MODE_COUNT) if mode != singlet_mode)
    return lab_b_mode_pair_swap_matrix(doublet_modes[0], doublet_modes[1])


def _orders_for_split(
    *,
    singlet_mode: int,
    singlet_order: int,
    doublet_orders: Sequence[int],
) -> tuple[int, int, int]:
    if len(doublet_orders) != 2:
        raise ValueError("Lab B split needs two doublet orders")
    orders = [0, 0, 0]
    orders[singlet_mode] = singlet_order
    doublet_iter = iter(doublet_orders)
    for mode in range(LAB_B_MODE_COUNT):
        if mode != singlet_mode:
            orders[mode] = next(doublet_iter)
    return orders[0], orders[1], orders[2]


def _layer(name: str, matrix: sp.Matrix) -> RuleLayerInput:
    return RuleLayerInput(name=name, matrix=matrix, support=(0,), locality_radius=0)


def _wall_layer(name: str, matrix: sp.Matrix) -> RuleLayerInput:
    return RuleLayerInput(name=name, matrix=matrix, support=(0, 1), locality_radius=1)


def _checked_frame_gauge_pairs(
    transition: sp.Matrix,
    *,
    left_singlet_mode: int,
    right_singlet_mode: int,
) -> tuple[tuple[sp.Matrix, sp.Matrix], ...]:
    """Return explicit gauge pairs after checking the independent right frame."""

    left_gauge = su2_l_u1_y_generators_r6_for_singlet(left_singlet_mode)
    right_gauge = su2_l_u1_y_generators_r6_for_singlet(right_singlet_mode)
    pairs = tuple(zip(left_gauge, right_gauge, strict=True))
    for left_generator, right_generator in pairs:
        transported = (transition * left_generator * transition.T).applyfunc(sp.simplify)
        if not _same_matrix(transported, right_generator):
            raise ValueError("transition does not map the left gauge frame to the declared right frame")
    return pairs


def lab_b_physical_wall_transition(internal_transition: sp.Matrix) -> sp.Matrix:
    """Return the two-site transition that transports left and right frames."""

    zero = sp.zeros(LAB_B_DIMENSION)
    return sp.Matrix.vstack(
        sp.Matrix.hstack(zero, internal_transition.T),
        sp.Matrix.hstack(internal_transition, zero),
    )


def lab_b_strict_regression_candidates() -> tuple[LabBRegressionCandidate, ...]:
    """Return the fixed small Route-1 regression panel for Lab B strict."""

    definitions = (
        (
            "route1_rot_3_3_4_swap",
            (
                _layer("rot_3_3_4", lab_b_pair_rotation_matrix((3, 3, 4))),
                _layer("doublet_swap", lab_b_doublet_swap_matrix()),
            ),
        ),
        (
            "route1_rot_3_4_3_swap",
            (
                _layer("rot_3_4_3", lab_b_pair_rotation_matrix((3, 4, 3))),
                _layer("doublet_swap", lab_b_doublet_swap_matrix()),
            ),
        ),
        (
            "route1_rot_3_3_4_swap_twist",
            (
                _layer("rot_3_3_4", lab_b_pair_rotation_matrix((3, 3, 4))),
                _layer("doublet_swap", lab_b_doublet_swap_matrix()),
                _layer("doublet_twist", lab_b_signed_twist_matrix((1, -1, 1))),
            ),
        ),
        (
            "route1_rot_4_3_4_swap",
            (
                _layer("rot_4_3_4", lab_b_pair_rotation_matrix((4, 3, 4))),
                _layer("doublet_swap", lab_b_doublet_swap_matrix()),
            ),
        ),
        (
            "route1_rot_3_4_4_swap_twist",
            (
                _layer("rot_3_4_4", lab_b_pair_rotation_matrix((3, 4, 4))),
                _layer("doublet_swap", lab_b_doublet_swap_matrix()),
                _layer("doublet_twist", lab_b_signed_twist_matrix((1, 1, -1))),
            ),
        ),
    )
    return tuple(
        LabBRegressionCandidate(
            name=f"lab_b_strict_{name}",
            layers=layers,
            metadata=(("panel", "strict"),),
        )
        for name, layers in definitions
    )


def iter_lab_b_strict_candidates(
    *,
    max_candidates: int | None = None,
) -> Iterable[LabBRegressionCandidate]:
    yielded = 0
    for candidate in lab_b_strict_regression_candidates():
        yield candidate
        yielded += 1
        if max_candidates is not None and yielded >= max_candidates:
            return


def lab_b_structural_wall_locality_constraint() -> LocalityConstraint:
    return LocalityConstraint(
        model=LocalityModel.BLOCK_SUPPORT,
        blocks=((0, LAB_B_DIMENSION),),
        allowed_block_pairs=((0, 0),),
    )


def lab_b_structural_wall_context(transition: sp.Matrix) -> WallContext:
    singlet, doublet = lab_b_singlet_doublet_projectors()
    gauge = su2_l_u1_y_generators_r6()
    j = lab_b_complex_structure()
    right_j = (transition * j * transition.T).applyfunc(sp.simplify)
    return WallContext(
        gauge_pairs=transported_gauge_pairs(transition, gauge),
        left_complex_structure=j,
        right_complex_structure=right_j,
        transition=transition,
        locality_constraint=lab_b_structural_wall_locality_constraint(),
        left_side_projector=singlet,
        right_side_projector=doublet,
    )


def lab_b_domain_wall_context(
    *,
    transition: sp.Matrix,
    left_singlet_mode: int,
    right_singlet_mode: int,
) -> WallContext:
    """Build an internal split-reassignment domain-wall context.

    This first Session 6 model is a single-carrier internal wall, not a
    two-site spatial wall. The side projectors are therefore identity masks and
    are explicitly marked as non-complementary in the context.
    """

    j = lab_b_complex_structure()
    gauge_pairs = _checked_frame_gauge_pairs(
        transition,
        left_singlet_mode=left_singlet_mode,
        right_singlet_mode=right_singlet_mode,
    )
    right_j = (transition * j * transition.T).applyfunc(sp.simplify)
    one = identity(LAB_B_DIMENSION)
    return WallContext(
        gauge_pairs=gauge_pairs,
        left_complex_structure=j,
        right_complex_structure=right_j,
        transition=transition,
        locality_constraint=lab_b_structural_wall_locality_constraint(),
        left_side_projector=one,
        right_side_projector=one,
        side_projectors_must_be_complementary=False,
    )


def lab_b_physical_domain_wall_context(
    *,
    transition: sp.Matrix,
    internal_transition: sp.Matrix,
    left_singlet_mode: int,
    right_singlet_mode: int,
) -> WallContext:
    """Build a two-site R^12 domain-wall context with real side projectors."""

    internal_pairs = _checked_frame_gauge_pairs(
        internal_transition,
        left_singlet_mode=left_singlet_mode,
        right_singlet_mode=right_singlet_mode,
    )
    gauge_pairs = tuple(
        (
            _block_diag(left_generator, right_generator),
            (transition * _block_diag(left_generator, right_generator) * transition.T).applyfunc(
                sp.simplify
            ),
        )
        for left_generator, right_generator in internal_pairs
    )
    j = lab_b_physical_wall_complex_structure()
    right_j = (transition * j * transition.T).applyfunc(sp.simplify)
    left_site, right_site = lab_b_physical_wall_site_projectors()
    return WallContext(
        gauge_pairs=gauge_pairs,
        left_complex_structure=j,
        right_complex_structure=right_j,
        transition=transition,
        locality_constraint=LocalityConstraint(
            model=LocalityModel.SITE_SUPPORT,
            sites=(0, 1),
        ),
        left_side_projector=left_site,
        right_side_projector=right_site,
    )


def _structural_wall_source_pairs() -> tuple[tuple[RuleLayerInput, RuleLayerInput], ...]:
    rotations = (
        _layer("rot_3_3_4", lab_b_pair_rotation_matrix((3, 3, 4))),
        _layer("rot_3_4_3", lab_b_pair_rotation_matrix((3, 4, 3))),
        _layer("rot_4_3_4", lab_b_pair_rotation_matrix((4, 3, 4))),
        _layer("rot_4_4_3", lab_b_pair_rotation_matrix((4, 4, 3))),
    )
    pairs = []
    for left, right in product(rotations, repeat=2):
        if left.name != right.name and spectrum_matches(left.matrix, right.matrix):
            pairs.append((left, right))
    return tuple(pairs)


def _domain_wall_source_pairs(
    *,
    singlet_orders: Sequence[int] = (3, 4),
    doublet_order_pairs: Sequence[tuple[int, int]] = ((4, 6), (3, 6)),
) -> tuple[tuple[RuleLayerInput, RuleLayerInput, int, int], ...]:
    pairs = []
    for left_singlet in range(LAB_B_MODE_COUNT):
        for right_singlet in range(LAB_B_MODE_COUNT):
            if left_singlet == right_singlet:
                continue
            for singlet_order in singlet_orders:
                for doublet_orders in doublet_order_pairs:
                    left_orders = _orders_for_split(
                        singlet_mode=left_singlet,
                        singlet_order=singlet_order,
                        doublet_orders=doublet_orders,
                    )
                    right_orders = _orders_for_split(
                        singlet_mode=right_singlet,
                        singlet_order=singlet_order,
                        doublet_orders=doublet_orders,
                    )
                    left = _layer(
                        f"domain_left_s{left_singlet}_o{'_'.join(str(item) for item in left_orders)}",
                        lab_b_pair_rotation_matrix(left_orders),
                    )
                    right = _layer(
                        f"domain_right_s{right_singlet}_o{'_'.join(str(item) for item in right_orders)}",
                        lab_b_pair_rotation_matrix(right_orders),
                    )
                    if spectrum_matches(left.matrix, right.matrix):
                        pairs.append((left, right, left_singlet, right_singlet))
    return tuple(pairs)


def iter_lab_b_structural_wall_candidates(
    *,
    max_candidates: int | None = None,
    max_pairs: int | None = None,
    max_transitions_per_pair: int = 2,
    include_generic: bool = False,
) -> Iterable[LabBStructuralWallCandidate]:
    if max_transitions_per_pair <= 0:
        raise ValueError("max_transitions_per_pair must be positive")

    yielded_pairs = 0
    yielded_candidates = 0
    locality = lab_b_structural_wall_locality_constraint()
    for left, right in _structural_wall_source_pairs():
        yielded_pairs += 1
        if max_pairs is not None and yielded_pairs > max_pairs:
            return
        transitions, tier = solve_t_intertwiner_orthogonal(
            left.matrix,
            right.matrix,
            locality_constraint=locality,
            include_generic=include_generic,
        )
        for transition_index, transition in enumerate(transitions[:max_transitions_per_pair]):
            transition_layer = _layer(
                f"structural_wall_T_{transition_index}_{tier}_{left.name}_to_{right.name}",
                transition,
            )
            name = (
                f"lab_b_structural_wall_{left.name}__to__{right.name}"
                f"__T{transition_index}_{tier}"
            )
            yield LabBStructuralWallCandidate(
                name=name,
                layers=(left, right, transition_layer),
                metadata=(
                    ("panel", "structural_wall"),
                    ("left", left.name),
                    ("right", right.name),
                    ("transition_tier", tier),
                    ("transition_index", str(transition_index)),
                ),
                wall_context=lab_b_structural_wall_context(transition),
            )
            yielded_candidates += 1
            if max_candidates is not None and yielded_candidates >= max_candidates:
                return


def iter_lab_b_domain_wall_candidates(
    *,
    max_candidates: int | None = None,
    max_pairs: int | None = None,
    max_transitions_per_pair: int = 2,
    include_generic: bool = False,
) -> Iterable[LabBDomainWallCandidate]:
    """Yield internal split-reassignment domain-wall candidates for Lab B."""

    if max_transitions_per_pair <= 0:
        raise ValueError("max_transitions_per_pair must be positive")

    yielded_pairs = 0
    yielded_candidates = 0
    locality = lab_b_structural_wall_locality_constraint()
    for left, right, left_singlet, right_singlet in _domain_wall_source_pairs():
        yielded_pairs += 1
        if max_pairs is not None and yielded_pairs > max_pairs:
            return
        transitions, tier = solve_t_intertwiner_orthogonal(
            left.matrix,
            right.matrix,
            locality_constraint=locality,
            include_generic=include_generic,
        )
        for transition_index, transition in enumerate(transitions[:max_transitions_per_pair]):
            doublet_swap_layer = _layer(
                f"domain_left_doublet_swap_s{left_singlet}",
                lab_b_doublet_swap_matrix_for_singlet(left_singlet),
            )
            transition_layer = _layer(
                f"domain_wall_T_{transition_index}_{tier}_{left.name}_to_{right.name}",
                transition,
            )
            name = (
                f"lab_b_domain_wall_s{left_singlet}_to_s{right_singlet}"
                f"__{left.name}__to__{right.name}__T{transition_index}_{tier}"
            )
            yield LabBDomainWallCandidate(
                name=name,
                layers=(left, right, transition_layer, doublet_swap_layer),
                metadata=(
                    ("panel", "domain_wall"),
                    ("domain_model", "internal_split_reassignment"),
                    ("left_singlet_mode", str(left_singlet)),
                    ("right_singlet_mode", str(right_singlet)),
                    ("transition_tier", tier),
                    ("transition_index", str(transition_index)),
                    ("left_doublet_mixer", doublet_swap_layer.name),
                ),
                wall_context=lab_b_domain_wall_context(
                    transition=transition,
                    left_singlet_mode=left_singlet,
                    right_singlet_mode=right_singlet,
                ),
                left_singlet_mode=left_singlet,
                right_singlet_mode=right_singlet,
                transition_tier=tier,
                transition_index=transition_index,
            )
            yielded_candidates += 1
            if max_candidates is not None and yielded_candidates >= max_candidates:
                return


def iter_lab_b_physical_domain_wall_candidates(
    *,
    max_candidates: int | None = None,
    max_pairs: int | None = None,
    max_transitions_per_pair: int = 2,
    include_generic: bool = False,
) -> Iterable[LabBPhysicalDomainWallCandidate]:
    """Yield two-site R^12 domain-wall candidates with complementary site masks."""

    if max_transitions_per_pair <= 0:
        raise ValueError("max_transitions_per_pair must be positive")

    yielded_pairs = 0
    yielded_candidates = 0
    locality = lab_b_structural_wall_locality_constraint()
    for left, right, left_singlet, right_singlet in _domain_wall_source_pairs():
        yielded_pairs += 1
        if max_pairs is not None and yielded_pairs > max_pairs:
            return
        transitions, tier = solve_t_intertwiner_orthogonal(
            left.matrix,
            right.matrix,
            locality_constraint=locality,
            include_generic=include_generic,
        )
        for transition_index, internal_transition in enumerate(
            transitions[:max_transitions_per_pair]
        ):
            try:
                physical_transition = lab_b_physical_wall_transition(internal_transition)
                wall_context = lab_b_physical_domain_wall_context(
                    transition=physical_transition,
                    internal_transition=internal_transition,
                    left_singlet_mode=left_singlet,
                    right_singlet_mode=right_singlet,
                )
            except ValueError:
                continue

            left_doublet_swap = lab_b_doublet_swap_matrix_for_singlet(left_singlet)
            right_doublet_swap = lab_b_doublet_swap_matrix_for_singlet(right_singlet)
            onsite_layer = _wall_layer(
                f"physical_sites_{left.name}_to_{right.name}",
                _block_diag(left.matrix, right.matrix),
            )
            transition_layer = _wall_layer(
                f"physical_wall_T_{transition_index}_{tier}_{left.name}_to_{right.name}",
                physical_transition,
            )
            doublet_swap_layer = _wall_layer(
                f"physical_doublet_swaps_s{left_singlet}_to_s{right_singlet}",
                _block_diag(left_doublet_swap, right_doublet_swap),
            )
            name = (
                f"lab_b_physical_domain_wall_s{left_singlet}_to_s{right_singlet}"
                f"__{left.name}__to__{right.name}__T{transition_index}_{tier}"
            )
            yield LabBPhysicalDomainWallCandidate(
                name=name,
                layers=(onsite_layer, transition_layer, doublet_swap_layer),
                metadata=(
                    ("panel", "physical_domain_wall"),
                    ("domain_model", "two_site_r12"),
                    ("left_singlet_mode", str(left_singlet)),
                    ("right_singlet_mode", str(right_singlet)),
                    ("transition_tier", tier),
                    ("transition_index", str(transition_index)),
                    ("right_frame_checked", "true"),
                    ("site_projectors", "complementary"),
                    ("center_contract", "known_center_only"),
                ),
                wall_context=wall_context,
                left_singlet_mode=left_singlet,
                right_singlet_mode=right_singlet,
                transition_tier=tier,
                transition_index=transition_index,
            )
            yielded_candidates += 1
            if max_candidates is not None and yielded_candidates >= max_candidates:
                return
