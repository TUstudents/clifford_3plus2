"""Session 01 exact three-clock infrastructure.

This sidecar starts deliberately smaller than the previous quark bath attempts.
It provides only the finite-clock algebra needed to experiment with a simpler
quark model:

    three independent cyclic clocks,
    exact shift/phase matrices,
    exact word composition and closure orders.

No quark masses, CKM angles, or sector assignments are derived here.  The point
is to freeze the clock algebra before adding physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from math import gcd, prod

import sympy as sp


def lcm_pair(left: int, right: int) -> int:
    """Return the least common multiple of two positive integers."""

    if left <= 0 or right <= 0:
        raise ValueError("lcm inputs must be positive")
    return left * right // gcd(left, right)


def lcm_many(values: tuple[int, ...]) -> int:
    """Return the least common multiple of one or more positive integers."""

    if not values:
        raise ValueError("need at least one value")
    return reduce(lcm_pair, values)


def matrix_equal(left: sp.Matrix, right: sp.Matrix) -> bool:
    """Return true when two matrices are exactly equal after simplification."""

    return all(sp.simplify(entry) == 0 for entry in left - right)


@dataclass(frozen=True)
class ClockSpec:
    """One finite cyclic clock."""

    name: str
    order: int

    def __post_init__(self) -> None:
        """Validate clock order."""

        if self.order < 2:
            raise ValueError("clock order must be at least 2")

    @property
    def omega(self) -> sp.Expr:
        """Return the exact primitive clock phase."""

        return sp.exp(2 * sp.pi * sp.I / self.order)


def shift_matrix(order: int) -> sp.Matrix:
    """Return the cyclic shift matrix ``X`` of a clock of given order."""

    matrix = sp.zeros(order, order)
    for index in range(order):
        matrix[(index + 1) % order, index] = 1
    return matrix


def phase_matrix(order: int) -> sp.Matrix:
    """Return the cyclic phase matrix ``Z`` of a clock of given order."""

    omega = sp.exp(2 * sp.pi * sp.I / order)
    return sp.diag(*(omega**index for index in range(order)))


def weyl_relation_holds(order: int) -> bool:
    """Return whether ``Z X = omega X Z`` exactly for one clock."""

    shift = shift_matrix(order)
    phase = phase_matrix(order)
    omega = sp.exp(2 * sp.pi * sp.I / order)
    return matrix_equal(phase * shift, omega * shift * phase)


@dataclass(frozen=True)
class ThreeClockSystem:
    """A product of exactly three finite cyclic clocks."""

    clocks: tuple[ClockSpec, ClockSpec, ClockSpec]

    @property
    def orders(self) -> tuple[int, int, int]:
        """Return the three clock orders."""

        return tuple(clock.order for clock in self.clocks)

    @property
    def names(self) -> tuple[str, str, str]:
        """Return the three clock names."""

        return tuple(clock.name for clock in self.clocks)

    @property
    def dimension(self) -> int:
        """Return the product Hilbert-space dimension."""

        return prod(self.orders)

    @property
    def closure_order(self) -> int:
        """Return the common closure order of the product clock."""

        return lcm_many(self.orders)

    def identity_word(self) -> ClockWord:
        """Return the identity word."""

        return ClockWord(system=self, exponents=(0, 0, 0))

    def word(self, exponents: tuple[int, int, int]) -> ClockWord:
        """Return a normalized word in the product clock."""

        return ClockWord(system=self, exponents=exponents)

    def tensor_shift(self, clock_index: int) -> sp.Matrix:
        """Return the product-space shift matrix for one clock."""

        return self._tensor_operator(clock_index, shift_matrix(self.orders[clock_index]))

    def tensor_phase(self, clock_index: int) -> sp.Matrix:
        """Return the product-space phase matrix for one clock."""

        return self._tensor_operator(clock_index, phase_matrix(self.orders[clock_index]))

    def _tensor_operator(self, clock_index: int, local: sp.Matrix) -> sp.Matrix:
        """Lift a local clock operator into the three-clock tensor product."""

        if clock_index not in (0, 1, 2):
            raise ValueError("clock_index must be 0, 1, or 2")
        factors: list[sp.Matrix] = []
        for index, order in enumerate(self.orders):
            factors.append(local if index == clock_index else sp.eye(order))
        return sp.kronecker_product(*factors)

    def independent_weyl_relations_hold(self) -> bool:
        """Return whether all three local Weyl relations hold."""

        return all(weyl_relation_holds(order) for order in self.orders)


@dataclass(frozen=True)
class ClockWord:
    """One element of a three-clock product group."""

    system: ThreeClockSystem
    exponents: tuple[int, int, int]

    def __post_init__(self) -> None:
        """Normalize exponents modulo their clock orders."""

        if len(self.exponents) != 3:
            raise ValueError("three-clock words need exactly three exponents")
        normalized = tuple(
            exponent % order for exponent, order in zip(self.exponents, self.system.orders)
        )
        object.__setattr__(self, "exponents", normalized)

    def __mul__(self, other: ClockWord) -> ClockWord:
        """Compose two words in the same clock system."""

        if self.system != other.system:
            raise ValueError("cannot multiply words from different systems")
        return ClockWord(
            system=self.system,
            exponents=tuple(left + right for left, right in zip(self.exponents, other.exponents)),
        )

    def inverse(self) -> ClockWord:
        """Return the inverse word."""

        return ClockWord(
            system=self.system,
            exponents=tuple(-exponent for exponent in self.exponents),
        )

    def is_identity(self) -> bool:
        """Return whether this word is the identity."""

        return self.exponents == (0, 0, 0)

    def order(self) -> int:
        """Return the exact closure order of this word."""

        local_orders: list[int] = []
        for exponent, clock_order in zip(self.exponents, self.system.orders):
            if exponent == 0:
                local_orders.append(1)
            else:
                local_orders.append(clock_order // gcd(clock_order, exponent))
        return lcm_many(tuple(local_orders))


def default_three_clock_system() -> ThreeClockSystem:
    """Return the initial three-clock prototype.

    The order-3 choice is only an infrastructure default.  It is not yet a
    quark-sector theorem.
    """

    return ThreeClockSystem(
        clocks=(
            ClockSpec("clock_a", 3),
            ClockSpec("clock_b", 3),
            ClockSpec("clock_c", 3),
        )
    )


@dataclass(frozen=True)
class ClockSpinePayload:
    """Session 01 three-clock infrastructure verdict."""

    final_verdict: str
    clock_count: int
    default_orders: tuple[int, int, int]
    default_dimension: int
    default_closure_order: int
    independent_weyl_relations_pass: bool
    word_composition_pass: bool
    word_inverse_pass: bool
    word_closure_order_pass: bool
    custom_orders_supported: bool
    no_quark_masses_claimed: bool
    interpretation: str


def custom_orders_supported() -> bool:
    """Return whether the infrastructure supports non-uniform clock orders."""

    system = ThreeClockSystem(
        clocks=(
            ClockSpec("z2_control", 2),
            ClockSpec("z3_control", 3),
            ClockSpec("z5_control", 5),
        )
    )
    word = system.word((1, 1, 1))
    return system.dimension == 30 and system.closure_order == 30 and word.order() == 30


def clock_spine_payload() -> ClockSpinePayload:
    """Return the Session 01 three-clock infrastructure payload."""

    system = default_three_clock_system()
    identity = system.identity_word()
    word_a = system.word((1, 0, 0))
    word_b = system.word((0, 1, 0))
    word_ab = system.word((1, 1, 0))

    word_composition = word_a * word_b == word_ab and identity * word_a == word_a
    word_inverse = (word_ab * word_ab.inverse()).is_identity()
    word_closure = (
        word_a.order() == 3
        and word_ab.order() == 3
        and system.word((1, 1, 1)).order() == 3
    )
    weyl = system.independent_weyl_relations_hold()
    custom = custom_orders_supported()

    checks_pass = (
        len(system.clocks) == 3
        and system.orders == (3, 3, 3)
        and system.dimension == 27
        and system.closure_order == 3
        and weyl
        and word_composition
        and word_inverse
        and word_closure
        and custom
    )

    if checks_pass:
        final_verdict = "THREECLOCKS_INFRASTRUCTURE_PASS"
        interpretation = (
            "The threeclocks sidecar now has a minimal exact finite-clock "
            "spine: three independent cyclic clocks, exact Weyl shift/phase "
            "matrices, word composition, inverses, and closure orders.  The "
            "default Z3^3 prototype is infrastructure only; no quark mass or "
            "mixing theorem is claimed."
        )
    else:
        final_verdict = "THREECLOCKS_INFRASTRUCTURE_KILL"
        interpretation = (
            "The three-clock count, Weyl relations, word group operations, "
            "closure orders, or custom-order controls failed."
        )

    return ClockSpinePayload(
        final_verdict=final_verdict,
        clock_count=len(system.clocks),
        default_orders=system.orders,
        default_dimension=system.dimension,
        default_closure_order=system.closure_order,
        independent_weyl_relations_pass=weyl,
        word_composition_pass=word_composition,
        word_inverse_pass=word_inverse,
        word_closure_order_pass=word_closure,
        custom_orders_supported=custom,
        no_quark_masses_claimed=True,
        interpretation=interpretation,
    )
