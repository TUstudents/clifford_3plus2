"""Verify the bare BB wedge-domain audit.

This docs-local certificate supports
``src/clifford_3plus2_d5/synthesis/BARE_BB_WEDGE_DOMAIN_AUDIT.md``.
It checks the exact integer geometry of the physical two-face edge:

* ``n = r1+r2`` and ``q = r1-r2`` give the wedge
  ``n >= |q|`` with matching parity;
* diagonal states ``q=0`` leak to ``q=+-2`` for ``n>=2``;
* the adjacent leakage states return to ``q=0``;
* face boundaries block only the outward relative hop, not the inward one.

Run:
    python docs/scripts/verify_bare_bb_wedge_domain_audit.py
"""

from __future__ import annotations


def to_nq(r1: int, r2: int) -> tuple[int, int]:
    return r1 + r2, r1 - r2


def to_r1r2(n: int, q: int) -> tuple[int, int]:
    return (n + q) // 2, (n - q) // 2


def in_wedge(n: int, q: int) -> bool:
    return n >= abs(q) and (n - q) % 2 == 0


def hop(n: int, q: int, sigma1: int, sigma2: int) -> tuple[int, int]:
    return n + sigma1 + sigma2, q + sigma1 - sigma2


def main() -> None:
    max_depth = 8
    wedge_points = [
        (n, q)
        for n in range(max_depth + 1)
        for q in range(-max_depth, max_depth + 1)
        if in_wedge(n, q)
    ]

    for n, q in wedge_points:
        r1, r2 = to_r1r2(n, q)
        assert r1 >= 0 and r2 >= 0
        assert to_nq(r1, r2) == (n, q)

    fixed_depth_counts = {
        n: len([q for q in range(-n, n + 1) if in_wedge(n, q)])
        for n in range(max_depth + 1)
    }
    assert fixed_depth_counts == {n: n + 1 for n in range(max_depth + 1)}

    transitions = {
        "same_out": (1, 1),
        "same_in": (-1, -1),
        "mixed_plus": (1, -1),
        "mixed_minus": (-1, 1),
    }

    for n in range(2, max_depth + 1, 2):
        assert in_wedge(n, 0)
        plus = hop(n, 0, *transitions["mixed_plus"])
        minus = hop(n, 0, *transitions["mixed_minus"])
        assert plus == (n, 2)
        assert minus == (n, -2)
        assert in_wedge(*plus)
        assert in_wedge(*minus)

        return_from_plus = hop(*plus, *transitions["mixed_minus"])
        return_from_minus = hop(*minus, *transitions["mixed_plus"])
        assert return_from_plus == (n, 0)
        assert return_from_minus == (n, 0)
        assert in_wedge(*return_from_plus)
        assert in_wedge(*return_from_minus)

    # The head is the only diagonal point where mixed leakage is geometrically
    # blocked by the wedge.
    assert hop(0, 0, *transitions["mixed_plus"]) == (0, 2)
    assert hop(0, 0, *transitions["mixed_minus"]) == (0, -2)
    assert not in_wedge(0, 2)
    assert not in_wedge(0, -2)

    for n in range(2, max_depth + 1):
        upper_face = (n, n)
        lower_face = (n, -n)
        assert in_wedge(*upper_face)
        assert in_wedge(*lower_face)

        upper_outward = hop(*upper_face, *transitions["mixed_plus"])
        upper_inward = hop(*upper_face, *transitions["mixed_minus"])
        lower_outward = hop(*lower_face, *transitions["mixed_minus"])
        lower_inward = hop(*lower_face, *transitions["mixed_plus"])

        assert not in_wedge(*upper_outward)
        assert in_wedge(*upper_inward)
        assert not in_wedge(*lower_outward)
        assert in_wedge(*lower_inward)

    print("checked wedge points through n =", max_depth)
    print("fixed-depth counts =", fixed_depth_counts)
    print("diagonal q=0 leaks to q=+-2 for all even n>=2")
    print("q=+-2 returns to q=0 for all even n>=2")
    print("head n=0 blocks mixed leakage only at the single point")
    print("face boundaries block outward mixed hops but keep inward hops")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
