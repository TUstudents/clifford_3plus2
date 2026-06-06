"""Verify that face-exterior absorption cannot remove the first BB return.

This docs-local certificate supports
``src/clifford_3plus2_d5/synthesis/BARE_BB_FACE_EXTERIOR_INSUFFICIENCY.md``.
It checks that outward hops at the wedge faces are not the same moves as the
first relative return ``q=+-2 -> q=0`` from the diagonal scar.

Run:
    python docs/scripts/verify_bare_bb_face_exterior_insufficiency.py
"""

from __future__ import annotations


def in_wedge(n: int, q: int) -> bool:
    return n >= abs(q) and (n - q) % 2 == 0


def hop(n: int, q: int, sigma1: int, sigma2: int) -> tuple[int, int]:
    return n + sigma1 + sigma2, q + sigma1 - sigma2


def main() -> None:
    mixed_plus = (1, -1)
    mixed_minus = (-1, 1)

    max_depth = 12

    for n in range(2, max_depth + 1, 2):
        assert in_wedge(n, 0)

        plus = hop(n, 0, *mixed_plus)
        minus = hop(n, 0, *mixed_minus)
        assert plus == (n, 2)
        assert minus == (n, -2)
        assert in_wedge(*plus)
        assert in_wedge(*minus)

        return_plus = hop(*plus, *mixed_minus)
        return_minus = hop(*minus, *mixed_plus)
        assert return_plus == (n, 0)
        assert return_minus == (n, 0)
        assert in_wedge(*return_plus)
        assert in_wedge(*return_minus)

        # At n=2, q=+-2 are face states, but the return moves inward. For
        # n>2, q=+-2 are interior relative states. Either way, the first return
        # does not cross the exterior boundary.
        if n == 2:
            assert plus == (n, n)
            assert minus == (n, -n)
        else:
            assert abs(plus[1]) < plus[0]
            assert abs(minus[1]) < minus[0]

    for n in range(2, max_depth + 1):
        upper_face = (n, n)
        lower_face = (n, -n)
        assert in_wedge(*upper_face)
        assert in_wedge(*lower_face)

        upper_out = hop(*upper_face, *mixed_plus)
        upper_in = hop(*upper_face, *mixed_minus)
        lower_out = hop(*lower_face, *mixed_minus)
        lower_in = hop(*lower_face, *mixed_plus)

        assert not in_wedge(*upper_out)
        assert in_wedge(*upper_in)
        assert not in_wedge(*lower_out)
        assert in_wedge(*lower_in)

    print("checked even diagonal depths through n =", max_depth)
    print("q=0 -> q=+-2 is allowed for n>=2")
    print("q=+-2 -> q=0 returns inside the wedge for n>=2")
    print("at n=2 the return is inward from a face, not outward to exterior")
    print("face exterior absorbs q=+n -> q=n+2 and q=-n -> q=-n-2 only")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
