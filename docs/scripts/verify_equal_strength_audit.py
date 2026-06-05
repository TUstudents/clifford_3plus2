"""Audit of the proposed equal-strength repair derivation.

Docs-local. Tests the two steps behind THEORY_REVIEW.md Section 11.7's
equal-strength question:

  (1) PATH-LOCALITY from one-tick height grading [D,X]=X  -> X = a E1 + b E2
      VERDICT: valid (C:9).
  (9) EQUAL-STRENGTH from "active-domain irreducibility => Schur"
      VERDICT: INVALID. Three independent reasons, all asserted below.

Conclusion: equal-strength |a|=|b| is a defect-homogeneity *assumption*, not a
theorem -- confirming the uniqueness audit's "permitted, not forced." The cut
that builds the generation hierarchy is the same cut that breaks the edge-
equating clock symmetry.

Run:  python docs/scripts/verify_equal_strength_audit.py
"""

from __future__ import annotations

import sympy as sp

I = sp.I


def main() -> None:
    # (1) one-tick height grading forces path-locality
    D = sp.diag(0, 1, 2)
    surviving = [
        (i, j) for i in range(3) for j in range(3) if D[i, i] - D[j, j] == 1
    ]
    assert surviving == [(1, 0), (2, 1)]                 # E1, E2 only; no shortcut
    # [D, E_k] = E_k explicitly
    E1 = sp.Matrix([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    E2 = sp.Matrix([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    assert D * E1 - E1 * D == E1 and D * E2 - E2 * D == E2

    # (9a) Schur does NOT force X^dag X scalar: it is not in the commutant
    sx = sp.Matrix([[0, 1], [1, 0]])
    XdX = sp.diag(1, 4)                                  # |a|^2=1, |b|^2=4
    assert (XdX * sx - sx * XdX) != sp.zeros(2)          # not in commutant of M2
    # so an irreducible algebra constrains its commutant, not diag(1,4)

    # (9b) the active SOURCE domain {|0>,|1>} has distinct heights -> reducible
    heights_active = (D[0, 0], D[1, 1])
    assert heights_active == (0, 1)                      # distinct => diagonal-only
    # End_local(P_act) contains diag(p,q) for all p,q != C*P_act

    # (9c) the clock Z3 that would equate E1,E2 is broken by the cut
    C3 = sp.Matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    assert sp.simplify(C3 * E1 * C3.inv()) == E2         # clock equates the edges
    cut_edge = sp.Matrix([[0, 0, 1], [0, 0, 0], [0, 0, 0]])  # |0><2|
    assert sp.simplify(C3 * E2 * C3.inv()) == cut_edge   # ...but sends E2 to the cut

    print("(1) [D,X]=X  -> X = a E1 + b E2  (no shortcut)            [VALID, C:9]")
    print("(9a) diag(1,4) not in commutant of M2: Schur cannot force it scalar  [CLAIM INVALID]")
    print("(9b) active sources |0>,|1> at heights", heights_active,
          "-> defect algebra diagonal -> REDUCIBLE  [premise FAILS]")
    print("(9c) C3 E1 C3^-1 = E2 (equates edges) BUT C3 E2 C3^-1 = |0><2| (cut)")
    print("     -> the edge-equating clock is broken by the cut  [no symmetry]")
    print("\nVERDICT: path-locality derived; equal-strength NOT forced.")
    print("|a|=|b| is a defect-homogeneity ASSUMPTION (permitted, not forced).")


if __name__ == "__main__":
    main()
