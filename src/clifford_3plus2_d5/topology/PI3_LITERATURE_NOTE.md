# Phase D-3 — π_3 literature note

**Verdict**: PI3 KILL — no carrier-relevant coset has Z/3 torsion in π_3.

## Context

The hypothesis was: maybe the three SM generations are a topological
invariant of some carrier-relevant coset ``G/H``, with ``π_3(G/H) =
Z/3`` providing the three generation count.  This note surveys
homotopy-group tables for the relevant cosets and confirms the
hypothesis fails.

## Source references

| Source | Use |
|---|---|
| Mimura, M., Toda, H. *Topology of Lie Groups, I and II*. AMS Translations of Math. Monographs, Vol. 91, 1991. | Tables of π_k(G) and π_k(G/H) for compact Lie groups. |
| Bott, R., Tu, L. *Differential Forms in Algebraic Topology*. Springer, 1982. | π_3 of classical groups via Lie algebra cohomology. |
| Husemoller, D. *Fibre Bundles*. Springer, 3rd ed., 1994. | π_n(G/H) via fibration exact sequence. |
| Bott, R. "Lectures on K-theory." | Stable homotopy groups π_n(U(∞)), π_n(O(∞)). |
| Wikipedia. *Homotopy groups of spheres*; *Topology of Lie groups*. | Quick reference tables. |

## Universal results for compact simple Lie groups

For any compact simple Lie group ``G``:

```text
π_3(G) = Z  (one generator: the Hopf-class winding number)
π_4(G) = Z/2 for G = SU(2), Sp(N); else 0 or Z
```

For ``π_3``, the result is universal: ``π_3(G) ≅ Z`` for every compact
simple Lie group.  This is because ``π_3`` of a Lie group counts
homotopy classes of maps ``S^3 → G``, classified by the integer
winding number around an SU(2) subgroup (the "instanton number").

**No 3-torsion (Z/3 factor) appears in π_3 of any simple compact Lie
group.**

## Carrier-relevant cosets

The fibration sequence ``H → G → G/H`` gives a long exact sequence in
homotopy:

```text
... → π_3(H) → π_3(G) → π_3(G/H) → π_2(H) → π_2(G) → ...
```

For compact Lie groups, ``π_2(G) = 0`` (Lie groups are 2-connected
beyond their fundamental group), so the relevant tail is:

```text
π_3(H) → π_3(G) → π_3(G/H) → π_2(H) → 0
```

For ``H`` connected, ``π_2(H) = 0`` too, giving ``π_3(G/H)`` as the
cokernel of ``π_3(H) → π_3(G)``.  Since both source and target are
``Z``-valued, the cokernel is ``Z/n`` for some integer ``n`` (the
"index" of the embedding).

### Specific cosets

| Coset | π_3 | Source / reasoning |
|---|---|---|
| ``Spin(10)``                          | Z  | universal for compact simple Lie. |
| ``Spin(10) / SU(5)``                  | 0 or Z (depending on embedding index) — NOT Z/3. | π_3(SU(5)) = π_3(Spin(10)) = Z; quotient is at most Z. |
| ``Spin(10) / (SU(4) × SU(2) × SU(2))`` (Pati-Salam) | 0 or Z. | All factors have π_3 = Z. |
| ``Spin(6) / SU(3)``                   | 0 or Z.   | Spin(6) ≅ SU(4); SU(4) / SU(3) ≅ S^7, π_3(S^7) = 0. |
| ``G_2 / SU(3)``                       | Z (the generator is S^6 ≅ G_2/SU(3) homotopy class). | π_3(S^6) = 0; π_3(G_2/SU(3)) = π_3(G_2) / image of π_3(SU(3)) = Z/Z = Z or 0. |
| ``F_4 / Spin(9)``                     | 0.        | F_4/Spin(9) ≅ OP^2 octonionic projective plane, π_3(OP^2) = 0. |
| ``E_6 / (Spin(10) × U(1))``           | 0.        | E_6/(Spin(10)×U(1)) is a homogeneous space whose π_3 is killed by the U(1) factor. |

### Universal pattern

In all relevant cosets, ``π_3`` is either ``0`` or ``Z`` — **never** Z/3
or any 3-torsion.  This is a deep fact rooted in:

1. ``π_3`` of compact simple Lie groups is always ``Z`` (the
   Eilenberg–MacLane generator).
2. Quotients ``G/H`` for compact simple ``G`` and connected ``H``
   produce ``π_3 = Z/n`` for some integer ``n``, but the index ``n``
   appears as a ratio of dual Coxeter numbers or root-length norms —
   never specifically ``3``.

## Where do 3-torsion homotopy groups appear?

For completeness: 3-torsion in homotopy groups does exist, but only in
higher dimensions or for non-Lie-group spaces:

- ``π_3(S^2)`` = Z (the Hopf invariant); no 3-torsion.
- ``π_7(S^2)`` = Z/12 (= Z/4 × Z/3); 3-torsion appears here, at
  dimension 7, on the 2-sphere — not relevant for our carriers.
- Stable homotopy groups: ``π_3^s = Z/24`` (= Z/8 × Z/3); 3-torsion
  appears in the third stable stem.
- Exotic spheres in dimension ≥ 7 can have 3-torsion in their groups
  of diffeomorphisms.

**None of these are relevant** to the BCC × chiral-16 carrier or any
of its natural cosets.

## Verdict

**PI3 KILL** — no natural ``π_3`` mechanism produces three generations
from the carrier topology.  The topological hypothesis "3 generations =
π_3(G/H) = Z/3 for some carrier-relevant ``G/H``" is closed at the
literature level.

The relevant ``π_3`` groups are uniformly ``0`` or ``Z`` (the integer
Hopf-class / instanton-number group), with no 3-torsion.  Three
generations is not a homotopy-class invariant of our carrier.

## What this does NOT close

This note addresses only the specific question of ``π_3`` torsion.  It
does NOT address:

- Higher homotopy ``π_n`` for ``n > 3`` (potentially Z/3-torsion at
  ``n = 7`` on ``S^2``, but irrelevant to our carriers).
- ``π_4`` (Witten's global SU(2) anomaly counts via ``π_4(SU(2)) =
  Z/2`` — relevant to Phase D-5, not three generations).
- K-theory / KO-theory of the carrier (different invariant; may have
  Z/3 elements but not natural carriers for generations).
- Cobordism / TQFT invariants of the lattice walk (Phase D-5
  territory).

These topological invariants live in different dimensions and would
require separate investigations.  The natural ``π_3`` hypothesis,
however, is closed.
