# Bold D — topology sidecar verdict report

**Date**: 2026-05-19
**Status**: CLOSED — all four phases produced negative verdicts.

## Load-bearing question

> Does any natural topological or discrete-subgroup mechanism on the
> existing BCC × chiral-16 carrier produce three SM-generation copies
> without inventing additional structure?

**Answer**: No, on every candidate examined.

## Phase-by-phase verdicts

### Phase D-1 — BCC body-diagonal Z_3 (TOPOLOGY KILL)

- The body-diagonal rotation about ``(1, 1, 1) / √3`` is the cyclic
  permutation ``R: (x, y, z) → (y, z, x)``, of order 3.
- Its induced action on the 8 BCC directions has cycle structure
  ``(3, 3, 1, 1)`` (two corner fixed points, two 3-cycles).
- Its Dirac spinor lift ``U_3 ∈ SU(2)`` cubes to ``-I`` (correct spin-½
  half-cover sign).
- **Bonus finding**: the Bialynicki-Birula BCC hops are NOT
  Z_3-equivariant in the spatial × Dirac sector — the BB construction
  picks a direction-asymmetric convention.  All 8 hop residuals are
  non-zero.  This is recorded as orthogonal to the three-generation
  question.
- **Three-generation verdict**: the chiral-16 internal carrier is built
  from Cl(0, 10) gammas with internal labels — NOT spatial coordinates.
  The spatial Z_3 acts trivially on the chiral-16 by construction.
  Decomposition: ``16 = 16 (trivial) + 0 (ω) + 0 (ω²)``.  No
  three-generation structure can emerge from spatial Z_3 alone.

### Phase D-2 — color SU(3)_c Z_3 center (COLOR Z_3 KILL)

- Color Z_3 center generator: ``g_3 = diag(1, ω, ω²)`` on color triplets,
  ``ω = exp(2π i / 3)``.
- Per-generation field count: ``6 + 3 + 3 + 2 + 1 + 1 = 16`` Weyl
  fermions.
- Decomposition under ``g_3``: ``(8, 4, 4)`` over
  ``(trivial, ω, ω²)``.
- The decomposition is asymmetric — leptons sit entirely in the
  trivial character while quarks distribute across all three.  Three
  equivalent generations would require ``(16/3, 16/3, 16/3)``, which is
  not an integer partition.
- **Verdict**: identifying spatial Z_3 with the SU(3)_c center does
  not produce three symmetric generation copies.

### Phase D-3 — π_3 literature check (PI3 KILL)

Documented as a markdown note (``PI3_LITERATURE_NOTE.md``).  Surveyed
``π_3(G/H)`` for every carrier-relevant coset:

- ``Spin(10) / SU(5)``, ``Spin(10) / (SU(4) × SU(2)²)`` (Pati-Salam),
  ``Spin(6) / SU(3)``, ``G_2 / SU(3)``, ``F_4 / Spin(9)``,
  ``E_6 / (Spin(10) × U(1))``.
- All are either ``0`` or ``Z`` — never Z/3 or any 3-torsion.
- Universal fact: ``π_3(G) ≅ Z`` for every compact simple Lie group
  (the instanton winding number); ``π_3(G/H)`` is the cokernel of
  ``Z → Z`` — never Z/3 specifically.

3-torsion homotopy groups exist (``π_7(S²) = Z/12``, stable stem
``π_3^s = Z/24``) but live in dimensions or spaces irrelevant to our
carrier.

### Phase D-5 — discrete anomaly forcing N = 3 (ANOMALY KILL)

Sub-phases:

| Sub-phase | Question | Result |
|---|---|---|
| FD-11/12 | Do continuous SM anomalies cancel per generation? | Yes: grav, U(1)_Y³, SU(2)²·Y, SU(3)²·Y all = 0 with the standard hypercharge assignments. |
| FD-13 | Does Witten's global SU(2) anomaly constrain N? | No: each generation contributes 4 SU(2)_L doublets (even), and 4N is even for every N. |
| FD-14 | Does any anomaly uniquely force N = 3? | No: the combined constraint reduces to ``0 = 0``, satisfied for every N ≥ 0. |

The admissible set of generation counts is the full non-negative
integers — three is admissible but not unique.

## Combined verdict

```text
D-1: TOPOLOGY KILL — chiral-16 internal is Z_3-trivial
D-2: COLOR Z_3 KILL — chiral-16 decomposes as 8 + 4 + 4
D-3: PI3 KILL — no carrier-relevant coset has Z/3 torsion in π_3
D-5: ANOMALY KILL — anomalies cancel for any N, do not force N = 3
```

The topology sidecar produces a clean negative across all four
candidates.  Combined with the prior negatives from triality (K1 fail),
broken_triality (BT-2 fail), and exceptional (all 4 candidates fail),
no algebraic or topological mechanism on the existing BCC × chiral-16
carrier produces three SM generations.

## What this does NOT close

- **D-4 (instanton / Pontryagin index Z → Z/3)**: deferred long-shot.
- **Higher π_n** for n ≥ 4: π_4 governs Witten's anomaly (already
  checked in D-5); higher π_n live in higher dimensions and on
  different spaces.
- **K-theory / KO-theory** of the carrier: a separate invariant that
  could have Z/3 elements but is not a natural carrier for generations.
- **Cobordism / TQFT** invariants of the lattice walk: D-5 territory
  beyond the global SU(2) anomaly.
- **Discrete modular constraints** from the BCC walk's effective
  action at higher orders in ε.

## Bonus findings (orthogonal to three-generation question)

1. **BB hops are not Z_3-equivariant on spatial × Dirac**.  All 8 hop
   residuals are non-zero.  The Bialynicki-Birula construction picks a
   direction-asymmetric convention.  Worth investigating in a separate
   walk-symmetry audit.
2. **Standard SM anomaly cancellation re-verified symbolically** for
   the per-generation hypercharge content.  Could be lifted into a
   shared utility if needed by future sidecars.

## Test summary

```bash
uv run pytest src/clifford_3plus2_d5/topology/tests -q
# 53 passed
```

## Files produced

```
topology/
  __init__.py
  PLAN.md
  STATUS.md
  parameter_ledger.md
  PI3_LITERATURE_NOTE.md
  SESSION_TOPOLOGY.md                (this file)
  reuse.py
  bcc_z3_rotation.py                 (D-1 rotation + Dirac lift)
  hop_equivariance.py                (D-1 equivariance audit — bonus)
  internal_triviality.py             (D-1 chiral-16 Z_3-trivial)
  color_center_z3.py                 (D-2 (8, 4, 4) decomposition)
  anomaly_cancellation/
    __init__.py
    bcc_anomaly_polynomial.py        (D-5 FD-11/12 SM anomalies)
    global_anomaly_check.py          (D-5 FD-13 Witten + mod-2)
    discrete_anomaly_constraint.py   (D-5 FD-14 constraint on N)
    anomaly_audit.py                 (D-5 combined verdict)
  tests/
    __init__.py
    test_bcc_z3_rotation.py          (D-1)
    test_hop_equivariance.py         (D-1 bonus)
    test_internal_triviality.py      (D-1)
    test_color_center_z3.py          (D-2)
    test_bcc_anomaly_polynomial.py   (D-5)
    test_global_anomaly_check.py     (D-5)
    test_discrete_anomaly_constraint.py (D-5)
    test_anomaly_audit.py            (D-5)
```

## Recommendation

The topology route is closed for the BCC × chiral-16 carrier.  Future
work seeking the three-generation mechanism should either:

1. Examine **D-4 (instantons / Pontryagin)** as a deferred long-shot.
2. Move to **higher-rank carriers** (E_6, E_8) — explicitly outside
   the scope of this sidecar.
3. Investigate **non-anomaly discrete invariants** (cobordism, TQFT,
   modular tensor categories).
4. Accept three generations as a **phenomenological input** rather
   than a derived consequence of the carrier.

The negative is decisive at the level of the BCC × chiral-16 carrier
itself.
