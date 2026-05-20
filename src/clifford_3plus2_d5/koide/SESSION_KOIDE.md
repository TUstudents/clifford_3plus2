# Koide — flavor 45°-cone audit verdict report

**Date**: 2026-05-20
**Status**: CLOSED — **KOIDE CONSISTENT** (PDG NOT IN LOCUS).

## Load-bearing question

> Does the BCC body-diagonal Z₃ acting on the σ^a-axes of the
> chiral-16 T_{2g} structure naturally place the charged-lepton
> mass-vector on the Koide 45° cone, given three generations as
> phenomenological input?

**Answer**: The BCC body-diagonal Z₃-equivariant Yukawa locus
**intersects** the Koide cone at a specific 1-parameter sub-family
(|v_t|/|v_o| = 3 + 2√2 ≈ 5.83), but does **not** lie entirely
inside the cone, and PDG charged-lepton masses are **not** in the
equivariant locus (their distinct values cannot be reproduced by a
2-fold-degenerate Yukawa).  Result: KOIDE CONSISTENT — the
structure admits Koide but does not predict PDG.

## Phase-by-phase findings

### Phase KO-1 — Empirical Koide + 45° cone geometry (done)

Verified K_PDG = (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)² =
0.666661 from PDG charged-lepton masses, deviation 6.16 × 10⁻⁶ from
2/3 = 0.666667.  Three equivalent geometric forms verified:

| Form | Statement | PDG value | Target |
|---|---|---|---|
| K-ratio | (Σm)/(Σ√m)² | 0.666661 | 0.666667 |
| Angle | (v·n̂)²/|v|² | 0.500005 | 0.500000 |
| Equipartition | |P_trace v|²/|P_traceless v|² | 1.000018 | 1.000000 |

n̂ = (1, 1, 1)/√3 is the trace direction.  18 tests passing.

### Phase KO-2 — BCC body-diagonal Z₃ on σ^a-axes (done)

The BCC body-diagonal Z₃ rotation
``R = [[0,1,0],[0,0,1],[1,0,0]]`` (reused from topology/SC-2):

- Fixes (1, 1, 1) / √3 (eigenvalue +1).
- Is order 3 (R³ = I), orthogonal (R^T R = I), det = +1.
- Commutes with both trace projector P_t and traceless projector P_o
  — these are the Z₃-irrep projectors:
  - 1-dim trivial irrep along (1,1,1) (the "Koide cone axis").
  - 2-dim non-trivial irrep in the orthogonal plane (eigenvalues
    ω, ω̄ where ω = e^{2πi/3}).

Identification convention pinned: σ^x ↔ e, σ^y ↔ μ, σ^z ↔ τ.  This
is one of three cyclic conventions; the others differ by relabelling
the generations.  Koide K is invariant under cyclic permutations of
the masses, so the choice is conventional.

Under this identification, the PDG mass-vector satisfies the Koide
equipartition condition |P_t v|² ≈ |P_o v|² to ~10⁻⁵.  12 tests
passing.

### Phase KO-3 — BCC-Z₃-orbit 3×3 Yukawa eigenvalue locus (done)

Built the broken_triality-pattern Yukawa with R = BCC body-diagonal
Z₃ replacing Spin(8) triality.  For a starting vector v_* ∈ ℝ³ in
σ^a-space:

```text
Y_{ij}  =  ⟨R^i v_*, R^j v_*⟩   (Euclidean inner product on ℝ³)
```

is a **circulant 3×3 matrix** with structure:

```text
Y_ii = |v_t|² + |v_o|²
Y_ij = |v_t|² − |v_o|²/2    (i ≠ j)
```

where v_t = P_trace v_* and v_o = P_traceless v_*.  Its eigenvalues
are:

```text
λ_1 = 3 |v_t|²              (Z₃-trivial, multiplicity 1)
λ_2 = λ_3 = (3/2) |v_o|²    (Z₃-non-trivial, multiplicity 2)
```

**Two of three eigenvalues are always degenerate** by Z₃
equivariance.  The mass-vector v_Y = (√3 |v_t|, √(3/2) |v_o|,
√(3/2) |v_o|) has a degenerate pair regardless of v_*.

Koide K = 2/3 holds IFF the special ratio condition

```text
|v_t| / |v_o|  =  3 + 2√2  ≈ 5.8284
```

is met.  At this ratio, the non-degenerate-to-degenerate mass ratio
is exactly

```text
m_1 / m_2 = 2(3 + 2√2)² = 2(17 + 12√2) ≈ 67.94.
```

PDG charged-lepton masses (m_τ/m_μ ≈ 16.8, m_μ/m_e ≈ 207) do **not**
match this ratio and (more fundamentally) are not 2-fold degenerate.
12 tests passing.

### Phase KO-4 — Cone vs. locus comparison (done)

Classified the Z₃-equivariant Yukawa locus

```text
L_Z3  =  {(3|v_t|², (3/2)|v_o|², (3/2)|v_o|²)  :  |v_t|, |v_o| ∈ ℝ₊}
```

against the Koide cone

```text
C  =  {(m_1, m_2, m_3) ∈ ℝ³₊  :  K(m) = 2/3}.
```

Findings:

| Property | Value |
|---|---|
| L_Z3 ∩ C non-empty? | Yes (1-param family at |v_t|/|v_o| = 3+2√2) |
| L_Z3 ⊂ C? | No (counter-example at |v_t| = |v_o|) |
| PDG ∈ L_Z3? | No (PDG has all-distinct masses; L_Z3 has 2-fold degeneracy) |
| Verdict | **KOIDE CONSISTENT** |
| Tag | PDG NOT IN LOCUS |

7 tests passing.

### Phase KO-5 — Combined audit (done)

Aggregated KO-1..KO-4 into ``KoideAuditPayload``.  5 tests passing.

## Combined verdict

```text
Empirical:    K_PDG = 0.666661, deviation 6 × 10⁻⁶ from 2/3 ✓
Structural:   BCC body-diagonal Z₃ has (1,1,1)/√3 as trivial-irrep axis ✓
Yukawa:       L_Z3 always has 2-fold degenerate eigenvalues
Geometry:     L_Z3 ∩ Cone = 1-parameter family at |v_t|/|v_o| = 3+2√2
PDG fit:      PDG ∉ L_Z3 (3 distinct masses inconsistent with degeneracy)
Verdict:      KOIDE CONSISTENT — admits Koide, does not predict PDG
```

## Significance

The audit settles the apparent BCC↔Koide coincidence with a clear
**positive structural verdict and a clear phenomenological limit**:

- **Positive**: the Koide cone direction (1,1,1)/√3 is the
  Z₃-trivial irrep direction of the BCC body-diagonal rotation —
  this is geometrically meaningful, not accidental.  The Z₃-
  equivariant Yukawa naturally has a 1-parameter sub-family on the
  cone.  The coincidence is real.
- **Limit**: the program's Z₃-equivariant Yukawa cannot produce
  three distinct masses (it always has 2-fold degeneracy), so it
  cannot reproduce the PDG mass pattern as a derived consequence.
  Three distinct masses require **Z₃-breaking input** beyond the
  carrier structure.

The natural follow-up is **Bold-B (dynamical Higgs sector)** — a
Higgs VEV alignment that tilts the Yukawa off the equivariant
locus would select a point on the cone (or near it) with three
distinct masses.  The audit identifies this as the load-bearing
mechanism for converting CONSISTENT to PREDICTED.

## What this audit does NOT close

1. **Quark Koide / mass-vector audit**.  Up-quark K ≈ 0.85, down-quark
   K ≈ 0.73 — Koide is generation-precise for charged leptons but
   imprecise for quarks.  An analog audit could pursue the partial
   pattern; out of scope here.
2. **Bold-B dynamical Higgs VEV alignment**.  Whether a natural VEV
   dynamics prefers the cone direction is a separate ~2-month
   investigation.
3. **Non-equivariant Yukawa scan**.  The audit focused on Z₃-
   equivariant Yukawas (broken_triality pattern).  Allowing
   Z₃-breaking projections widens the locus and could include
   PDG, but doesn't address the structural question of whether
   the program PREDICTS Koide.
4. **Neutrino Koide**.  m_ν ≪ m_e; neutrino sector requires
   separate modelling.
5. **Identification convention dependence**.  The σ^x↔e / σ^y↔μ /
   σ^z↔τ choice is one of three cyclic conventions.  Koide is
   permutation-invariant, so the choice is conventional, but it
   should be made explicit in any follow-up that uses generation
   labels directly.

## Test summary

```bash
uv run pytest src/clifford_3plus2_d5/koide/tests -q
# 54 passed
```

(KO-1: 18, KO-2: 12, KO-3: 12, KO-4: 7, KO-5 audit: 5)

## Verdict callable

```bash
uv run --no-sync python -c "
from clifford_3plus2_d5.koide.koide_audit import koide_audit_payload
p = koide_audit_payload()
print(p.verdict)
print(p.interpretation)
"
```

Output:

```text
KOIDE AUDIT — KOIDE CONSISTENT (PDG NOT IN LOCUS)

Phase KO-1 verified Koide empirically: K_PDG = 0.666661 vs 2/3 = 0.666667,
deviation -6.16e-06 (~10⁻⁵).  ...
```
