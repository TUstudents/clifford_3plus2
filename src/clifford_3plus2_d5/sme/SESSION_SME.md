# Bold A — SME experimental-bound verdict on ε

**Date**: 2026-05-19
**Status**: CLOSED — UNFALSIFIABLE PASS at ε ≲ 2 × 10⁻³³ m
(~10² ℓ_P, ~2 orders of magnitude above the Planck length).

## Load-bearing question

> Given H^(1)'s explicit T_{2g} CP-odd structure at O(ε), what is the
> maximum allowed ε from current experimental bounds on the
> corresponding dim-5 non-minimal SME coefficients, and which of three
> pre-named scale verdicts applies?

**Answer**: `UNFALSIFIABLE PASS` — the program is observation-consistent
but predicts no near-term measurable effects at the representative
d^{(5)} bound used.

## Phase-by-phase findings

### Phase A-1 — SME framework identification

H^(1) satisfies all five symmetry classes simultaneously:

| Property                            | Value |
| ---                                 | ---   |
| Hermitian                           | True  |
| Chirality-preserving                | True  |
| Momentum-bilinear (degree 2 in k)   | True  |
| CP-odd fraction of ||H^(1)||²       | 1     |
| T_{2g} fraction of ||H^(1)||²       | 1     |

Identified target: **dim-5 non-minimal SME, fermion sector, CP-odd
spin-tensor** (Kostelecky-Mewes, arXiv:1308.4973).

### Phase A-2 — H^(1) → SME tensor mapping

The (CP-odd, T_{2g}) cell of H^(1), per chirality, decomposes
exactly as

```text
H^(1)_chirality(k) = σ^x · k_y k_z  −  σ^y · k_x k_z  +  σ^z · k_x k_y
```

with three non-zero entries.  The identity-Pauli component vanishes
(ruling out a chirality-blind scalar correction).  The upper and lower
chirality blocks are equal, confirming the natural axial-vector
Lagrangian structure

```text
L_correction ⊃ ε · T^{aij} · ψ̄ γ^a γ^5 ∂_i ∂_j ψ
```

This maps to the **d^{(5)}_{αβγ}** Kostelecky-Mewes dim-5 coefficient
(axial-vector × two derivatives, CPT-even, CP-odd).  Three specific
tensor entries carry non-zero coefficients of magnitude 1 (with signs
+1, −1, +1 along the spin-axes x, y, z respectively).

### Phase A-3 — SME experimental bounds

See ``SME_LITERATURE_NOTE.md``.  The tightest applicable representative
bound on the d^{(5)} fermion-sector coefficient, drawn from electron-
sector atomic-clock + co-magnetometer sidereal-modulation analyses, is

```text
|d^{(5)}| ≲ 10⁻¹⁷ GeV⁻¹  ≈ 2 × 10⁻³³ m  (using ℏc / 1 GeV = 1.97 × 10⁻¹⁶ m).
```

Specific Kostelecky-Russell entry-id verification and 2024-2025 atom-
interferometry cross-checks are flagged as follow-up.  The
representative bound is conservative; a verified bound an order of
magnitude tighter would push the verdict toward PLANCK-CONSISTENT.

### Phase A-4 — Symbolic ε constraint

For each of the three d^{(5)} components, the constraint is

```text
ε · |coefficient_i|  ≤  |d^{(5)}|_bound .
```

All three coefficients have magnitude 1, so the tightest face is
simply

```text
ε  ≲  2 × 10⁻³³ m  =  ~ 10² × Planck length
                  =  ~ 10⁸ × 10⁻²⁵ m  (current observable threshold).
```

This lies in the **UNFALSIFIABLE PASS** verdict class per the PLAN:

| Verdict class       | ε range                                              |
| ---                 | ---                                                  |
| SUB-PLANCK KILL     | ε ≤ ℓ_P  ≈ 1.6 × 10⁻³⁵ m                            |
| PLANCK-CONSISTENT   | ℓ_P < ε ≤ 10 ℓ_P                                    |
| **UNFALSIFIABLE PASS** | **10 ℓ_P < ε ≤ 10⁻²⁵ m  (← this verdict)**       |
| OBSERVABLE POSITIVE | ε > 10⁻²⁵ m                                          |

## Combined verdict

```text
ε_max  ≲  2 × 10⁻³³ m         (UNFALSIFIABLE PASS)
log₁₀(ε_max / ℓ_P)  ≈ 2.09     (~ 100 × Planck length)
```

The program is **alive** under current experimental bounds on the
identified dim-5 non-minimal SME d^{(5)} components.  It predicts a
small ε-suppressed cubic-anisotropic CP-violating correction that is
~10⁸ times smaller than current observational reach in this channel —
the program is unfalsifiable in the d^{(5)} channel at present
sensitivity.

## What this audit does NOT close

1. **Photon-sector (k_F)^(5) bounds**.  These are 10⁻²⁹ GeV⁻¹ (tighter
   than fermion-sector by ~10¹² from GRB photon-dispersion), but they
   constrain a different sector of SME.  An analogous audit on a
   future gauge-sector H^(1) would be needed before applying these
   bounds.

2. **2024-2025 atom-interferometry tightening**.  Recent experiments
   may improve the fermion d^{(5)} bound; the literature note flags
   this for follow-up.  A tightened bound by 2-3 orders of magnitude
   would push the verdict to PLANCK-CONSISTENT.

3. **Strong-CP / θ_QCD**.  Original brainstorm item 10.  Orthogonal
   sector; deferred.

4. **Field-redefinition triviality (F-sme-5)**.  If the H^(1) cubic-
   anisotropic structure falls in a d^{(5)} direction that is field-
   redefinition equivalent to dim-3 b^μ, the constraint on ε could
   be vacuous and the program would be unfalsifiable in this channel
   for a different reason.  Worth a follow-up dedicated audit.

5. **Three-generation considerations**.  Bold A treats one
   generation.  Generations closed negative independently
   (triality / broken_triality / exceptional / topology).

6. **Higher-order corrections (O(ε²), O(ε³))**.  Original brainstorm
   item 8.  Orthogonal; would give independent bounds at higher
   non-minimal SME order.

## Recommendation

Bold A returns a clean PASS in the UNFALSIFIABLE class at the present
representative bound.  Natural next moves:

1. **Verify the Kostelecky-Russell entry ids** for the three d^{(5)}
   components and recompute the bound with verified entries.  If the
   verified bound is 2-3 orders of magnitude tighter, the verdict
   changes to PLANCK-CONSISTENT.
2. **Cross-check against 2024-2025 atom-interferometry results**.
   Pull recent arXiv papers and update ``DIM5_FERMION_BOUND_GEV_INVERSE``
   in ``epsilon_constraint.py`` if applicable.
3. **Investigate field-redefinition triviality** (F-sme-5).  A short
   follow-up note checking whether the (CP-odd, T_{2g}) d^{(5)}
   direction is equivalent under SME field redefinitions to dim-3
   b^μ would close the F-sme-5 failure mode definitively.
4. **Pivot to photon-sector audit** as a separate Bold A-prime
   sidecar, applying the much tighter (k_F)^(5) bounds to a
   gauge-sector H^(1) correction (if one exists).

The program is structurally consistent with current bounds at the
representative ε scale.  No experimental discrepancy forces a re-
examination of the BCC × chiral-16 carrier or the cp/ dual-positive
result.

## Test summary

```bash
uv run pytest src/clifford_3plus2_d5/sme/tests -q
# 37 passed
```

## Verdict callable

```bash
uv run --no-sync python -c "
from clifford_3plus2_d5.sme.sme_audit import sme_audit_payload
p = sme_audit_payload()
print(p.verdict)
print(p.interpretation)
"
```

Output:

```text
SME AUDIT — UNFALSIFIABLE PASS

Phase A-1 identified the SME sector as dim-5 non-minimal SME, fermion
sector, CP-odd spin-tensor.  Phase A-2 mapped H^(1) to 3 non-zero
components of d^{(5)}_{αβγ}.  Phase A-3 (literature note) supplied
the tightest representative bound on d^(5).  Phase A-4 produced an ε
bound of 1.973e-33 m, which is 2.09 orders of magnitude above the
Planck length.  Final scale verdict: UNFALSIFIABLE PASS.
```
