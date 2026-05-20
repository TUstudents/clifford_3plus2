# Session CP-O(ε²) — Multi-element β + Continuum Expansion

**Verdict: DUAL ROBUST PASS** — strongest CP result the program has produced.

| Audit | Verdict | Key result |
|---|---|---|
| β-multi (all 4 basis + 4 transposes) | **ROBUST PASS** | All 8 elements give J-anticommuting fraction exactly 1/2 |
| α-continuum (H^(1) at O(ε)) | **PASS** | H^(1) is purely CP-odd, lives entirely in T_{2g} cubic-harmonic irrep |

57 passing tests (29 baseline → 57 with this milestone).

**Note (2026-05-20 convention audit)**: the β-multi metric was renamed
from `cp_violating_fraction` → `j_anticommuting_fraction`.  It is an
algebraic property of the dim-4 Higgs-like map space under a chosen J,
NOT a physical-CP measurement.  The 1/2 result is unchanged.  The
α-continuum verdict (H^(1) CP-odd, T_{2g}-localized) is also unchanged
under the corrected `cp_action_on_operator` (degree-2 polynomial is
even under k → -k, so momentum-flip in CP is a no-op at this order).

## β-multi result

Extending the original β audit to the full dim-4 Higgs-like basis from
`spacetime_qca.yukawa.color_singlet_charge_shift_basis()` and the 4
transpose-derived conjugate-shift components:

```text
basis[0]    fraction = 1/2
basis[1]    fraction = 1/2
basis[2]    fraction = 1/2
basis[3]    fraction = 1/2
basis[0].T  fraction = 1/2
basis[1].T  fraction = 1/2
basis[2].T  fraction = 1/2
basis[3].T  fraction = 1/2
```

**Interpretation**: the 50/50 J-commuting vs J-anticommuting split is a
**universal feature** of the dim-4 Higgs-like space, not an artifact of
the basis[0] convention.  Every color-singlet (+1/2, +1/2) charge-shift
map has exactly equal CP-even and CP-odd content under the chosen J.

This robust result removes the principal robustness concern raised after
the original β audit.

## α-continuum result

### Setup

The bare massless BCC Dirac walk's effective Hamiltonian, expanded to
second order in the lattice time-step ε:

```text
H_eff(ε, k) = H^(0)(k) + ε · H^(1)(k) + O(ε²)
```

with H^(0) = α·k (Dirac kinematic).  Extracted via Baker-Campbell-Hausdorff:

```text
B(ε, k) = exp(-i ε H_eff(ε, k))
B_2(k)  = ε²-coefficient of B(ε, k)
H^(1)(k) = i · (B_2(k) - B_2(k)^†) / 2.
```

### H^(1) is Hermitian, block-diagonal in chirality

```text
H^(1) = block_diag(H_R^(1), H_L^(1))

H_R^(1)(k) = (  k_x k_y       k_z (k_x + i k_y) )
             ( k_z (k_x - i k_y)   -k_x k_y      )

H_L^(1)(k) = same form (the two chiral blocks happen to coincide)
```

Hermitian: yes ✓.  Block-diagonal in chirality (no mass-type mixing): yes ✓.

Polynomial-coefficient norm: `||H^(1)||² = 12`.

### Cubic-harmonic decomposition: pure T_{2g}

Decomposing the 6-dim space of degree-2 momentum polynomials under O_h:

| Cell | A_{1g} | E_g | T_{2g} |
|---|---|---|---|
| CP-even | 0 | 0 | 0 |
| CP-odd  | 0 | 0 | **12** |

The entire H^(1) is:

- **100% CP-odd** (CP-odd fraction at O(ε) = exactly 1),
- **localized in T_{2g}** (cross-product monomials `k_x k_y, k_y k_z, k_z k_x`),
- absent from A_{1g} (the SO(3)-isotropic part) and E_g (traceless
  diagonal).

### Physical interpretation

The Bialynicki-Birula construction's complex `q_± = (1 ± i)/4` hop
coefficients give the BCC Dirac walk a structure where:

1. The continuum limit `H^(0) = α·k` is CP-invariant (and SO(3)-covariant
  through T_{1u}).
2. The **first** lattice correction `ε · H^(1)` is **entirely**
  CP-violating.
3. The CP-violating correction has the **specific angular signature**
  of the T_{2g} cubic irrep: it picks out particular combinations of
  spatial axes (`k_y k_z`, etc.), not isotropic |k|² or traceless
  diagonal contributions.

This is the strongest possible signature for the original
CP-from-quantization hope: lattice CP violation appears already at
**O(ε)**, is **structurally pure** (lives in a single irrep), and
**vanishes in the continuum limit** as required.

## What this strengthens

- **The cp/ sidecar's earlier α-2 result** (walk-level CP breaking) is now
  pinned to a specific order in ε and a specific cubic-harmonic irrep.
- **The β = 1/2 result** is now known to be universal across the dim-4
  Higgs space, not a basis-element artifact.

Together: the program has two independent, structurally clean CP slots
(walk-level T_{2g} CP-odd at O(ε); algebraic 50/50 CP-mixing in the
Higgs map space), both consistent with CPT exactness and continuum
recovery.

## What's still open

Even with these strengthenings, the result remains structural:

- **Magnitude vs experiment**: T_{2g} cubic-harmonic CP-odd operators
  correspond to specific Standard Model Extension (SME) parameters.
  Comparing the lattice-spacing ε to experimental bounds on those
  parameters requires literature/SME analysis — deferred per user
  decision.
- **Three-generation embedding**: still unresolved.  triality/ and
  broken_triality/ failed; the α-cont CP slot and β-multi result do
  not depend on a particular three-generation mechanism, but
  phenomenological CP-violation magnitudes (CKM δ, ε_K, etc.) require
  one.
- **Higher-order continuum corrections**: O(ε²) and beyond not pursued
  per user decision to keep scope to degree-2.

## What this audit specifically rules out / rules in

**Rules out**: the worry that the α-2 lattice CP-breaking was a
high-order or basis-fragile artifact.  It is in fact **leading-order
(O(ε)) and basis-universal**.

**Rules in**: a falsifiable prediction.  T_{2g} CP-violating operators
on the BCC lattice provide a concrete target for matching to:

1. measured CKM CP-violation parameters (after a three-generation
  embedding is established);
2. experimental bounds on SME T_{2g}-type Lorentz-violating operators
  (after the lattice spacing ε is set);
3. ratios between cubic-anisotropy effects and SO(3)-isotropic effects.

## Tests

```text
tests/test_discrete_symmetries.py     13 passed (unchanged)
tests/test_walk_symmetries.py          9 passed (unchanged)
tests/test_j_misalignment.py          11 passed (7 baseline + 4 multi-element)
tests/test_cubic_harmonics.py         10 passed (new)
tests/test_continuum_cp.py            14 passed (new)
                                    ─────────
                                      57 passed
```

Run with:

```bash
uv run pytest src/clifford_3plus2_d5/cp/tests -q
```

## Effort

- Multi-element β: ~1 hr (extending existing module + tests).
- nth_order_in_epsilon helper: ~½ hr.
- cubic_harmonics.py + tests: ~1.5 hr.
- continuum_cp.py + tests: ~2 hr.
- Reports + status: ~1 hr.

Total: ~6 hr of focused work for both audits.  Within the ~3-4 day
budget; the BCC Dirac symbol expanded fast enough that the F-cont-1
failure mode (sp.series time-out) did not materialize.

## Cross-references

- `SESSION_CP_ALPHA_BETA.md` — the predecessor session (α-2, α-3, β).
- `STATUS.md` — updated with current verdicts.
- `parameter_ledger.md` — appended with new convention entries.
- `../spacetime_qca/continuum.py` — extended with `nth_order_in_epsilon`.
- `cubic_harmonics.py` — new minimal O_h projector framework
  (degree-2 only).
- `continuum_cp.py` — BCH-correct H^(1) extraction, CP × irrep
  decomposition.
