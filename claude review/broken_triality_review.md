# broken_triality — review

_Reviewer: Claude (Opus 4.8). Independent assessment. See `REVIEW_PLAN.md` for rubric._

## TL;DR

`broken_triality` (~0.55k lines, 17 tests) is the follow-up to `triality`: even
though exact triality fails K1, does the *broken* cycle produce a forced flavor
structure with mass hierarchy? Answer: **no — BT-2 FAIL**. The 3×3 Yukawa built
as the Gram matrix of the projected triality orbit of `v_* = Y'` has eigenvalues
`{5/7, 31/72, 0}` (rank 2), and the two non-zero eigenvalues differ by only
`360/217 ≈ 1.66×` — essentially flat versus the SM's 10²–10⁵ within-sector
hierarchies. I verified all of this exactly. The flatness is *structurally
robust* (an orthogonal order-3 orbit yields equal-norm vectors whose Gram matrix
cannot spread far), which makes the kill solid.

- **Verdict:** clean negative; same structural wall as `triality` (and `koide`):
  3-fold-symmetric/orbit constructions give flat-or-degenerate spectra, no
  hierarchy without breaking. BT-1 "PASS" is a generous technical waypoint, not
  a positive result.
- **Confidence:** high that BT-2 fails and that the flatness is robust; the BT-1
  "PASS" is charitable (it counts a zero eigenvalue as one of "3 distinct").

## What it claims

STATUS: *"Can a broken Z/3 cycle … produce the measured mass hierarchy and CKM
CP phase with O(few) free parameters, while preserving CPT? Answer: no, under
the natural Spin(8) embedding and hypercharge-aligned v_*."* BT-1 PASS, BT-2 FAIL,
BT-3/BT-4 skipped (program closed at BT-2).

## Progress / soundness (verified)

| Kill | Verdict | Verified |
|---|---|---|
| BT-1 Yukawa overlap structure | PASS | eigenvalues `{5/7, 31/72, 0}`, **rank 2**, off-diagonals non-zero ✔ |
| BT-2 mass hierarchy | **FAIL** | non-zero ratio `360/217 ≈ 1.659`; fail threshold 10, pass 100 ✔ |
| BT-3 / BT-4 | skipped | correct — closed at BT-2 |

The construction `Y_ij = ⟨Π_SM(τ^i v_*), Π_SM(τ^j v_*)⟩` is exact symbolic linear
algebra (a Gram matrix of three projected orbit vectors), and I reproduced the
spectrum and the 1.66 ratio. Sound.

**Why BT-2's flatness is robust (not just a v_* artifact):** triality τ is
orthogonal, so the orbit `{v_*, τv_*, τ²v_*}` consists of equal-norm vectors; a
Gram matrix of equal-norm, symmetry-related vectors has tightly bounded
eigenvalue spread. A large hierarchy is essentially impossible from such an
orbit. This is the **same structural lesson** as `koide` (Z₃-equivariance forces
a degenerate pair) and `triality` (democracy vs the 6+4 split): symmetric
constructions can't manufacture hierarchy.

## Assumptions / inputs

1. **`v_* = Y'`** (restricted hypercharge) — physically motivated as a Higgs
   aligned with `U(1)_Y`, but it carries a residual `H_1↔H_2` swap symmetry that
   forces the **rank-2 deficiency** (one massless eigenvalue). So the rank deficit
   *is* a v_*-specific artifact (honestly flagged); the BT-2 flatness is not.
2. Natural Spin(8) embedding + the `triality` module's `T_cartan`.
3. Imports only from `triality` (which re-exports `lepton`) — no new algebra.

## Where I'd temper the framing

- **BT-1 "PASS" is charitable.** "3 distinct eigenvalues" includes `0`, so the
  Yukawa is rank 2 = one massless generation; it is really 2 non-zero masses, not
  3. The module is honest about this ("passes the literal kill condition; the
  rank deficit is forced by …"). The reader should treat the *whole sidecar* as a
  negative result (no hierarchy), with BT-1 a technical waypoint that exists to
  justify running BT-2 — not as evidence of viable flavor structure.
- The thresholds (fail < 10, pass > 100) are author-chosen but reasonable given
  the SM's 10²–10⁵ within-sector ratios; 1.66 fails by a wide margin regardless.

## Novelty

Negative and modest. It reinterprets the `triality` K1 failure as "forced flat
flavor" and confirms it. The genuine value is the *consolidation*: together with
`triality` and `koide`, it establishes that the program's natural 3-fold-symmetric
routes to flavor all produce flat/degenerate spectra. No new physics.

## Gaps

1. **Conditional on `v_* = Y'` and the natural embedding** — a non-natural `v_*`
   without the `H_1↔H_2` symmetry could give rank 3, but (per the orbit argument)
   would still be unlikely to give a large hierarchy. Untested, honestly named.
2. Doesn't address `A_4`/`T'` discrete flavor (no Spin(8) origin), the Furey
   octonion three-copy program, or the "approximate embedding from BCC anisotropy"
   hope — all orthogonal and explicitly out of scope.

**Highest-leverage next step:** none for this module — correctly closed at BT-2.
The live alternatives (discrete flavor groups, octonion copies) are different
programs, not extensions of this one.

## Confidence (calibrated)

- BT-2 FAIL (flat spectrum): **high** — verified and structurally robust.
- BT-1 "PASS": technically met but **charitable** (rank-2, one massless).
- "Broken triality can't give hierarchy" for the natural setup: **high**; for
  arbitrary `v_*`/embeddings: **medium** (honestly out of scope).

## Verdict

`broken_triality` is a correct, cheap, well-disciplined negative that closes the
"maybe the broken cycle still works" escape from `triality`'s K1 failure. The math
is exact and I reproduced it: the triality-orbit Yukawa is rank 2 with non-zero
eigenvalues differing by only ~1.66×, far below any reasonable hierarchy bar, and
the flatness is structurally robust because an orthogonal order-3 orbit cannot
produce a widely-spread Gram spectrum. The one thing to read carefully is that
BT-1 "PASS" is a generous technical waypoint (it counts a zero eigenvalue as a
distinct one), so the sidecar as a whole is a negative result, not a partial
success — which the module's own notes make clear. Its real contribution is
consolidating, with `triality` and `koide`, the project's sharpest recurring
structural finding: **3-fold-symmetric/orbit constructions on this carrier yield
flat or degenerate spectra, so the observed mass hierarchy requires explicit
symmetry breaking that none of these modules supplies.**
