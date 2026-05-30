# strongcp — review

_Reviewer: Claude (Opus 4.8). Independent assessment. See `REVIEW_PLAN.md` for rubric._

## TL;DR

`strongcp` (~2.3k lines, 69 tests) argues that the centrosymmetric BCC
Bialynicki-Birula walk does **not spuriously inject a strong-CP θ-term**: its
effective-Hamiltonian corrections live in parity irreps (`T_{2g}`, `T_{1u}`)
that the O_h selection rule keeps out of the pseudoscalar `A_{2u}` (θ-channel,
`k_xk_yk_z`), and they are *vector* (tr γ⁵H^(n)=0), so they induce no chiral
rotation of the fermion measure. I verified both legs to O(ε²). The mechanism
(centrosymmetry forbids pseudoscalar responses) is sound and is textbook crystal
physics applied to the QCA. **Two honest caveats:** the "direct computation"
(SC-4) confirms on a *spatial-only* sector where the topological charge is
identically zero by construction, and the headline "satisfies the strong-CP
problem without an axion" **overstates** — this shows the lattice doesn't
*create* a spurious θ̄, not that it *solves* the SM strong-CP problem.

- **Verdict:** sound, well-tested "the QCA discretization is strong-CP-clean"
  result. Not a resolution of the Standard Model strong-CP problem.
- **Confidence:** high that no spurious θ̄ is generated to O(ε²) and that
  higher orders are suppression-safe; low on the "solves strong-CP" framing.

## What it claims

PROJECT_STATUS: *"STRONG-CP TRIVIAL … the program naturally satisfies the
strong-CP problem without invoking an axion or accidental cancellation."*
STATUS (more careful): *"TRIVIAL at O(ε) and O(ε²); SAFE at higher orders; SC-4
direct lattice-gauge computation CONFIRMS."* The STATUS wording is accurate; the
PROJECT_STATUS wording is the overclaim (see Gaps).

## Progress

| Phase | Verdict |
|---|---|
| SC-1 degree-3 cubic harmonics | `A_{2u} ⊕ T_{2u} ⊕ T_{1u}` projectors (idempotent, orthogonal, complete) |
| SC-2 BCC centrosymmetry | lattice + BB walk + Bloch symbol all centrosymmetric |
| SC-3 higher-order parity | **H^(2) is 100% T_{1u}, zero A_{2u}** (verified) |
| SC-4 lattice topological charge | spatial Q ≡ 0 dimensionally; 6-dim plaquette rep parity-even ⇒ A_{2u}=0 for SU(2)_L/SU(2)_R/SU(4)_PS |
| SC-5 chiral anomaly + θ̄ | **tr(γ⁵H^(n))=0 for n=1,2** (verified); cross-term ≠0 at O(ε³) |
| SC-6 combined + θ̄ bound | aggregated TRIVIAL/SAFE |

69 tests. I verified SC-3 and SC-5 directly.

## Assumptions / inputs

1. `cp`'s H^(1), BB walk + BCC plaquettes from `spacetime_qca` — sound upstream.
2. **θ-channel ↔ A_{2u} identification.** `A_{2u}` (`k_xk_yk_z`) is genuinely the
   O_h pseudoscalar irrep, so identifying the parity-odd θ-density's angular
   signature with it is correct group theory.
3. **Spatial-only gauge sector** — no temporal Wilson plaquettes (explicitly
   deferred). This is load-bearing for SC-4 (below).
4. Neutron-EDM bound `|θ̄| ≤ 10⁻¹⁰` (Abel et al. 2020, real and correctly cited).
5. **Inherits `sme`'s ε bound** (~10⁻³³ m, itself unverified) for the O(ε³)
   suppression normalization — though the suppression is so large this barely
   matters (see Gaps 3).

## Soundness

- **Selection rule** `g×g=g, u×u=g, g×u=u` — correct O_h parity Clebsch-Gordan.
- **SC-3 verified:** H^(2) has zero A_{2u} component, entirely T_{1u}; H^(1)∈T_{2g}.
- **SC-5 verified:** `tr(γ⁵H^(1)) = tr(γ⁵H^(2)) = 0`, both purely vector; the
  **cross-term `tr(γ⁵H^(1)H^(2)) = −8k_xk_y³k_z/3 ≠ 0`** is non-zero at O(ε³).
  This confirms the precise claim: *exactly* trivial only to O(ε²); a non-zero
  chiral trace appears at O(ε³), argued safe by ε-suppression.
- **The cleanest leg is SC-5**, and it is the one most directly tied to the
  standard mechanism: a θ̄ shift comes from a chiral rotation of the fermion
  measure (Fujikawa); vector corrections (tr γ⁵=0) induce none. Sound.
- The cubic-harmonic O_h machinery is standard and correctly built.

**Weak leg — SC-4 "direct computation."** Its own ledger states: *"spatial-only
Q convention: 3 spatial directions; ε^{μνρσ} requires 4 distinct spacetime
indices → Q ≡ 0 dimensionally."* So the "direct lattice-gauge computation" that
"CONFIRMS" is evaluating a quantity that is **identically zero by construction**
(there are no temporal gauge links). It confirms by *absence of the relevant
sector*, not by computing a 4D topological charge and finding it zero. The
genuine 4D computation is deferred. The plaquette-parity argument (6-shape rep
is inversion-even ⇒ tr(F·F) ⊂ g-irreps ⇒ no A_{2u}) is correct on its own terms,
but it is still the *spatial* sector.

## Novelty

- **Mechanism is textbook, application is reasonable.** That a *centrosymmetric*
  lattice cannot support pseudoscalar (parity-odd) responses is standard crystal
  physics (cf. no piezoelectricity / no linear magnetoelectric effect in
  centrosymmetric crystals). Applying it to argue the BB walk can't populate the
  A_{2u} θ-channel is a clean, specific use, not a new principle.
- `tr(γ⁵ H^(n)) = 0` for vector operators is elementary.
- **Genuine value:** a tidy self-consistency result — the QCA discretization does
  not introduce a spurious strong-CP problem. Useful, modest, correctly the
  "SAFE/clean" kind of result.

## Gaps

1. **Does not address the actual SM strong-CP problem.** There are no dynamical
   quark mass phases (`arg det M_q`), no QCD instanton vacuum, and no topological
   susceptibility here. The well-posed question answered is "does the BB walk
   introduce a *lattice-artifact* θ-term?" (→ no), which is **not** "why is the
   physical θ̄ ≈ 0." The PROJECT_STATUS "satisfies the strong-CP problem without
   an axion" conflates the two.
2. **SC-4 is on a trivially-zero spatial sector** — temporal plaquettes / a 4D Q
   are deferred, so "direct computation confirms" is weaker than it sounds.
3. **O(ε³) chiral trace is non-zero** — exact triviality holds only to O(ε²);
   beyond that, safety rests on ε-suppression (inherits `sme`'s unverified ε).
   *However*, the suppression is so enormous (`ε/Λ_QCD⁻¹ ~ 10⁻¹⁸` per power, cubed)
   that even at the Planck scale θ̄ would sit absurdly far below 10⁻¹⁰ — so this
   particular safety is robust regardless of the exact ε bound.

**Highest-leverage next step:** the deferred temporal-plaquette / 4D topological-
charge computation, which would turn SC-4 from "trivial by absent sector" into a
genuine "computed and found parity-protected" result — and would let the module
state precisely *which* question it answers (lattice-artifact cleanliness),
distinct from the SM strong-CP problem it does not solve.

## Confidence (calibrated)

- No spurious θ̄ generated to O(ε²) (selection rule + tr γ⁵=0): **high** — verified.
- Higher-order suppression safety: **high** — enormous, robust to the ε bound.
- SC-4 "direct confirmation": **low–medium** — trivially-zero spatial sector.
- "Solves/satisfies SM strong-CP without an axion": **low** — framing overreach.

## Verdict

`strongcp` is a clean, well-tested structural argument with a sound core: a
centrosymmetric BCC walk's lattice corrections sit in parity irreps the O_h
selection rule excludes from the pseudoscalar θ-channel, and those corrections
are vector so they don't chirally rotate the measure — both verified, to O(ε²),
with a non-zero but hugely-suppressed O(ε³) remainder. The mechanism is textbook
crystal-symmetry physics applied correctly to the QCA, and the cleanest leg
(tr γ⁵H^(n)=0) is properly tied to how a θ̄ shift would actually arise. The
caveats are framing, not correctness: the "direct computation" confirms on a
spatial-only sector where the topological charge vanishes by construction, and —
more importantly — the result shows the QCA discretization does not *create* a
spurious θ-term, **not** that it *solves* the Standard Model's strong-CP problem
(no quark mass phases, no instantons are in play). Stated as "the BB lattice is
strong-CP-clean," it is a sound and useful consistency result; stated as
"satisfies strong-CP without an axion," it claims more than it shows.
