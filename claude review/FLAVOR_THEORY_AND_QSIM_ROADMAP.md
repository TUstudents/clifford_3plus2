# Roadmap — A Complete Flavor Theory and a BSM Quantum Simulator

_Brainstorm/strategy document. Claude (Opus 4.8), 2026-05-30. Grounded in this
session's reviews and calculations. Confidence tags: **[SECURE]** verified/derived,
**[OPEN]** plausible but unproven, **[BLOCKED]** known obstruction, **[SPEC]**
speculative._

_Revision note (2026-05-30): §0 and §6 updated to reflect (a) the energy-inert
selector result — the A₂ᵤ branch term is real in the dispersion but cancels on
BZ integration into `V_eff(h)` — and (b) the ε-provenance correction
(`ε_silver ≠ ε_lattice`; see `docs/epsilon_provenance.md`)._

---

## 0. The premise and the honest baseline

**The bet:** flavor is the *boundary spectral response* (Feshbach/Schur
self-energy) of a vacuum-framed BCC Weyl QCA — `Σ(z)=V†(z−H_Q)⁻¹V` — and the
Standard Model's flavor parameters are spectral data of an integrated-out
boundary, not fundamental constants.

**What is actually secure today:**
- **[SECURE]** `ε=√2−1` is the decaying root of the residual `K₃` triangle
  (degree 2), derived from the BCC body-diagonal orbit (V26/V27), using the
  *real* Bialynicki-Birula walk.
- **[SECURE]** Neutrino theorem: `Δm²₂₁/Δm²₃₁ = ε⁴ = 17−12√2 ≈ 0.0294` (vs obs
  0.0295), `m₁=0`, `sin²θ₁₃ = ¾ε⁴ ≈ 0.0221`. Conditional only on "the boundary
  sector is chiral" — i.e. on neutrinos being Weyl.
- **[SECURE/negative]** Three generations do **not** come from the algebra or
  topology (`triality`, `broken_triality`, `exceptional`, `topology` all kill).
- **[SECURE/negative]** The vacuum-framing *branch sign* is **not energy-driven**
  (for the natural h-coupling): the A₂ᵤ selector is real in the dispersion but
  cancels on BZ integration into `V_eff(h)` — the same vanishing that protects
  strong-CP. (Computed this session; Dirac-sector correctness check passes.)

**What is not:**
- **[OPEN/postdiction]** CKM/PMNS textures (depths 0,2,6; Clebsches 4/3,√2,1/√2;
  the spin-Coxeter phase word) — assigned, not derived.
- **[BLOCKED]** Absolute scales (`v`, `M_boundary`) — irreducible dimensionful
  inputs, in this framework and in the SM alike.
- **[BLOCKED]** Charged-lepton hierarchy — *not* clean `ε`-powers
  (`m_τ/m_μ≈ε^{−3.2}`), and the Z₃-symmetric Yukawa is degenerate (`koide`).

**The method that works (keep it):** kill-disciplined gates — one question,
explicit controls, derive-don't-assign, a shrinking declared-input ledger, and
*loud* honesty when a probe is blind (the lesson of V35: a parity-even,
polynomial probe cannot see a parity-odd, angular term).

---

## 1. Design principles

1. **Separate scale from pattern.** Every mass is `v·y_f`; only the dimensionless
   `y_f` *pattern* is ever predictable. Never claim "absolute masses."
2. **Derive or count.** Each texture input is either derived from `H_Q`/`V` or
   enters an explicit free-parameter ledger. No silent postdictions.
3. **One boundary, all sectors.** `H_Q` must be the *same* operator for ν, e, u,
   d. Cross-sector consistency is the sharpest internal falsifier.
4. **Falsifiability first.** Prioritize what experiment can kill near-term
   (neutrino ordering, `Σm_ν`, `0νββ`) over what is merely elegant.
5. **Probe parity/analyticity before trusting a zero.** (V35 lesson.)

---

## 2. Completeness checklist — what a "complete flavor theory" owes

| Observable | SM free params | Framework status | Route |
|---|---|---|---|
| 3 generations | (assumed) | **[BLOCKED]** algebra/topology killed | accept as input, or A5 |
| ν mass ratio, ordering, `m₁` | — | **[SECURE]** `ε⁴`, NO, `m₁=0` | done |
| ν absolute scale `Σm_ν` | — | **[BLOCKED]** needs `M_boundary` | A6 seesaw |
| ν nature (Dirac/Majorana) | — | **[OPEN]** sterile-edge ⇒ Majorana | A6 / C |
| PMNS angles + δ | 4 | **[OPEN/postdiction]** | A3 |
| charged-lepton masses | 3 | **[BLOCKED]** scale + hierarchy + degeneracy | A4 (Koide only) |
| up/down quark masses | 6 | **[BLOCKED]** scale; **[OPEN]** ratios | A3 |
| CKM angles + δ | 4 | **[OPEN/postdiction]** | A3 |
| Koide `K=2/3` | (relation) | **[OPEN]** consistent, not derived | A4 |
| flavor↔Lorentz↔CP link | — (BSM) | **[OPEN]** shared BCC origin; CP↔Lorentz share one *lattice* ε (distinct from flavor's silver ε — see `docs/epsilon_provenance.md`) | C / Track C |

The honest scoreboard: of the **~20** SM flavor parameters (9 charged-fermion
masses + 3 ν masses + 4 CKM + 4 PMNS), **one parameter-free relation is genuinely
predicted** — the ν mass ratio, which with `m₁=0` fixes the neutrino mass pattern
up to one scale. The roadmap is about converting **[OPEN]** rows into derivations
or honest counts, and shipping the BSM predictions the structure forces.

---

## 3. Track A — The flavor theory (analytic / symbolic)

### A1 — Consolidate & publish the neutrino theorem  **[SECURE, do now]**
Write up the secure chain `BCC orbit → K₃ → ε → K_ν → {ratio, m₁=0, θ₁₃, Σm_ν}`.
This is the one novel, falsifiable, near-secure result. **Acceptance:** a
standalone derivation with the conditional ("chiral boundary") stated, and the
near-term falsifiers (ordering, `Σm_ν`) listed. *Ship before anything else.*

### A2 — Cross-sector universality gate  **[OPEN, cheap, decisive]**
Verify the *same* `H_Q` (labels, the flavor invariant `ε_silver`, chirality)
governs ν, e, u, d. The same `ε_silver = √2−1` already appears in `koide`'s
charged-lepton ratio `(1+√2)² = 1/ε_silver²` and in the ν sector, across modules
that don't import each other (this is the dimensionless flavor invariant, *not*
the lattice ε — see `docs/epsilon_provenance.md`). **Acceptance:**
`UNIVERSAL_BOUNDARY_PASS` if one operator reproduces all sectors;
`SECTOR_DEPENDENT_BOUNDARY_KILL` otherwise. *If this kills, the whole bet is in
trouble — run it early and cheaply.*

### A3 — Textures: derive or count  **[OPEN, the real frontier]**
For CKM/PMNS, replace assignment with derivation or an honest ledger:
- depth hierarchy (0,2,6) ← boundary-shell geometry (same `K₃` coordination?);
- Clebsches (4/3, √2, 1/√2) ← actual color/BCC contractions;
- phase word ← **boundary holonomy / Berry phase** (not a hand-built word).
**Acceptance:** each factor tagged `derived` or `declared`, with a final count
`N_declared` vs `N_observables`. *This is the line between "ν theorem + numerology"
and "flavor theory."*

### A4 — Koide as the charged-lepton constraint  **[OPEN, hard]**
The one clean charged-lepton relation: `K=2/3` (6×10⁻⁶). `koide` showed the cone
axis **is** the BCC body-diagonal Z₃ axis but the equivariant Yukawa is
degenerate → needs Z₃-breaking. **Goal:** derive a specific BCC Z₃-breaking that
yields `K=2/3` *and* the observed ratios. **Acceptance:** `KOIDE_DERIVED_PASS` if
the breaking is forced (not fitted); else honest `KOIDE_BREAKING_IS_INPUT`.

### A5 — Generation count `N=3`  **[BLOCKED, high-risk optional]**
Algebra/topology routes are dead. Remaining: **Callias/Atiyah-Singer index of the
boundary Dirac operator** (`N_fam = Ind 𝒟_∂`), or cobordism/TQFT of the BCC walk.
**Acceptance:** an index theorem giving exactly 3, or an honest "3 is empirical
input." *Do not gate the program on this.*

### A6 — Scales and the geometric seesaw  **[OPEN]**
The sterile boundary **is** a seesaw: `m_ν ~ m_D²/M_boundary`. Pin the *relation*
between the ν scale, the charged-lepton Dirac scale, and `M_boundary`.
**Acceptance:** a derived dimensionless relation among scales (absolute values
still need one input); predicts `0νββ` effective mass and ν nature.

### A7 — Topological CP-branch selection  **[OPEN, deep, secondary]**
The A₂ᵤ selector is energy-inert but *real in the dispersion* (Berry-curvature-
like). Test whether a **θ-term/Chern coupling** of the framing field selects the
branch without energy (consistent with strong-CP). **Acceptance:** `BERRY_BRANCH_
SELECTED` (fixes the leptonic CP-phase *sign*) or `BRANCH_IS_TOPOLOGICAL_INPUT`.
*Only affects CP-sign/handedness, not the mass ratio — low priority.*

---

## 4. Track B — The dynamical BCC simulator (classical)

Goal: turn `spacetime_qca` from an *arena* into an instrument that **dynamically
reproduces the analytic flavor structure** (the real test of the bet).

### B1 — Constrain the dynamics  **[engineering, prerequisite]**
Exact Gauss-law projection (not diagnostic descent); always-on backreaction.
**Acceptance:** a gauge-fermion-Higgs run that conserves the constraint to
tolerance on a non-trivial lattice. *Without this it animates, it doesn't simulate.*

### B2 — Boundary/sterile sector + dynamical self-energy  **[the key build]**
Implement the unresolved `Q` chain in the JAX arena; measure the dynamical
self-energy pole. **Acceptance:** `ε=√2−1` emerges as a *measured* pole ratio,
and the **finite-size correction to `17−12√2`** is quantified (prediction #1).

### B3 — Dynamical mass generation  **[OPEN]**
Couple the Higgs condensate + Yukawa to the boundary recirculation; read masses
as poles. **Acceptance:** the ν mass *pattern* (`0:ε²:1`) appears dynamically;
charged-lepton masses appear *given* the Yukawa structure (mechanical, not a
prediction — say so).

### B4 — Scale-up & Lorentz recovery  **[engineering]**
Larger lattices, long-time stability; boost covariance beyond the `O(ε⁴)`
free-dispersion audit. **Acceptance:** stable production runs; measured dispersion
isotropy vs lattice size.

### B5 — Convergence test (Track A ⊗ Track B)  **[the verdict]**
Does the dynamical simulation reproduce the analytic Schur-complement flavor
structure (poles, framing, ratios)? **Acceptance:** `MECHANISM_DYNAMICALLY_
CONFIRMED` or a documented discrepancy (which would teach us where the analytic
factorization assumptions break).

---

## 5. Track C — The quantum-hardware QCA simulator (the genuinely "beyond" piece)

A QCA is **natively a quantum circuit** — local unitary update on a lattice. The
BCC Weyl/Dirac walk + boundary coin is directly a quantum-simulation target. This
is where "quantum simulator" is literal, not metaphorical.

### C1 — Circuit compilation  **[SPEC, novel]**
Compile the BB walk + `Q`-boundary coin to a gate set (Trotterized shift·coin).
**Deliverable:** a circuit whose classical limit matches Track B.

### C2 — Analog/digital quantum execution  **[SPEC]**
Run small instances on available hardware (digital NISQ, or analog lattice
fermion platforms — ultracold atoms / photonic walks already realize Dirac QCAs).
**What you'd learn that's classically hard:** real-time boundary recirculation
dynamics, entanglement structure of the framed vacuum, sign-problem-free
real-time CP/Lorentz-violation signatures.
**Acceptance:** a measured walk observable (dispersion, zitterbewegung frequency,
CP-odd anisotropy) matching the BCC prediction on hardware.

### C3 — Quantum simulation of BSM physics  **[SPEC, the vision]**
Use the quantum device to probe regimes the classical sim can't: interacting
boundary dynamics, vacuum framing as a real-time symmetry-breaking quench, sterile
sector entanglement. *This is a genuinely new use of quantum simulators — testing
a candidate BSM theory's dynamics, not just a condensed-matter model.*

---

## 6. The BSM predictions (the payoff — what makes this "beyond")

Near-term-falsifiable first (#1, #3), then structural/consistency claims that are
real but not near-term killable (#2, #4, #5):

1. **[SECURE, near-term] Neutrino sector:** normal ordering, `m₁=0`,
   `Σm_ν ≈ 0.058–0.059 eV`, `Δm²₂₁/Δm²₃₁ = 17−12√2`. Sharpest single falsifier:
   **a massless lightest neutrino.** Tests: JUNO (ordering, ~now), cosmology
   `Σm_ν` (DESI/Euclid/CMB-S4), `0νββ` (ν nature, via A6).
2. **[structural — established, not near-term falsifiable] Shared-origin
   cross-sector structure** _(corrected — the earlier "single-ε correlation" was a
   symbol collision; see `docs/epsilon_provenance.md`)._ The flavor, CP, and
   Lorentz sectors share a
   common origin (the BB walk + BCC `O_h` symmetry), and the CP and Lorentz sectors
   share *one lattice scale* `ε_lattice` (cp's T₂g `H⁽¹⁾` → sme's `d⁽⁵⁾ = ε_lattice`).
   **Caveat that kills the naive version:** the flavor invariant `ε_silver = √2−1`
   (dimensionless, sets the ν mass *ratio*) is a *distinct object* from `ε_lattice`
   (a length ~10⁻³³ m); they differ by ~32 orders of magnitude. So fixing the ν
   mass ratio does **not** fix the Lorentz/CP scale — there is no single number
   tying all three. The genuine distinctive claims are: (i) CP and Lorentz
   violation are locked by one lattice ε; (ii) all three sectors are organized by
   the same BCC symmetry group; (iii) the leptonic CP phase and the T₂g
   CP-violation share the BB chiral coin `q_±=(1±i)/4` as their source.
3. **[OPEN, BSM] Sterile/dark sector:** the unresolved `Q`-boundary contains
   sterile-like states; A6/B2 would give their mass/mixing pattern — natural dark
   or sterile-ν candidates.
4. **[structural] Strong-CP *cleanliness* (not a solution to strong-CP):** BCC
   centrosymmetry forbids the A₂ᵤ θ-channel in the vector sector (`strongcp`), so
   the QCA discretization does **not** spuriously generate `θ̄` — and the *same*
   vanishing makes the chiral selector energy-inert (this session). **Caveat (per
   the `strongcp` review):** this is lattice-artifact cleanliness, **not** a
   resolution of the SM strong-CP problem (no dynamical quark-mass phases or QCD
   instantons are in play); no axion is invoked because none is needed for the
   narrower statement.
5. **[SPEC] Lorentz-violation signature:** a specific T₂g cubic-anisotropy in
   fermion dispersion at scale `ε` — currently ~10⁸ below reach (`sme`), but a
   definite shape if reach improves.

---

## 7. Risk register / honest kill points

| Risk | Track | Consequence if it fails |
|---|---|---|
| Cross-sector `ε` differs | A2 | the "one boundary" bet dies — *run first* |
| Textures need ~as many inputs as observables | A3 | "flavor theory" → "ν theorem + numerology" (still publishable, honestly) |
| Z₃-breaking for Koide is free/fitted | A4 | charged leptons stay unpredicted |
| Index theorem doesn't give 3 | A5 | `N=3` is permanent empirical input |
| Constrained sim unstable / can't scale | B1–B4 | no dynamical confirmation; theory stays analytic-only |
| Sim contradicts analytic flavor structure | B5 | the factorization assumptions (equal returns, etc.) were wrong |
| Hardware too noisy for boundary dynamics | C2 | quantum-sim vision deferred |

**Meta-risk (the one to watch):** *framing drift* — declaring a derivation when a
gate only renamed an input or used a blind probe. Every gate must state the
parity/analyticity its probe can see, and the declared-input ledger must only
shrink for real reasons.

---

## 8. Sequencing — the critical path

```
   A1 (ship ν theorem)           ── do immediately, independent
        │
   A2 (cross-sector ε)  ◄── cheap, decisive; gates everything below
        │  pass
   ├── A3 (textures: derive or count)   ── the real research program
   ├── A4 (Koide breaking)              ── charged-lepton clean target
   └── A6 (seesaw scales, ν nature)     ── feeds 0νββ prediction
        │
   B1 → B2 (constrain sim; boundary sector)  ── parallel engineering track
        │
   B3 → B5 (dynamical mass; convergence test) ── the dynamical verdict
        │
   C1 → C2 → C3 (quantum-hardware QCA)        ── long-horizon, novel
```

**Do first (this quarter, low cost, high information):** A1 (write the secure
result) and A2 (the cheap decisive falsifier). These two determine whether the
program is "a clean neutrino theorem" or "a candidate flavor theory."

**Do next (the real program):** A3 + A4 + A6 (analytic) in parallel with B1→B2
(make the simulator real and test that `ε` emerges dynamically).

**Long horizon (the genuinely novel "beyond"):** Track C — compile the BCC QCA to
quantum hardware and use it to probe BSM boundary dynamics no classical machine
can. That is the version of "quantum simulator beyond the Standard Model" that is
both literal and new.

---

## 9. One-paragraph honest framing for any external write-up

> A vacuum-framed BCC Weyl quantum cellular automaton has a residual three-channel
> boundary whose self-energy fixes the neutrino mass ratio to `(√2−1)⁴ = 17−12√2`
> with no free parameter (matching data to 0.2%), predicts a massless lightest
> neutrino and normal ordering, and shares a common BCC-symmetry origin with the
> Lorentz- and CP-violation sectors (which are themselves locked together by one
> lattice scale — distinct from the flavor invariant; see
> `docs/epsilon_provenance.md`). It does **not** derive the absolute
> mass scale, the generation count (three is empirical), or the charged-lepton and
> quark mass hierarchies (those remain conditional textures). The path forward is
> to (i) test whether one boundary operator governs all sectors, (ii) derive or
> honestly count the mixing textures, and (iii) realize the automaton as a quantum
> simulator — classical first, then on quantum hardware — to confirm the boundary
> mechanism dynamically. The honest claim is a falsifiable neutrino theorem inside
> a disciplined research program, not a finished theory of flavor.
