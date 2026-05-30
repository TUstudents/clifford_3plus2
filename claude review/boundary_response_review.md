# boundary_response — review

_Reviewer: Claude (Opus 4.8). Independent assessment. See `REVIEW_PLAN.md` for rubric._
_Note: reviewed interactively early in the session at V23/V25; this written version is updated to the current **V31** state (280 tests pass)._

## TL;DR

`boundary_response` is the most actively-developed and methodologically
disciplined sidecar: a "prove-or-kill" program that derives flavor structure as
the boundary spectral response (Feshbach/Schur self-energy) of a BCC Weyl QCA.
Its **strongest result is genuine and verified** — the neutrino core
`K_ν = ε²P_u + P_b` with `ε = √2−1`, giving `Δm²₂₁/Δm²₃₁ = ε⁴ = 17−12√2 ≈
0.0294` and `sin²θ₁₃ = ¾ε⁴`, both matching NuFIT at the percent level. Since my
interactive review, gates **V26–V31** substantially strengthened this: `ε` is now
*derived* from the residual `K₃` graph degree (not an asserted recurrence),
vacuum framing is an exact BCC orbit + selector-potential construction, and the
whole chain is honestly reduced to **one** irreducible physical input — that a
tetrahedral selector order parameter condenses. The persistent caveats are
unchanged: the PMNS/CKM textures (V7–V14) remain honest **postdictions** with
substantial researcher degrees of freedom, and the "flat ergodicity" assumption,
though re-expressed as a clean equal-rank fiber theorem (V30), is still an
assumption.

- **Verdict:** sound, novel neutrino-core theorem chain — now reduced to a single
  named physical assumption, achieved with exemplary honesty (it sidesteps the
  over-claim trap and respects its own no-gos). PMNS/CKM are conditional
  postdictions. The framing ("conditional program, not a theory") is accurate.
- **Confidence:** high on the neutrino-core math and the V26–V31 structural
  reduction; low on the CKM/PMNS textures as derivations (high as fits).

## What it claims

The mature claim (research note): *"Flavor is the sector-dependent boundary
spectral response of a vacuum-framed BCC Weyl QCA."* The compact invariant is
`ε = √2−1`; the headline falsifiable consequence is
`Δm²₂₁/Δm²₃₁ = (√2−1)⁴`. PMNS and CKM are explicitly labeled conditional.

## Progress (V1–V31, 280 tests)

| Block | Gates | Result |
|---|---|---|
| Neutrino core | V1–V6 | S₃ no-go (V1), framed-sterile ansatz, exact semi-infinite **Weyl-function theorem** (V6): `Σ/m = ε²P_u + P_b` at `z=2√2` |
| Charged-lepton / PMNS | V7–V10 | leakage `sin θ_e = √(3/2)ε²`, spin-Coxeter phase word, conditional PMNS assembly |
| Quark / CKM | V11–V14 | `Cl₅` flat coin, depth hierarchy (0,2,6→ε²,ε⁴,ε⁶), color/BCC Clebsches, conditional CKM |
| Rigidity / ergodicity | V15–V25 | honest no-gos (V15–V18: flatness *not* forced by unitarity/symmetry); Jaynes/microcanonical/conserved-label re-derivations of the flat ratio |
| **Source derivations (new)** | **V26–V31** | `ε` from `K₃` degree; BCC orbit framing; selector Hamiltonian; unit continuation; equal-rank fiber; tetrahedral selector potential |

## Soundness (verified)

**The core math is exact and correct** (verified interactively + here):

- **`ε` recurrence** `ε²+2ε−1=0 → √2−1`. **V26** now derives the coefficient `2`
  as `deg(K₃)`, with `K₂`/`K₄`/scaled controls giving different roots
  (`K₂`→golden, `K₄`→(√13−3)/2 — I confirmed the family). Crucially, **V26 does
  not over-claim** that the radial quotient *is* the unit half-line Weyl chain
  (the det = −1 vs +1 / on-site-shift trap I flagged when this gate was
  proposed) — it keeps the unit-continuation normalization as a *named* input
  (then pins it in V29 as `M = I₃`). This is the honest route.
- **V6 Weyl function** `m(z) = (z−√(z²−4))/2` is the genuine semi-infinite
  tight-binding Green function; at `z=2√2`, `m=√2−1=ε`. Sound.
- **S₃ no-go (V1)** — by Schur on `3 = 1⊕2`, an S₃-invariant operator is
  `αP_u+βP_D` and cannot produce `K_ν`; the module deploys this *against its own*
  headline and is explicit that `K_ν` needs `S₃→S₂`. (Same Schur mechanism as
  `koide`/`broken_triality`.)
- **V27/V28/V31 (vacuum framing), verified here:** the 4 BCC primitive exits form
  a regular tetrahedron (`v_i·v_j = −1/3` ✔); a rank-one selector `E_i=−h·v_i`
  gives spectrum `(−1,⅓,⅓,⅓)`, gap 4/3 ✔; the tetrahedral cubic invariant
  `C(h)=Σ(h·v_i)³` has `C(v_i)=8/9`, with the antipodal branch correctly rejected
  ✔. Together these upgrade "select one exit" from a bare `S₄→S₃` declaration to
  a concrete symmetry-breaking gate.
- **Empirical fit (verified vs PDG/NuFIT):** `Δm²₂₁/Δm²₃₁ = ε⁴ ≈ 0.0294`
  (obs ~0.0296); `sin²θ₁₃ = ¾ε⁴ ≈ 0.0221` (obs ~0.0222); `|V_us|≈0.2255`,
  `|V_cb|≈0.0416`, `|V_ub|≈0.0036`, `J≈3.0×10⁻⁵`, `δ_q=atan√5=65.9°≈γ`. Percent-
  level across the board.

## Assumptions — and the V26–V31 reduction

The signature feature is the **shrinking "remaining declared inputs" ledger**,
gate by gate. Tracking it:

- early (V6): the transfer recurrence + probe `z=2√2` + equal sterile returns.
- after V25: `vacuum_framing`, `unit_chain_normalization`, `regular_fiber_or_max_entropy_prior`.
- **after V31: essentially one physical input** — `tetrahedral_selector_order_parameter_condenses` — plus the structural fiber-isomorphism of V30.

This is real progress: `ε` (V26+V29), vacuum framing (V27/V28/V31), and the
equal-degeneracy flat ratio (V30) are now structural theorems rather than
declarations. **But two things remain genuinely assumed:**

1. **Condensation (V31's own caveat):** "does not derive microscopic condensation
   of that order parameter." That the vacuum *develops* the tetrahedral selector
   field is the irreducible physical input the whole neutrino core now rests on.
2. **Fiber isomorphism (V30):** `H_Q = C⁶_label ⊗ B_local` assumes every
   conserved-label fiber carries an *isomorphic* local patch (equal degeneracy).
   This is the old "flat ergodicity" assumption re-expressed — cleaner (an
   equal-rank theorem that respects the V22 dynamical no-go), but the physical
   content ("why label-independent fibers rather than sector-dependent ones?") is
   still an assumption, not a derivation.

## The unchanged caveat: PMNS/CKM are postdictions

V26–V31 strengthened the **neutrino-core / ε** sector; they did **not** touch the
quark/lepton **texture** assembly (V7–V14). Those still carry substantial
researcher degrees of freedom chosen with the targets known:

- the quark family **depth assignment** (0, 2, 6) → ε², ε⁴, ε⁶;
- the **Clebsch factors** 4/3 (color `C_F`), √2, 1/√2;
- the **spin-Coxeter phase word** `−q_{A₃}q_{A₂}` with `e^{iπ/4}·e^{iπ/3}`.

Each gate has rejection controls (good practice), but the winning options were
selected knowing the answer, and the count of tuned discrete choices is
comparable to the number of observables reproduced. The module is honest about
this — CKM is "not a blind prediction… lower-confidence consistency success."

## Novelty

- **Genuinely novel:** the boundary-response (Feshbach self-energy) framing of
  flavor in a QCA; the arithmetic link `ε=√2−1` (silver ratio) →
  `Δm²₂₁/Δm²₃₁ = 17−12√2`; and now the derivation of that `ε` from the residual
  `K₃` graph degree of a BCC vacuum orbit. I am not aware of this mass-ratio
  relation or its graph-theoretic origin elsewhere.
- **Notable cross-module echo:** the same silver ratio appears in `koide`'s special
  ratio `(1+√2)²` and mass ratio `2/ε⁴`, in a module that does not import this one
  — flagged in the `koide` review as a shared-BCC-`√2` question worth a deliberate
  cross-check.
- **Not novel:** the Feshbach/Schur machinery is textbook; the S₄→S₃→TBM lineage
  is standard discrete-flavor territory; the CKM piece is structured numerology.

## Gaps

1. **One irreducible physical input remains** — tetrahedral order-parameter
   condensation (V31). The neutrino core is "derived modulo the vacuum condensing
   the right way."
2. **Fiber isomorphism = flat ergodicity, renamed** (V30) — cleaner, but still an
   assumption about label-independent boundary fibers.
3. **PMNS/CKM textures are postdictions** (V7–V14) — unchanged; the highest
   concentration of researcher DOF.
4. **No three generations** — the module derives ratios/mixings within a
   one-generation-style boundary picture; the generation count is the unsolved
   problem the kill-sidecars closed negatively.
5. **`sin²θ₂₃ ≈ 0.489`** is the weakest PMNS fit (near maximal; obs best-fits sit
   ~0.45 or ~0.57).

**Highest-leverage next step:** the module has *done* the one I flagged earlier
(derive `ε` and framing from the graph/orbit — V26–V31). The next is the same
discipline applied to the **texture sector**: derive (not assign) the quark depth
hierarchy and the Clebsch/phase-word factors from the boundary shell, or
explicitly count and report the texture sector's free parameters against the
observables it reproduces. Secondarily, the V31 condensation assumption is the
natural frontier — a microscopic reason the tetrahedral selector forms.

## Confidence (calibrated)

- Neutrino core (`ε⁴`, `¾ε⁴`) and the V26–V31 structural reduction: **high** —
  exact, verified, and honestly scoped.
- `ε` as derived (not asserted): **high** now (V26/V29), with the unit-chain
  identity correctly *not* over-claimed.
- CKM/PMNS as derivations: **low**; as percent-level fits: **high**.
- The program as "a theory of flavor": **not claimed** — it is a conditional
  program reduced, in the neutrino sector, to one physical assumption.

## Verdict

`boundary_response` is the workspace's most ambitious and most disciplined
sidecar, and since my first pass it has done exactly the right thing: instead of
stacking another texture layer, it attacked the foundations, and gates V26–V31
turned the central invariant `ε = √2−1` from an asserted recurrence into the
decaying root of the residual `K₃` graph quotient of a BCC vacuum orbit, with the
vacuum framing itself promoted to a concrete tetrahedral selector-potential gate.
It did this **honestly** — declining to over-claim the unit-chain identity where
a determinant/on-site subtlety lurks, keeping each normalization as a named input
until it is separately pinned, and respecting its own dynamical no-go (V22) when
re-deriving the flat ratio as a structural equal-rank fiber theorem (V30). The net
effect is that the falsifiable neutrino-core prediction (`Δm²₂₁/Δm²₃₁ = 17−12√2`,
`m₁=0`, `Σm_ν≈0.058 eV`, `sin²θ₁₃=¾ε⁴`) now rests on essentially a single physical
assumption — that a tetrahedral order parameter condenses — which is a remarkably
tight place to stand and an excellent, genuinely novel, testable result. The
honest counterweights are that this last assumption is undeniably physical and
underived, that V30's fiber isomorphism is the flat-ergodicity hypothesis in
better clothes, and that the PMNS/CKM textures remain conditional postdictions
with researcher freedom comparable to their output. The module's own framing —
"the disciplined route from beautiful numerology toward a testable physical
theory" — is, unusually for this workspace, neither an over- nor an under-
statement of what it has achieved.
