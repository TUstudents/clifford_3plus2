# triality — review

_Reviewer: Claude (Opus 4.8). Independent assessment. See `REVIEW_PLAN.md` for rubric._

## TL;DR

`triality` (~0.75k lines, 21 tests) asks one load-bearing question — *can the
explicit Spin(8) triality automorphism produce three equivalent SM-generation
carriers without declaring three generations by hand?* — and kills it cleanly:
**K1 FAIL**. I verified the two things that matter: the triality matrix is a
**genuine root-system automorphism** (it preserves all 24 so(8) roots, is
orthogonal, order-3, det +1 — not an arbitrary order-3 rotation), and it
**provably maps all three SM-inside-Spin(8) Cartan generators outside** the SM
Cartan subspace. So the three triality-rotated chiral-16 copies sit at
inequivalent SM positions and cannot be three equivalent generations.

- **Verdict:** an exemplary kill-disciplined negative — cheap, decisive,
  correctly scoped, no overclaim. The closure is robust to the τ/τ² choice and
  honestly limited to the natural embedding.
- **Confidence:** high that K1 fails for this embedding and that triality is
  genuine; medium that "triality can't give 3 generations" generally (non-natural
  embeddings and broken triality are explicitly out of scope).

## What it claims

STATUS: *"Can explicit Spin(8) triality produce three equivalent SM-generation
carriers without declaring three generations by hand? Answer: no, under the
natural Pati-Salam-aligned Spin(8) embedding."* K1 (Cartan necessary condition)
fails on all 3 SM Cartan generators.

## Progress

| Test | Result |
|---|---|
| triality matrix well-formed | orthogonal, order-3, det +1, **preserves the 24 so(8) roots** (verified) |
| K1 (Cartan subspace preserved?) | **FAIL** — 3 of 3 SM-in-Spin(8) Cartan generators map out |
| K2 (Y' spectrum) | diagnostic context only |

21 tests. Everything downstream (mass/Yukawa, CP, CKM/PMNS fit) deferred because
K1 failed — correct kill discipline.

## Assumptions / inputs

1. **Pinned embedding** `Spin(8) ⊂ Spin(10)` via Cl(0,10) gamma indices `{0..7}`,
   Cartan `H_k = ½γ_{2k}γ_{2k+1}`. Declared, "natural" Pati-Salam-aligned.
2. **Triality acts via its Cartan matrix** `T_cartan` (the full 28-generator
   extension is not built — and correctly is not needed: if the Cartan subspace
   isn't preserved, no full-level τ can preserve the subalgebra).
3. **SM-inside-Spin(8)** is a *projection*: `SU(3)_c` survives (Cl(0,6)⊂Cl(0,8)),
   `SU(2)_L` does **not** survive into Spin(8), and `U(1)_Y` survives only as the
   projection `Y'` onto the Spin(8) Cartan. So the tested object is already a
   reduced SM. (Honestly stated in `sm_restriction.py`.)
4. All algebra imported from `lepton` via `reuse.py` — no duplication.

## Soundness

Everything I checked is exact and correct, and the key worry (is `T_cartan`
*really* triality?) is dispatched.

- **`T_cartan` is a genuine triality.** I verified it is orthogonal, order-3,
  det +1, **and maps the 24-element so(8) root system onto itself** — the
  definitive check that distinguishes a real Dynkin automorphism from an
  arbitrary order-3 orthogonal matrix. Its +1 eigenspace is 2-dimensional
  (eigenvalues `{1,1,ω,ω̄}`), consistent with cyclically permuting the three
  outer D₄ nodes while fixing the central one.
- **K1 FAIL verified.** All three SM-in-Spin(8) Cartan generators (2 from
  `SU(3)_c` + `Y'`) leave the 3-dim SM Cartan subspace under `T_cartan`.
  `Y' = (1/3,1/3,1/3,1/2)` and the residual `‖Y − Y'‖² = 2` confirms physical
  hypercharge genuinely has out-of-Spin(8) content (the `γ_8γ_9`/electroweak
  direction).
- **Robust to the τ vs τ² choice.** A subspace is invariant under τ iff under the
  whole cyclic group ⟨τ⟩ (since τ² = τ⁻¹), so failing for τ fails for τ² too —
  the closure isn't an artifact of picking one of the two order-3 elements.
- **The structural reason is sound and is the real content:** triality acts
  *democratically* on `H_0..H_3`, but the Pati-Salam factorization
  `Cl(0,6)⊗Cl(0,4)` singles out `H_3` (the electroweak/Cl(0,4) Cartan). Any
  triality that cyclically mixes three of the four Cartan directions necessarily
  mixes the color side with the electroweak side, so it cannot respect the 6+4
  split that defines the SM embedding. K1 is the concrete witness of this.

No soundness problems.

## Novelty

- The "three generations from Spin(8)/SO(8) triality" idea exists in the
  literature; this module's contribution is a **clean, explicit falsification**
  of the natural-embedding version, with the precise structural reason
  (democracy-vs-6+4 mismatch). Modest but genuine negative knowledge.
- The machinery (triality, Cartan, D₄ roots) is textbook; the value is the sharp
  application as a kill test.

## Gaps

1. **Tested object is a reduced SM.** `SU(2)_L` doesn't survive into Spin(8) at
   all, so K1 concerns `SU(3)_c ⊕ U(1)_{Y'}`, not the full SM. (Honest, but it
   means even the "thing not preserved" is a projection.)
2. **Conditional on the natural embedding.** Non-natural `Spin(8) ⊂ Spin(10)`
   embeddings are untested (flagged). Robust to τ/τ² but not to conjugation by
   embeddings.
3. **Doesn't address approximate/broken triality** — that is `broken_triality`'s
   separate program (and it too fails, at BT-2).

**Highest-leverage next step:** none for this module — it is correctly closed.
The honest open question (could a *non-natural* embedding pass K1?) is a
different, larger search; the cheap decisive result here is that the *natural*
route is dead, and the project rightly moved on.

## Confidence (calibrated)

- `T_cartan` is a genuine triality: **high** — root-system preservation verified.
- K1 FAIL for the natural embedding: **high** — verified, robust to τ/τ².
- "Triality cannot give 3 SM generations" in general: **medium** — specifically
  the natural embedding; non-natural embeddings and broken triality are out of
  scope (and honestly named).

## Verdict

`triality` is a model of the project's kill-disciplined method: it poses one
sharp, falsifiable question, builds exactly enough machinery to answer it,
verifies that the machinery is the real object (the matrix genuinely is Spin(8)
triality — it preserves the full root system), and reports a clean K1 FAIL with
the correct structural explanation (triality is democratic across the four
Cartan directions while the Pati-Salam embedding privileges the electroweak one,
so the SM Cartan cannot be triality-invariant). The result is robust to the τ/τ²
choice and honestly scoped to the natural embedding, with everything downstream
correctly deferred rather than half-built. Unlike some other modules, there is no
framing gap to flag — the claim ("no three equivalent generations from natural
Spin(8) triality") is exactly what the verified computation shows, no more. This
is what a negative result should look like.
