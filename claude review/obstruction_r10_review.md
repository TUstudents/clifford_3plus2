# obstruction_r10 — review

_Reviewer: Claude (Opus 4.8). Independent assessment. See `REVIEW_PLAN.md` for rubric._

## TL;DR

`obstruction_r10` is the project's trunk and its most intellectually honest
piece. It **inverts** the usual SO(10) GUT exercise: instead of "Spin(10)
contains the SM, QED" (textbook), it asks whether a **local QCA update rule**
can *force* the carrier structure — a complex structure `J`, a `6+4` split, and
a Standard-Model-safe gate algebra — that the spinor construction needs, **without
hand-placing it**. The answer it reaches is a disciplined **no, within these
primitive classes**: five propositions + two corollaries close specified
bounded rule classes, backed by exact finite censuses. Every load-bearing
boolean is honestly `false` (`load_bearing_qca_bridge: false`,
`notation_only`/`candidate_only`). **The math I spot-checked is sound; the
framing is novel; the scope is narrow and the authors say so.**

- **Verdict:** a sound, well-fenced *finite-class no-go* program. Not a universal
  impossibility theorem, not a derivation. Publishable as an obstruction paper.
- **Confidence:** high that the proved propositions are correct and correctly
  scoped; medium that the obstruction is "deep" rather than an artifact of a
  strict self-imposed acceptance bar.

## What it claims

The headline (PROJECT_STATUS): *"Five propositions and two corollaries closing
specified primitive classes for the carrier-first QCA derivation of the Spin(10)
chiral-16 on R^10."* The bridge is accepted **only if** QCA rule data produce,
before invoking Spin(10):

```
K_x ≅ R^10;  J ∈ SO(K_x), J² = −I;  K_x = K_3 ⊕ K_2 (dim 6, 4);  J K_i = K_i;
A_geom ⊆ Comm(SU(3)×SU(2)×U(1))   [the "no-locking" gate condition]
```

and only then `W = (K_x,J) ≅ C^5`, `S⁺ = Λ^even(W)`, `Y = −N_3/3 + N_2/2`.

The claim is **not** that this is achieved. It is that natural primitive classes
of QCA rules **cannot** achieve it, and the propositions prove that for each
class.

## Progress

The project pivoted (documented in `docs/REORG_PLAN.md`) from a *derivation
attempt* to an *obstruction program*. Current state:

| # | Statement | Status | Basis |
|---|---|---|---|
| Prop 1 | Coarse symmetric primitives (clocks, block reflections, mode perms) yield no rule-derived 6+4 center | structurally closed | E1/E2 search, no surviving candidate |
| Prop 2 | Commuting non-scalar 2nd layer ⇒ lower-rank central idempotent | structurally closed | Bezout + commutativity proof |
| Prop 3 | Floquet-α block-preserving noncommuting on-site rule ⇒ lower locking, full addressability, or commutativity | structurally closed | Wedderburn proof + 3840-twist exhaustive scan |
| Cor 3.1 | On-site bridge closure | consequence of 1–3 | — |
| Prop 4a | Full 2400-candidate projector-free monomial-hop Bloch census: 1320 closed, all split-real centralizers, zero compatible-J | finite-census closed | `bloch_path_a_stepwise.py`, committed scan artifacts |
| Lemma 4b.1 | Split-real projected centralizer admits no real orthogonal J | structurally closed | one-line character proof |
| Prop 4b | Coprime monomial-hop incompatibility (general) | **conjectural** + finite witness | missing invariant named |
| Prop 5 | Route-1 compatible-J orbits are SM-inequivalent; gauge-equiv relaxation fails | structurally closed | Hodge-complement / hypercharge mismatch + `gauge_equivalence_check.py` |

Tooling: a large catalogue of exact `--check` scripts is the actual commit
gate (the README is explicit that full `pytest` is "a slow archival regression
suite, not the normal commit gate"). The exact symbolic work runs over `QQ`
and `QQ(ζ₁₂)` with certified polynomial identities — not floating-point
sampling.

## Assumptions

Cleanly separable into three tiers (the module does this itself; I'm checking
the separation holds):

**Textbook, correctly flagged as not load-bearing.** The Spin(10) ⊃ SM spinor
branching and `Y = −N_3/3 + N_2/2`. I recomputed it independently:
`dim Λ^even(C^5) = 16` with spectrum `{0:1, 1:1, 1/6:6, −2/3:3, 1/3:3, −1/2:2}`
— exactly the standard SO(10) 16. `theory.md` states verbatim: *"This is a
representation identity. It is not evidence that QCA geometry supplies the split
or J."* ✔ honest.

**The acceptance standard (self-imposed and consequential).** The bridge
requires `J` to be (i) rule-generated, (ii) unique up to *global* `±J`, with
(iii) the gate algebra strictly inside `C P_3 ⊕ C P_2` (no rank-one
addressability — "no-locking"). This bar is what makes the problem hard, and
**the negative results are partly a function of how demanding it is** (see
Gaps). A canonical spectral/monodromy `J` *does* exist (alpha-plus produces one
for all 10 Floquet patterns); the program rejects it because it isn't unique in
the full compatible variety. Defensible, but it is the single most important
judgment call in the module.

**Bounded primitive classes (per-proposition, explicitly narrow).** Global
clocks/reflections/permutations (P1); commuting second layers (P2);
Floquet-α "one irreducible quadratic factor per coarse block," block-preserving,
on-site (P3); projector-free monomial-hop Bloch on a single R^10 (P4a/4b);
Route-1 signed-twist (P5). Higher-dim carriers, genuine translation-breaking
defects, and parameterized rule families are **out of scope by hypothesis**.

## Soundness

I read `theory.md` in full and spot-checked the two pivotal computations; the
proofs use standard, correctly-applied tools.

- **Prop 2** — `[U,V]=0` ⇒ `R[U,V]` commutative ⇒ V's real spectral projectors
  are polynomials in V (Bezout on the real minimal polynomial) ⇒ central; a
  non-scalar V on the rank-6 block has a spectral component of rank `<6`.
  Correct.
- **Prop 3** — Wedderburn decomposition of the real block algebra `A|_{P_α}`;
  `J ∈ Z(A)`, `J²=−I` forces a `C` factor; on `R^6 ≅ C^3` the only central-simple
  options without a lower idempotent are scalar `C` or full `M_3(C)`, giving the
  trichotomy (lower locking / full addressability / blockwise commutativity).
  This is a clean, correct case analysis **within** the stated one-quadratic-
  factor hypothesis, and Cor 3.1 correctly limits the claim to that class. The
  3840-twist → 96-class → 0-bridge exhaustive run is the matching finite
  witness.
- **Prop 4a** — exact finite classification (2400 candidates), exact SymPy
  algebra closure, certified central-idempotent rank profiles
  (`[0,2,8,10]` for the dim-26 branch → rank-2 locking; `[0,4,6,10]` for dim-34
  → coarse center but zero rule-local `J`). Sound as a finite census; not a
  sampled numerical claim.
- **Lemma 4b.1** — trivial and correct: in `R^m` (split commutative semisimple),
  `J²=−P` would need a real coordinate with square `−1`; none exists.
- **Prop 5** — I verified the load-bearing step directly: the η-block Hodge flip
  `N_2 → 2−N_2` sends the fixed table `{ν^c:0, e^c:1, u^c:−2/3, d^c:1/3, …}` to
  `{1, 0, 1/3, −2/3, …}` — it **scrambles the SM assignments** (ν^c↔e^c,
  u^c↔d^c), so the four compatible `J`s are genuinely SM-inequivalent and the
  strict `±J` standard cannot be relaxed for this family. Correct, and the
  "load-bearing negative is the hypercharge mismatch" framing is accurate.

No soundness problems found. The proofs prove exactly what they state — no more.

## Novelty

- **Genuinely novel — the inversion.** The vast SO(10) GUT literature (Georgi
  1975; Fritzsch–Minkowski 1975) establishes the *positive* embedding; that part
  here is textbook and labeled as such. What is new is the sharp **inverse
  dynamical question**: can a *local finite-depth update rule* force `J` and the
  `3+2` split, rather than an author choosing them? That carrier-first,
  rule-must-force-`J` framing is, to my knowledge, not in the QCA literature
  (Bisio–D'Ariano–Perinotti reconstruct free Weyl/Dirac/QED dynamics from QCA
  principles; 't Hooft's CA interpretation is foundational but not about Spin(10)
  carrier rigidity).
- **Genuinely novel — the obstruction theorems themselves.** Props 2/3/4a/5 are
  specific new results (modest, but real) about when block-preserving
  orthogonal rules can/can't generate a central complex structure without
  collapsing into addressability or locking. The "no-locking" discipline (gates
  must be block scalars `C P_3 ⊕ C P_2`) is a clean, reusable operationalization
  of "don't smuggle in the answer."
- **Not novel.** The spinor branching, the hypercharge formula, and Wedderburn/
  Bezout/Schur themselves are standard. The contribution is the *question and
  the closure*, not the algebraic machinery.

## Gaps

1. **The bridge is not derived — and now claimed locally impossible.** Every
   check is `candidate_only`/`notation_only`. The positive QCA→Spin(10) bridge
   does not exist in this repo; the result is a *negative* one about specific
   classes.
2. **Not a universal no-go.** The honestly-named open routes survive:
   (a) translation-symmetry-breaking **defect** QCA with different α-patterns
   on each side; (b) **parameterized rule families**; (c) **higher-dimensional
   coarse-graining carriers**; (d) the **Prop 4b structural extension**. The
   program rules out the *easy/natural* routes and points at the hard ones — it
   does not prove QCA can never bridge.
3. **Prop 4b is conjectural.** The strongest single statement (empty
   compatible-`J` in the joint centralizer for coprime monomial-hop) rests on a
   finite witness; the promoting invariant ("projected compatible centralizer is
   split-real") is named but unproven. Finding one coprime monomial-hop rule
   whose projected centralizer contains an orthogonally-acting `C` factor would
   refute it and *become* the next bridge candidate — a clean falsifier.
4. **Strictness sensitivity.** Because a canonical spectral `J` exists and is
   rejected only on uniqueness grounds, an external reader may reasonably ask
   whether a weaker-but-physical acceptance standard would already "pass." The
   module argues (Prop 5) that relaxation fails *for Route 1* on hypercharge
   grounds — but that is one family, not a general defense of the strict bar.
   This is the place a referee will push.
5. **One generation only.** Three families, mass/Yukawa, mirror decoupling, and
   continuum dynamics are explicitly elsewhere (`lepton`, `spacetime_qca`, the
   closed sidecars). The bridge question is strictly one-generation carrier
   rigidity.

**Highest-leverage next step:** attack the genuinely-open *defect / non-
translation-invariant* class (named in §8 of the publication plan), since the
on-site and monomial-hop classes are now closed and Prop 4b's gap is precisely
about projector-free rules where `P_α/P_η` are *outputs* not inputs. Either
prove the 4b invariant (closing the coprime class) or find the refuting rule
(which becomes the bridge). That is the bifurcation the whole program now hinges
on.

## Confidence (calibrated)

- Propositions 1–3, 4a, 5 are **correct and correctly scoped**: high.
- The finite censuses are exact (not sampled): high.
- The obstruction reflects something structural rather than an artifact of the
  strict `±J` bar: **medium** — defensible but not settled; depends on whether a
  weaker physical standard is admissible.
- "QCA cannot derive the Spin(10) carrier": **not claimed**, and correctly so —
  the open classes are real.

## Verdict

This is the strongest module in the workspace as *science*, precisely because
it is the most willing to conclude "no." It takes a question that is usually
answered by hand-waving ("the SM fits in Spin(10), so a fundamental theory
should produce it") and converts it into exact, falsifiable, finite statements
about what local QCA rules can and cannot force — then proves the negative for
several natural classes and refuses to overclaim the rest. The mathematics I
checked is sound and the scope discipline is exemplary (the `load_bearing` and
`notation_only` flags are doing real work). Its limits are equally clear: it is
a finite-class no-go, conditional on a strict self-imposed acceptance standard,
for one generation, with the most interesting routes (defects, parameterized
families, higher-dim carriers, Prop 4b) still open. As an obstruction paper it
is ready and honest; as a claim about QCA and the Standard Model it is a
well-built fence around the easy answers, not a derivation.
