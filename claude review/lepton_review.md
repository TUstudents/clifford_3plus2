# lepton — review

_Reviewer: Claude (Opus 4.8). Independent assessment. See `REVIEW_PLAN.md` for rubric._

## TL;DR

`lepton` is the project's **positive pillar**: it exhibits the full Standard
Model gauge content — `SU(3)_c × SU(2)_L × U(1)_Y`, a compatible complex
structure `J`, and the correct one-generation hypercharge spectrum — as an
**explicit, exact, matrix-level construction** on the chiral-16 of Spin(10),
via the Pati-Salam factorization `Cl(0,10) = Cl(0,6) ⊗ Cl(0,4)`, plus a 1+1D
massless Dirac/Weyl continuum under background gauge links. **The construction
is correct and I verified its key outputs.** But it is *positive by
construction from a long list of declared choices*, and the headline physics
(PS/SO(10) one-generation embedding) is textbook. Its scientific value is as a
rigorous concrete realization and as the complement to `obstruction_r10`'s
negative result — **not** as a derivation of the SM from QCA.

- **Verdict:** sound, exact, honestly-labeled realization of the standard
  Pati-Salam/SO(10) one-generation embedding + a modest dynamics lift. Real
  engineering, not new physics, and conditional on exactly the inputs
  `obstruction_r10` shows are not QCA-forced.
- **Confidence:** high that the construction is correct; high that it is
  textbook rep theory rather than a derivation; low as standalone evidence for
  QCA→SM.

## What it claims

STATUS: *"derives the Standard Model gauge content (`SU(3) × SU(2)_L × U(1)_Y`),
a compatible complex structure, and the correct one-generation hypercharge
spectrum on the chiral-16 of Spin(10), with 1+1D massless Dirac/Weyl continuum
dynamics under background gauge links, using the Pati-Salam factorization of
Cl(0,10)."* The word **"derives"** is load-bearing and qualified throughout by
an explicit "Choice" column.

## Progress

Sessions 7–19b, ~9.4k lines, 120+ tests (I ran 32 core tests → all pass).

| Object | Source | Status (module's own) |
|---|---|---|
| chiral-16 carrier (real dim 32) | volume element of Cl(0,10) | **derived** |
| PS algebra `SU(4)×SU(2)_L×SU(2)_R` | `Cl(0,6)⊗Cl(0,4)` | derived *once factorization chosen* |
| compatible `J` | right-quaternionic Cl(0,4) commutant | structurally distinguished; **H-unit choice declared** |
| octonion structure | Fano table | **declared** (1 of 480) |
| imaginary `e_7` | choice in Im(O) | **declared** |
| `SU(3)_c×SU(2)_L×U(1)_Y` | PS→SM breaking | conventional |
| hypercharge spectrum | `Y = T_{3R} + (B−L)/2` | **declared** norms (T₃ᴿ×½, B−L×⅔) |
| 1+1D Dirac/Weyl continuum | Feynman checkerboard ⊗ R³² | derived |
| background gauge covariance | lattice gauge link | derived |

## Assumptions

The declared-input list is long and the module lists it openly (this is to its
credit):

1. `Cl(0,10)` signature; 2. the `Cl(0,6)⊗Cl(0,4)` PS factorization;
3. the Fano octonion table (1 of 480); 4. the `e_7` imaginary direction;
5. the **right-quaternionic `J`** (H-unit choice); 6. T₃ᴿ factor ½ and
B−L factor ⅔; 7. the PS→SM breaking pattern.

**The single most important cross-module fact:** the `J` declared here is the
*same object* `obstruction_r10` proves is **not forced** by any QCA rule in its
studied classes. So `lepton`'s positive result is conditional on precisely the
structure the trunk shows QCA cannot supply. The two modules are **consistent**
(one says "if declared, the SM follows"; the other says "QCA rules don't force
the declaration"), but a reader must not read `lepton` as closing the QCA→SM
gap. It doesn't, and STATUS doesn't claim it does.

## Soundness

Everything I checked is exact symbolic computation and correct.

- **Clifford layer (Session 18):** 8×8 real `Cl(0,6)`, 8×8 real `Cl(0,4)`,
  64×64 `Cl(0,10)` tensor rep; relations pass; chirality ranks `(32,32)`;
  `su4`=15, `su2_l`=3, `su2_r`=3, `[su2_l,su2_r]=0`. ✔
- **`J` compatibility:** `J²=−I`, `JᵀJ=I`, `[J,su4]=[J,su2_l]=[J,su2_r]=0`. The
  comparison control (a simple Spin(4) bivector also squares to −I but breaks
  both SU(2)s) is a genuine negative test that distinguishes the right-
  quaternionic `J`. ✔
- **SM extraction (19a):** B−L = sum of three Spin(6) Cartan bivectors;
  centralizer in `su4` = 9; `su3_c` = 8 and closes; `8+3+1=12`;
  `[su3_c,su2_l]=[su3_c,Y]=[su2_l,Y]=0`; `J` commutes with all. ✔
- **Hypercharge (19b)** — I ran the audit. The **raw** spectrum is
  `{7/4:1, −1/4:1, −3/4:2, 3/4:3, −5/4:3, 1/4:6}` and **does not** match SM by a
  single common scale (`raw_common_scale_matches_sm = False`). Two independent
  component normalizations are applied (T₃ᴿ×½, B−L×⅔), after which the spectrum
  and the joint `(Y,T₃ᴸ)` table match SM exactly.

  **Critical reading of the normalization:** I checked that the two factors are
  the *standard* canonical normalizations, not arbitrary tuning — T₃ᴿ raw
  eigenvalues ±1 → ±½ (canonical weak-isospin), B−L raw ±½,±3/2 → ±⅓,±1
  (canonical quark/lepton B−L) — and the `Y = T₃ᴿ + (B−L)/2` coefficient is the
  textbook PS formula, *not* fitted. Crucially, **the multiplicity multiset
  `{1,1,2,3,3,6}` is identical in the raw and normalized spectra** — i.e. the
  normalization only relabels charge *values*; the *structural* content (the
  multiplicities) is invariant and is the genuine result. That structural
  content is exactly the `Λ^even(C⁵)=16` branching already verified in
  `obstruction_r10`. So: multiplicities = real structural fact (= textbook
  SO(10) 16); charge values = standard normalization of the PS formula. Sound,
  and the module discloses the two-factor normalization explicitly.

No soundness problems. The arithmetic is exact and the controls are real.

## Novelty

- **Not new physics.** The content — `SU(4)×SU(2)_L×SU(2)_R` from
  `Cl(0,6)⊗Cl(0,4)`, breaking to `SU(3)_c×SU(2)_L×U(1)_Y`, `Y=T₃ᴿ+(B−L)/2`,
  chiral-16 = one SM generation with the standard hypercharges — is the
  Pati-Salam (1974) / SO(10) GUT (Georgi, Fritzsch–Minkowski 1975) embedding.
  Decades old and textbook. The module says so ("derived once factorization is
  chosen", "conventional", "declared").
- **The octonion/`G_2 ⊃ SU(3)` color thread** (Sessions 13–14) is in the
  tradition of octonionic SM model-building (Günaydin–Gürsey; and recent work by
  Furey, Dubois-Violette, Todorov). Also not new, but a legitimate research
  line.
- **What is genuinely contributed:** a *fully explicit, exact-arithmetic,
  test-backed* matrix realization of this embedding on a single concrete
  Clifford carrier, wired to a QCA-motivated checkerboard so the same carrier
  carries 1+1D massless gauge-covariant propagation. The value is rigor and
  concreteness (every claim is an exact matrix identity with a test), not
  conceptual novelty.

## Gaps

The open list (module-stated, accurate) is large:

1. **No mass / Yukawa** — no coupling between chiralities; the carrier is
   massless (Floquet gapless only at `k=0`).
2. **No dynamical gauge fields** — gauge enters only as background links
   (Yang-Mills/plaquette dynamics live in `spacetime_qca`).
3. **1+1D only** — no 3+1D Lorentz recovery (deferred to `spacetime_qca`).
4. **One generation** — three generations is exactly what the closed sidecars
   (`triality`, `broken_triality`, `exceptional`, `topology`) all fail to
   produce; `lepton` does not address it.
5. **Color orientation (3 vs 3̄) not audited** — multiplicities identify the
   triplet sectors but not their chirality/orientation (a Casimir/weight audit
   is deferred).
6. **The deepest gap is conditional-ness:** the positive result is granted by
   declaring `J`, the factorization, and the octonion/normalization choices.
   Per `obstruction_r10`, the `J` and `6+4` split are *not* QCA-forced. So
   `lepton` answers "what gauge content lives in the chiral-16 *if* you accept
   the Clifford framework," which was essentially already known — it does not
   answer "does QCA produce the SM."

**Highest-leverage next step:** the honest frontier is *mass*, since everything
above the gauge-content layer (Yukawa, Higgs as condensate, generations) hangs
off it — and it is the one place the construction would have to *predict* rather
than *exhibit*. This is also where it would connect to `koide`/`boundary_response`
(which already make falsifiable mass-ratio statements). The gauge-content
extraction itself is finished and does not need more work.

## Confidence (calibrated)

- Construction correctness (Clifford relations, PS/SM dims, `J` compatibility,
  hypercharge multiplicities + values): **high** — exact and test-backed.
- "Correct one-generation SM content": **high that it is correct**, but it is
  textbook rep theory made explicit, **not** a derivation.
- As standalone evidence for QCA → SM: **low** — conditional on declared
  choices that the trunk shows are not forced.

## Verdict

`lepton` does exactly what a careful person would want the "positive" half of
this program to do: take the algebraic framework seriously, build it explicitly
in exact arithmetic, and verify — with real negative controls — that the
Standard Model's one-generation gauge content, a compatible complex structure,
and the correct hypercharge spectrum genuinely live inside the Spin(10)
chiral-16 via Pati-Salam, then show the carrier supports massless gauge-covariant
1+1D propagation. All of that is correct. The honest framing is that this is a
*construction from declared choices*, not a derivation: the physics is the
1970s PS/SO(10) embedding, the long declared-input list (signature,
factorization, Fano table, `e_7`, right-quaternionic `J`, two charge norms,
breaking pattern) is the price, and the `J` it declares is precisely what
`obstruction_r10` proves QCA rules do not force. Read as "the SM gauge content
is exactly realizable on this QCA-motivated carrier given the Clifford
framework," it is a clean, rigorous, true result. Read as "QCA derives the SM,"
it overreaches — and the module itself is careful never to make that second
claim.
