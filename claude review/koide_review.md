# koide — review

_Reviewer: Claude (Opus 4.8). Independent assessment. See `REVIEW_PLAN.md` for rubric._

## TL;DR

`koide` (~1.5k lines, 54 tests) takes a tempting coincidence — the Koide-formula
cone direction `(1,1,1)/√3` is exactly the BCC body-diagonal Z₃-trivial axis —
and **resolves it honestly**: a Z₃-equivariant Yukawa built from the BCC orbit is
forced (by Schur) to have a **2-fold degenerate mass pair**, lies on the Koide
cone only at the special ratio `|v_t|/|v_o| = (1+√2)² = 3+2√2`, and therefore
**cannot reproduce the three distinct PDG lepton masses** — those require
Z₃-breaking input the module does not supply. I verified every numerical claim.
Verdict **KOIDE CONSISTENT (PDG NOT IN LOCUS)** is accurate and appropriately
modest: the geometry *permits* Koide but does *not predict* the masses.

- **Verdict:** clean, fully-verified, honestly negative-leaning result — it kills
  its own coincidence. One of the more scientifically satisfying sidecars.
- **Confidence:** high on all verified facts; the "not accidental" framing of the
  axis coincidence is true but somewhat tautological (see Novelty).

## What it claims

STATUS / PROJECT_STATUS: *"KOIDE CONSISTENT … the BCC body-diagonal Z₃ structure
admits Koide-satisfying Yukawa solutions but does not uniquely predict the PDG
mass triple … the Koide cone direction (1,1,1)/√3 IS the BCC body-diagonal Z₃-
trivial axis (not accidental)."*

## Progress

| Phase | Verdict |
|---|---|
| KO-1 empirical Koide + cone geometry | `K_PDG = 0.666661`, dev 6×10⁻⁶ from 2/3; three forms agree |
| KO-2 BCC body-diagonal Z₃ on flavor | R fixes `(1,1,1)/√3`; trace + 2D irrep projectors |
| KO-3 Z₃-orbit 3×3 Yukawa locus | **always 2-fold degenerate**; cone IFF ratio = 3+2√2 |
| KO-4 cone vs locus | `L ∩ C` = 1-param family; `L ⊄ C`; **PDG ∉ L** |
| KO-5 combined | KOIDE CONSISTENT |

54 tests. I verified KO-1/KO-3 directly.

## Assumptions / inputs

1. **Koide formula** (empirical; Koide 1981) — real, `K = (Σm)/(Σ√m)² = 2/3`.
2. **BCC body-diagonal Z₃ rotation R** from `topology` (fixes `(1,1,1)/√3` — its
   trivial/Perron eigenvector).
3. **3×3 Yukawa-from-orbit template** from `broken_triality`.
4. **σ^a-flavor ↔ generation identification** — a declared mapping.
5. PDG lepton masses (cited).

## Soundness

Everything I checked is exact and correct.

- **Circulant Yukawa, always-degenerate triple:** generic `v=[2,1,0]` gives
  eigentriple `(9,3,3)` — verified. The structure
  `λ_1 = 3|v_t|²`, `λ_2 = λ_3 = (3/2)|v_o|²` is a direct consequence of Z₃
  equivariance: under `3 = 1 ⊕ 2`, Schur forces the 2D-irrep eigenvalues equal.
  (Same Schur mechanism as `boundary_response`'s S₃ no-go.) Sound.
- **Special ratio:** `r* = 3+2√2 = (1+√2)²` verified; on-cone mass ratio
  `m_1/m_2 = 2(3+2√2)² = 34+24√2 ≈ 67.94` verified.
- **`K_PDG = 0.6666605`** vs `2/3 = 0.6666667` — verified (dev ~6×10⁻⁶).
- **The honest conclusion is correct:** PDG masses are all distinct, the
  equivariant locus forces `m_2 = m_3`, so PDG ∉ locus; three distinct masses
  require Z₃-breaking. Cleanly and correctly stated.

**Notable internal echo (genuine):** the special ratio is exactly the silver
ratio squared `(1+√2)²` and the on-cone mass ratio is `2(1+√2)⁴ = 2/ε⁴` with
`ε = √2−1` — the *same* silver-ratio invariant that drives `boundary_response`'s
neutrino sector (`Δm²₂₁/Δm²₃₁ = ε⁴ = 17−12√2`). `koide` does **not** import
`boundary_response`; both trace independently to `√2` in BCC body-diagonal/face
geometry. This is either a real structural feature of the BCC lattice worth
chasing, or a coincidence of where `√2` enters — worth flagging for the
cross-cutting synthesis either way.

## Novelty

- **The coincidence is real but its framing is overstated.** That the Koide cone
  axis `(1,1,1)/√3` equals the BCC Z₃-trivial axis is *true*, but it is close to
  tautological: any 3-fold-symmetric structure singles out the symmetric `(1,1,1)`
  direction (it is the Perron eigenvector of the 3-cycle), and the Koide formula
  is *defined* relative to `(1,1,1)` in `√m`-space. So "not accidental" is
  correct but means "both distinguish the symmetric direction for the same
  3-fold-symmetry reason," not a deep numerical surprise.
- **No new Koide derivation.** The Koide formula has a large numerology
  literature; this module does not add a derivation — it tests whether the BCC
  structure *predicts* it and honestly finds "consistent, not predictive."
- **Genuine value:** the clean rep-theoretic result (Z₃-equivariant ⇒ forced
  degeneracy ⇒ cannot be PDG) and the discipline of killing one's own coincidence.

## Gaps

1. **Not predictive of PDG** — the headline limit. Z₃-equivariant always gives a
   degenerate pair; observed masses need Z₃-breaking, which is not provided.
2. **No Z₃-breaking mechanism** — Higgs-VEV alignment is suggested as a "Bold-B"
   follow-up but not implemented; without it the module cannot reach real masses.
3. **σ^a ↔ generation mapping is declared**, not derived.
4. **Inherits the unsolved generation problem** — like the other flavor sidecars,
   no actual three-generation origin.

**Highest-leverage next step:** the suggested Bold-B — a concrete Z₃-breaking
input (e.g. a Higgs-VEV alignment that tilts the Yukawa off the trace/traceless
eigenspaces) and a check whether the *broken* locus can pass through PDG while
staying near the cone. Without that, "consistent" is the ceiling. Secondarily,
the `√2`/silver-ratio echo with `boundary_response` is worth a deliberate
cross-module check — is there a shared BCC invariant, or is it coincidence?

## Confidence (calibrated)

- Eigenvalue structure / degeneracy / special ratio / `K_PDG`: **high** — verified.
- "Cone axis = BCC Z₃ axis": **high** it's true; but it's the generic 3-fold
  symmetric direction, so less remarkable than "striking coincidence" suggests.
- "KOIDE CONSISTENT but not predictive": **high** — correct and honest.
- As evidence the BCC structure *explains* Koide: **low** — it explicitly cannot
  reproduce PDG.

## Verdict

`koide` is among the more scientifically satisfying sidecars precisely because it
takes a seductive coincidence and refuses to oversell it. The mathematics is
exact and fully verified: the BCC-Z₃-equivariant Yukawa is a circulant matrix
whose `3 = 1 ⊕ 2` structure forces a degenerate mass pair, it touches the Koide
cone only at the silver-ratio point `(1+√2)²`, and it therefore cannot produce
the three distinct charged-lepton masses without Z₃-breaking. The "KOIDE
CONSISTENT (PDG NOT IN LOCUS)" verdict is exactly right and admirably modest. Two
caveats for the reader: the "cone axis = BCC axis, not accidental" claim is true
but near-tautological (both single out the symmetric direction any 3-fold
symmetry distinguishes), so it is structural rather than a deep numerical
coincidence; and there is a genuine, unexplained echo — the same silver ratio
`ε = √2−1` that governs `boundary_response`'s neutrino mass ratios reappears here
in the Koide special ratio and mass ratio, across modules that do not import each
other. Whether that is a real BCC invariant or a `√2` coincidence is the most
interesting thing the module surfaces, and it is left open.
