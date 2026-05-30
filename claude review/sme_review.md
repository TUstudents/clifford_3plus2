# sme — review

_Reviewer: Claude (Opus 4.8). Independent assessment. See `REVIEW_PLAN.md` for rubric._

## TL;DR

`sme` is a small (~1.2k lines), honest falsifiability audit that maps `cp`'s
T₂g CP-odd lattice term `H^(1)` into the Standard-Model-Extension dim-5 fermion
sector and bounds the lattice scale: **ε ≲ 2×10⁻³³ m (~10² Planck lengths) →
UNFALSIFIABLE PASS**. The dimensional analysis and arithmetic are sound and the
code is candid about its own weaknesses. But the headline rests on (i) an
explicitly **unverified** representative experimental bound and (ii) a lattice→SME
matching coefficient **set to exactly 1** without derivation — and crucially the
verdict sits only ~1–2 orders from the **SUB-PLANCK KILL** boundary, so verifying
the real Kostelecký–Russell bound could flip this from a *pass* into a *kill*.

- **Verdict:** correct, honest, order-of-magnitude consistency check. The robust
  conclusion is the weaker one: the program's lattice CP violation is *neither
  observable nor currently excluded*; the precise pass/kill placement is
  unverified.
- **Confidence:** high on "not observable / not killed"; low on the specific
  number and on the pass-vs-kill class.

## What it claims

PROJECT_STATUS / STATUS: *"UNFALSIFIABLE PASS at ε ≲ 2×10⁻³³ m (~10² ×
Planck length), ~10⁸ above current observational reach."* The verdict name is
itself honest — the program survives but predicts nothing observable in this
channel.

## Progress

| Phase | Done | Result |
|---|---|---|
| A-1 framework ID | ✔ | dim-5 non-minimal SME, fermion sector, CP-odd spin-tensor |
| A-2 H^(1) → SME tensor | ✔ | 3 non-zero `d^(5)` components (axial-vector × 2 derivatives, CPT-even) |
| A-3 literature bound | ✔ | representative `|d^(5)| ≲ 10⁻¹⁷ GeV⁻¹` (**KR entry-id verification pending**) |
| A-4 ε constraint | ✔ | `ε ≲ 2×10⁻³³ m` → UNFALSIFIABLE PASS |
| A-5 combined audit | ✔ | alive but unfalsifiable in this channel |

37 tests. I ran the constraint payload and confirmed the numbers.

## Assumptions / inputs

1. **`cp`'s H^(1)** (CP-odd, T₂g) — verified upstream; the 3 mapped components
   `(σ^x: k_yk_z, +1), (σ^y: k_xk_z, −1), (σ^z: k_xk_y, +1)` faithfully match
   `cp`'s result.
2. **SME identification** — H^(1) ↔ dim-5 fermion-sector `d^(5)` (Kostelecký–
   Mewes 2013 basis). Reasonable; the operator structure (CPT-even, CP-odd,
   axial-vector × 2 derivatives) is internally consistent.
3. **The load-bearing input — `|d^(5)| ≲ 10⁻¹⁷ GeV⁻¹` — is REPRESENTATIVE and
   UNVERIFIED.** The code flags this explicitly: `KR_ENTRY_IDS_VERIFIED = False`,
   `bound_is_representative = True`.
4. **Matching coefficient set to 1** — the identification is `d^(5) = ε × (1
   unit)`; the O(1) lattice-to-SME-operator normalization is taken as unity with
   no derivation (the H^(1) tensor entries are ±1, but the *relative* normalization
   to the SME operator basis is an unaddressed matching problem).
5. **Author-defined verdict thresholds** — SUB-PLANCK KILL (≤ℓ_P) / PLANCK-
   CONSISTENT (≤10ℓ_P) / UNFALSIFIABLE PASS (≤10⁻²⁵ m) / OBSERVABLE POSITIVE.

## Soundness

- **Dimensional analysis correct.** In natural units ε is a length = GeV⁻¹, and
  a dim-5 SME coefficient also has units GeV⁻¹, so `d^(5) ~ ε` is dimensionally
  consistent. The Hamiltonian bookkeeping (`ε H^(1) ~ ε k²` must have mass
  dimension 1 ⇒ [ε] = −1) checks out.
- **Arithmetic verified.** `10⁻¹⁷ GeV⁻¹ × 1.97327×10⁻¹⁶ m/GeV⁻¹ = 1.97×10⁻³³ m`;
  `log₁₀(ε/ℓ_P) ≈ 2.09` ⇒ ~123 ℓ_P ≈ "10² ℓ_P". ✔
- **Sector choice is conservative and correct.** The note records that the
  photon-sector dim-5 bound (~10⁻²⁹ GeV⁻¹ from GRB dispersion) is far tighter but
  *does not apply* to a fermion-sector `d^(5)`; using the looser fermion bound is
  the right call.
- **The code is honest** — representative/unverified flags surface in the payload,
  and the verdict explanation states the placement is conditional on KR
  verification.

**The soundness caveat is that this is an order-of-magnitude estimate, and at
least three uncertainties stack:** (a) the unverified literature bound (could be
1–2 orders tighter); (b) the matching coefficient set to 1 (no derivation of the
lattice→SME normalization — the quietest of the three); (c) BCC-lattice-axis vs
Earth/sidereal-frame alignment, an explicit O(1). Together these put ≳2 orders of
slack on the number.

**The structurally important point (under-stated in the headline):** the verdict
is *near a class boundary*. UNFALSIFIABLE PASS runs 10⁻³⁴–10⁻²⁵ m; the current
2×10⁻³³ m is only ~10× above 10ℓ_P. So a verified bound ~10× tighter → PLANCK-
CONSISTENT, and ~100× tighter → **SUB-PLANCK KILL**. The same audit, with a
verified (rather than representative) bound, could therefore turn from a pass into
a kill. It *cannot* move the other way to OBSERVABLE POSITIVE (that needs the
bound ~10⁸ looser). The note acknowledges "1–2 orders tighter would push ε into
PLANCK-CONSISTENT or below" — "or below" being the kill.

## Novelty

- **Not new physics.** It applies the established SME framework (Kostelecký et
  al.) to bound a specific lattice-anisotropy coefficient. The machinery is
  standard; the H^(1) → `d^(5)` identification is reasonable but routine.
- **Genuine value:** it is the project's *honest external-consistency check* —
  "is the BCC walk's lattice CP violation already excluded by Lorentz-violation
  experiments?" Answer: no, the scale is far below reach. Naming the result
  "UNFALSIFIABLE PASS" (a survival that predicts nothing) is admirably
  non-triumphal.

## Gaps

1. **The load-bearing experimental bound is unverified** — the single biggest
   gap, and the verdict *class* depends on it (pass ↔ kill).
2. **Lattice→SME matching coefficient = 1 by fiat** — no derivation; an O(1)–O(10)
   error here moves the result.
3. **BCC-axis / sidereal-frame alignment** — unhandled O(1).
4. **No flavor/3-generation context** — no link to physical CKM CP violation.
5. **Verdict thresholds are author-chosen** — reasonable but not canonical.

**Highest-leverage next step:** the obvious one the module itself tracks — verify
the actual Kostelecký–Russell `d^(5)` entry IDs (and the 2024–2025 atom-
interferometry updates). This is cheap and decides whether the channel is a
pass, a Planck-consistent pass, or a kill. Secondarily, derive (don't assume) the
lattice→SME matching coefficient.

## Confidence (calibrated)

- "Lattice CP violation is not currently observable and not currently excluded":
  **high** — robust to ≳2 orders of slack (observable needs ~10⁸).
- The specific `ε ≲ 2×10⁻³³ m`: **low–medium** — unverified bound + matching=1.
- The pass-vs-kill *class*: **low** — sits ~1–2 orders from the kill boundary;
  genuinely contingent on KR verification.

## Verdict

`sme` does the right, modest thing: it takes `cp`'s clean structural CP result
and asks whether experiment already rules it out, using the standard SME
framework, with sound dimensional analysis and verified arithmetic, and reports a
deliberately humble "UNFALSIFIABLE PASS." The implementation is honest — it flags
its representative bound as unverified and its verdict as conditional. The two
things a reader must keep in view are that (1) the number rests on an unverified
literature bound and an undisclosed-derivation matching factor of 1, and (2) the
result sits close enough to the Planck/sub-Planck boundary that *verifying* the
real bound could convert this pass into a kill. The durable takeaway is the
weaker, robust one: in this channel the program is neither observable nor
excluded — and the honest follow-up (verify the Kostelecký–Russell entries) is
both cheap and potentially decisive.
