# Claude Review — Plan & Methodology

_Working folder for independent reviews of the `clifford-3plus2-d5` workspace.
These are temporary notes, not part of the project deliverables._

Reviewer: Claude (Opus 4.8). Started 2026-05-28.

---

## 0. Purpose

The workspace is a QCA-derivation lab for the Standard Model, organized as
self-contained "kill-disciplined" sidecars. The goal of this review is an
**independent, adversarial-but-fair assessment** of each module along five
axes the user asked for:

1. **Progress** — what is actually done vs. claimed.
2. **Assumptions** — what is declared/input vs. derived; what is load-bearing.
3. **Soundness** — is the math/physics correct; do proofs match their claims.
4. **Novelty** — what is genuinely new vs. textbook/rediscovery.
5. **Gaps** — what is open, and what is the highest-leverage next step.

Plus a sixth synthesis axis: **confidence** (calibrated, per-claim) and a
one-line **verdict**.

The governing principle is the workspace's own motto (`obstruction_r10/README`):
**"Do not fool yourself."** A representation identity is not a theorem about
QCA geometry; a chosen `n=3` is not a derived `n=3`; a boolean is only true if
code computes it or a proof derives it.

---

## 1. Per-module rubric (reusable template)

Each module review is a standalone markdown file `claude review/<module>_review.md`
with this structure:

```
# <module> — review

## TL;DR            (3-5 lines: what it is, verdict, confidence)
## What it claims   (the headline result, in the project's own words)
## Progress         (sessions/gates done; test count; pass/fail)
## Assumptions      (declared inputs vs derived; the load-bearing ones flagged)
## Soundness        (proof/code spot-checks; what I verified myself)
## Novelty          (new vs textbook vs rediscovery; prior-art pointers)
## Gaps             (open problems; honest scope limits; highest-leverage next step)
## Confidence       (per-sector calibration: high/med/low + why)
## Verdict          (one paragraph)
```

### Scoring discipline
- Separate **"correct"** from **"meaningful"**. A proof can be sound and still
  prove something narrow. State both.
- Separate **derivation** from **postdiction**. If discrete choices were made
  knowing the target, count the researcher degrees of freedom.
- Distinguish **structurally closed** / **finite-census closed** / **conjectural**
  / **conditional pass** / **declared input** — the project already uses these
  tags; honor them and check they're applied correctly.
- Always note **what would falsify** the claim.

---

## 2. Methodology (per module)

1. **Read the narrative docs first**: `README`, `STATUS`, `PLAN`/`roadmap`,
   `theory`, `falsifiers`, session notes. Extract the headline claim verbatim.
2. **Read the core code**, not just docs — the load-bearing
   functions/payloads, and the controls (negative tests) that are supposed to
   reject alternatives.
3. **Independently verify the key claims**: recompute the central identity
   symbolically/numerically; check numbers against external data (PDG/NuFIT,
   literature bounds) where the module makes empirical contact.
4. **Run the tests** (`uv run pytest <module>/tests -q`) and note pass count.
5. **Cross-check scope**: does the proof's hypothesis match the claimed
   conclusion's breadth? Is anything smuggled in via a "declared input"?
6. **Locate in prior art**: is the framing/result novel, or a known
   construction? (SO(10) GUT branching, discrete flavor symmetry, Caldeira–
   Leggett/Feshbach, QCA reconstruction literature, etc.)

---

## 3. Module ordering

Review trunk-first, then dependents, then closed sidecars (cheapest to
re-verify last). Dependency-aware so shared claims are assessed once.

| Order | Module | Role | Status (project's own) |
|---|---|---|---|
| 1 | `obstruction_r10` | **main QCA trunk** | frozen for publication |
| 2 | `lepton` | SM gauge content from Cl(0,10) | positive, active |
| 3 | `spacetime_qca` | 3+1D BCC simulation arena | in progress |
| 4 | `cp` | CP/T audit, shared import surface | dual-positive |
| 5 | `boundary_response` | flavor boundary-response sidecar | V1–V25 (✅ reviewed separately) |
| 6 | `sme` | SME bound on ε | unfalsifiable pass |
| 7 | `strongcp` | strong-CP triviality | trivial/safe |
| 8 | `koide` | Koide ↔ BCC body diagonal | consistent |
| 9 | `triality` | 3 generations via Spin(8) triality | K1 FAIL |
| 10 | `broken_triality` | flavor from broken triality | BT-2 FAIL |
| 11 | `exceptional` | 3 gen via exceptional algebras | FULL KILL |
| 12 | `topology` | 3 gen via topology | all KILL |
| 13 | `sim` | shared JAX infra | infrastructure |

`boundary_response` has already been reviewed interactively (neutrino-core
`ε⁴` theorem sound; PMNS/CKM honest postdictions; ergodicity closure V19–V23
is a reframing of the central assumption). A written version can be back-filled.

---

## 4. Cross-cutting questions (whole workspace)

- **The honest-superposition claim**: the README says no single module derives
  the SM; the workspace is "the honest superposition" of partial answers. Is
  that framing actually honest, or does it let weak links hide behind strong
  ones?
- **Generation count (N=3)**: triality / broken_triality / exceptional /
  topology all KILL the algebraic/topological routes. Is the cumulative
  negative result genuinely conclusive, or are there unexamined routes?
- **The two pillars**: `obstruction_r10` (what *can't* be derived) and `lepton`
  (what *is* constructed from declared choices). Do they fit together without
  double-counting assumptions?
- **Methodology as the real product**: the kill-disciplined sidecar pattern
  (one gate, dataclass payload, controls + verdict) — is it being applied
  rigorously, or does "PASS" sometimes mean "the construction I built does what
  I built it to do"?

---

## 5. Coverage tracker

| Module | Reviewed | File | Verdict (mine) |
|---|---|---|---|
| obstruction_r10 | ✅ | `obstruction_r10_review.md` | sound finite-class no-go; honest |
| boundary_response | ✅ | `boundary_response_review.md` | sound neutrino-core theorem chain (now V31, ε derived from K₃); PMNS/CKM honest postdictions; reduced to 1 physical input |
| lepton | ✅ | `lepton_review.md` | correct exact construction; textbook physics; conditional on declared J |
| spacetime_qca | ✅ | `spacetime_qca_review.md` | sound, well-controlled simulation arena; apparatus not result; tiny-lattice |
| cp | ✅ | `cp_review.md` | clean exact CP result (T₂g O(ε)); self-honest; "dual" branding half-earned |
| sme | ✅ | `sme_review.md` | sound order-of-mag check; bound unverified; pass sits near kill boundary |
| strongcp | ✅ | `strongcp_review.md` | sound "lattice strong-CP-clean" to O(ε²); not an SM strong-CP solution; SC-4 trivially-zero sector |
| koide | ✅ | `koide_review.md` | verified; kills its own coincidence honestly; silver-ratio echo with boundary_response |
| triality | ✅ | `triality_review.md` | exemplary kill; genuine triality verified; K1 FAIL robust; no framing gap |
| broken_triality | ✅ | `broken_triality_review.md` | clean kill (BT-2 flat); BT-1 "PASS" charitable; same wall as triality/koide |
| exceptional | ✅ | `exceptional_review.md` | soundest-class FULL KILL; textbook E₆ branching; honest dim-vs-rep distinction |
| topology | ✅ | `topology_review.md` | 4-phase kill all correct; D-5 (anomaly) substantive, D-1/D-2 near-trivial; scope honestly bounded |
| sim | ✅ | `sim_review.md` | pure infrastructure; clean, well-bounded; no physics claims |
