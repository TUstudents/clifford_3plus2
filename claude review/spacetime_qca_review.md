# spacetime_qca — review

_Reviewer: Claude (Opus 4.8). Independent assessment. See `REVIEW_PLAN.md` for rubric._

## TL;DR

`spacetime_qca` is the program's **experimental apparatus**: a large
(~17k lines, 84 files, 43 sessions), carefully-engineered BCC-lattice
simulation arena that takes a genuine published QCA walk (Bialynicki-Birula
1994) and standard lattice-gauge machinery and assembles them into a coupled
fermion + gauge + Higgs prototype over `lepton`'s chiral-16 PS/SM carrier, with
exact-arithmetic reference backends and JAX numerical parity. **The kernels I
verified are correct and the engineering discipline (oracle paths, gauge-
covariance and reversibility controls, parity tests) is high.** But its outputs
are *infrastructure, not physics conclusions*: every dynamical layer is an
explicit prototype/control, lattices are tiny, the constraint structure is
diagnostic-only, and it inherits `lepton`'s one-generation declared carrier.

- **Verdict:** a solid, honest simulation arena — the "apparatus" half of the
  project. It is ready to host experiments but has not yet produced a physics
  result that constrains the SM. Much of the late session count is performance
  engineering, not physics advance.
- **Confidence:** high that the implemented kernels are correct; medium-high
  that it's a usable arena (tiny-lattice only); low as standalone evidence for
  QCA→SM.

## What it claims

STATUS: *"builds the 3D spatial side of the QCA: a BCC Weyl walk
(Bialynicki-Birula 1994) and its chiral assembly into a 4D Dirac carrier, with
a constant-background tensor lift against `lepton`'s chiral-16 internal
carrier."* It does **not** claim derived dynamics, constraint solving, Lorentz
invariance, or masses — each session report ends with an explicit "this is not
yet…" disclaimer.

## Progress

Sessions 20–62. The physics scaffold is concentrated in 20–42; 43–62 is
predominantly stability/profiling/Gauss-diagnostic tooling.

| Layer | Sessions | What |
|---|---|---|
| BCC Weyl/Dirac kinematics | 20–22 | BB 8-hop walk → `σ·k` / `α·k`; finite real-space step |
| mass / Yukawa (rep-level) | 21, 23, 25, 38 | scalar mass slot; static Higgs-doublet charge structure |
| gauge covariance / plaquettes | 24, 24b | position-dependent links, BCC plaquette holonomy |
| JAX backend | 26 | numerical step matching exact SymPy |
| Wilson observables/action/force | 27–35 | SO(2)/SU(2)/SU(3)/SU(4)/PS/SM compact force + leapfrog |
| fermion/gauge coupling | 36–37 | no-backreaction wrapper; Gauss residual + source kick |
| dynamical Higgs/Yukawa | 39–40, 45, 57 | site-local Higgs field; exact local unitary Yukawa |
| anomaly / Lorentz audits | 41–42 | one-gen anomaly cancellation; O(ε⁴) dispersion anisotropy |
| simulator + profiling | 43–62 | tiny runner, scan-backed simulator, force/Yukawa perf, Gauss descent |

Tests: ~299 full / 162 fast (module-reported). I verified the load-bearing
kernels directly (below) rather than re-running the multi-minute full suite.

## Assumptions / inputs

1. **BB 1994 walk convention** — pinned and faithfully implemented (`q_± =
   (1±i)/4`, the four rank-one projectors P₁–P₄). Real, citable QCA result.
2. **`lepton`'s chiral-16 carrier** (real `R^32` form) — so this module
   *inherits all of `lepton`'s declared choices*, including the right-
   quaternionic `J` that `obstruction_r10` shows is not QCA-forced. The arena
   sits atop the same un-derived carrier.
3. **`lepton`'s Session-19b SM charge table** — reused directly by the anomaly
   audit.
4. **Trusted, not proven:** all-momentum BCC Bloch-symbol unitarity and the
   fundamental-zone no-doubling property are *sample-checked* and otherwise
   inherited from the BB source convention (explicitly open in STATUS).

## Soundness

Everything I checked is correct, and the control discipline is the module's
strongest feature.

- **BB walk → continuum:** I confirmed the right Weyl block's first-order
  effective Hamiltonian equals `σ·k` exactly, and the chiral Dirac assembly is
  the expected `4×4` `α·k`. ✔
- **Anomaly audit (Session 41):** I ran it — exact one-generation SM
  cancellation: `Tr Y = 0`, `Tr Y³ = 0`, `SU(3)²-U(1) = 0`, `SU(2)²-U(1) = 0`,
  `SU(3)³ = 0`, even SU(2) doublet count (4), and matrix-trace diagnostics
  match the field-table sums. ✔ **Caveat:** this is a *representation-level*
  check on the reused charge table, **not** a lattice-regulator anomaly theorem
  (the module says exactly this).
- **Lorentz recovery (Session 42):** a genuine, correct finite-spacing
  dispersion result — the right/left Weyl cubic anisotropies cancel under
  helicity pairing, pushing the first BCC Dirac trace-cosine residual to
  `O(ε⁴)` (directional coefficients `q⁴/24` face, `q⁴/18` body), versus the
  naive hypercube control's `O(ε²)`. This is a real (if narrow) physics
  observation, honestly labeled "free-dispersion audit, not interacting Lorentz
  proof."
- **Lattice-gauge machinery:** standard and correctly built — Wilson
  action `1 − Re Tr(H)/N`, compact left-trivialized force with pure-gauge
  flatness and gauge-covariance controls, reversible leapfrog (forward/back
  recovery), and an analytic-staple force validated against finite-difference
  oracles. JAX backends are tested for parity against exact SymPy.

No soundness problems found. Negative controls are real (e.g. non-scalar mass
breaks an SM generator; color-only sector rejected for Higgs coupling).

## Novelty

- **Not novel (correctly attributed):** the BB BCC Weyl walk (1994); the entire
  lattice-gauge-theory toolkit — Wilson action/plaquettes (Wilson 1974), compact
  link dynamics, leapfrog/HMC-style updates (Creutz et al.) — is textbook.
- **Genuine contribution:** the *assembly* — a concrete, exact-arithmetic-backed,
  JAX-parallel coupled fermion+gauge+Higgs simulation arena over the *specific*
  chiral-16 Pati-Salam/SM carrier, with a clean prototype-vs-production split.
  This is careful computational-physics engineering and a reusable apparatus,
  not a new physics result.
- The `O(ε⁴)` helicity-paired dispersion isotropy is a modestly interesting
  specific lattice-fermion observation for this walk.

## Gaps

Large and (to the module's credit) explicitly stated:

1. **No exact Gauss-law projection** — only a diagnostic residual and bounded
   descent (Sessions 60–62). The theory is not constrained.
2. **No genuine backreaction** — Higgs/fermion currents are finite-difference
   and **off by default**. The coupled step is a "research control."
3. **Tiny lattices only** — `(1,1,1)`, `(2,1,1)`. No production-scale runs, no
   long-time stability results, no continuum/scaling extraction.
4. **No 3+1D Lorentz boost recovery** beyond the free-dispersion anisotropy
   audit; no interacting Lorentz invariance.
5. **All-momentum BCC unitarity + no-doubling unproven** (trusted/sample-checked).
6. **No masses / Yukawa hierarchy / generations** — inherits `lepton`'s
   one-generation limit and rep-level Higgs/Yukawa (low-rank background probes,
   not realistic mass matrices).
7. **Inherited carrier dependence** — the whole arena rests on `lepton`'s
   declared `J`/factorization, which `obstruction_r10` shows QCA rules don't
   force.

**Honest note on session count:** roughly half the sessions (49–58 especially)
are performance/profiling work (force chunking, analytic staple, Yukawa fast
path, cold/warm JAX timing). Legitimate and well-documented, but it should not
be read as physics progress — the physics scaffold is essentially complete by
Session 42, and the apparatus has not since produced a new physics result.

**Highest-leverage next step:** an **exact Gauss-law constraint solve** (not
descent) plus a genuine **always-on backreaction** at a lattice large enough to
measure something — until the dynamics is constrained and self-consistent, the
arena can run but cannot test a physical claim. Equivalently, pick *one*
falsifiable target (e.g. confinement onset, or a dispersion/Lorentz-restoration
measurement at increasing lattice size) and drive the apparatus to it.

## Confidence (calibrated)

- Implemented-kernel correctness (walk, Dirac, Wilson, force, anomaly): **high**
  — verified key ones; strong control/oracle discipline.
- Usable simulation arena: **medium-high** — exists and is tested, but tiny-
  lattice only; production readiness unproven.
- Evidence for QCA → SM physics: **low** — it's apparatus, not result.

## Verdict

`spacetime_qca` is exactly what the project needs as its lab bench, and it is
built with real care: a faithful implementation of a published QCA walk, a
correct chiral Dirac assembly, a full and properly-controlled lattice-gauge
toolkit, and a coupled fermion/gauge/Higgs prototype, all with exact reference
backends and JAX parity. The mathematics and code I checked are sound and the
discipline around oracles, covariance, and reversibility is genuinely good.
What it is *not* — and never claims to be — is a source of physics conclusions:
the dynamics is unconstrained (Gauss is diagnostic-only), backreaction is
off-by-default, the lattices are minimal, Lorentz/mass/generation questions are
untouched, and the entire arena inherits the declared chiral-16 carrier that
the trunk shows QCA does not force. Read as "a correct, reusable BCC simulation
apparatus for the chiral-16 SM carrier," it succeeds. Read as "a 3+1D QCA
derivation of Standard-Model dynamics," it is, by its own honest account, not
there yet — it is the instrument, awaiting the experiment.
