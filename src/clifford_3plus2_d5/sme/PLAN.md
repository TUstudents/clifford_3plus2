# Plan: Bold A sidecar — SME experimental-bound verdict on ε

## Context

The cp/ sidecar produced a DUAL ROBUST PASS: H^(1), the O(ε) BCC walk
correction to the free-Dirac Hamiltonian, is 100% CP-odd and lives
entirely in the T_{2g} cubic-harmonic irrep — with explicit symbolic
matrix entries that are bilinear in spatial momentum (k_x k_y, k_y k_z,
k_z k_x).  This is a structural positive: the program predicts a small,
cubic-anisotropic, CP-violating correction to free Dirac dynamics at
scale ε.

Bold A asks: now that we have an explicit prediction, what does the
data say?  The Standard Model Extension (SME) framework provides
experimental bounds (atomic-clock, GRB photon-dispersion, accelerator,
and atom-interferometry data) on Lorentz/CPT-violating coefficients.
Because H^(1) is bilinear in momentum — k_i k_j rather than k_i — the
natural target is the **dim-5 non-minimal SME** (Kostelecky-Mewes,
arXiv:1102.4068), not the more familiar minimal SME.

**The single load-bearing question**:

> Given H^(1)'s explicit T_{2g} CP-odd structure at O(ε), what is the
> maximum allowed ε from current experimental bounds on the
> corresponding dim-5 non-minimal SME coefficients, and which of three
> pre-named scale verdicts applies?

Three pre-named outcomes (from the original gap analysis):

1. **Planck-consistent**: ε ≲ Planck length (~10⁻³⁵ m).  Program is
   observation-consistent but unfalsifiable at current reach.  Honest
   PASS.
2. **Sub-Planck**: ε must be pushed below Planck length to satisfy
   bounds.  Structurally inconsistent — clean KILL.
3. **Observable**: ε allowed at scales ≥ 10⁻²⁵ m (or any
   experimentally accessible scale).  Program predicts near-future
   measurable effects.  Publishable POSITIVE.

## User-confirmed scope decisions

- **Sidecar name**: `sme/`.
- **SME sector**: dim-5 non-minimal SME (Kostelecky-Mewes).  Honest fit
  to H^(1)'s bilinear-momentum structure.
- **ε bound style**: symbolic ratio with cited numerical bound — matches
  cp/'s all-symbolic style and keeps the audit reproducible.
- **Experimental-bound deliverable**: markdown literature note
  (`SME_LITERATURE_NOTE.md`), same pattern as topology/`s
  `PI3_LITERATURE_NOTE.md`.  No tests on the note itself.
- **Budget**: ~2 weeks, including cross-checks against Mattingly /
  Liberati reviews and any 2024-2025 atom-interferometry bounds.

## Existing infrastructure to reuse (via `sme/reuse.py`)

Verified from cp/ exploration:

**From cp/continuum_cp.py**:
- `symbolic_momentum()` → (ε, kx, ky, kz) as SymPy symbols (ε
  positive; k real).
- `effective_hamiltonian_first_correction()` → H^(1), the 128×128
  Hermitian first-order correction.  This is the load-bearing object.
- `cp_irrep_decomposition()` → dict[(parity, irrep), Matrix] of 2×3
  cells.  We will use the (CP-odd, T_{2g}) cell.
- `cp_irrep_norm_table()` → already-verified norms (||(CP-odd, T_{2g})||²
  = 12, all other cells = 0).

**From cp/cubic_harmonics.py**:
- `projector_T2g()` → 6×6 rational matrix.
- `decompose_matrix_of_polynomials()` for elementwise irrep
  decomposition.
- T_{2g} basis: (k_y k_z, k_z k_x, k_x k_y).

**Explicit H^(1) form** (from `cp/SESSION_CP_ORDER_EPS2.md`):

```text
H^(1)_R(k) = ( k_x k_y          k_z (k_x + i k_y) )
             ( k_z (k_x - i k_y)   -k_x k_y       )

H^(1)_L(k) = same form (block-diagonal in chirality)
```

This is the input.  All Bold A work consumes this and adds NO new
lattice-level computation.

## Decision tree

```text
Phase A-1 (SME framework identification, ~2 days)
  ├── Pin down which sector of non-minimal SME applies to a free Dirac
  │   Hamiltonian correction bilinear in momentum.
  ├── Specifically Kostelecky-Mewes dim-5 fermion-sector coefficients
  │   m^(5), a^(5), b^(5), c^(5), d^(5), e^(5), f^(5), g^(5), H^(5).
  └── Determine the symmetry class of H^(1): CP-odd, T_{2g} rotation-
      anisotropic, chirality-preserving, momentum-bilinear.
        │
        ▼
Phase A-2 (Map H^(1) → SME tensor components, ~3-4 days)
  ├── Decompose (CP-odd, T_{2g}) H^(1) cells into the dim-5 SME basis.
  ├── Identify the specific tensor-index combinations carrying the
  │   k_y k_z, k_z k_x, k_x k_y signature.
  ├── Confirm CP-odd → CPT-violating vs. CPT-preserving subclass.
  └── Output: symbolic mapping table.
        │
        ▼
Phase A-3 (Experimental bounds literature, ~3-4 days)
  ├── Primary source: Kostelecky-Russell "Data Tables for Lorentz and
  │   CPT Violation", arXiv:0801.0287 (current revision).
  ├── Cross-check #1: Mattingly "Modern Tests of Lorentz Invariance",
  │   Living Reviews in Relativity 8 (2005) 5.
  ├── Cross-check #2: Liberati "Tests of Lorentz invariance: a 2013
  │   update", Class. Quantum Grav. 30 (2013) 133001.
  ├── Cross-check #3: scan 2024-2025 atom-interferometry / clock-
  │   comparison papers for tightened bounds on the relevant
  │   coefficients.
  └── Output: SME_LITERATURE_NOTE.md with tightest cited bound per
      identified component.
        │
        ▼
Phase A-4 (Symbolic ε constraint, ~2 days)
  ├── Compute ε ≤ (SME bound) / (H^(1) coefficient magnitude).
  ├── Express the bound in length / inverse-energy units.
  ├── Classify into the three pre-named outcomes (Planck-consistent /
  │   sub-Planck / observable).
  └── Output: epsilon_constraint.py with a single symbolic-bound +
      verdict-class function.
        │
        ▼
Phase A-5 (Combined audit + SESSION report, ~2 days)
  ├── sme_audit.py aggregates A-1..A-4 into a single payload.
  ├── SESSION_SME.md final verdict report.
  └── Update STATUS.md and parameter_ledger.md.
```

## Phase A-1 — SME framework identification (~2 days)

### FA-1: SME sector

The chiral-16 BCC Bialynicki-Birula walk is a free Dirac walk in its
continuum limit; H^(1) is the leading O(ε) lattice correction.  The
correction modifies the dispersion relation by an extra k·k term —
this is the defining signature of dim-5 fermion-sector SME.

Following Kostelecky-Mewes (arXiv:1102.4068), the dim-5 fermion
Lagrangian extension includes coefficients:

- `m^(5)_αβγ`     (scalar bilinear, CP/T properties tabulated)
- `a^(5)_αβγ`     (vector bilinear)
- `b^(5)_αβγ`     (axial-vector bilinear)
- `c^(5)_αβγ`     (symmetric tensor)
- `d^(5)_αβγ`     (antisymmetric tensor)
- `e^(5)_αβγ`     (extra spin)
- `f^(5)_αβγ`     (extra spin)
- `g^(5)_αβγ`     (extra spin-tensor)
- `H^(5)_αβγ`     (extra magnetic-moment-like)

Each carries three spacetime indices for dim-5 (compared to two for
dim-4 minimal SME).  The CP and T transformation properties of each
coefficient are tabulated in Table I of Kostelecky-Mewes 2011.

### FA-2: Symmetry-class identification

H^(1)'s symmetry classes:

- **Hermitian** (already confirmed by `effective_hamiltonian_first_correction()`).
- **Chirality-preserving** (block-diagonal in chiral basis).
- **CP-odd** (verified by `cp_irrep_decomposition()` — 100% in
  (CP-odd, T_{2g})).
- **T_{2g} cubic-anisotropic** (off-diagonal symmetric momentum products).
- **Momentum-bilinear** (degree 2 in k, hence dim-5 lattice
  contribution).

Output: a small data module `sme_framework_identification.py` exposing:

- `H1_symmetry_class()` → dataclass listing the five symmetry properties.
- `dim5_sme_sector_label()` → string identifier of the natural target
  sector (most likely a subset of `g^(5)` / `H^(5)`, modulo Phase A-2
  verification).

## Phase A-2 — Map H^(1) T_{2g} CP-odd cells to SME tensor components (~3-4 days)

### FA-3: dim-5 SME basis decomposition

The (CP-odd, T_{2g}) cell of H^(1) has the explicit form

```text
H^(1)_T2g(k) = k_y k_z · M_{yz} + k_z k_x · M_{zx} + k_x k_y · M_{xy}
```

where `M_{ij}` are 4×4 (per-chirality) constant Hermitian matrices
extracted from the cp/ cell.

We project each `M_{ij}` onto the standard set of γ-matrix bilinears

```text
{ 1, γ⁰, γⁱ, γ⁵, γ⁰γⁱ, γⁱγʲ, γ⁰γ⁵, γⁱγ⁵, γ⁰γⁱγ⁵ }
```

(in the chiral basis).  The dim-5 SME bilinear structure `ψ̄ Γ ψ ∂ ∂`
maps each γ-bilinear projection to a specific SME coefficient class.

### FA-4: Identify the CP-odd subclass

Per Kostelecky-Mewes Table I, of the dim-5 coefficients:

- Some are CPT-even and CP-even (e.g., `c^(5)`).
- Some are CPT-even and CP-odd.
- Some are CPT-odd and CP-odd.
- Some are CPT-odd and CP-even.

H^(1) being CP-odd narrows the target to two of these.  Combined with
the T_{2g} rotation class, we expect only 1-3 specific tensor-index
combinations to receive non-zero coefficients.

Output: `sme_tensor_mapping.py` exposing:

- `h1_t2g_cp_odd_cell()` → SymPy Matrix (already computable from cp/).
- `dim5_sme_components_with_nonzero_coefficient()` → tuple of (tensor,
  index_pattern, symbolic_coefficient) triples.
- `mapping_audit_payload()` → dataclass summarising the identification.

## Phase A-3 — Pull experimental bounds (~3-4 days)

Deliverable: `SME_LITERATURE_NOTE.md`, same pattern as topology/'s
`PI3_LITERATURE_NOTE.md`.  Structure:

```markdown
# Phase A-3 — SME experimental bounds note

## Source references

| Source                       | Use                                |
| ----                         | ----                               |
| Kostelecky, Russell.  Data Tables for Lorentz and CPT Violation, arXiv:0801.0287 (current revision). | Comprehensive tabulated bounds.   |
| Kostelecky, Mewes.  Lorentz and CPT violation in the Standard Model with a Higher-Derivative Sector, arXiv:1102.4068. | Dim-5 SME definitions.            |
| Mattingly.  Modern Tests of Lorentz Invariance, Living Reviews in Relativity 8 (2005) 5. | Cross-check.                      |
| Liberati.  Tests of Lorentz invariance: a 2013 update, Class. Quantum Grav. 30 (2013) 133001. | Cross-check.                      |
| 2024-2025 atom-interferometry / clock papers (named individually).  | Tighter recent bounds if any.     |

## Bounds for identified components

| SME component          | Tightest bound      | Source                            |
| ----                   | ----                | ----                              |
| (component from FA-4)  | (cited bound)       | (Kostelecky-Russell entry id)     |
| ...                    | ...                 | ...                               |

## Verdict on the bound

(One paragraph: tightest applicable bound; uncertainty; whether
multiple components contribute and we take the tightest face.)
```

No code, no tests on the note itself.

## Phase A-4 — Symbolic ε constraint (~2 days)

### FA-5: Bound expression

For each identified SME component `c_i` with bound `b_i`:

```text
ε · |H^(1) coefficient for c_i| ≤ b_i
=> ε ≤ b_i / |H^(1) coefficient for c_i|
```

Taking the tightest face:

```text
ε_max := min_i { b_i / |H^(1) coefficient for c_i| }
```

The H^(1) coefficient is a SymPy rational (e.g., 1, ±1/2, etc. — to be
extracted from the (CP-odd, T_{2g}) cell).  `b_i` is a numerical
constant cited from Phase A-3 in length or inverse-energy units.

### FA-6: Scale-verdict classification

Compare `ε_max` to:

- Planck length `ℓ_P ≈ 1.6 × 10⁻³⁵ m`.
- Currently observable scale (atom-interferometry, optical-clock,
  GRB-time-of-flight, accelerator): typically `10⁻²⁰` to `10⁻³⁰ m`
  depending on the SME component.

Verdict classes:

| Case                          | Verdict             |
| ----                          | ----                |
| `ε_max ≤ ℓ_P`                 | SUB-PLANCK KILL    |
| `ε_max ~ ℓ_P` (within 10×)    | PLANCK-CONSISTENT  |
| `ℓ_P < ε_max ≤ 10⁻²⁵ m`       | UNFALSIFIABLE PASS |
| `ε_max > 10⁻²⁵ m`             | OBSERVABLE POSITIVE |

Output: `epsilon_constraint.py` exposing:

- `epsilon_bound_symbolic(component_index: int) -> sp.Expr` — single-
  component bound.
- `epsilon_bound_tightest_face() -> sp.Expr` — minimum over components.
- `scale_verdict(epsilon_bound: sp.Expr) -> str` — verdict class.
- `epsilon_constraint_payload() -> EpsilonConstraintPayload`.

### FA-7: Pass / kill / positive

| Outcome              | Action                                     |
| ----                 | ----                                       |
| SUB-PLANCK KILL      | Major negative.  Document.  Bold A closes the program at the ε-scale level. |
| PLANCK-CONSISTENT    | Honest PASS.  Document as unfalsifiable but alive. |
| UNFALSIFIABLE PASS   | Honest PASS, slightly looser than Planck. |
| OBSERVABLE POSITIVE  | Major positive.  Investigate specific experimental signatures (atomic clocks, GRB, etc.).  Open follow-up. |

## Phase A-5 — Combined audit + SESSION report (~2 days)

`sme_audit.py` aggregates A-1..A-4 into a single `SMEAuditPayload`
dataclass with:

- `framework_class` (from FA-2)
- `mapping_components` (from FA-4)
- `epsilon_bound_symbolic`, `epsilon_bound_numerical`
- `scale_verdict`
- `interpretation` (string)

`SESSION_SME.md` is the final 1-2 page verdict, structure analogous to
`SESSION_TOPOLOGY.md`:

```markdown
# Bold A — SME experimental-bound verdict on ε

**Status**: CLOSED — {verdict}

## Load-bearing question
{...}

## Phase-by-phase findings
{...}

## Combined verdict
ε_max = {value}; scale verdict = {SUB-PLANCK KILL / PLANCK-CONSISTENT / ... }

## What this does NOT close
{...}

## Test summary
uv run pytest src/clifford_3plus2_d5/sme/tests -q
```

## Pre-named failure modes

**F-sme-1**: H^(1) maps to dim-5 SME coefficients that have no
experimental bound yet.  Diagnosis: document the gap honestly; the
verdict becomes "UNCONSTRAINED — no current bound applies".  This is a
real possibility for some non-minimal SME components.

**F-sme-2**: Multiple SME components receive non-zero coefficients,
and the constraint becomes a polytope rather than a single bound.
Diagnosis: report the tightest face and note that the actual bound on
ε could be looser if the H^(1) prediction is a single direction in
component space.

**F-sme-3**: ε bound comes out at an awkward intermediate scale (e.g.,
10⁻³⁰ m, between Planck and current observability).  Diagnosis:
report honestly; this is still informative.

**F-sme-4**: Recent 2024-2025 bounds tighten existing tabulated bounds
by orders of magnitude.  Diagnosis: use the tighter bound and note
the specific citation.

**F-sme-5**: H^(1) is found to map to dim-5 SME components that are
*not* in the standard tabulated list (e.g., because they vanish
identically by SME field redefinitions).  Diagnosis: investigate
whether the cp/ T_{2g} signature is itself a redefinition-trivial
direction; if so, it predicts NO observable effect and the program
is unfalsifiable in this channel.

## Critical files

To create (under `src/clifford_3plus2_d5/sme/`):

```text
sme/
  __init__.py
  PLAN.md                              (this plan as reference)
  STATUS.md
  parameter_ledger.md
  reuse.py                             # imports from cp/, lepton, spacetime_qca
  sme_framework_identification.py      # Phase A-1: symmetry class + sector
  sme_tensor_mapping.py                # Phase A-2: H^(1) → SME components
  SME_LITERATURE_NOTE.md               # Phase A-3: bounds + citations
  epsilon_constraint.py                # Phase A-4: symbolic bound + verdict
  sme_audit.py                         # Phase A-5: combined payload
  SESSION_SME.md                       # final verdict
  tests/
    __init__.py
    test_sme_framework_identification.py
    test_sme_tensor_mapping.py
    test_epsilon_constraint.py
    test_sme_audit.py
```

To consult / read-only:

```text
src/clifford_3plus2_d5/cp/continuum_cp.py       # H^(1) source
src/clifford_3plus2_d5/cp/cubic_harmonics.py    # T_{2g} projector
src/clifford_3plus2_d5/cp/SESSION_CP_ORDER_EPS2.md # explicit H^(1) form
src/clifford_3plus2_d5/cp/parameter_ledger.md   # ε convention
```

## What this plan does NOT include (deferred)

- Higher-order continuum corrections (O(ε²), O(ε³)) — original
  brainstorm item 8.  Orthogonal to Bold A; could come next.
- Strong-CP / θ_QCD computation from BCC walk — original item 10.
  Orthogonal; would build on a similar SME mapping but for the gauge
  sector.
- Photon-sector (k_F)^(5) coefficients — Bold A focuses on the
  fermion sector because H^(1) is a fermion Hamiltonian correction.
  Photon-sector bounds would require an independent gauge-field
  analysis.
- Cosmic-ray / astrophysical bounds on Lorentz violation (GRB
  photon-dispersion is included in scope; ultra-high-energy cosmic-ray
  threshold analyses are deferred).
- Three-generation considerations.  Bold A treats one generation; the
  generation question is independently closed negative by triality /
  broken_triality / exceptional / topology.

## Verification

End-to-end test sequence after each phase:

```bash
# Phase A-1
uv run pytest src/clifford_3plus2_d5/sme/tests/test_sme_framework_identification.py -q

# Phase A-2
uv run pytest src/clifford_3plus2_d5/sme/tests/test_sme_tensor_mapping.py -q

# Phase A-3 (markdown note; no tests)
cat src/clifford_3plus2_d5/sme/SME_LITERATURE_NOTE.md

# Phase A-4
uv run pytest src/clifford_3plus2_d5/sme/tests/test_epsilon_constraint.py -q

# Phase A-5
uv run pytest src/clifford_3plus2_d5/sme/tests/test_sme_audit.py -q

# Full sidecar
uv run pytest src/clifford_3plus2_d5/sme/tests -q

# Regression
uv run pytest src/clifford_3plus2_d5/cp/tests -q
uv run pytest src/clifford_3plus2_d5/topology/tests -q
```

Verdict callable:

```bash
uv run --no-sync python -c "
from clifford_3plus2_d5.sme.sme_audit import sme_audit_payload
payload = sme_audit_payload()
print(payload.scale_verdict)
print(payload.epsilon_bound_symbolic, '=', payload.epsilon_bound_numerical)
print(payload.interpretation)
"
```

Expected outcome (a priori unknown — that is the point of the audit):
one of the four pre-named verdicts.  Honest engagement with experimental
data, irrespective of which class the program lands in.

## Effort budget

- Phase A-1: ~2 days.
- Phase A-2: ~3-4 days.
- Phase A-3: ~3-4 days (literature work, including 2024-2025 cross-check).
- Phase A-4: ~2 days.
- Phase A-5: ~2 days.

**Total committed budget**: ~2 weeks.

**Worst case**: ~2 weeks for a verdict in any class.  Bold A is
designed so that every outcome class is publishable — SUB-PLANCK KILL,
PLANCK-CONSISTENT, UNFALSIFIABLE PASS, and OBSERVABLE POSITIVE all
provide actionable information about the program's status.

The kill-discipline applies less directly here than for the algebraic /
topological sidecars (Bold A's "kill" outcome is one of several
honest outcomes, not a binary fail).  But the load-bearing question
returns a single scale verdict in all cases, which keeps the sidecar
tight.
