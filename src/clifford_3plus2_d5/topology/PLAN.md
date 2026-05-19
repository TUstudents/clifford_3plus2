# Plan: Bold D sidecar — three-generation topology audit

## User-confirmed scope decisions

- **Sidecar name**: ``topology/``.
- **Phase D-5 (anomaly forcing N=3)**: committed from start, not just
  conditional.  ~4-5 weeks total budget.
- **Phase D-3 (π_3 literature check)**: a documented markdown literature
  note, NOT a code module with tests.  Reduces overhead on bookkeeping.

## Context

The triality (K1 fail), broken_triality (BT-2 fail), and exceptional (all 4
candidates fail) sidecars have closed negative on algebraic mechanisms for
three SM generations.  Bold D pivots to topological mechanisms.

The general "topology gives three" claim is too vague to test directly.
The plan decomposes it into concrete candidates ordered cheapest-first.
The expected outcome is another clean negative — but the candidates are
sharply distinct from the algebraic ones, so the negative would be
meaningful new evidence.

The single load-bearing question:

> Does any natural topological or discrete-subgroup mechanism on the
> existing BCC × chiral-16 carrier produce three SM-generation copies
> without inventing additional structure?

Worst case: ~1 week for a clean negative on the two cheap candidates.
Best case: ~3-4 weeks if Phase D-5 (anomaly forcing N=3) is pursued and
reveals something.

## Existing infrastructure to reuse (via `topology/reuse.py`)

Verified by exploration:

**BCC geometry & Bialynicki-Birula walk** (`spacetime_qca/`):
- `bcc_weyl.bialynicki_birula_directions()`: 8 BCC body-diagonal tuples
  in `product((1, -1), repeat=3)` order.
- `bcc_weyl.bialynicki_birula_hops()`: 8 hop matrices (2×2 complex).
- `bcc_weyl.opposite_helicity_hops()`: parity-flipped hops.
- `dirac.gamma_matrices()`: chiral-basis γ⁰..γ³ (4×4 SymPy).
- `dirac.alpha_matrices()`: α^i = γ⁰γ^i.

**Color SU(3) on chiral-16** (`lepton/patisalam_sm.py`):
- `su3_c_generators_from_su4()`: 8 generators on chiral-16 (32×32 skew).
- `b_minus_l_generator_from_su4()`: B-L generator.

**Existing Z_3 / cubic structure** (`cp/cubic_harmonics.py`,
`triality/spin8_triality.py`):
- `cubic_harmonics.projector_A1g()`, etc.: O_h irrep projectors (degree-2
  polynomials).  Useful for context but doesn't give the 3×3 spatial
  rotation matrix itself.
- `triality.triality_cartan_matrix()`: an order-3 4×4 orthogonal matrix
  (for Spin(8) Cartan, NOT spatial).  Demonstrates the pattern of
  building Z_3 actions symbolically.

**What must be built new**:
- 3×3 cyclic rotation matrix for ``(x, y, z) → (y, z, x)`` (the body-diagonal Z_3).
- Induced permutation action on the 8 BCC directions.
- Spinor lift of the Z_3 to the Dirac 4-spinor.
- Audit comparing W_{σh} to ``U_3 W_h U_3^{-1}``.
- Color Z_3-center action on chiral-16 (= diag(1, ω, ω²) on color triplets, 32×32 unitary).
- chiral-16 decomposition under color Z_3 center.

## Decision tree

```text
Phase D-1 (BCC body-diagonal Z_3 on existing carrier, ~2 days)
  ├── Verify 8 BCC directions cycle under (x,y,z) → (y,z,x).
  ├── Verify the BB hops are Z_3-equivariant under the spin-1/2 lift.
  └── Show chiral-16 internal is Z_3-TRIVIAL by construction.
        │
        ▼  [expected: trivial fail on three generations]
Phase D-2 (color Z_3 center identification, ~1-2 days)
  ├── Identify spatial Z_3 with color Z_3 center: g = diag(1, ω, ω²) on color triplets.
  ├── Decompose chiral-16 under color Z_3.
  └── Expected: 16 = 8 + 4 + 4 (color-trivial + ω + ω²), NOT three equal generations.
        │
        ▼  [expected: 8 + 4 + 4 asymmetric → fail]
Phase D-3 (π_3 literature check, ~3 days)
  Look up homotopy groups π_3(G/H) for relevant carrier-related cosets.
  ├── Spin(10)/SU(5),  Spin(10)/(SU(4) × SU(2) × SU(2)), G_2/SU(3), etc.
  └── Search for any natural 3-torsion (π_3 = Z/3 or Z/3-quotient).
        │
        ▼  [expected: no 3-torsion in carrier-relevant cosets]
Phase D-5 (optional, ~3-4 weeks)
  Discrete anomaly cancellation forcing N_generations = 3.
  ├── Compute global SM anomaly contribution per generation in BCC walk.
  ├── Check whether any anomaly forces N=3.
  └── Either confirms N=3 is forced, or remains a negative.
```

A negative at any of D-1, D-2, D-3 documents that specific topological
mechanism as closed.  D-5 is the only one that could plausibly produce
a positive — and it's optional / deep.

## Phase D-1 — BCC body-diagonal Z_3 quick check (~2 days)

### FD-1 — Build the rotation

3-fold rotation around `(1,1,1)/√3`:

```text
σ : (x, y, z) → (y, z, x)
3×3 rotation matrix: R = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]
R³ = I, R^T R = I, det(R) = +1.
```

This is the cleanest Z_3 element of O_h ⊂ SO(3).

### FD-2 — Permutation on the 8 BCC directions

The 8 directions are `(±, ±, ±)`.  Under `R`, each is permuted to
another direction.  Compute the explicit permutation:

```text
(+,+,+) → (+,+,+)   (fixed)
(-,-,-) → (-,-,-)   (fixed)
The 6 other points form two 3-cycles.
```

**Test**: verify the permutation has cycle structure (1, 1, 3, 3).

### FD-3 — Spin-1/2 lift to Dirac

The Z_3 rotation about `(1,1,1)/√3` lifts to a spinor transformation
`U_3 ∈ SU(2)`.  Compute as

```text
U_3 = exp(- i (2π/3) (n · σ) / 2)
where n = (1, 1, 1) / √3.
```

`U_3` is a 2×2 SU(2) matrix.  Extended to 4-spinor Dirac:
`U_3^{Dirac} = block_diag(U_3, U_3)` (acts the same on both chiralities
in the chiral basis since axial rotations preserve chirality).

**Test**: `(U_3^{Dirac})^3 = ± I`.  (In SU(2), the spinor cube is -I,
not I, due to spin 1/2 vs SO(3).)

### FD-4 — Equivariance check on hops

For each BCC direction `h` with `σ(h) = h'`, check:

```text
W_{h'} = U_3^{Dirac} · W_h · (U_3^{Dirac})^{-1}
```

where `W_h` is the Dirac-assembled BB hop (chiral block of right and
left Weyl hops).

**Pass**: walk has Z_3 symmetry on the spatial × Dirac sector.
**Fail**: BB construction breaks body-diagonal Z_3 (would be surprising).

### FD-5 — Chiral-16 internal triviality

The chiral-16 internal carrier (32×32 real-skew SU(3)_c × SU(2)_L × U(1)_Y
generators) does NOT carry spatial coordinates.  By construction, the
spatial Z_3 of body diagonals acts on the chiral-16 trivially.

**Verdict (expected)**: under the body-diagonal Z_3 acting via spatial
rotations alone, the chiral-16 decomposes as `16 × (trivial Z_3 rep) =
16 × 1`.  No three-generation structure.

This is the expected D-1 negative.

## Phase D-2 — Color Z_3 center identification (~1-2 days)

### FD-6 — Color Z_3 center on chiral-16

The SU(3)_c color has a Z_3 center `{I, ωI, ω²I}` where
`ω = exp(2π i / 3)`.  On the chiral-16 (in the color-triplet basis), the
generator is `diag(1, ω, ω²)` on the color components.

Build this as a 32×32 unitary by:
1. Identify color triplet components in chiral-16.
2. Apply phase `1, ω, ω²` to each.

### FD-7 — Decompose chiral-16 under color Z_3

Compute the chiral-16's decomposition under color Z_3 center:

```text
Per generation (16 Weyl fermions):
  Q_L (color triplet × SU(2)_L doublet) = 6 fields: 2 in each of {1, ω, ω²}
  u_R (color triplet) = 3 fields: 1 in each
  d_R (color triplet) = 3 fields: 1 in each
  L_L (color singlet × doublet) = 2 fields: both in trivial
  e_R = 1 field: trivial
  ν_R = 1 field: trivial

Total: trivial char = 8, ω char = 4, ω² char = 4.
```

**Test**: build the 32×32 Z_3-center matrix.  Compute eigenspaces.
Verify the multiplicities are (8, 4, 4) — NOT (16/3, 16/3, 16/3).

### FD-8 — Verdict

Multiplicities `(8, 4, 4)` are NOT three equivalent generations.  The
color Z_3 center identification with spatial Z_3 fails to produce three
symmetric generation copies.

Even if the identification is forced by some mechanism (Kaluza-Klein,
discrete gauge invariance), the resulting "three generations" would be
asymmetric — leptons in one orbit, half of each quark in each of the
other two orbits — which is not what experiment shows.

## Phase D-3 — π_3 literature check (~3 days, documented note only)

Per the user's scope decision, this phase produces a Markdown literature
note at ``topology/PI3_LITERATURE_NOTE.md``, not a code module.

### Sources to consult

```text
- Mimura, Toda. Topology of Lie groups, I and II. AMS, 1991.
- Husemoller. Fibre Bundles. Tables in appendix.
- Bott. Lectures on K-theory.
- Wikipedia tables of homotopy groups of Lie groups.
```

### Cosets to look up

```text
π_3(Spin(10))                 = ?
π_3(Spin(10) / SU(5))         = ?
π_3(Spin(10) / (SU(4) × SU(2)²))   (Pati-Salam coset)
π_3(Spin(6) / SU(3))                (color-only embedding)
π_3(G_2 / SU(3))                    (octonion-relevant)
π_3(F_4 / Spin(9))                  (exceptional Jordan-relevant)
π_3(E_6 / (Spin(10) × U(1)))        (Boyle 27-relevant)
```

### Format of the deliverable

A 1-3 page markdown document with:

- Tabulated π_3 values from cited sources.
- Highlight any 3-torsion (expected: none).
- One-paragraph verdict: D-3 closed at the literature level.

No tests, no Python code.

## Phase D-5 — anomaly forcing N=3 (~3-4 weeks, committed)

Per user scope decision: included from the start as a committed phase,
not deferred.

The hypothesis: a discrete anomaly (global anomaly, modular invariance
constraint, or modular-tensor-category consistency) requires
N_generations = 3 for the BCC walk on chiral-16.

### Step-by-step plan (multi-week scope)

**FD-11 — Standard SM anomaly review (~3 days)**
- Review the standard SU(3)×SU(2)×U(1) anomaly conditions per generation.
- Verify cancellation: ``Σ_fermions Y³ = 0``, ``Σ_quarks Y · T² = 0``, etc.
- Document the per-generation anomaly polynomial.

**FD-12 — BCC walk anomaly contribution (~1 week)**
- Compute the BCC Bialynicki-Birula walk's contribution to the
  triangle anomaly.  The lattice-level computation requires the
  effective action.  Use the existing α-cont infrastructure (continuum
  expansion at O(ε) and beyond) from ``cp/continuum_cp.py``.
- Compare to the continuum SM result.

**FD-13 — Global / discrete anomaly check (~1-2 weeks)**
- Witten's global SU(2) anomaly:  ``π_4(SU(2)) = Z/2``  →  even
  number of SU(2)_L doublets required.  Verify our chiral-16 satisfies
  this per generation.
- Mod-2 anomaly count under reduction to lattice spacetime.
- Discrete anomaly polynomial coefficients (if non-trivial).

**FD-14 — Constraint on N_generations (~3 days)**
- From the discrete anomaly coefficients, extract any constraint on
  N_generations.
- Standard expectation: anomaly cancellation per generation makes the
  constraint vacuous for any N.  Unless there's a NEW constraint
  specific to the lattice / BCC structure.

### Pass/fail

| Outcome | Verdict |
|---|---|
| Some discrete anomaly forces N=3 (or constrains N to a specific value)        | **PASS** — major positive result.  Investigate further. |
| Anomaly cancellation is satisfied for any N (standard SM result)              | **FAIL** — anomalies don't force three generations. |
| BCC lattice introduces a new discrete anomaly that the chiral-16 doesn't cancel | **STRUCTURAL ISSUE** — would indicate the BCC walk is inconsistent at the lattice level. |

### Implementation files

```text
topology/anomaly_cancellation/
  bcc_anomaly_polynomial.py        # per-generation SM anomaly + BCC contribution
  global_anomaly_check.py          # Witten + mod-2 anomaly count
  discrete_anomaly_constraint.py   # full discrete anomaly polynomial
  anomaly_audit.py                 # combined verdict payload
```

### Effort: ~3-4 weeks committed

## Pre-named failure modes

**F-topo-1**: BCC body-diagonal Z_3 permutation has unexpected cycle
structure (not 1+1+3+3).  Diagnosis: probably basis-order bug.

**F-topo-2**: BB hops are NOT Z_3-equivariant.  Diagnosis: the
Bialynicki-Birula construction picks a specific direction-ordering that
breaks the symmetry, contrary to expectation.  Investigate before
concluding.

**F-topo-3**: chiral-16 internal carries non-trivial Z_3 action under
spatial rotations.  EXTREMELY UNEXPECTED (internal doesn't see spatial).
Indicates either a bug in `patisalam_chiral16_basis_matrix` or an
unintended coupling.  Investigate immediately.

**F-topo-4**: color Z_3 decomposition gives (≠ 8, 4, 4).  Diagnosis:
either a color-triplet identification bug or a non-standard
SU(3)_c basis.  Cross-check against existing SM tests in lepton.

**F-topo-5**: π_3 of some Pati-Salam / Spin(10) coset has Z/3 torsion.
UNEXPECTED.  Investigate; could be the actual three-generation
mechanism.

**F-topo-6**: π_3 literature check inconclusive (sources contradict /
no clean answer).  Diagnosis: pick the standard textbook reference and
proceed.

## Critical files

To create (under `src/clifford_3plus2_d5/topology/`):

```text
topology/
  __init__.py
  PLAN.md                                  (this plan as reference)
  STATUS.md
  parameter_ledger.md
  reuse.py                                 # imports from spacetime_qca + lepton
  bcc_z3_rotation.py                       # Phase D-1: 3×3 rotation + Dirac lift
  hop_equivariance.py                      # Phase D-1: BB hops Z_3-equivariance
  internal_triviality.py                   # Phase D-1: chiral-16 Z_3-trivial check
  color_center_z3.py                       # Phase D-2: color Z_3 center decomposition
  PI3_LITERATURE_NOTE.md                   # Phase D-3: documented note
  anomaly_cancellation/                    # Phase D-5: discrete anomaly
    __init__.py
    bcc_anomaly_polynomial.py
    global_anomaly_check.py
    discrete_anomaly_constraint.py
    anomaly_audit.py
  SESSION_TOPOLOGY.md                      # final verdict
  tests/
    __init__.py
    test_bcc_z3_rotation.py
    test_hop_equivariance.py
    test_internal_triviality.py
    test_color_center_z3.py
    test_bcc_anomaly_polynomial.py
    test_global_anomaly_check.py
    test_discrete_anomaly_constraint.py
    test_anomaly_audit.py
```

To consult / read-only:

```text
src/clifford_3plus2_d5/spacetime_qca/bcc_weyl.py
src/clifford_3plus2_d5/spacetime_qca/bcc_geometry.py
src/clifford_3plus2_d5/spacetime_qca/dirac.py
src/clifford_3plus2_d5/lepton/clifford_patisalam.py
src/clifford_3plus2_d5/lepton/patisalam_sm.py
src/clifford_3plus2_d5/cp/cubic_harmonics.py
src/clifford_3plus2_d5/triality/spin8_triality.py
```

## What this plan does NOT include (deferred)

- Phase D-4 (instantons / Pontryagin index Z → Z/3 mechanism).  User's
  brainstorm flagged this as a long-shot; deferred.
- Phase D-2 alternative variants (binary tetrahedral subgroup 2T of
  SU(2)_L, etc.) — flagged as too unmotivated without further prompting.
- Full E_6 / F_4 / E_8 topology (would be much larger sidecars).
- Phenomenological magnitude matching to CKM / PMNS even if D-5 closes
  positive — magnitude matching is its own separate investigation.

## Verification

End-to-end test sequence after each phase:

```bash
# Phase D-1 (rotation + hops + internal triviality)
uv run pytest src/clifford_3plus2_d5/topology/tests/test_bcc_z3_rotation.py -q
uv run pytest src/clifford_3plus2_d5/topology/tests/test_hop_equivariance.py -q
uv run pytest src/clifford_3plus2_d5/topology/tests/test_internal_triviality.py -q

# Phase D-2 (color Z_3 center)
uv run pytest src/clifford_3plus2_d5/topology/tests/test_color_center_z3.py -q

# Phase D-3 (homotopy literature note — no tests, just a markdown read)
cat src/clifford_3plus2_d5/topology/PI3_LITERATURE_NOTE.md

# Phase D-5 (anomaly cancellation)
uv run pytest src/clifford_3plus2_d5/topology/tests/test_bcc_anomaly_polynomial.py -q
uv run pytest src/clifford_3plus2_d5/topology/tests/test_global_anomaly_check.py -q
uv run pytest src/clifford_3plus2_d5/topology/tests/test_discrete_anomaly_constraint.py -q
uv run pytest src/clifford_3plus2_d5/topology/tests/test_anomaly_audit.py -q

# Full sidecar
uv run pytest src/clifford_3plus2_d5/topology/tests -q

# Regression: existing tests stay green
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests -q
uv run pytest src/clifford_3plus2_d5/exceptional/tests -q
```

Verdict callable:

```bash
uv run --no-sync python -c "
from clifford_3plus2_d5.topology.internal_triviality import internal_z3_triviality_payload
from clifford_3plus2_d5.topology.color_center_z3 import color_z3_decomposition_payload
from clifford_3plus2_d5.topology.anomaly_cancellation.anomaly_audit import anomaly_audit_payload
print(internal_z3_triviality_payload().verdict)
print(color_z3_decomposition_payload().verdict)
print(anomaly_audit_payload().verdict)
"
```

Expected outcomes:

> D-1: TOPOLOGY KILL — chiral-16 internal is Z_3-trivial; spatial Z_3
> alone gives no three-generation structure.
>
> D-2: COLOR Z_3 KILL — chiral-16 decomposes as 8 + 4 + 4 under color
> Z_3 center, asymmetric (not three equal generations).
>
> D-3 (literature note): no carrier-relevant coset has Z/3 torsion in π_3.
>
> D-5: ANOMALY KILL — standard SM anomaly conditions cancel per
> generation, no discrete anomaly forces N=3.  (Or a positive result if
> a new BCC-specific anomaly is found.)

## Effort budget

- Phase D-1 (BCC Z_3 quick check): ~2 days.
- Phase D-2 (color Z_3 identification): ~1-2 days.
- Phase D-3 (π_3 literature note): ~2-3 days (reduced — no code).
- Phase D-5 (anomaly forcing N=3, COMMITTED): ~3-4 weeks.

**Total committed budget**: ~4-5 weeks.

**Worst case**: ~4-5 weeks for a clean negative on all four candidates.
The D-5 anomaly check is the only candidate that could plausibly produce
a positive — committing to it from the start means the sidecar produces
a decisive verdict across the topology family in ~one month.

The kill-discipline ordering still applies: D-1 and D-2 close in days,
D-3 in less than a week, D-5 takes the bulk of the budget.  Any of
D-1/D-2/D-3 producing a surprise would change D-5's scope (e.g.,
shifting effort toward investigating the surprise).
