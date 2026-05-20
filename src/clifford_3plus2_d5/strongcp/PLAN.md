# Plan: strongcp sidecar — full Strong-CP / θ_QCD lattice audit

## Context

The cp/ sidecar verified that H^(1), the O(ε) BCC walk correction to
the free-Dirac Hamiltonian, is 100% CP-odd and entirely in the T_{2g}
cubic-harmonic irrep.  sme/ closed Bold A at UNFALSIFIABLE PASS for
the fermion-sector dim-5 SME coefficient ``d^(5)``.

The Strong-CP problem asks: why is the QCD vacuum-angle parameter
``θ_QCD`` ≤ 10⁻¹⁰ (from neutron-EDM measurements) despite being a
free O(1) parameter in continuum QCD?  The standard SM has no
mechanism — it requires an axion or some accidental cancellation.

**Key structural observation**: H^(1) is in T_{2g}, which is a
**g-irrep** (parity-even under ``k → -k``) of the cubic group O_h.
The θ_QCD operator ``E·B = ε^{μνρσ} tr(F_{μν} F_{ρσ})`` is a
**pseudoscalar** lattice operator — its degree-3 momentum-shape
``k_x k_y k_z`` lives in **A_{2u}**, a u-irrep (parity-odd).

Parity selection rule on O_h: ``g × g = g``, ``g × u = u``,
``u × u = g``.  No power of g-irrep operators produces a u-irrep.
So if H^(n) at all orders stays in g-irreps, the BCC walk's
contribution to θ_QCD is **exactly zero by structural symmetry** —
the program naturally satisfies strong-CP without an axion.

The single load-bearing question:

> Does the BCC Bialynicki-Birula walk's effective action contribute
> to θ_QCD, and if so at what order in ε relative to the neutron-EDM
> bound |θ_QCD| ≤ 10⁻¹⁰?

Three pre-named outcomes (user-confirmed three-band scheme):

1. **STRONG-CP TRIVIAL**: θ = 0 to all orders by structural lattice
   symmetry.  Publishable positive — the program naturally solves
   strong-CP without invoking an axion.
2. **STRONG-CP SAFE**: θ = O(ε^n) for n ≥ 1 → far below 10⁻¹⁰
   neutron-EDM bound.  Honest pass.
3. **STRONG-CP TENSION**: θ > 10⁻¹⁰.  Conflicts with neutron-EDM
   bound.  Kill.

## User-confirmed scope decisions

- **Sidecar name**: ``strongcp/``.
- **Audit depth**: full lattice topological-charge audit — symbolic
  selection-rule argument **plus** direct lattice computation of
  Q = (1/32π²) ε^{μνρσ} tr(F_{μν} F_{ρσ}) and the chiral-anomaly
  channel.  ~3-4 weeks.
- **Cubic harmonics extension**: full degree-3 O_h decomposition
  (all 8 irreps A_{1g}, A_{2u}, E_g, E_u, T_{1g}, T_{1u}, T_{2g},
  T_{2u}), kept in cp/cubic_harmonics.py (or a side module imported
  by strongcp/), so the full machinery is reusable.
- **Verdict thresholds**: three bands aligned with the neutron-EDM
  bound (TRIVIAL / SAFE / TENSION).

## Existing infrastructure to reuse (via `strongcp/reuse.py`)

Verified from prior exploration:

**From cp/**:
- ``cp/cubic_harmonics.py`` — degree-2 O_h projectors (A_{1g}, E_g,
  T_{2g}); will be extended with degree-3 module (see Phase SC-1).
- ``cp/continuum_cp.py`` — H^(1) at O(ε), nth_order_in_epsilon
  wrapper, CP-action machinery.
- ``cp/discrete_symmetries.py`` — P, T, C spinor matrices and
  composites.

**From spacetime_qca/**:
- ``bcc_geometry.py`` — BCC lattice site structure.
- ``bcc_weyl.py`` — Bialynicki-Birula walk hops.
- ``jax_wilson.py`` — Wilson plaquette action density.
- ``jax_gauge_force.py`` — SU(3) generators, plaquette forces.
- ``plaquette.py`` — ``canonical_bcc_plaquette_shapes()`` for BCC
  elementary plaquettes.
- ``continuum.py`` — ``nth_order_in_epsilon``, BCH expansion.

**What must be built new**:
- Degree-3 cubic-harmonic decomposition (10-dim monomial space → 8
  irreps).
- BCC centrosymmetry verification (lattice + walk both invariant
  under ``r → -r``).
- Higher-order H^(n) audit for n = 2, 3 (extend cp/'s O(ε) audit).
- Lattice topological-charge density Q(x) from BCC plaquettes.
- Chiral-anomaly direction: does H^(1) shift the fermion measure?
- Combined audit + verdict + neutron-EDM constraint comparison.

## Decision tree

```text
Phase SC-1 (degree-3 cubic harmonics extension, ~2-3 days)
  ├── Build 10-dim degree-3 polynomial basis.
  ├── Construct projectors for all 8 degree-3 O_h irreps.
  ├── Identify A_{2u} (the k_x k_y k_z direction) as the θ-term irrep.
  └── Verify projectors satisfy orthogonality + completeness.
        │
        ▼
Phase SC-2 (BCC centrosymmetry audit, ~2-3 days)
  ├── Verify BCC lattice is centrosymmetric (inversion ``r → -r``
  │   maps sites to sites).
  ├── Verify Bialynicki-Birula walk is parity-covariant on the
  │   spinor: under spatial inversion combined with Dirac parity P,
  │   the walk is invariant.
  └── Verify the gauged walk (SU(3) Wilson links) preserves
      centrosymmetry — links transform appropriately under r → -r.
        │
        ▼  [parity selection rule applies]
Phase SC-3 (higher-order H^(n) parity audit, ~3-4 days)
  ├── Extend cp/continuum_cp.py infrastructure to H^(2), H^(3).
  ├── Decompose H^(2), H^(3) into degree-{2,3,4} cubic-harmonic
  │   irreps using Phase SC-1's projectors.
  ├── Verify H^(2) lives in g-irreps (degree-2 monomials projected
  │   on A_{1g}, E_g, T_{2g}).
  ├── Verify H^(3) lives in u-irreps but NOT in A_{2u} (i.e., not in
  │   the pseudo-scalar k_x k_y k_z direction).
  └── Apply g × u parity selection rule: products cannot generate
      A_{2u} in the effective action's θ-channel.
        │
        ▼  [selection rule argument complete]
Phase SC-4 (lattice topological-charge density, ~5-7 days)
  ├── Build Q(x) = (1/32π²) ε^{μνρσ} tr(F_{μν} F_{ρσ}) from BCC
  │   plaquettes via F_{μν} = (1/i) log(P_{μν}) (or equivalent
  │   compact discretization).
  ├── For BCC, the natural ε^{μνρσ} pairing is (time × spatial-pair)
  │   × (spatial-pair).  Pure-spatial Q from BCC plaquettes is
  │   identically zero (3 spatial directions cannot supply 4 indices).
  ├── Compute the time-component contribution via the discrete-time
  │   walk's structure.
  └── Decompose Q into cubic-harmonic irreps; verify A_{2u}
      component vanishes by Phase SC-3's selection rule.
        │
        ▼
Phase SC-5 (chiral anomaly + θ̄ shift, ~4-5 days)
  ├── Standard SM: θ̄ = θ + arg(det M_q).  A chiral rotation
  │   ψ → e^{iα γ^5} ψ shifts θ̄ by 2 α N_f.
  ├── Check whether H^(1) acts as a chiral rotation on the fermion
  │   measure: compute ``tr γ^5 H^(1)`` (or its lattice equivalent).
  ├── H^(1) = diag(σ^a, σ^a) × momentum: identical action on both
  │   chiralities → vector, not axial.  Expected ``tr γ^5 H^(1) = 0``.
  ├── Repeat for H^(2), H^(3) using SC-3's irrep decomposition.
  └── Combine: total θ̄ shift from BCC walk = sum of chiral-rotation
      contributions across all orders.
        │
        ▼
Phase SC-6 (combined audit + SESSION report, ~3-4 days)
  ├── Aggregate SC-1..SC-5 into a single ``StrongCPAuditPayload``.
  ├── Compare the cumulative θ̄ contribution to 10⁻¹⁰ neutron-EDM
  │   bound; classify into TRIVIAL / SAFE / TENSION.
  ├── SESSION_STRONGCP.md final verdict report.
  └── Update STATUS.md, parameter_ledger.md, PROJECT_STATUS.md.
```

## Phase SC-1 — Degree-3 cubic harmonics (~2-3 days)

### FS-1: 10-dim monomial basis

Degree-3 monomials in (k_x, k_y, k_z):

```text
{ k_x³, k_y³, k_z³,
  k_x² k_y, k_x² k_z, k_y² k_x, k_y² k_z, k_z² k_x, k_z² k_y,
  k_x k_y k_z }
```

(10 monomials.)  All are parity-odd under k → -k → live in u-irreps.

### FS-2: Decomposition under O_h

Standard result (Tinkham, Group Theory and Quantum Mechanics):

```text
10-dim degree-3 space = A_{2u}  ⊕  E_u  ⊕  T_{1u}  ⊕  T_{2u}
                       (1)     (2)    (3)    (3)        = 10 ✓
```

Wait — let me re-verify.  Standard cubic-harmonic table for O_h
applied to degree-3 polynomials gives:

```text
A_{1u}: 0
A_{2u}: 1   (basis: k_x k_y k_z)
E_u:    2   (basis: combinations of k_x³, k_y³, k_z³ minus trace)
T_{1u}: 3   (basis: {k_x (k_y² + k_z²), k_y (k_z² + k_x²), k_z (k_x² + k_y²)})
T_{2u}: 3   (basis: {k_x (k_y² - k_z²), k_y (k_z² - k_x²), k_z (k_x² - k_y²)})
total:  9   ✗ short by 1
```

Re-examine: the diagonal cubes ``k_x³, k_y³, k_z³`` decompose into
A_{1u} (trace) + E_u (traceless symmetric).  But A_{1u} of O_h is
empty (totally-symmetric scalar with parity-odd is impossible in
3D — no cubic-invariant pseudoscalar of degree 1 in each axis).

Actually the trace ``k_x³ + k_y³ + k_z³`` is not totally symmetric
under O_h because under x↔y, it's invariant, but full O_h includes
all permutations and sign-flips of axes.  Under reflection across a
diagonal of the cube, x³ → ±y³ etc.  Need to check the character.

Conclusion: the precise degree-3 decomposition of O_h is well-known
but technical.  Phase SC-1 implements it by direct character-theory
construction using O_h character table.

### FS-3: Implementation

Add a new module ``strongcp/cubic_harmonics_degree3.py`` (or extend
``cp/cubic_harmonics.py``) exposing:

- ``degree3_monomial_basis(k_symbols) -> tuple`` — fixed-order 10-tuple.
- ``projector_<IRREP>()`` for each of the 8 O_h irreps (with empty
  projectors returned as 10×10 zeros for the irreps that don't appear
  at degree 3, e.g., A_{1g} doesn't appear at degree 3).
- ``decompose_degree3_polynomial(poly, k_symbols)`` — return
  dict[irrep, projected polynomial].
- Tests verifying:
  - Projectors sum to identity (completeness).
  - Projectors are mutually orthogonal.
  - ``k_x k_y k_z`` projects entirely into A_{2u}.

## Phase SC-2 — BCC centrosymmetry audit (~2-3 days)

### FS-4: Lattice centrosymmetry

The BCC lattice has inversion centres at every site.  Verify
programmatically:

- Lattice sites: ``Z³ ∪ (Z + 1/2)³``.
- Inversion: ``(x, y, z) → (-x, -y, -z)``.
- For any site ``s ∈ Z³`` or ``s ∈ (Z + 1/2)³``: ``-s`` is also a
  lattice site.  ✓ trivial.

### FS-5: BB walk parity covariance

The Bialynicki-Birula walk on BCC:

```text
ψ(x + h, t + 1) = W_h · ψ(x, t)
```

where ``h`` runs over the 8 BCC body-diagonal directions and ``W_h``
is the per-direction hop matrix.

Spatial inversion sends ``h → -h``.  Inversion symmetry requires:

```text
P · W_h · P^{-1}  =  W_{-h}
```

where ``P`` is the Dirac parity operator (= ``γ⁰`` in chiral basis,
or its appropriate spinor representation).

The BB hops have explicit form ``W_h = (1 - i α · ĥ) / sqrt(2)``
roughly; their parity-covariance is a quick algebraic check.

### FS-6: Gauged-walk centrosymmetry

When the BB walk is gauged with SU(3) Wilson links, the link
``U_h(x)`` from site ``x`` in direction ``h`` should transform as

```text
U_h(x)  →  U_{-h}(-x)^{-1} = U_h(-x - h)^{-1}
```

under spatial inversion.  Verify this is consistent with the
Wilson action.

## Phase SC-3 — Higher-order H^(n) parity audit (~3-4 days)

### FS-7: H^(2), H^(3) extraction

Extend ``cp/continuum_cp.py``'s ``effective_hamiltonian_first_correction``
infrastructure to:

- ``effective_hamiltonian_second_correction()`` → H^(2) (degree 3 in k).
- ``effective_hamiltonian_third_correction()`` → H^(3) (degree 4 in k).

The BCH expansion at higher orders is well-defined; ``nth_order_in_epsilon``
already supports this.  The challenge is computational (more terms).

### FS-8: Irrep decomposition

For each H^(n):

- Project each matrix entry's polynomial onto the degree-n cubic-
  harmonic irreps.
- Compute Frobenius-norm-squared in each irrep cell.
- Identify which irreps carry non-zero content.

### FS-9: Parity selection rule application

For each H^(n):

- Even n: degree-even polynomial → g-irreps (A_{1g}, E_g, T_{2g}, ...).
- Odd n: degree-odd polynomial → u-irreps (A_{2u}, E_u, T_{1u}, T_{2u}, ...).

**Critical check**: at odd n, does H^(n) populate the A_{2u} cell?

H^(3) has 4 possible u-irreps for degree-3 monomials: A_{2u}, E_u,
T_{1u}, T_{2u}.  If H^(3) has zero A_{2u}-component, the leading
non-zero contribution to θ_QCD would be from higher orders (H^(5),
H^(7), ...), each O(ε^(2k+1)) and additionally suppressed by
intermediate g-irrep products.

## Phase SC-4 — Lattice topological charge density (~5-7 days)

### FS-10: BCC plaquette field strength

For BCC plaquettes, the Wilson loop around a closed plaquette gives:

```text
P_{μν}(x) = U_μ(x) U_ν(x + ê_μ) U_μ(x + ê_ν)^{-1} U_ν(x)^{-1}
```

The field strength is extracted as ``F_{μν}(x) = -i log(P_{μν}(x))``
or the compact ``F_{μν} ≈ (P_{μν} - P_{μν}^†) / (2 i a²)`` where
``a`` is the lattice spacing.

For BCC there are several plaquette shapes; ``canonical_bcc_plaquette_shapes``
in spacetime_qca/plaquette.py enumerates them.

### FS-11: Topological charge density

```text
Q(x) = (1/32π²) ε^{μνρσ} tr( F_{μν}(x) F_{ρσ}(x) )
```

The natural index pairing is (μ, ν) = (time × space) and (ρ, σ) =
(space × space) — i.e., ``E · B``.

For pure-spatial plaquettes on BCC there are only 3 spatial
directions, so ``ε^{ijk0}`` cannot supply 4 distinct spatial indices
— pure-spatial Q is zero.

### FS-12: Q in irrep decomposition

The discrete ``Q(x)`` lives in some specific cubic-group irrep
of the lattice.  Verify it is in A_{2u} (the natural θ-term
irrep).  Then apply Phase SC-3's selection rule to conclude
that the BCC walk's contribution to Q is zero at all orders.

## Phase SC-5 — Chiral anomaly + θ̄ shift (~4-5 days)

### FS-13: Fermion-measure phase

Under a chiral rotation ``ψ → e^{iα γ^5} ψ``:

- The classical fermion action is invariant (free Dirac).
- The path integral measure is NOT invariant: it picks up a phase
  factor proportional to the topological charge:

```text
log det(γ^μ D_μ + m e^{i α γ^5})  =  ... + i α (2 N_f) ∫ Q(x) d⁴x
```

This shifts the effective θ̄ by ``2 α N_f``.

### FS-14: Does H^(n) induce a chiral rotation?

H^(n) acts on the fermion in a specific spinor structure.  A chiral
rotation has spinor structure ``e^{iα γ^5}`` — diagonal in chirality
with opposite sign on left and right.

H^(1) per chirality: ``diag(σ^a, σ^a)`` — IDENTICAL on both
chiralities → vector, not axial.  Expected ``tr(γ^5 H^(1)) = 0``.

Verify by direct computation:

```text
tr( γ^5 H^(1)(k) )  =  ?
```

Should be zero if H^(1) is purely vector.

### FS-15: Extend to H^(2), H^(3)

Repeat the ``tr(γ^5 H^(n))`` check at higher orders.  The lattice's
parity selection rule + CPT-invariance suggests these all vanish.

If any non-zero ``tr(γ^5 H^(n))`` is found, the magnitude pins the
contribution to θ̄: ``δθ̄ = 2 N_f α``, where ``α`` is the effective
chiral-rotation angle ``α ∝ ε^n / m^n``.

## Phase SC-6 — Combined audit + SESSION report (~3-4 days)

### FS-16: StrongCPAuditPayload

Aggregate:

- ``cubic_harmonics_degree3_consistency``: from SC-1.
- ``bcc_centrosymmetry_verified``: from SC-2.
- ``h2_h3_in_safe_irreps``: from SC-3.
- ``Q_in_A2u``: from SC-4.
- ``chiral_anomaly_theta_bar_shift``: from SC-5.
- ``cumulative_theta_bound_at_each_order``: symbolic expressions in ε.
- ``final_verdict``: TRIVIAL / SAFE / TENSION.

### FS-17: SESSION_STRONGCP.md

Structure analogous to ``SESSION_TOPOLOGY.md`` and ``SESSION_SME.md``:

```markdown
# Strong-CP — θ_QCD audit verdict report

**Status**: CLOSED — {TRIVIAL / SAFE / TENSION}

## Load-bearing question

## Phase-by-phase findings (SC-1 through SC-5)

## Combined verdict (with ε-suppression analysis)

## What this does NOT close

## Test summary

uv run pytest src/clifford_3plus2_d5/strongcp/tests -q
```

## Pre-named failure modes

**F-strongcp-1**: BCC lattice fails centrosymmetry.  Diagnosis:
structural issue — would indicate a setup bug, not a real lack of
inversion symmetry.

**F-strongcp-2**: H^(2) or H^(3) populates A_{2u} subspace.  Diagnosis:
program contributes to θ at order n; compute magnitude.  Likely
SAFE if n ≥ 2; TENSION only if n = 1 and the coefficient is O(1).

**F-strongcp-3**: ``tr(γ^5 H^(n))`` non-zero for some n.  Diagnosis:
chiral-rotation contribution to θ̄ — compute shift; classify.

**F-strongcp-4**: BCC plaquette ``Q(x)`` not in A_{2u}.  Diagnosis:
the lattice's natural θ-term irrep is different; revisit the
selection rule argument.

**F-strongcp-5**: Discrete-time walk introduces a new anomaly direction
not captured by spatial centrosymmetry.  Diagnosis: extend the
analysis to include time-reversal and discrete-time invariance.

**F-strongcp-6**: Degree-3 O_h irrep decomposition fails the
character-table consistency check.  Diagnosis: basis or convention
bug; cross-check against Tinkham character tables.

## Critical files

To create (under ``src/clifford_3plus2_d5/strongcp/``):

```text
strongcp/
  __init__.py
  PLAN.md                                  (this plan as reference)
  STATUS.md
  parameter_ledger.md
  reuse.py                                 # imports from cp/, spacetime_qca
  cubic_harmonics_degree3.py               # Phase SC-1
  bcc_centrosymmetry.py                    # Phase SC-2
  higher_order_parity.py                   # Phase SC-3
  topological_charge_density.py            # Phase SC-4
  chiral_anomaly_check.py                  # Phase SC-5
  theta_bar_constraint.py                  # combined θ̄ bound vs neutron-EDM
  strong_cp_audit.py                       # Phase SC-6
  SESSION_STRONGCP.md                      # final verdict
  tests/
    __init__.py
    test_cubic_harmonics_degree3.py
    test_bcc_centrosymmetry.py
    test_higher_order_parity.py
    test_topological_charge_density.py
    test_chiral_anomaly_check.py
    test_theta_bar_constraint.py
    test_strong_cp_audit.py
```

To consult / read-only:

```text
src/clifford_3plus2_d5/cp/cubic_harmonics.py       # degree-2 baseline
src/clifford_3plus2_d5/cp/continuum_cp.py          # H^(1) source, BCH machinery
src/clifford_3plus2_d5/cp/discrete_symmetries.py   # P, T, C
src/clifford_3plus2_d5/spacetime_qca/bcc_geometry.py
src/clifford_3plus2_d5/spacetime_qca/bcc_weyl.py
src/clifford_3plus2_d5/spacetime_qca/jax_wilson.py
src/clifford_3plus2_d5/spacetime_qca/plaquette.py
src/clifford_3plus2_d5/spacetime_qca/continuum.py
```

## What this plan does NOT include (deferred)

- **Photon-sector (k_F)^(5) analogue audit** — would target the U(1)
  electromagnetic θ-angle independently.  Orthogonal sector;
  ~3-4 weeks if pursued separately.
- **Axion mechanism integration**.  This audit asks whether the
  program needs an axion; if STRONG-CP TRIVIAL, no.  If SAFE, no.
  Only TENSION would force an axion-mechanism investigation.
- **Higher-order H^(n) for n > 3**.  The selection-rule argument is
  expected to close at n ≤ 3; recurse only if SC-3 finds something
  unexpected.
- **Three-generation extension**.  Bold A and topology closed the
  three-generation question; this audit treats one generation
  (per-generation θ̄ contribution scales linearly with N_gen and
  doesn't change the verdict class).
- **Pati-Salam / SM gauge sector**.  Phase SC-4 uses SU(3)_c
  alone; the analogous Pati-Salam analysis would require gauging
  the larger group.

## Verification

End-to-end test sequence after each phase:

```bash
# Phase SC-1
uv run pytest src/clifford_3plus2_d5/strongcp/tests/test_cubic_harmonics_degree3.py -q

# Phase SC-2
uv run pytest src/clifford_3plus2_d5/strongcp/tests/test_bcc_centrosymmetry.py -q

# Phase SC-3
uv run pytest src/clifford_3plus2_d5/strongcp/tests/test_higher_order_parity.py -q

# Phase SC-4
uv run pytest src/clifford_3plus2_d5/strongcp/tests/test_topological_charge_density.py -q

# Phase SC-5
uv run pytest src/clifford_3plus2_d5/strongcp/tests/test_chiral_anomaly_check.py -q

# Phase SC-6
uv run pytest src/clifford_3plus2_d5/strongcp/tests/test_strong_cp_audit.py -q

# Full sidecar
uv run pytest src/clifford_3plus2_d5/strongcp/tests -q

# Regression: prior sidecars stay green
uv run pytest src/clifford_3plus2_d5/cp/tests -q
uv run pytest src/clifford_3plus2_d5/sme/tests -q
uv run pytest src/clifford_3plus2_d5/topology/tests -q
```

Verdict callable:

```bash
uv run --no-sync python -c "
from clifford_3plus2_d5.strongcp.strong_cp_audit import strong_cp_audit_payload
p = strong_cp_audit_payload()
print(p.verdict)
print(p.cumulative_theta_bar_symbolic, '<=', p.neutron_edm_bound)
print(p.interpretation)
"
```

Expected outcome (a priori unknown — that is the point of the audit):
one of the three pre-named verdicts.  Most-likely outcome based on
the parity-selection-rule logic: **STRONG-CP TRIVIAL** (θ = 0 to all
orders), but SC-3 and SC-5 must verify this without loopholes.

## Effort budget

- Phase SC-1: ~2-3 days.
- Phase SC-2: ~2-3 days.
- Phase SC-3: ~3-4 days.
- Phase SC-4: ~5-7 days.
- Phase SC-5: ~4-5 days.
- Phase SC-6: ~3-4 days.

**Total committed budget**: ~3-4 weeks.

**Worst case**: ~3-4 weeks for a verdict in any class.  Bold positive
case (TRIVIAL) requires SC-3 + SC-5 to both close clean.  The bulk of
the cost is in SC-4 (lattice topological-charge computation) and SC-5
(chiral anomaly direct check).  The kill-discipline ordering puts the
cheapest selection-rule arguments first; if SC-3 closes positive, SC-4
and SC-5 confirm by direct computation.

A surprise at SC-3 or SC-5 (some H^(n) populates A_{2u} or some
``tr(γ^5 H^(n))`` is non-zero) would shift the verdict toward SAFE
(magnitude small enough) or potentially TENSION (magnitude near or
above neutron-EDM bound).  In all cases the audit produces an
actionable verdict.
