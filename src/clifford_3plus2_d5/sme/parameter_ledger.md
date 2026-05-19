# SME Sidecar Parameter Ledger

Filled in as each phase pins choices.

## Inherited from cp/ (read-only)

1. H^(1) extraction via Baker-Campbell-Hausdorff at O(ε)
   (`continuum_cp.effective_hamiltonian_first_correction`).
2. CP action: spinor CP matrix from `cp.discrete_symmetries.parity_spinor`
   composed with `charge_conjugation_spinor`; antiunitary on operators
   (M → CP · M* · CP⁻¹).
3. T_{2g} basis ordering: `(k_y k_z, k_z k_x, k_x k_y)`.
4. Chiral-basis γ matrices from `spacetime_qca.dirac`.
5. ε is symbolic positive (SymPy).

## Phase A-1 choices

1. Five symmetry classes used to identify the SME sector: Hermitian,
   chirality-preserving (block-diagonal in 4-spinor chiral basis),
   momentum-bilinear (total degree 2 in (kx, ky, kz)), CP-odd fraction
   = 1, T_{2g} fraction = 1.
2. SME sector label: "dim-5 non-minimal SME, fermion sector, CP-odd
   spin-tensor" (Kostelecky-Mewes arXiv:1308.4973).

## Phase A-2 choices

1. Pauli basis convention: {I, σ^x, σ^y, σ^z} as 2×2 SymPy matrices,
   projection ``c_a = (1/2) tr(σ^a · M)``.
2. T_{2g} momentum-pair ordering: (i ≤ j) — namely (y, z), (x, z),
   (x, y).
3. dim-5 SME target label: ``d^{(5)}_{αβγ}`` (axial-vector × 2
   derivatives, CPT-even, CP-odd) per Kostelecky-Mewes 2013.
4. CPT class: CPT-even, derived from BCC walk CPT-invariance combined
   with cp/'s CP-odd verdict (since CPT = CP · T, CP-odd implies T-odd
   given CPT-even).
5. Three non-zero T^{aij} entries: (a, i, j) ∈ {(x, y, z), (y, x, z),
   (z, x, y)} with coefficients (+1, -1, +1).

## Phase A-3 choices

1. Primary literature source: Kostelecky-Russell "Data Tables for
   Lorentz and CPT Violation", arXiv:0801.0287 (current revision).
2. Cross-check sources: Mattingly Living Reviews 8 (2005) 5;
   Liberati Class. Quantum Grav. 30 (2013) 133001.
3. Representative bound: ``|d^{(5)}| ≲ 10⁻¹⁷ GeV⁻¹`` from electron-
   sector sidereal-modulation atomic-clock + co-magnetometer analyses.
   Specific Kostelecky-Russell entry ids pending verification.
4. Convention: dim-5 fermion SME coefficients have units of GeV⁻¹.

## Phase A-4 choices

1. Planck length constant: ``ℓ_P = 1.616255 × 10⁻³⁵ m`` (CODATA).
2. Conversion ``GeV⁻¹ = 1.97327 × 10⁻¹⁶ m`` via ℏc.
3. Observable threshold separating UNFALSIFIABLE PASS from OBSERVABLE
   POSITIVE: ``10⁻²⁵ m``.
4. Verdict classes:
   - SUB-PLANCK KILL: ε ≤ ℓ_P
   - PLANCK-CONSISTENT: ℓ_P < ε ≤ 10 ℓ_P
   - UNFALSIFIABLE PASS: 10 ℓ_P < ε ≤ 10⁻²⁵ m
   - OBSERVABLE POSITIVE: ε > 10⁻²⁵ m
5. Tightest-face rule: ``ε_max = min_i { bound_i / |coefficient_i| }``.
   With three unit coefficients, the tightest face equals the single
   bound.

## Continuous parameters

Two:

- ε (lattice scale, symbolic positive).  The single continuous
  parameter constrained by Bold A.
- (None other introduced by this sidecar.)

Numerical citations from Phase A-3 (experimental bounds) appear as
tabulated constants in `epsilon_constraint.py`, sourced from named
papers — they are inputs, not parameters of the program.
