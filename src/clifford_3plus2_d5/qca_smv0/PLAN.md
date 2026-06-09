# QCA_SMv0 Plan

## Purpose

Build a focused simulator sidecar for the next Standard-Model QCA prototype.

The first rule is:

```text
infrastructure first, theory second
```

## Session 00 - Infrastructure

Pass only if:

- the sidecar package exists;
- `README.md`, `PLAN.md`, and `STATUS.md` exist;
- `scripts/` and `tests/` exist;
- the package exposes only sidecar metadata;
- the placeholder test verifies only infrastructure;
- the central repo docs link to the sidecar;
- no simulator theory, mass model, or hidden ledger is introduced.

Verdict:

```text
QCA_SMV0_INFRASTRUCTURE_READY
```

## Stage 1 - Free BCC Weyl/Dirac walk

Pass only if:

- the Weyl state has shape `(nx, ny, nz, 2)`;
- the eight BCC source offsets are exactly `{+-1}^3`;
- the Pauli projectors satisfy `P_j^- + P_j^+ = I` and are orthogonal;
- the hop matrices are `A_h=P_x^{h_x}P_y^{h_y}P_z^{h_z}`;
- `sum_h A_h^dagger A_h = I`;
- the position kernel uses the repository pull convention `psi[x+h]`;
- the Bloch symbol from the split product equals the summed hop symbol;
- the Bloch symbol is unitary;
- the periodic-lattice step preserves norm;
- the step is JIT-compatible;
- the small-momentum eigenphase has Weyl speed `1` near the origin;
- the directional phase-speed spread decreases under momentum halving;
- the four-component Dirac step is assembled from opposite-chirality Weyl
  blocks;
- Dirac hops satisfy `sum_h D_h^dagger D_h = I_4`;
- the Dirac symbol is unitary;
- the Dirac step preserves norm and is JIT-compatible;
- no boundary, gauge, Higgs, flavor, or recirculation rule is introduced.

Verdict:

```text
QCA_SMV0_STAGE1_FREE_BCC_PASS
```

## Stage 2 - Static SM Gauge Background

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the internal register has shape `32`, interpreted as one SM chiral-16 label
  set duplicated over the Dirac spin block;
- local anti-Hermitian `SU(3)_c x SU(2)_L x U(1)_Y` generators are explicit;
- finite static BCC links have shape `(nx, ny, nz, 8, 32, 32)`;
- link exponentiation produces unitary matrices;
- identity links reduce the gauged Dirac update to the Stage 1 free Dirac
  update applied independently to each internal component;
- the static-link update preserves norm on a periodic lattice;
- the state/link transformation law gives local gauge covariance:
  `U_G(psi^Omega, A^Omega) = Omega U_G(psi, A)`;
- identity and pure-gauge backgrounds have zero Wilson action, while a
  deterministic non-pure background has non-zero Wilson action;
- normalized Wilson traces are gauge-invariant;
- weak-link transport agrees with the first-order covariant-derivative
  expansion with quadratic residual scaling under `epsilon -> epsilon/2`;
- the gauged step is JIT-compatible;
- no dynamic gauge-field update, Higgs, Yukawa, flavor, boundary, or
  recirculation rule is introduced.

Verdict:

```text
QCA_SMV0_STAGE2_STATIC_SM_GAUGE_PASS
```

## Stage 3 - Pure Dynamic SM Gauge Fields

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- link momenta have shape `(nx, ny, nz, 8, 12)` in the same SM generator basis
  as the Stage 2 links;
- projection from algebra matrices to SM coordinates round-trips momenta;
- momenta transform by target-site adjoint gauge action;
- identity and pure-gauge links have zero Wilson force;
- deterministic non-flat links have nonzero Wilson force;
- momentum updates keep links unitary;
- the pure-gauge leapfrog step approximately conserves
  `K(P) + beta S_W(U)` at small step size;
- leapfrog reversibility holds under momentum flip;
- electric divergence is zero for zero momenta and transforms covariantly;
- zero-momentum pure-gauge configurations preserve the Gauss constraint under
  one leapfrog step;
- weak-field BCC plaquette holonomies linearize as `W = I + epsilon F` with
  quadratic residual scaling;
- Wilson action matches the weak-field Yang-Mills density
  `epsilon^2 * mean(0.5 Tr(F^dag F)/32)`;
- the no-backreaction fermion/gauge wrapper preserves spectator fermion norm;
- the leapfrog update is JIT-compatible;
- no matter backreaction, Higgs, Yukawa, flavor, boundary, or recirculation
  rule is introduced.

Verdict:

```text
QCA_SMV0_STAGE3_DYNAMIC_SM_GAUGE_PASS
```

## Stage 4 - Local Higgs/Yukawa Collision

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the Higgs field has shape `(nx, ny, nz, 2)`;
- a constant unitary-gauge helper returns `H=(0,v/sqrt(2))`;
- the conjugate doublet `H_tilde=i sigma_2 H^*` is implemented;
- the local one-generation Yukawa matrix is Hermitian;
- in unitary gauge, `H_tilde` opens the up/neutrino doors and `H` opens the
  down/electron doors, while the wrong weak components vanish;
- the local collision is the exact finite rotation
  `exp(-i step_size beta Y(H))`;
- zero step and zero Higgs are identity controls;
- the local collision preserves fermion norm to numerical precision;
- the collision creates a chirality-flipped component from a left seed;
- a constant Higgs background gives the expected small-momentum massive
  dispersion `E(k) ~= sqrt(|k|^2 + m^2)`;
- the collision is JIT-compatible;
- no dynamic Higgs potential, matter backreaction, flavor/FN recirculation,
  boundary rule, or CP rule is introduced.

Verdict:

```text
QCA_SMV0_STAGE4_HIGGS_YUKAWA_PASS
```

## Stage 5 - FN Recirculation Paths

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the family register dimension is `3`;
- the FN attenuation `lambda` is an explicit simulator input, with both
  empirical Wolfenstein and shear-candidate modes supported;
- a two-state beam splitter
  `[[sqrt(1-lambda^2), -lambda], [lambda, sqrt(1-lambda^2)]]` is unitary;
- a finite hidden path of length `n` is a product of local beam splitters;
- the endpoint transfer amplitude of that path is `lambda^n`;
- quark path lengths are generated from explicit integer charges:
  `n^u_ij=Q_i+U_j` and `n^d_ij=Q_i+D_j`;
- the default simulator charges are `Q=(3,2,0)`, `U=(5,2,0)`,
  `D=(1,0,0)`;
- the diagonal scalings reproduce the standard FN orders
  `m_u:m_c:m_t ~ lambda^8:lambda^4:1` and
  `m_d:m_s:m_b ~ lambda^4:lambda^2:1`;
- the left-frame scaling matrix is `lambda^abs(Q_i-Q_j)`, giving the
  Wolfenstein hierarchy pattern;
- effective Yukawa matrices are built as
  `Y_ij=c_ij lambda^(Q_i+R_j)`;
- singular masses and the CKM-like left-frame mismatch are read from the same
  generated matrices;
- the generated CKM-like matrix is unitary to numerical precision;
- the effective Yukawa generator is JIT-compatible;
- no claim is made that the BCC bulk derives the charges, `lambda`, or the
  order-one coefficients;
- no center-holonomy CP, matter backreaction, boundary rule, or dynamic Higgs
  field is introduced.

Verdict:

```text
QCA_SMV0_STAGE5_FN_RECIRCULATION_PASS
```

## Stage 6 - Center-Holonomy CP

Pass only if:

- the only upstream runtime imports are from local `qca_smv0` modules;
- center phases are values in the `SU(3)_c` center:
  `omega=exp(2 pi i / 3)`;
- the center-power matrices are explicit simulator inputs;
- all center phases have unit modulus and satisfy `phase^3=1`;
- center phases decorate only the FN order-one coefficients and preserve their
  magnitudes;
- quark Yukawas and antiparticle Yukawas are related by complex conjugation;
- quark and antiquark singular masses agree to numerical precision;
- the CKM-like matrix is still the left-frame mismatch of the generated up/down
  Yukawas and remains unitary;
- the center-decorated matrices give a nonzero CKM Jarlskog quartet;
- the all-real coefficient control gives zero Jarlskog;
- the commutator trace `Im Tr([Yu Yu^dag,Yd Yd^dag]^3)` is nonzero for the
  center-decorated matrices and zero for the all-real control;
- the center-holonomy phase map is JIT-compatible;
- no claim is made that the BCC bulk derives the center powers;
- no matter backreaction, boundary rule, dynamic Higgs field, or full
  three-family Higgs collision is introduced.

Verdict:

```text
QCA_SMV0_STAGE6_CENTER_HOLONOMY_CP_PASS
```

## Stage 7 - Three-Family Higgs/Yukawa Collision

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the family-extended fermion state has shape `(nx, ny, nz, 4, 32, 3)`;
- the local internal/family register has dimension `32 x 3 = 96`;
- the Stage 5/6 generated quark Yukawa matrices are embedded into the local
  Higgs/Yukawa matrix, not re-entered as scalars;
- the local three-family Yukawa matrix is Hermitian;
- in unitary gauge, the up block extracted from the local matrix reproduces
  the generated up Yukawa matrix and the down block reproduces the generated
  down Yukawa matrix;
- wrong weak-component doors vanish as in Stage 4;
- the CKM-like left-frame mismatch computed from the embedded blocks agrees
  with the mismatch computed from the generated matrices, up to singular-vector
  phase conventions;
- the collision is the exact local unitary
  `exp(-i step_size beta Y_family(H))`;
- zero step and zero Higgs are identity controls;
- the collision preserves norm to numerical precision;
- the collision creates a chirality-flipped component from a left family seed;
- the family collision is JIT-compatible;
- no dynamic Higgs potential, matter backreaction, boundary rule, or derivation
  of charges/phases is introduced.

Verdict:

```text
QCA_SMV0_STAGE7_FAMILY_HIGGS_YUKAWA_PASS
```

## Stage 8 - Dynamic Higgs-Field Evolution

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- Higgs electroweak generators act on the Higgs doublet with
  `SU(2)_L x U(1)_Y` anti-Hermitian generators;
- finite Higgs electroweak BCC links have shape `(nx, ny, nz, 8, 2, 2)`;
- finite Higgs links are unitary;
- Higgs fields and momenta have shape `(nx, ny, nz, 2)`;
- the Higgs Hamiltonian includes momentum kinetic energy, a gauge-covariant BCC
  gradient density, and a local quartic potential;
- the unitary-gauge vacuum has zero force with identity links;
- pure-gauge transformed vacuum fields have zero covariant-gradient energy;
- the Higgs force transforms covariantly under site-local electroweak gauges;
- the Higgs Hamiltonian density is gauge-invariant under simultaneous field,
  momentum, and link transforms;
- the no-fermion Higgs leapfrog step is reversible under momentum flip;
- the leapfrog step has small Hamiltonian drift at small step size;
- the leapfrog step is JIT-compatible;
- no fermion backreaction, boundary rule, quantized scalar register, or
  derivation of Higgs potential parameters is introduced.

Verdict:

```text
QCA_SMV0_STAGE8_HIGGS_DYNAMICS_PASS
```

## Stage 9 - Gauge-Higgs Backreaction

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- Higgs electroweak link momenta have shape `(nx, ny, nz, 8, 4)`;
- projection from Higgs algebra matrices to electroweak coordinates
  round-trips momenta;
- Higgs link momenta transform by target-site adjoint gauge action;
- the Higgs gauge current is computed as the left-trivialized derivative of
  the gauge-covariant Higgs gradient energy with respect to each BCC link;
- the Higgs link force vanishes for a covariantly constant vacuum and is
  nonzero for a deterministic non-vacuum/nonflat field;
- the Higgs link force transforms covariantly under site-local electroweak
  gauges;
- the local Higgs charge density is implemented and transforms covariantly;
- the Higgs Gauss diagnostic is electric divergence minus Higgs charge and
  transforms covariantly;
- the vacuum with zero field and link momenta has zero Higgs Gauss residual;
- Higgs electroweak momenta embed into the full SM 12-coordinate momentum
  layout with zeros on color and components on generators `8..11`;
- the coupled no-fermion Higgs/gauge leapfrog keeps links unitary;
- the coupled leapfrog is reversible under simultaneous momentum flips;
- the coupled Hamiltonian density has small drift at small step size;
- the coupled leapfrog is JIT-compatible;
- no fermion backreaction, boundary rule, quantized scalar register, dynamic
  full-SM gauge update sourced by fermions, or derivation of Higgs potential
  parameters is introduced.

Verdict:

```text
QCA_SMV0_STAGE9_GAUGE_HIGGS_BACKREACTION_PASS
```
