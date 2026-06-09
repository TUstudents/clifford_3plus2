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

## Stage 10 - Local Fermion/Higgs Backreaction

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the local source uses the same three-family Yukawa matrix as Stage 7;
- the local Yukawa energy density is
  `E_Y=Re psi^dagger beta Y(H) psi`;
- the Higgs source is the complex-field force `-dE_Y/dH*`, computed from
  real/imaginary automatic differentiation;
- zero fermion state gives zero Higgs source;
- a deterministic Yukawa bilinear source gives nonzero Higgs force;
- the local Yukawa-door gauge convention is explicit: the source covariance
  audit uses physical right-handed electroweak hypercharges for the Yukawa
  doors, while the Stage 2 transport carrier remains the left-handed conjugate
  chiral-16 convention;
- Yukawa energy is invariant under the local Yukawa-door electroweak gauge;
- the Higgs source transforms covariantly under the same gauge;
- the Higgs momentum kick is reversible under `step_size -> -step_size`;
- the collision-plus-kick wrapper preserves fermion norm to float32 precision;
- the source and kick are JIT-compatible;
- no BCC streaming fermion current, full gauge-link fermion backreaction,
  quantized scalar register, boundary rule, or derivation of the Yukawa inputs
  is introduced.

Verdict:

```text
QCA_SMV0_STAGE10_FERMION_HIGGS_BACKREACTION_PASS
```

## Stage 11 - BCC Streaming Fermion Gauge Current

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the current uses the same BCC pull-convention gauged Dirac transport as
  Stage 2;
- the local streaming bilinear is
  `E_stream=Re sum_x,h psi[x]^dag D_h U_h[x] psi[x+h]`;
- the fermion gauge current is the left-trivialized derivative of
  `E_stream` with respect to each BCC link;
- zero fermion state gives zero current;
- a deterministic fermion state gives nonzero current;
- `E_stream` is invariant under site-local SM gauge transforms;
- the current transforms by target-site adjoint action;
- local fermion charge density is implemented and transforms covariantly;
- the fermion Gauss diagnostic is electric divergence minus fermion charge and
  transforms covariantly;
- zero state with zero momenta has zero fermion Gauss residual;
- the fermion-current momentum kick is reversible under
  `step_size -> -step_size`;
- kick-then-transport keeps links unitary and preserves spectator fermion norm
  to float32 precision;
- the current and kick are JIT-compatible;
- no family-summed current, full Higgs+fermion sourced SM integrator,
  quantized gauge/scalar registers, boundary rule, or derivation of simulator
  inputs is introduced.

Verdict:

```text
QCA_SMV0_STAGE11_FERMION_GAUGE_CURRENT_PASS
```

## Stage 12 - Coupled Sourced SM Gauge Tick

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- SM transport links and Higgs electroweak links are carried side by side as
  two representations of one 12-coordinate gauge-momentum field;
- Higgs electroweak site/link coordinates embed into the full SM generator
  layout only on components `8..11`;
- the sourced SM link force is the sum of the Stage 3 Wilson force, the Stage 9
  embedded Higgs gauge force, and the Stage 11 BCC streaming fermion current;
- zero fermion state, vacuum Higgs field, identity SM links, and identity Higgs
  links give zero sourced link force to float32 precision;
- deterministic nonzero fields give a nonzero sourced link force;
- the sourced link force transforms by target-site adjoint action under a
  shared electroweak gauge transformation;
- the full sourced Gauss diagnostic is electric divergence minus fermion charge
  minus embedded Higgs charge and transforms covariantly;
- zero fermion state, zero gauge momenta, vacuum Higgs field, and zero Higgs
  momenta have zero sourced Gauss residual;
- the sourced momentum kick is reversible under `step_size -> -step_size`;
- updating SM and Higgs-representation links from the same momentum field keeps
  both link fields unitary;
- the coupled tick transports the fermion through the sourced SM links,
  advances the Higgs field with the Higgs force, and preserves fermion norm to
  float32 precision;
- the coupled tick is JIT-compatible;
- no family-summed BCC streaming current, full Yukawa/Higgs source merge,
  quantized gauge/scalar register, boundary rule, or derivation of simulator
  inputs is introduced.

Verdict:

```text
QCA_SMV0_STAGE12_SOURCED_SM_TICK_PASS
```

## Stage 13 - Family-Summed BCC Fermion Gauge Current

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the family state layout is the Stage 7 layout `(nx, ny, nz, 4, 32, 3)`;
- SM gauge links remain family-blind and have the same shape as Stage 2;
- embedding one Stage 11 state into one family and extracting it round-trips;
- with only one occupied family, the family streaming energy, current, charge,
  and transport reduce to the Stage 11 one-carrier results to float32
  precision;
- the family streaming bilinear is
  `E_stream=Re sum_x,h,f psi_f[x]^dag D_h U_h[x] psi_f[x+h]`;
- the family-summed current is the left-trivialized derivative of that
  bilinear with respect to each BCC link;
- zero family state gives zero current;
- deterministic family state gives nonzero current;
- family streaming energy is invariant under site-local SM gauge transforms;
- family current transforms by target-site adjoint action;
- family-summed local fermion charge transforms covariantly;
- the family fermion Gauss diagnostic is electric divergence minus
  family-summed fermion charge and transforms covariantly;
- the family-current momentum kick is reversible under
  `step_size -> -step_size`;
- kick-then-transport keeps links unitary and preserves family-state spectator
  norm to float32 precision;
- the family current and transport are JIT-compatible;
- no family-sourced Higgs/gauge production tick, Yukawa-source merge, boundary
  rule, quantized register, or derivation of simulator inputs is introduced.

Verdict:

```text
QCA_SMV0_STAGE13_FAMILY_GAUGE_CURRENT_PASS
```

## Stage 14 - Family-Sourced SM Gauge Tick

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the state layout is the Stage 7 family layout `(nx, ny, nz, 4, 32, 3)`;
- SM transport links and Higgs electroweak links are still carried as two
  representations of one 12-coordinate SM momentum field;
- the family-sourced link force is the sum of the Stage 3 Wilson force, the
  Stage 9 embedded Higgs gauge force, and the Stage 13 family-summed BCC
  fermion current;
- one occupied family reduces the family-sourced force, sourced Gauss
  diagnostic, and tick to the Stage 12 one-carrier sourced tick to float32
  precision;
- zero family state, vacuum Higgs field, identity SM links, and identity Higgs
  links give zero sourced link force to float32 precision;
- deterministic nonzero family/Higgs/gauge fields give a nonzero sourced link
  force;
- the family-sourced link force transforms by target-site adjoint action under
  a shared electroweak gauge transformation;
- the full family-sourced Gauss diagnostic is electric divergence minus
  family-summed fermion charge minus embedded Higgs charge and transforms
  covariantly;
- zero family state, zero gauge momenta, vacuum Higgs field, and zero Higgs
  momenta have zero sourced Gauss residual;
- the family-sourced momentum kick is reversible under
  `step_size -> -step_size`;
- updating SM and Higgs-representation links from the same momentum field keeps
  both link fields unitary;
- the coupled tick transports the family state through the sourced SM links,
  advances the Higgs field with the Higgs force, and preserves family-state
  norm to float32 precision;
- the family-sourced tick is JIT-compatible;
- no local Yukawa/Higgs source merge, boundary rule, quantized register, or
  derivation of simulator inputs is introduced.

Verdict:

```text
QCA_SMV0_STAGE14_FAMILY_SOURCED_TICK_PASS
```

## Stage 15 - Full Family Production Tick

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the state layout remains the Stage 7 family layout
  `(nx, ny, nz, 4, 32, 3)`;
- the tick composes the Stage 14 family-sourced gauge tick with the Stage 10
  local three-family Yukawa/Higgs source and the exact Stage 7 family Yukawa
  collision;
- setting all quark and lepton Yukawa matrices to zero reduces the full
  production tick to the Stage 14 family-sourced gauge tick to float32
  precision;
- the Higgs momentum force is the sum of the Stage 8 Higgs force and the
  Stage 10 local Yukawa Higgs source;
- deterministic nonzero family/Higgs fields give a nonzero Yukawa source;
- zero family state and vacuum Higgs give zero Yukawa source;
- the Higgs momentum source kick is reversible for frozen fields under
  `step_size -> -step_size`;
- the production tick preserves family-state norm to float32 precision and
  keeps SM and Higgs-representation links unitary;
- the default production tick differs from the Stage 14 tick in both the
  local family state and Higgs momenta, so the Yukawa merge is active;
- the production tick is JIT-compatible;
- the Stage 10 local Higgs-door gauge convention remains explicitly separate
  from the Stage 14 transport/current gauge convention;
- no boundary rule, quantized register, or derivation of simulator inputs is
  introduced.

Verdict:

```text
QCA_SMV0_STAGE15_FAMILY_PRODUCTION_TICK_PASS
```

## Stage 16 - Gauge-Convention Bridge Audit

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the Stage 2 transport electroweak generators and Stage 10 physical
  Yukawa-door electroweak generators are both explicit;
- exponentiating the Stage 16 Yukawa-door generators reproduces the Stage 10
  finite Yukawa-door gauge helper;
- the transport and Yukawa-door generators agree exactly on duplicated left
  doublet labels `Q,L`;
- both conventions have zero `SU(2)_L` action on singlet labels;
- on duplicated right-singlet labels, the transport and Yukawa-door
  hypercharges are charge-conjugate: `Y_transport + Y_yukawa = 0`;
- the full electroweak generator sets differ nontrivially;
- the sorted hypercharge spectra differ, proving no unitary similarity on the
  fixed 32-component internal carrier can identify the two conventions;
- the local Yukawa energy is gauge-covariant in the physical Yukawa-door
  convention;
- the same local Yukawa energy is not gauge-invariant if transformed with the
  Stage 2 transport electroweak convention;
- the energy-residual audit is JIT-compatible;
- the stage does not claim to unify the conventions, introduce boundary rules,
  quantized registers, or derive simulator inputs.

Verdict:

```text
QCA_SMV0_STAGE16_GAUGE_CONVENTION_BRIDGE_PASS
```

## Stage 17 - Antiunitary Singlet Bridge

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the physical-right SM generators are built from the Stage 2 transport
  generators by a projector rule:
  `T_phys = T_transport` on left doublets and
  `T_phys = conj(T_transport)` on right singlets;
- the bridge is exact on generators: zero residual on the left-linear and
  right-antilinear projector blocks;
- the electroweak slice of the physical-right generators reproduces the
  Stage 10 / Stage 16 physical Yukawa-door generators;
- finite physical-right site gauges equal the projected finite bridge
  `P_L G_transport P_L + P_R conj(G_transport) P_R`;
- finite bridge gauges are unitary to float32 precision;
- the physical-right generators remain nontrivially different from the Stage 2
  transport generators, so the bridge is not a unitary identification;
- the full `SU(3)_c x SU(2)_L x U(1)_Y` physical-right gauge restores local
  Yukawa energy covariance;
- the unbridged Stage 2 transport gauge remains non-invariant for the same
  local Yukawa energy;
- the full bridge energy-residual audit is JIT-compatible;
- the stage does not claim a microscopic BCC derivation, boundary rule,
  quantized register, or rewrite of the production tick carrier.

Verdict:

```text
QCA_SMV0_STAGE17_ANTIUNITARY_BRIDGE_PASS
```

## Stage 18 - Physical-Right Bridged Transport

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- Stage 2 transport-convention finite BCC links can be bridged edgewise to
  physical-right links by
  `U_phys=P_L U_transport P_L + P_R conj(U_transport) P_R`;
- identity links are unchanged by the bridge;
- finite bridged links remain unitary to float32 precision;
- gauge-transforming transport links and then bridging them agrees with
  bridging first and transforming by the physical-right site gauge;
- the family BCC Dirac transport through identity bridged links reduces to the
  existing identity-link transport;
- physical-right family transport is covariant under the bridged site gauge;
- nontrivial physical-right transport differs from the unbridged Stage 13
  transport convention, so the bridge is active;
- physical-right transport preserves family-state norm to float32 precision;
- the transport kernel is JIT-compatible;
- no bridged current, sourced tick, production tick rewrite, boundary rule,
  quantized register, or derivation of simulator inputs is introduced.

Verdict:

```text
QCA_SMV0_STAGE18_PHYSICAL_RIGHT_TRANSPORT_PASS
```

## Stage 19 - Physical-Right Fermion Gauge Current

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the Stage 13 family streaming bilinear is lifted to the Stage 18
  physical-right bridged transport carrier;
- current coordinates remain the shared 12 transport-coordinate labels used by
  the Stage 2 gauge links and Stage 3 link momenta;
- the physical-right streaming energy is real to float32 precision;
- the zero family state gives zero physical-right fermion current;
- deterministic nonzero family/gauge fields give a nonzero physical-right
  current;
- the physical-right current differs nontrivially from the unbridged Stage 13
  transport-convention family current;
- streaming energy, current, charge density, and Gauss diagnostic are
  covariant under the bridged physical-right site gauge;
- the Gauss diagnostic is electric divergence minus physical-right
  family-charge density;
- the physical-right current momentum kick is reversible under
  `step_size -> -step_size`;
- the kick-then-transport wrapper updates transport links, preserves bridged
  link unitarity, and preserves spectator family-state norm to float32
  precision;
- the current and physical-right transport kernels are JIT-compatible;
- no sourced tick rewrite, production tick rewrite, boundary rule, quantized
  register, or derivation of simulator inputs is introduced.

Verdict:

```text
QCA_SMV0_STAGE19_PHYSICAL_RIGHT_CURRENT_PASS
```

## Stage 20 - Physical-Right Sourced SM Tick

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the Stage 14 sourced gauge tick is lifted from the transport-convention
  family current to the Stage 19 physical-right family current;
- the pure Wilson force and embedded Higgs gauge force remain the existing
  transport-coordinate source terms;
- the sourced link force is
  `Wilson + embedded Higgs gauge force + physical-right family current`;
- the physical-right sourced link force differs nontrivially from the Stage 14
  transport-convention family-sourced force;
- the physical-right sourced Gauss diagnostic is physical-right electric
  divergence minus physical-right family charge minus embedded Higgs charge;
- zero family state, vacuum Higgs field, identity SM links, and identity Higgs
  links give zero sourced force to float32 precision;
- deterministic nonzero family/Higgs/gauge fields give a nonzero sourced
  force;
- sourced force and sourced Gauss transform covariantly under the bridged
  physical-right site gauge;
- zero family state, zero gauge momenta, vacuum Higgs field, and zero Higgs
  momenta have zero sourced Gauss residual;
- the physical-right sourced momentum kick is reversible under
  `step_size -> -step_size`;
- the coupled tick transports the family state through physical-right bridged
  links, advances the Higgs field, keeps SM and Higgs-representation links
  unitary, and preserves family-state norm to float32 precision;
- the physical-right sourced tick differs nontrivially from the Stage 14
  transport-convention sourced tick;
- the physical-right sourced tick is JIT-compatible;
- no production tick rewrite, boundary rule, quantized register, or derivation
  of simulator inputs is introduced.

Verdict:

```text
QCA_SMV0_STAGE20_PHYSICAL_RIGHT_SOURCED_TICK_PASS
```

## Stage 21 - Physical-Right Production Tick

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the Stage 15 production tick ordering is lifted onto the Stage 20
  physical-right sourced gauge tick;
- the local Yukawa/Higgs source remains the explicit simulator input from
  Stages 7/10/15;
- setting all quark and lepton Yukawa matrices to zero reduces the
  physical-right production tick to the Stage 20 physical-right sourced tick
  to float32 precision;
- the production Higgs force is the sum of the Stage 8 Higgs force and the
  Stage 10 local Yukawa Higgs source;
- deterministic nonzero family/Higgs fields give a nonzero Yukawa source;
- zero family state and vacuum Higgs give zero Yukawa source;
- the Higgs momentum source kick is reversible for frozen fields under
  `step_size -> -step_size`;
- the production tick uses symmetric local half-collision,
  physical-right BCC family transport, and a second local half-collision;
- the default physical-right production tick differs from the Stage 20
  physical-right sourced tick in both the local family state and Higgs
  momenta, so the Yukawa merge is active;
- the default physical-right production tick differs nontrivially from the
  Stage 15 transport-convention production tick, so the carrier rewrite is
  active;
- the production tick preserves family-state norm to float32 precision and
  keeps SM and Higgs-representation links unitary;
- the production tick is JIT-compatible;
- no boundary rule, quantized register, or derivation of simulator inputs is
  introduced.

Verdict:

```text
QCA_SMV0_STAGE21_PHYSICAL_RIGHT_PRODUCTION_TICK_PASS
```

## Stage 22 - Physical-Right Production Rollout

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the Stage 21 physical-right production tick is wrapped as one explicit
  rollout state carrying family fermions, Higgs fields/momenta, SM links and
  momenta, and Higgs-representation links;
- one-step rollout agrees with a direct Stage 21 tick to float32 precision;
- sparse observations are recorded through the shared `sim.runner`
  loop/scan interface, not a sidecar-specific recording scheme;
- loop and scan recorded rollouts agree;
- recorded observables include family norm, Higgs norm, Higgs momentum norm,
  SM momentum norm, and SM/Higgs link-unitarity residuals;
- the multi-tick rollout keeps family norm drift and SM/Higgs link unitarity
  controlled on the deterministic small certificate state;
- the default production rollout remains distinguishable from a zero-Yukawa
  rollout, so production is active under iteration;
- all recorded states and observables are finite;
- no boundary rule, quantized register, performance benchmark, or derivation
  of simulator inputs is introduced.

Verdict:

```text
QCA_SMV0_STAGE22_PHYSICAL_RIGHT_PRODUCTION_ROLLOUT_PASS
```
