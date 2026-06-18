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
- the simulator internal register has shape `32`, interpreted as a doubled
  practical representation of one SM chiral-16 label set.  The compact
  theory-side one-generation matter carrier is Spin(10) chiral `C^16`; the
  separate Dirac axis is spacetime spin for BCC transport;
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
- the full sector is an explicit direct-sum hidden unitary network containing
  one local path for every `L_i -> R_j` entry;
- the FN power matrix is read from source-to-sink transfer amplitudes of that
  hidden network, not inserted as a bare power table;
- visible left-family channels enter the hidden path sources, propagate through
  the unitary recirculation network, and exit through right-family channels;
- the visible transfer has an exact finite unitary dilation, so the nonunitary
  Yukawa block can be embedded in a local simulator gate;
- a local FN collision primitive applies that dilation to visible-plus-auxiliary
  family states, preserving norm while exposing the normalized visible transfer
  block;
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
  `Y_ij=c_ij A_ij`, where `A_ij` is the measured hidden-network transfer;
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
- the local simulator internal/family register has dimension `32 x 3 = 96`
  before the separate four-component spacetime Dirac spin axis;
- the Stage 5/6 generated quark Yukawa matrices are embedded into the local
  Higgs/Yukawa matrix, not re-entered as scalars;
- the default quark Yukawa source is explicitly reconstructed from
  center-holonomy coefficients and the Stage 5 visible-hidden-visible FN
  recirculation readout before embedding;
- finite FN unitary dilations expose the Higgs-scaled visible transfers for
  the up/down quark doors;
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
- the default quark source is the recirculation-backed FN source from Stage 7;
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

## Stage 23 - Physical-Right Production Gauss Monitor

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the monitor reads the existing physical-right sourced Gauss diagnostic
  `electric divergence - physical-right family charge - embedded Higgs charge`
  on Stage 21/22 production rollout states;
- the zero family state, unitary-gauge Higgs vacuum, zero momenta, identity SM
  links, and identity Higgs links have zero initial Gauss norm;
- that zero-source vacuum control remains zero-Gauss under a short production
  rollout;
- deterministic nonzero production fields give a finite, nonzero Gauss history;
- the monitored rollout keeps family norm and SM/Higgs link unitarity
  controlled;
- the default production rollout is distinguishable from a zero-Yukawa rollout
  in the final Gauss observable;
- all recorded Gauss-monitor observables are finite;
- no Gauss projection, boundary rule, quantized register, performance
  benchmark, or derivation of simulator inputs is introduced.

Verdict:

```text
QCA_SMV0_STAGE23_PHYSICAL_RIGHT_PRODUCTION_GAUSS_PASS
```

## Stage 24 - Physical-Right Production Energy Monitor

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the monitor reads only energy functions already introduced in earlier stages:
  pure SM gauge Hamiltonian, Higgs Hamiltonian, physical-right streaming
  bilinear, and local three-family Yukawa energy;
- the monitored total energy is the explicit sum of those four components;
- the zero family state, unitary-gauge Higgs vacuum, zero momenta, identity SM
  links, and identity Higgs links have zero monitored total energy to
  numerical precision;
- deterministic nonzero production fields give finite nonzero component
  energies;
- the short production rollout has controlled monitored-total drift on the
  deterministic certificate state;
- the default production rollout is distinguishable from a zero-Yukawa rollout
  at the monitored total-energy level;
- family norm remains controlled during the monitored rollout;
- all recorded energy-monitor observables are finite;
- no new dynamics, conservation claim, boundary rule, quantized register,
  performance benchmark, Gauss projection, or derivation of simulator inputs is
  introduced.

Verdict:

```text
QCA_SMV0_STAGE24_PHYSICAL_RIGHT_PRODUCTION_ENERGY_PASS
```

## Stage 25 - Physical-Right Production Variational Audit

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the production Higgs force is normalized as `-dE/dphi*` for the Stage 24
  Higgs Hamiltonian density plus local Yukawa energy;
- the physical-right sourced link force decomposes exactly into Wilson force,
  embedded Higgs gauge force, and physical-right family current;
- the physical-right production Higgs force decomposes exactly into Higgs
  Hamiltonian force and local Yukawa Higgs source;
- a selected color-link coordinate agrees with a central finite difference of
  the Wilson plus physical-right streaming energy;
- a selected complex Higgs component agrees with real and imaginary central
  finite differences of the Higgs plus Yukawa energy using the
  `force=-dE/dphi*` convention;
- the zero-source production vacuum has negligible link and Higgs force norms;
- deterministic nonzero production fields have nonzero link and Higgs force
  norms;
- all variational residuals are finite;
- no new dynamics, boundary rule, conservation claim, Gauss projection,
  quantized register, performance benchmark, or derivation of simulator inputs
  is introduced.

Verdict:

```text
QCA_SMV0_STAGE25_PHYSICAL_RIGHT_PRODUCTION_VARIATIONAL_PASS
```

## Stage 26 - Physical-Right Production Refinement Limitation Audit

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the audit compares two deterministic production rollouts at the same physical
  time, using `dt` and `dt/2` with doubled step count;
- the compared histories use the Stage 24 monitored total energy;
- the fixed physical times match exactly to numerical precision;
- the coarse rollout has controlled monitored total-energy drift;
- the refined rollout remains finite, link-unitary, and family-norm
  controlled;
- the zero-source production vacuum keeps zero monitored total-energy drift
  under the refined rollout;
- the audit explicitly detects that the current hybrid production tick does
  not improve monitored total-energy drift under timestep halving;
- the detected limitation is documented as an integrator limitation, not as a
  physics conservation result;
- no new dynamics, boundary rule, conservation claim, Gauss projection,
  quantized register, performance benchmark, or derivation of simulator inputs
  is introduced.

Verdict:

```text
QCA_SMV0_STAGE26_PHYSICAL_RIGHT_PRODUCTION_REFINEMENT_LIMITATION_PASS
```

## Stage 27 - Physical-Right Production Adjoint Limitation Audit

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the physical-right BCC family transport has an explicit adjoint step;
- the adjoint transport restores a deterministic state to float32 precision;
- the local family Yukawa collision is inverted by `step_size -> -step_size`
  on frozen Higgs fields;
- the frozen fermion substage of the production tick
  `(old-Higgs half collision, updated-link transport, updated-Higgs half
  collision)` has an explicit adjoint and restores the input state;
- the frozen fermion substage preserves norm to float32 precision;
- a full production tick followed by another production tick with negative
  timestep is explicitly detected as **not** an inverse of the full map;
- both the forward and naive negative-step rollouts keep SM and Higgs links
  unitary;
- the detected limitation is documented as a missing full adjoint/reversible
  production integrator, not as a failure of the frozen fermion substage;
- no new dynamics, boundary rule, conservation claim, Gauss projection,
  quantized register, performance benchmark, or derivation of simulator inputs
  is introduced.

Verdict:

```text
QCA_SMV0_STAGE27_PHYSICAL_RIGHT_PRODUCTION_ADJOINT_LIMITATION_PASS
```

## Stage 28 - Explicit Physical-Right Production Inverse

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the inverse reconstructs the half-step momenta from the final production
  forces, runs the sourced link update backward, rewinds the Higgs field, and
  applies the frozen fermion-stage adjoint from Stage 27;
- a deterministic forward production tick followed by the explicit inverse
  restores all rollout-state fields to float32 precision;
- the restored state, advanced again with the forward tick, reproduces the
  original final state;
- the explicit inverse improves over the naive negative-timestep production
  tick by a large margin;
- the restored family norm and SM/Higgs link unitarity remain controlled;
- the inverse is compatible with JIT;
- the result is documented as an exact inverse helper for the current discrete
  map, not as an energy-conservation, timestep-convergence, boundary,
  Gauss-projection, quantized-register, performance, or microscopic-input
  derivation claim.

Verdict:

```text
QCA_SMV0_STAGE28_PHYSICAL_RIGHT_PRODUCTION_EXPLICIT_INVERSE_PASS
```

## Stage 29 - Physical-Right Production Trajectory Reversibility

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the Stage 28 explicit inverse can be composed into a multi-step inverse
  rollout;
- a deterministic multi-step production rollout followed by the inverse
  rollout restores the initial state to float32 precision;
- the restored initial state, advanced again for the same number of steps,
  reproduces the original final state;
- each intermediate inverse restore matches the stored forward trajectory to
  float32 precision;
- the explicit inverse trajectory improves strongly over a multi-step naive
  negative-timestep rollout;
- forward and inverse trajectories keep family norm and SM/Higgs link
  unitarity controlled;
- the inverse rollout is compatible with JIT for fixed step count;
- the result is documented as a trajectory-level audit of the current
  discrete map, not as an energy-conservation, timestep-convergence, boundary,
  Gauss-projection, quantized-register, performance, or microscopic-input
  derivation claim.

Verdict:

```text
QCA_SMV0_STAGE29_PHYSICAL_RIGHT_PRODUCTION_TRAJECTORY_REVERSIBILITY_PASS
```

## Stage 30 - Physical-Right Production Loschmidt Echo

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- a short deterministic production trajectory is advanced and then exactly
  rewound with the Stage 29 inverse rollout;
- a small local final-time SM momentum perturbation has a finite nonzero echo
  on the restored initial surface;
- doubling that final-time perturbation doubles the echo to local linear
  precision;
- the unperturbed base roundtrip remains at float32 precision;
- the perturbed inverse path keeps SM/Higgs link unitarity controlled;
- the result is documented as a Loschmidt/stability diagnostic for the current
  discrete production map, not as a new dynamics rule, energy-conservation
  claim, timestep-convergence claim, boundary rule, Gauss projection,
  quantized register, performance benchmark, or microscopic-input derivation.

Verdict:

```text
QCA_SMV0_STAGE30_PHYSICAL_RIGHT_PRODUCTION_LOSCHMIDT_ECHO_PASS
```

## Stage 31 - Physical-Right Production Tangent Echo

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- two independent final-time perturbations are applied to a short production
  trajectory: one SM momentum coordinate and one Higgs momentum coordinate;
- each perturbation has a finite nonzero inverse echo on the restored initial
  surface;
- the combined perturbation echo matches the sum of the two separate echoes to
  local finite-difference precision;
- the unperturbed base roundtrip remains at float32 precision;
- the combined perturbed inverse state keeps SM/Higgs link unitarity
  controlled;
- the result is documented as a finite tangent-response diagnostic for the
  current discrete production map, not as a new dynamics rule,
  energy-conservation claim, timestep-convergence claim, boundary rule, Gauss
  projection, quantized register, performance benchmark, or microscopic-input
  derivation.

Verdict:

```text
QCA_SMV0_STAGE31_PHYSICAL_RIGHT_PRODUCTION_TANGENT_ECHO_PASS
```

## Stage 32 - Physical-Right Production Echo Gram

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- at least three independent final-time perturbations are pulled back through
  the inverse production trajectory;
- the initial-surface echo Gram matrix is symmetric to numerical precision;
- all echo norms are finite and nonzero;
- the Gram matrix has a positive minimum eigenvalue and finite condition
  number on the certificate trajectory;
- off-diagonal echo correlations remain bounded, so the chosen local probes do
  not collapse to the same tangent direction;
- all inverse-pulled echo states keep SM/Higgs link unitarity controlled;
- the result is documented as a finite local tangent-metric diagnostic for the
  current discrete production map, not as a continuum stability theorem, new
  dynamics rule, energy-conservation claim, timestep-convergence claim,
  boundary rule, Gauss projection, quantized register, performance benchmark,
  or microscopic-input derivation.

Verdict:

```text
QCA_SMV0_STAGE32_PHYSICAL_RIGHT_PRODUCTION_ECHO_GRAM_PASS
```

## Stage 33 - Physical-Right Production Echo-Gram Scale Stability

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the Stage 32 echo-Gram diagnostic is evaluated at two perturbation sizes,
  `epsilon` and `2 epsilon`;
- echo norms scale linearly with perturbation size to local finite-difference
  precision;
- Gram eigenvalues scale quadratically with perturbation size to local
  finite-difference precision;
- dimensionless Gram data, including condition number and off-diagonal
  correlation, remain stable between the two perturbation sizes;
- both base roundtrips and all inverse-pulled links remain controlled;
- the result is documented as a finite scale-stability audit for the local
  echo-Gram diagnostic, not as a continuum stability theorem, new dynamics
  rule, energy-conservation claim, timestep-convergence claim, boundary rule,
  Gauss projection, quantized register, performance benchmark, or
  microscopic-input derivation.

Verdict:

```text
QCA_SMV0_STAGE33_PHYSICAL_RIGHT_PRODUCTION_ECHO_GRAM_SCALE_PASS
```

## Stage 34 - Physical-Right Production Echo Horizon

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the Stage 32 echo-Gram diagnostic is evaluated at two trajectory horizons,
  one production tick and two production ticks;
- inverse-pulled echo gains are finite, nonzero, and bounded at both
  horizons;
- gain growth between horizons remains controlled on the certificate
  trajectory;
- dimensionless Gram data, including condition number and off-diagonal
  correlation, remain controlled between horizons;
- both base roundtrips and all inverse-pulled links remain controlled;
- the result is documented as a finite-horizon stability diagnostic for the
  current discrete production map, not as a continuum Lyapunov theorem, new
  dynamics rule, energy-conservation claim, timestep-convergence claim,
  boundary rule, Gauss projection, quantized register, performance benchmark,
  or microscopic-input derivation.

Verdict:

```text
QCA_SMV0_STAGE34_PHYSICAL_RIGHT_PRODUCTION_ECHO_HORIZON_PASS
```

## Stage 35 - Physical-Right Production Finite Stencil

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the BCC transport stencil is the eight body-diagonal hops and is closed under
  inversion;
- local Higgs/collision terms have radius zero and Higgs-gradient terms remain
  inside the one-hop BCC envelope;
- plaquette/current force terms are conservatively bounded by the two-hop BCC
  envelope;
- the full one-tick production envelope has finite radius `2`;
- the two-tick inverse echo envelope has finite radius `4`, giving linear
  radius growth per tick on the static stencil certificate;
- the origin remains in the tick stencil, so local/contact terms are explicitly
  represented;
- the result is documented as a static finite-stencil locality audit for the
  current discrete production map, not as a numerical large-lattice spatial
  echo measurement, continuum causal-cone theorem, new dynamics rule,
  energy-conservation claim, timestep-convergence claim, boundary rule, Gauss
  projection, quantized register, performance benchmark, or microscopic-input
  derivation.

Verdict:

```text
QCA_SMV0_STAGE35_PHYSICAL_RIGHT_PRODUCTION_STENCIL_PASS
```

## Stage 36 - Physical-Right Production Dense Workload

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the implemented dense rollout-state bytes per site are computed directly
  from the current register dimensions;
- the `3^3` certificate lattice remains small in raw state storage, separating
  data size from the observed large-lattice echo failure;
- the finite-difference Wilson-force coordinate count is computed from
  `sites x 8 BCC links x 12 SM generators`;
- the global action-evaluation count and plaquette-holonomy workload are
  computed for the current finite-difference force implementation;
- the workload is shown to scale quadratically in the site count while a local
  analytic staple-force target would scale linearly;
- the result is documented as a dense-workload scalability audit and next-step
  selector, not as a performance benchmark, new dynamics rule,
  energy-conservation claim, timestep-convergence claim, boundary rule, Gauss
  projection, quantized register, continuum causal-cone theorem, or
  microscopic-input derivation.

Verdict:

```text
QCA_SMV0_STAGE36_PHYSICAL_RIGHT_PRODUCTION_WORKLOAD_PASS
```

## Stage 37 - Physical-Right Production Local Wilson Force

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the legacy centered finite-difference Wilson force remains available as an
  explicit diagnostic oracle;
- the production `sm_left_wilson_force` uses the local BCC plaquette-staple
  derivative rather than global action finite differences;
- the local force vanishes on identity and pure-gauge links and responds to a
  deterministic nonflat link field;
- the local force transforms covariantly under site-local SM gauge rotations;
- representative force coordinates match automatic differentiation of the
  exact Wilson action under left link updates;
- a coarse legacy finite-difference comparison remains bounded but is not used
  as the defining proof because float32 cancellation is visible at small
  coordinate scales;
- the local-force plaquette workload is linear in the site count and the Stage
  37 certificate reports a large reduction relative to the legacy finite
  difference path;
- the result is documented as a production force replacement and scalability
  fix, not as a boundary rule, Gauss projection, exact energy-conservation
  theorem, timestep-convergence theorem, continuum causal-cone theorem, or
  microscopic-input derivation.

Verdict:

```text
QCA_SMV0_STAGE37_PHYSICAL_RIGHT_PRODUCTION_LOCAL_FORCE_PASS
```

## Stage 38 - Physical-Right Production Local-Force Rollout

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the implemented production tick is run on a lattice larger than the one-site
  Stage 21/22 default;
- the final rollout state is finite and the SM/Higgs links remain unitary;
- the tick produces nonzero Higgs-field, SM-link, and SM-momentum changes;
- changing the legacy `wilson_epsilon` keyword leaves the production step
  unchanged, proving that the old finite-difference Wilson-force knob is no
  longer active in the tick;
- the local-force plaquette workload is recorded on the certificate lattice
  and remains a large reduction relative to the legacy finite-difference path;
- the result is documented as a local-force rollout smoke test, not as a
  large-lattice spatial echo measurement, continuum causal-cone theorem,
  performance benchmark, exact energy-conservation theorem,
  timestep-convergence theorem, boundary rule, or microscopic-input
  derivation.

Verdict:

```text
QCA_SMV0_STAGE38_PHYSICAL_RIGHT_PRODUCTION_LOCAL_ROLLOUT_PASS
```

## Stage 39 - Physical-Right Production Local-Force Recorded Rollout

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the implemented production tick is recorded for more than one step on a
  lattice larger than the one-site Stage 21/22 default;
- the Python loop runner and `lax.scan` runner agree on both final state and
  recorded observations;
- all recorded observations remain finite;
- family norm drift and SM/Higgs link unitarity remain controlled over the
  recorded history;
- Higgs-field, SM-link, and SM-momentum changes remain nonzero over the
  rollout;
- the result is documented as a sparse-runner stability check on the local
  force path, not as a large-lattice spatial echo measurement, continuum
  causal-cone theorem, performance benchmark, exact energy-conservation
  theorem, timestep-convergence theorem, boundary rule, or microscopic-input
  derivation.

Verdict:

```text
QCA_SMV0_STAGE39_PHYSICAL_RIGHT_PRODUCTION_LOCAL_RECORDED_PASS
```

## Stage 40 - Physical-Right Production Local-Force Profile

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the shared `sim.profile_callable` helper produces an eager profile payload
  for one local-force production step;
- the payload is finite and JSON-safe through the shared profiler contract;
- a closed-over JIT version of the same step produces separate compile and
  cached-run timing fields without tracing `step_size` through Python control
  flow;
- eager and JIT outputs agree within float32 tolerance;
- final state finiteness, family norm drift, and SM/Higgs link unitarity remain
  controlled;
- all timing fields are documented as machine-dependent diagnostics, not as
  strict performance claims or physics derivations;
- the result is documented as a profile smoke test, not as a large-lattice
  spatial echo measurement, continuum causal-cone theorem, exact
  energy-conservation theorem, timestep-convergence theorem, boundary rule, or
  microscopic-input derivation.

Verdict:

```text
QCA_SMV0_STAGE40_PHYSICAL_RIGHT_PRODUCTION_LOCAL_PROFILE_PASS
```

## Stage 41 - Physical-Right Production Local-Force Support

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the local Wilson force is measured numerically after a single localized
  SM-link perturbation on a lattice large enough to contain sites outside the
  Stage 35 two-hop force radius;
- the measured support radius does not exceed the two-hop force envelope;
- at least one center site and multiple neighboring sites respond, so the
  perturbation is not a zero or trivial measurement;
- detected support outside the predicted force radius is zero within the
  configured numerical threshold;
- the result is documented as a local-force support measurement, not as a full
  production-map large-lattice spatial echo measurement, continuum causal-cone
  theorem, performance benchmark, exact energy-conservation theorem,
  timestep-convergence theorem, boundary rule, or microscopic-input
  derivation.

Verdict:

```text
QCA_SMV0_STAGE41_PHYSICAL_RIGHT_PRODUCTION_FORCE_SUPPORT_PASS
```

## Stage 42 - Physical-Right Production Local Fermion Current

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the legacy physical-right fermion current is preserved as an explicit
  finite-difference oracle, not used as the production current;
- the production physical-right fermion current is the analytic local
  left-trivialized derivative of the existing physical-right streaming
  bilinear;
- the local current agrees with the one-site finite-difference oracle within
  finite-difference tolerance;
- the zero-state local current vanishes and a deterministic nonzero state
  sources a nonzero current;
- the production current is independent of the old `fermion_current_epsilon`
  knob;
- a small production tick using the local current keeps family norm drift and
  SM/Higgs link unitarity controlled;
- the certificate records the estimated dense finite-difference workload
  reduction on the production smoke lattice;
- the result is documented as a local-current production implementation
  stage, not as a full production-map large-lattice spatial echo measurement,
  continuum causal-cone theorem, performance benchmark, exact
  energy-conservation theorem, timestep-convergence theorem, boundary rule, or
  microscopic-input derivation.

Verdict:

```text
QCA_SMV0_STAGE42_PHYSICAL_RIGHT_PRODUCTION_LOCAL_CURRENT_PASS
```

## Stage 43 - Physical-Right Production One-Tick Spatial Support

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the assembled physical-right production tick is run on a lattice large enough
  to contain sites outside the conservative Stage 35 one-tick stencil radius;
- each production state sector is perturbed locally: family state, Higgs field,
  Higgs momentum, SM link, SM momentum, and Higgs link;
- the combined per-site response after one full production tick is measured for
  each perturbation;
- the maximum measured support radius does not exceed the Stage 35 one-tick
  production stencil radius;
- detected support outside the predicted radius is zero within the configured
  numerical threshold;
- every perturbation gives a nonzero response, so the audit is not a trivial
  zero measurement;
- the result is documented as a one-tick finite-support measurement of the
  implemented discrete production map, not as a multi-step spatial echo,
  continuum causal-cone theorem, performance benchmark, exact
  energy-conservation theorem, timestep-convergence theorem, boundary rule, or
  microscopic-input derivation.

Verdict:

```text
QCA_SMV0_STAGE43_PHYSICAL_RIGHT_PRODUCTION_SPATIAL_SUPPORT_PASS
```

## Stage 44 - Physical-Right Production Family Cone

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- a localized family-state perturbation is evolved by the assembled
  physical-right production tick for more than one tick;
- the measured support radii over steps `1, 2, 3` are exactly `1, 2, 3`;
- the measured support is zero outside the discrete step-count cone at each
  horizon;
- support counts grow monotonically over the measured horizons;
- the response remains nonzero and finite over the measured horizons;
- the result is documented as a finite-horizon discrete family-cone support
  audit, not as a continuum light-cone theorem, Lyapunov estimate, all-sector
  propagation theorem, performance benchmark, exact energy-conservation
  theorem, timestep-convergence theorem, boundary rule, or microscopic-input
  derivation.

Verdict:

```text
QCA_SMV0_STAGE44_PHYSICAL_RIGHT_PRODUCTION_CONE_PASS
```

## Stage 45 - Physical-Right Production All-Sector Cones

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- localized perturbations are applied independently to family state, Higgs
  field, Higgs momentum, SM link, SM momentum, and Higgs link;
- every perturbed state is evolved by the assembled physical-right production
  tick for two ticks on a lattice containing the two-step cone;
- the combined per-site response is measured against the unperturbed two-tick
  baseline for every sector;
- every perturbation produces a finite nonzero response;
- the maximum measured support radius over all sectors is at most the
  two-step discrete cone radius;
- detected support outside the two-step cone is zero within the configured
  numerical threshold;
- the result is documented as a finite-horizon all-sector cone support audit,
  not as a continuum light-cone theorem, Lyapunov estimate, performance
  benchmark, exact energy-conservation theorem, timestep-convergence theorem,
  boundary rule, or microscopic-input derivation.

Verdict:

```text
QCA_SMV0_STAGE45_PHYSICAL_RIGHT_PRODUCTION_SECTOR_CONES_PASS
```

## Stage 46 - Physical-Right Production Gauss Projection

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the existing physical-right production Gauss residual is treated as
  `G(P)=div_E(P)-rho` at fixed links and matter fields;
- the correction acts only on SM link momenta, leaving family state, Higgs
  field, Higgs momenta, SM links, and Higgs links unchanged;
- the descent direction is the automatic-differentiated gradient of
  `0.5 ||G||^2` with respect to SM link momenta;
- the line step is the exact one-dimensional least-squares minimizer along
  that gradient direction for the affine fixed-link Gauss map;
- the zero-source vacuum has zero Gauss residual and remains unchanged by the
  relaxation step;
- a deterministic production state has nonzero initial Gauss residual and the
  relaxation step strictly reduces it;
- the relaxed state keeps SM and Higgs links unitary to float32 precision;
- the relaxation helper is JIT-compatible;
- the result is documented as a momentum-only Gauss relaxation/projection
  precursor, not as a full nonlinear gauge-orbit projection, Gauss-preserving
  production integrator, boundary rule, exact energy-conservation theorem,
  timestep-convergence theorem, or microscopic-input derivation.

Verdict:

```text
QCA_SMV0_STAGE46_PHYSICAL_RIGHT_PRODUCTION_GAUSS_PROJECTION_PASS
```

## Stage 47 - Physical-Right Production Gauss Solver

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the Stage 46 momentum-only relaxation is iterated for a fixed finite number
  of steps with an explicit residual history;
- the recorded history contains the initial Gauss norm, every post-step Gauss
  norm, every line step, and every gradient norm;
- the zero-source vacuum has zero Gauss residual and remains unchanged by the
  solver;
- a deterministic production state has nonzero initial Gauss residual and the
  solver reduces it monotonically over the recorded history;
- the total reduction is large enough to be a solver signal, not only a
  one-step smoke check;
- the correction acts only on SM link momenta, leaving family state, Higgs
  field, Higgs momenta, SM links, and Higgs links unchanged;
- the relaxed state keeps SM and Higgs links unitary to float32 precision;
- the solver helper is JIT-compatible with bounded float32 eager/JIT drift over
  the finite iteration history;
- the result is documented as a finite iterated relaxation solver for the
  current frozen production state, not as a full nonlinear gauge-orbit
  projection, Gauss-preserving production integrator, boundary rule, exact
  energy-conservation theorem, timestep-convergence theorem, or
  microscopic-input derivation.

Verdict:

```text
QCA_SMV0_STAGE47_PHYSICAL_RIGHT_PRODUCTION_GAUSS_SOLVER_PASS
```

## Stage 48 - Physical-Right Production Gauss-Projected Step

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the current physical-right production tick remains available as the raw
  unprojected step;
- a new wrapper advances one raw production tick and then applies the Stage 47
  finite momentum-only Gauss relaxation to the post-tick state;
- the zero-source vacuum has zero post-tick Gauss residual and remains
  unchanged by the projection;
- the deterministic post-tick Gauss residual is nonzero and the projected
  state has a smaller Gauss residual;
- the recorded projection history is monotone over the finite relaxation
  iterations and its final residual equals the projected-state residual;
- the projection changes only SM link momenta relative to the unprojected
  production step, leaving family state, Higgs field, Higgs momenta, SM links,
  and Higgs links unchanged;
- the projected state keeps SM and Higgs links unitary to float32 precision;
- the projected helper is JIT-compatible with bounded float32 eager/JIT drift
  over the projected momenta and finite residual history;
- the result is documented as an explicit projected-step wrapper, not as a
  full nonlinear gauge-orbit projection, rewritten production tick,
  Gauss-preserving integrator, boundary rule, exact energy-conservation
  theorem, timestep-convergence theorem, or microscopic-input derivation.

Verdict:

```text
QCA_SMV0_STAGE48_PHYSICAL_RIGHT_PRODUCTION_PROJECTED_STEP_PASS
```

## Stage 49 - Physical-Right Production Gauss-Projected Rollout

Pass only if:

- the only upstream runtime imports are from `sim` and local `qca_smv0`
  modules;
- the Stage 48 projected tick can be iterated over a finite horizon without
  changing the raw production tick helper;
- the helper records raw-rollout Gauss norms and projected-rollout Gauss norms
  from the same initial state;
- every projected tick records its raw post-tick Gauss norm, final projected
  Gauss norm, projection-reduction fraction, projection momentum delta, and
  finite relaxation-history consistency checks;
- the zero-source vacuum remains Gauss-free over the projected rollout;
- a deterministic projected rollout ends with a smaller final Gauss residual
  than the raw rollout over the same horizon;
- every projected step has positive Gauss reduction and monotone finite
  relaxation history;
- projected SM and Higgs links remain unitary to float32 precision over the
  recorded horizon;
- the result is documented as a finite projected-rollout audit, not as a full
  nonlinear gauge-orbit projection, Gauss-preserving integrator, exact
  energy-conservation theorem, timestep-convergence theorem, continuum
  stability theorem, boundary rule, or microscopic-input derivation.

Verdict:

```text
QCA_SMV0_STAGE49_PHYSICAL_RIGHT_PRODUCTION_PROJECTED_ROLLOUT_PASS
```

## Stage 50 - Compact Field-QCA Rollout Runner

Pass only if:

- a production-facing `sm_run_qca_rollout(config, initial_state, steps)` helper
  exists;
- the helper supports free BCC Dirac states, SM-internal states with optional
  static gauge links, and family SM-internal states with optional local
  Higgs/FN collision;
- the update order is BCC/gauge streaming followed by optional site-local
  Higgs/FN collision;
- the runner records norm and density observables without invoking the heavy
  dynamic production-tick machinery;
- center-CP quark Yukawas can be supplied through the generated Wilson-rule
  powers;
- a calibrated constructor builds a rollout config from quark masses, CKM,
  FN charges, and the center-CP phenomenology layer;
- focused tests cover free rollout, identity-gauge rollout, center-CP
  Higgs/FN collision, and calibrated masses/CKM configuration;
- no claim is made that the compact runner evolves dynamic gauge or Higgs
  fields; those remain in the heavier production-tick modules.

Verdict:

```text
QCA_SMV0_STAGE50_COMPACT_ROLLOUT_RUNNER_PASS
```
