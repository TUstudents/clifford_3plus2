# QCA_SMv0

Simulator sidecar for focused Standard-Model QCA experiments.

This sidecar is the place to build a smaller, recent-sidecar-informed simulator
without changing the shared `sim` package or the broader `spacetime_qca`
production prototype.

## Scope

`QCA_SMv0` should be used for:

- focused QCA simulator experiments tied to the recent boundary/flavor sidecars;
- small, JAX-native kernels that can later graduate into `spacetime_qca`;
- explicit session notes, local scripts, and focused tests.

It should not be used for:

- broad regression bookkeeping;
- importing theory claims before a session states them;
- replacing the shared `sim` infrastructure.

## Current State

Physics verdict:

- [`FN_CENTER_CP_VERDICT.md`](FN_CENTER_CP_VERDICT.md) records the current
  quark-flavor result: a constructive FN recirculation texture with order-one
  magnitudes and discrete `SU(3)_c` center phases fits quark masses, CKM
  moduli, and Jarlskog at percent-or-better accuracy for the default charge
  assignment.
- [`FN_CENTER_CP_ROBUSTNESS.md`](FN_CENTER_CP_ROBUSTNESS.md) records the
  robustness sweep: the frozen magnitudes are not universal, but the same
  discrete center powers remain viable after bounded order-one magnitude refits
  across realistic mass, CKM, and lambda variations.
- [`FN_CENTER_POWER_STRUCTURE.md`](FN_CENTER_POWER_STRUCTURE.md) decomposes the
  successful `Z3` center-power tables into row/column coboundaries, invariant
  plaquette fluxes, and the rank-one relative down/up center defect.  The same
  note records the current minimal Wilson-rule candidate that regenerates those
  powers.

Simulator front door:

- `sm_run_qca_rollout(...)` is the compact production-facing field-QCA runner.
  It composes BCC streaming, optional static SM gauge links, optional local
  Higgs/FN collision, center-CP quark Yukawas, and norm/density observables.
- `sm_qca_rollout_config_from_masses_ckm(...)` builds a rollout config from
  masses, CKM, FN charges, and the generated center-CP Wilson powers.  This is
  the constructive simulator path; dynamic gauge/Higgs field evolution remains
  in the heavier production-tick modules.

Stage 49 implements the free BCC Weyl/Dirac bulk walk, static
Standard-Model gauge-background transport, pure dynamic SM gauge fields, and a
site-local Higgs/Yukawa collision, explicit hidden-network FN recirculation, and
center-holonomy CP coefficients, a three-family Higgs/Yukawa collision, and
dynamic Higgs-field evolution with scalar gauge and local fermion/Higgs
backreaction, plus BCC streaming fermion gauge currents, the merged
family-production tick, the gauge-convention bridge audit, and the
antiunitary singlet bridge, physical-right bridged transport, and the
physical-right bridged fermion current, sourced gauge tick, and production
tick, sparse recorded rollout of that production tick, a physical-right Gauss
monitor on the production rollout, an energy-component monitor, and a
variational force-provenance audit, plus refinement and adjoint limitation
audits, an explicit inverse helper for the current production tick, a
trajectory-level reversibility audit, a Loschmidt echo diagnostic, and a
finite tangent-response, echo-Gram, echo-Gram scale, and echo-horizon audit
suite, a finite-stencil locality audit, a dense-workload scaling audit, a
local analytic Wilson staple-force replacement, a local analytic
physical-right fermion-current replacement, and a local-force production
rollout smoke test, a multi-step local-force recorded-rollout audit, a
local-force production profiling certificate, a numerical local-force
spatial-support audit, a one-tick full-production spatial-support audit, a
three-tick family-cone audit, a two-tick all-sector cone audit, and a
momentum-only Gauss relaxation/projection precursor with an iterated solver, a
Gauss-projected production-step wrapper, and a finite projected-rollout audit:

```text
qca_smv0/
  README.md
  PLAN.md
  STATUS.md
  __init__.py
  bulk_bcc.py
  sm_gauge.py
  sm_dynamics.py
  sm_higgs.py
  sm_fn.py
  sm_cp.py
  sm_family_higgs.py
  sm_higgs_dynamics.py
  sm_gauge_higgs.py
  sm_fermion_higgs.py
  sm_fermion_gauge.py
  sm_family_sourced_tick.py
  sm_family_production_tick.py
  sm_gauge_convention_bridge.py
  sm_antiunitary_bridge.py
  sm_physical_right_transport.py
  sm_physical_right_current.py
  sm_physical_right_sourced_tick.py
  sm_physical_right_production_tick.py
  sm_physical_right_production_rollout.py
  sm_physical_right_production_gauss.py
  sm_physical_right_production_gauss_projection.py
  sm_physical_right_production_gauss_solver.py
  sm_physical_right_production_projected.py
  sm_physical_right_production_projected_rollout.py
  sm_physical_right_production_energy.py
  sm_physical_right_production_variational.py
  sm_physical_right_production_refinement.py
  sm_physical_right_production_adjoint.py
  sm_physical_right_production_inverse.py
  sm_physical_right_production_reversibility.py
  sm_physical_right_production_echo.py
  sm_physical_right_production_tangent.py
  sm_physical_right_production_echo_gram.py
  sm_physical_right_production_echo_scale.py
  sm_physical_right_production_echo_horizon.py
  sm_physical_right_production_stencil.py
  sm_physical_right_production_workload.py
  sm_physical_right_production_local_force.py
  sm_physical_right_production_local_rollout.py
  sm_physical_right_production_local_recorded.py
  sm_physical_right_production_local_profile.py
  sm_physical_right_production_force_support.py
  sm_physical_right_production_local_current.py
  sm_physical_right_production_spatial_support.py
  sm_physical_right_production_cone.py
  sm_physical_right_production_sector_cones.py
  scripts/
    session_01_bare_bcc_walk.py
    session_02_static_sm_gauge_background.py
    session_03_dynamic_sm_gauge.py
    session_04_higgs_yukawa_collision.py
    session_05_fn_recirculation.py
    session_06_center_holonomy_cp.py
    session_07_family_higgs_yukawa.py
    session_08_higgs_dynamics.py
    session_09_gauge_higgs_backreaction.py
    session_10_fermion_higgs_backreaction.py
    session_11_fermion_gauge_current.py
    session_12_sourced_tick.py
    session_13_family_gauge_current.py
    session_14_family_sourced_tick.py
    session_15_family_production_tick.py
    session_16_gauge_convention_bridge.py
    session_17_antiunitary_bridge.py
    session_18_physical_right_transport.py
    session_19_physical_right_current.py
    session_20_physical_right_sourced_tick.py
    session_21_physical_right_production_tick.py
    session_22_physical_right_production_rollout.py
    session_23_physical_right_production_gauss.py
    session_46_physical_right_production_gauss_projection.py
    session_47_physical_right_production_gauss_solver.py
    session_48_physical_right_production_projected.py
    session_49_physical_right_production_projected_rollout.py
    session_24_physical_right_production_energy.py
    session_25_physical_right_production_variational.py
    session_26_physical_right_production_refinement.py
    session_27_physical_right_production_adjoint.py
    session_28_physical_right_production_inverse.py
    session_29_physical_right_production_reversibility.py
    session_30_physical_right_production_echo.py
    session_31_physical_right_production_tangent.py
    session_32_physical_right_production_echo_gram.py
    session_33_physical_right_production_echo_scale.py
    session_34_physical_right_production_echo_horizon.py
    session_35_physical_right_production_stencil.py
    session_36_physical_right_production_workload.py
    session_37_physical_right_production_local_force.py
    session_38_physical_right_production_local_rollout.py
    session_39_physical_right_production_local_recorded.py
    session_40_physical_right_production_local_profile.py
    session_41_physical_right_production_force_support.py
    session_42_physical_right_production_local_current.py
    session_43_physical_right_production_spatial_support.py
    session_44_physical_right_production_cone.py
    session_45_physical_right_production_sector_cones.py
  tests/
    test_bulk_bcc.py
    test_sm_gauge.py
    test_sm_dynamics.py
    test_sm_higgs.py
    test_sm_fn.py
    test_sm_cp.py
    test_sm_family_higgs.py
    test_sm_higgs_dynamics.py
    test_sm_gauge_higgs.py
    test_sm_fermion_higgs.py
    test_sm_fermion_gauge.py
    test_sm_family_sourced_tick.py
    test_sm_family_production_tick.py
    test_sm_gauge_convention_bridge.py
    test_sm_antiunitary_bridge.py
    test_sm_physical_right_transport.py
    test_sm_physical_right_current.py
    test_sm_physical_right_sourced_tick.py
    test_sm_physical_right_production_tick.py
    test_sm_physical_right_production_rollout.py
    test_sm_physical_right_production_gauss.py
    test_sm_physical_right_production_energy.py
    test_sm_physical_right_production_variational.py
    test_sm_physical_right_production_refinement.py
    test_sm_physical_right_production_adjoint.py
    test_sm_physical_right_production_inverse.py
    test_sm_physical_right_production_reversibility.py
    test_sm_physical_right_production_echo.py
    test_sm_physical_right_production_tangent.py
    test_sm_physical_right_production_echo_gram.py
    test_sm_physical_right_production_echo_scale.py
    test_sm_physical_right_production_echo_horizon.py
    test_sm_physical_right_production_stencil.py
    test_sm_physical_right_production_workload.py
    test_sm_physical_right_production_local_force.py
    test_sm_physical_right_production_local_rollout.py
    test_sm_physical_right_production_local_recorded.py
    test_sm_physical_right_production_local_profile.py
    test_sm_physical_right_production_force_support.py
    test_sm_physical_right_production_local_current.py
    test_sm_physical_right_production_spatial_support.py
    test_sm_physical_right_production_cone.py
```

The implemented Weyl kernel is a two-component periodic BCC bulk walk:

```text
psi_next[x] = sum_h A_h psi[x + h]
```

with eight BCC source offsets `h in {+-1}^3` and

```text
A_h = P_x^{h_x} P_y^{h_y} P_z^{h_z}.
```

With this pull convention the Bloch symbol is exactly

```text
U(k)=S_x(k_x) S_y(k_y) S_z(k_z),
S_j(k_j)=P_j^+ exp(i k_j)+P_j^- exp(-i k_j).
```

Verdict:

```text
QCA_SMV0_STAGE1_FREE_BCC_PASS
```

Stage 1 also includes:

- exact hop-completeness checks for Weyl and Dirac hops;
- periodic norm-conservation tests;
- split-product versus hop-sum Bloch-symbol consistency;
- small-momentum Weyl/Dirac dispersion checks;
- directional anisotropy scaling under momentum halving;
- massless Dirac assembly from opposite-chirality Weyl blocks;
- JIT compatibility checks.

Stage 2 adds a static background gauge layer:

- a local 32-component internal SM carrier: one left-handed chiral-16 label set
  duplicated across the four-component Dirac spin block;
- local anti-Hermitian `SU(3)_c x SU(2)_L x U(1)_Y` generators;
- finite BCC edge links with shape `(nx, ny, nz, 8, 32, 32)`;
- a gauge-covariant Dirac BCC step through those static links;
- site-local gauge transformations of states and links;
- selected BCC plaquette Wilson traces;
- weak-link linearization as the simulator covariant-derivative check;
- JIT compatibility for the gauged transport step.

Stage 2 verdict:

```text
QCA_SMV0_STAGE2_STATIC_SM_GAUGE_PASS
```

Stage 3 adds a pure dynamic gauge layer:

- real link momenta with shape `(nx, ny, nz, 8, 12)` in the SM generator basis;
- projection between algebra matrices and SM generator coordinates;
- target-site adjoint momentum gauge transforms;
- Wilson-force finite differences under left link perturbations;
- Hamiltonian density `K(P) + beta S_W(U)`;
- reversible pure-gauge leapfrog update;
- electric divergence / Gauss diagnostic for the pure gauge field;
- pure-gauge zero-momentum Gauss-preservation audit;
- weak-field plaquette field-strength linearization;
- Wilson action matching to the weak-field Yang-Mills density;
- no-backreaction fermion/gauge wrapper: evolve links and momenta, then
  transport spectator fermions through the updated links;
- JIT compatibility for the leapfrog update.

Stage 3 verdict:

```text
QCA_SMV0_STAGE3_DYNAMIC_SM_GAUGE_PASS
```

Stage 4 adds a local Higgs/Yukawa collision:

- local Higgs doublet fields with shape `(nx, ny, nz, 2)`;
- the constant unitary-gauge Higgs helper `H=(0,v/sqrt(2))`;
- `H_tilde=i sigma_2 H^*`;
- a Hermitian one-generation Yukawa matrix on the local SM carrier;
- unitary-gauge doors: `H_tilde` opens up/neutrino couplings while `H` opens
  down/electron couplings;
- exact site-local rotation `exp(-i step_size beta Y(H))`;
- zero-step and zero-Higgs identity controls;
- norm preservation, chirality-flip, massive-dispersion, and JIT audits.

Stage 4 verdict:

```text
QCA_SMV0_STAGE4_HIGGS_YUKAWA_PASS
```

Stage 5 adds FN recirculation paths:

- family dimension `3`;
- explicit simulator inputs `lambda`, `Q`, `U`, `D`, and order-one
  coefficients;
- empirical Wolfenstein mode and shear-candidate mode for `lambda`;
- finite local beam-splitter chains whose endpoint transfer is `lambda^n`;
- explicit direct-sum hidden unitary networks with one path for every
  `L_i -> R_j` entry;
- FN power matrices measured from hidden-network source-to-sink transfers;
- visible-hidden-visible readout maps that send left-family channels through
  the hidden recirculation network into right-family channels;
- exact finite unitary dilations of the visible FN transfer blocks;
- a local visible-plus-auxiliary FN collision primitive that applies the
  dilation unitarily and exposes the normalized visible transfer;
- quark path lengths `n^u_ij=Q_i+U_j` and `n^d_ij=Q_i+D_j`;
- default charges `Q=(3,2,0)`, `U=(5,2,0)`, `D=(1,0,0)`;
- standard diagonal FN orders `lambda^8:lambda^4:1` and
  `lambda^4:lambda^2:1`;
- left-frame Wolfenstein scaling `lambda^abs(Q_i-Q_j)`;
- generated matrices `Y_ij=c_ij A_ij`, where `A_ij` is the hidden-network
  transfer amplitude;
- singular masses and CKM-like left-frame mismatch read from the same
  generated matrices.

Stage 5 verdict:

```text
QCA_SMV0_STAGE5_FN_RECIRCULATION_PASS
```

Stage 6 adds center-holonomy CP coefficients:

- `SU(3)_c` center phases `omega^k`, `omega=exp(2 pi i / 3)`;
- explicit center-power matrices decorating only the FN order-one coefficients;
- unit-modulus and `phase^3=1` audits;
- quark/antiquark Yukawa conjugation with unchanged singular masses;
- CKM-like CP from the left-frame mismatch of the generated matrices;
- nonzero center-decorated Jarlskog and zero all-real control;
- nonzero CP-odd commutator trace for the center-decorated matrices.

Stage 6 verdict:

```text
QCA_SMV0_STAGE6_CENTER_HOLONOMY_CP_PASS
```

Stage 7 adds a three-family Higgs/Yukawa collision:

- family-extended fermion states with shape `(nx, ny, nz, 4, 32, 3)`;
- local internal/family dimension `96`;
- default quark Yukawa source built from center-holonomy coefficients feeding
  the Stage 5 visible-hidden-visible FN recirculation readout;
- finite FN unitary dilations for the up/down quark Higgs doors, with exposed
  Higgs-scaled visible transfers;
- recirculation-backed quark Yukawa matrices embedded in the local Higgs doors;
- explicit diagonal placeholder lepton matrices as simulator inputs;
- Hermitian local three-family Yukawa matrix;
- unitary-gauge block extraction showing the embedded up/down matrices match
  the generated matrices;
- wrong weak-component door controls;
- CKM-like left-frame mismatch consistency from the embedded blocks;
- exact local unitary collision `exp(-i step_size beta Y_family(H))`;
- zero-step, zero-Higgs, norm, chirality-flip, and JIT audits.

Stage 7 verdict:

```text
QCA_SMV0_STAGE7_FAMILY_HIGGS_YUKAWA_PASS
```

Stage 8 adds dynamic Higgs-field evolution:

- Higgs electroweak `SU(2)_L x U(1)_Y` generators on the doublet;
- finite Higgs BCC links with shape `(nx, ny, nz, 8, 2, 2)`;
- Higgs field and momentum layout `(nx, ny, nz, 2)`;
- kinetic, gauge-covariant BCC gradient, and quartic-potential energy terms;
- covariant Higgs force;
- unitary-gauge vacuum force audit;
- pure-gauge vacuum gradient-energy audit;
- force covariance and Hamiltonian gauge-invariance checks;
- reversible no-fermion Higgs leapfrog update;
- small-step Hamiltonian drift and JIT checks.

Stage 8 verdict:

```text
QCA_SMV0_STAGE8_HIGGS_DYNAMICS_PASS
```

Stage 9 adds gauge-Higgs backreaction:

- Higgs electroweak link momenta with shape `(nx, ny, nz, 8, 4)`;
- projection between Higgs algebra matrices and electroweak coordinates;
- target-site adjoint transforms for Higgs link momenta;
- left-trivialized Higgs gauge force from the covariant-gradient energy;
- zero force for covariantly constant vacuum controls;
- nonzero force for deterministic non-vacuum/nonflat fields;
- covariant Higgs charge density and Gauss diagnostic
  `electric divergence - Higgs charge`;
- embedding of Higgs electroweak momenta into the full SM 12-generator layout;
- coupled no-fermion Higgs/gauge leapfrog update;
- link unitarity, reversibility, small Hamiltonian drift, and JIT checks.

Stage 9 verdict:

```text
QCA_SMV0_STAGE9_GAUGE_HIGGS_BACKREACTION_PASS
```

Stage 10 adds local fermion/Higgs backreaction:

- local Yukawa energy density `E_Y=Re psi^dagger beta Y(H) psi`;
- default quark source inherited from the Stage 7 recirculation-backed FN
  Yukawa matrices;
- Higgs source force `-dE_Y/dH*` computed from real/imaginary autodiff;
- deterministic source states with nonzero Yukawa bilinears;
- zero-state and nonzero-source controls;
- explicit Yukawa-door electroweak gauge helper using physical right-handed
  hypercharges for local Yukawa covariance;
- energy gauge-invariance and source covariance checks for that local
  Yukawa-door convention;
- reversible Higgs-momentum kick;
- local collision-plus-kick wrapper;
- fermion norm, JIT, and source-after-collision audits.

Stage 10 verdict:

```text
QCA_SMV0_STAGE10_FERMION_HIGGS_BACKREACTION_PASS
```

Stage 11 adds the BCC streaming fermion gauge current:

- local streaming bilinear
  `E_stream=Re sum_x,h psi[x]^dag D_h U_h[x] psi[x+h]`;
- fermion current as the left-trivialized derivative of `E_stream` with
  respect to each SM BCC link;
- zero-state and nonzero-current controls;
- streaming-energy gauge-invariance audit;
- target-site adjoint covariance for the current;
- local fermion charge density and Gauss diagnostic
  `electric divergence - fermion charge`;
- reversible fermion-current momentum kick;
- kick-then-transport wrapper;
- link unitarity, spectator norm, and JIT audits.

Stage 11 verdict:

```text
QCA_SMV0_STAGE11_FERMION_GAUGE_CURRENT_PASS
```

Stage 12 adds the coupled sourced SM gauge tick:

- SM transport links and Higgs electroweak links are carried as two
  representations of one 12-coordinate link-momentum field;
- the sourced link force is
  `Wilson + embedded Higgs gauge force + BCC streaming fermion current`;
- zero-source and deterministic nonzero-source controls;
- sourced force and sourced Gauss covariance under a shared electroweak gauge;
- full Gauss diagnostic
  `electric divergence - fermion charge - embedded Higgs charge`;
- reversible sourced momentum kick;
- SM/Higgs link update from the same momentum field;
- coupled tick with fermion transport, Higgs-field advance, link unitarity,
  fermion norm, and JIT audits.

Stage 12 verdict:

```text
QCA_SMV0_STAGE12_SOURCED_SM_TICK_PASS
```

Stage 13 lifts the BCC streaming fermion gauge current to the family carrier:

- family state layout `(nx, ny, nz, 4, 32, 3)`;
- family-blind SM gauge links;
- single-family embedding/extraction controls;
- one-family reduction to the Stage 11 streaming energy, current, charge, and
  transport;
- family-summed streaming bilinear and left-trivialized current;
- family-summed charge and fermion Gauss diagnostic;
- gauge covariance, reversible momentum kick, link unitarity, family-state
  norm, and JIT audits.

Stage 13 verdict:

```text
QCA_SMV0_STAGE13_FAMILY_GAUGE_CURRENT_PASS
```

Stage 14 lifts the sourced gauge tick to the family carrier:

- family state layout `(nx, ny, nz, 4, 32, 3)`;
- family-sourced link force
  `Wilson + embedded Higgs gauge force + family-summed BCC current`;
- one-family reduction to the Stage 12 sourced tick;
- family-sourced Gauss diagnostic
  `electric divergence - family charge - embedded Higgs charge`;
- gauge covariance, reversible momentum kick, SM/Higgs link unitarity,
  family-state norm, Higgs-field advance, and JIT audits.

Stage 14 verdict:

```text
QCA_SMV0_STAGE14_FAMILY_SOURCED_TICK_PASS
```

Stage 15 merges the local three-family Yukawa/Higgs source into the
family-sourced gauge tick:

- full family production tick with Stage 14 gauge/Higgs/fermion-current
  sources plus the Stage 10 local Yukawa Higgs source;
- exact local Stage 7 family Yukawa collision inserted in a symmetric
  half-collision / transport / half-collision order;
- zero-Yukawa reduction to the Stage 14 family-sourced tick;
- Higgs momentum force
  `Higgs force + local Yukawa Higgs source`;
- deterministic nonzero Yukawa-source and zero-state/vacuum-source controls;
- reversible Higgs momentum source kick for frozen fields;
- family-state norm, SM/Higgs link unitarity, and JIT audits;
- explicit boundary that the Stage 10 local Higgs-door gauge convention is
  still separate from the Stage 14 transport/current gauge convention.

Stage 15 verdict:

```text
QCA_SMV0_STAGE15_FAMILY_PRODUCTION_TICK_PASS
```

Stage 16 audits the transport-vs-Yukawa gauge convention boundary:

- explicit Stage 2 transport electroweak generators and Stage 10 physical
  Yukawa-door electroweak generators;
- finite generator exponentiation reproduces the Stage 10 Yukawa-door helper;
- exact agreement on duplicated left doublet labels `Q,L`;
- zero `SU(2)_L` action on singlet labels in both conventions;
- right-singlet hypercharges are charge-conjugate across the two conventions;
- the full generator sets and sorted hypercharge spectra differ, proving no
  unitary similarity on the fixed 32-component carrier identifies them;
- Yukawa energy is covariant in the physical Yukawa-door convention and
  deliberately non-invariant under the transport convention;
- JIT audits for the energy residuals.

Stage 16 verdict:

```text
QCA_SMV0_STAGE16_GAUGE_CONVENTION_BRIDGE_PASS
```

Stage 17 implements the exact antiunitary singlet bridge:

- physical-right SM generators built from Stage 2 transport generators by
  `T_phys=T_transport` on left doublets and
  `T_phys=conj(T_transport)` on right singlets;
- exact left-linear and right-antilinear projector residuals;
- electroweak slice equal to the Stage 10/16 physical Yukawa-door generators;
- finite physical-right site gauge equal to
  `P_L G_transport P_L + P_R conj(G_transport) P_R`;
- finite bridge unitarity audit;
- full `SU(3)_c x SU(2)_L x U(1)_Y` physical-right gauge restores local
  Yukawa energy covariance, while the unbridged transport convention remains
  non-invariant;
- JIT audits for the full bridge energy residuals.

Stage 17 verdict:

```text
QCA_SMV0_STAGE17_ANTIUNITARY_BRIDGE_PASS
```

Stage 18 applies the antiunitary bridge to finite BCC transport:

- transport-convention BCC links bridged edgewise to physical-right links;
- identity links unchanged by the bridge;
- finite bridged links unitary to float32 precision;
- link covariance under bridged physical-right site gauges;
- family BCC transport through bridged links;
- identity-link reduction to the existing family transport;
- nontrivial difference from the unbridged transport convention;
- family-state norm and JIT audits.

Stage 18 verdict:

```text
QCA_SMV0_STAGE18_PHYSICAL_RIGHT_TRANSPORT_PASS
```

Stage 19 lifts the BCC streaming fermion gauge current to the physical-right
bridged carrier:

- physical-right streaming bilinear through bridged finite BCC links;
- left-trivialized finite-difference current in the same 12 transport
  coordinates;
- zero-state and nonzero-current controls;
- nontrivial difference from the unbridged Stage 13 transport-convention
  family current;
- physical-right charge density and Gauss diagnostic
  `electric divergence - physical-right family charge`;
- covariance checks for streaming energy, current, charge density, and Gauss
  under the bridged physical-right site gauge;
- reversible physical-right current momentum kick;
- kick-then-physical-right-transport wrapper;
- link unitarity, spectator family-state norm, and JIT audits.

Stage 19 verdict:

```text
QCA_SMV0_STAGE19_PHYSICAL_RIGHT_CURRENT_PASS
```

Stage 20 replaces the Stage 14 transport-convention family fermion source with
the Stage 19 physical-right current inside the sourced gauge tick:

- sourced link force
  `Wilson + embedded Higgs gauge force + physical-right family current`;
- physical-right sourced Gauss diagnostic
  `physical-right electric divergence - physical-right family charge -
  embedded Higgs charge`;
- covariance checks for the sourced force and Gauss diagnostic under the
  bridged physical-right site gauge;
- zero-source and deterministic nonzero-source controls;
- reversible physical-right sourced SM momentum kick;
- sourced tick using physical-right bridged family transport;
- nontrivial difference from the Stage 14 transport-convention sourced tick;
- SM/Higgs link unitarity, family-state norm, Higgs-field advance, and JIT
  audits.

Stage 20 verdict:

```text
QCA_SMV0_STAGE20_PHYSICAL_RIGHT_SOURCED_TICK_PASS
```

Stage 21 merges the local three-family Yukawa/Higgs production source into the
physical-right sourced gauge tick:

- symmetric local half-collision / physical-right BCC family transport /
  second half-collision production ordering;
- zero-Yukawa reduction to the Stage 20 physical-right sourced tick;
- production Higgs force
  `Higgs force + local Yukawa Higgs source`;
- deterministic nonzero Yukawa-source and zero-state/vacuum-source controls;
- reversible Higgs momentum source kick for frozen fields;
- nontrivial difference from both the Stage 20 physical-right sourced tick and
  the Stage 15 transport-convention production tick;
- family-state norm, SM/Higgs link unitarity, and JIT audits.

Stage 21 verdict:

```text
QCA_SMV0_STAGE21_PHYSICAL_RIGHT_PRODUCTION_TICK_PASS
```

Stage 22 wraps the Stage 21 physical-right production tick in the shared
`sim.runner` sparse-recorded rollout interface:

- `PhysicalRightProductionRolloutState` carries family fermions, Higgs fields,
  SM links/momenta, and Higgs-representation links as one rollout state;
- one-step rollout matches the direct Stage 21 tick;
- loop and scan recorded runners agree to float32 precision;
- sparse observations record family norm, Higgs norms, momentum norms, and
  SM/Higgs link unitarity;
- multi-tick rollout keeps family norm and link unitarity controlled;
- zero-Yukawa rollout remains distinguishable from the default production
  rollout, so the local production source stays active under iteration.

Stage 22 verdict:

```text
QCA_SMV0_STAGE22_PHYSICAL_RIGHT_PRODUCTION_ROLLOUT_PASS
```

Stage 23 adds a physical-right sourced Gauss monitor on top of the production
rollout:

- reads the existing Stage 20 Gauss diagnostic
  `electric divergence - physical-right family charge - embedded Higgs charge`
  on Stage 21/22 production states;
- provides an exact zero-source vacuum control whose Gauss norm remains zero
  under rollout;
- records deterministic nonzero Gauss history for the production rollout;
- checks that the default production rollout is distinguishable from a
  zero-Yukawa rollout at the Gauss observable level;
- keeps family norm and SM/Higgs link unitarity controlled during the monitored
  rollout.

Stage 23 verdict:

```text
QCA_SMV0_STAGE23_PHYSICAL_RIGHT_PRODUCTION_GAUSS_PASS
```

Stage 24 adds an energy-component monitor on top of the physical-right
production rollout:

- records pure SM gauge Hamiltonian density;
- records Higgs Hamiltonian density;
- records the physical-right BCC streaming bilinear;
- records local three-family Yukawa energy;
- records the sum as a monitored total energy without claiming exact
  conservation of the full hybrid tick;
- keeps an exact vacuum-zero control and a default-vs-zero-Yukawa contrast.

Stage 24 verdict:

```text
QCA_SMV0_STAGE24_PHYSICAL_RIGHT_PRODUCTION_ENERGY_PASS
```

Stage 25 adds a variational audit for the physical-right production forces:

- normalizes the Higgs force as `-dE/dphi*` for the Stage 24 Higgs
  Hamiltonian density;
- checks exact decomposition of the sourced link force into Wilson force,
  embedded Higgs gauge force, and physical-right family current;
- checks exact decomposition of the production Higgs force into Higgs
  Hamiltonian force plus local Yukawa Higgs source;
- compares one color-link coordinate against a central finite difference of
  the Wilson plus physical-right streaming energy;
- compares one complex Higgs component against real and imaginary central
  finite differences of the Higgs plus Yukawa energy;
- keeps zero-source vacuum force controls and deterministic nonzero-force
  controls.

Stage 25 verdict:

```text
QCA_SMV0_STAGE25_PHYSICAL_RIGHT_PRODUCTION_VARIATIONAL_PASS
```

Stage 26 adds a fixed-time timestep-refinement audit for the monitored
production energy:

- compares `dt` against `dt/2` with doubled step count at the same physical
  time;
- keeps the refined rollout finite, link-unitary, and family-norm controlled;
- keeps the zero-source production vacuum at zero monitored energy drift;
- records that the current hybrid production tick does not improve monitored
  total-energy drift under timestep halving;
- treats that as an integrator limitation, not as a conservation theorem.

Stage 26 verdict:

```text
QCA_SMV0_STAGE26_PHYSICAL_RIGHT_PRODUCTION_REFINEMENT_LIMITATION_PASS
```

Stage 27 adds an adjoint audit for the production fermion substage:

- implements the adjoint of the physical-right BCC family transport;
- verifies that adjoint transport restores a deterministic family state;
- verifies that the frozen production fermion substage has an explicit
  adjoint and preserves norm;
- verifies that the local family Yukawa collision inverts under
  `step_size -> -step_size` on frozen Higgs fields;
- records that a full production tick followed by a negative-timestep
  production tick is not the inverse of the full map;
- treats that as a missing full reversible production integrator, not as a
  failure of the frozen fermion substage.

Stage 27 verdict:

```text
QCA_SMV0_STAGE27_PHYSICAL_RIGHT_PRODUCTION_ADJOINT_LIMITATION_PASS
```

Stage 28 adds an explicit inverse helper for the current physical-right
production tick:

- reconstructs final half-step momenta from the final production forces;
- rewinds the sourced link update and Higgs field update;
- uses the Stage 27 frozen fermion-stage adjoint to rewind the local
  collision / bridged transport / local collision substage;
- recovers the initial momenta from the reconstructed first production forces;
- verifies forward-then-inverse and inverse-then-forward roundtrips to float32
  precision;
- keeps family norm and SM/Higgs link unitarity controlled;
- records a large improvement over the naive negative-timestep production
  tick;
- treats the result as an inverse of the current discrete map, not as an
  energy-conservation or timestep-convergence theorem.

Stage 28 verdict:

```text
QCA_SMV0_STAGE28_PHYSICAL_RIGHT_PRODUCTION_EXPLICIT_INVERSE_PASS
```

Stage 29 composes the explicit inverse over a short production trajectory:

- rewinds a multi-step physical-right production rollout with repeated
  Stage 28 inverse steps;
- checks that each inverse step lands on the stored forward trajectory;
- checks forward-then-inverse and inverse-then-forward trajectory roundtrips;
- compares against a multi-step naive negative-timestep rollout;
- keeps family norm and SM/Higgs link unitarity controlled across the forward
  and inverse paths;
- verifies fixed-step inverse rollout JIT compatibility;
- treats the result as a reversibility audit of the current discrete map, not
  as an energy-convergent Hamiltonian integrator theorem.

Stage 29 verdict:

```text
QCA_SMV0_STAGE29_PHYSICAL_RIGHT_PRODUCTION_TRAJECTORY_REVERSIBILITY_PASS
```

Stage 30 uses the inverse trajectory as a Loschmidt echo diagnostic:

- advances a short deterministic physical-right production trajectory;
- applies a small local SM-momentum perturbation at the final time;
- rewinds the perturbed final state to measure the initial-surface echo;
- checks that the echo is finite and nonzero;
- doubles the final-time perturbation and checks that the echo doubles to
  local linear precision;
- keeps the unperturbed roundtrip and perturbed inverse link unitarity
  controlled;
- treats the result as a stability diagnostic for the current discrete map,
  not as a new dynamics rule or energy-convergence theorem.

Stage 30 verdict:

```text
QCA_SMV0_STAGE30_PHYSICAL_RIGHT_PRODUCTION_LOSCHMIDT_ECHO_PASS
```

Stage 31 extends the echo into a finite tangent-response audit:

- applies independent final-time kicks in one SM momentum coordinate and one
  Higgs momentum coordinate;
- pulls each kick back with the explicit inverse trajectory;
- pulls back the combined kick and compares it with the sum of the separate
  inverse echoes;
- verifies finite nonzero echoes and a small superposition residual;
- keeps the unperturbed roundtrip and combined perturbed inverse link unitarity
  controlled;
- treats the result as a local finite-difference tangent diagnostic, not as a
  new dynamics rule or conservation theorem.

Stage 31 verdict:

```text
QCA_SMV0_STAGE31_PHYSICAL_RIGHT_PRODUCTION_TANGENT_ECHO_PASS
```

Stage 32 assembles a local echo Gram matrix:

- pulls back three independent final-time perturbations through the explicit
  inverse trajectory;
- uses the resulting initial-surface echoes to form a real Gram matrix;
- checks finite nonzero echo norms, symmetry, positive minimum eigenvalue, and
  finite condition number;
- checks that off-diagonal correlations remain bounded;
- keeps all inverse-pulled echo states link-unitary;
- treats the result as a finite local tangent-metric diagnostic, not as a
  continuum stability theorem.

Stage 32 verdict:

```text
QCA_SMV0_STAGE32_PHYSICAL_RIGHT_PRODUCTION_ECHO_GRAM_PASS
```

Stage 33 checks that the echo Gram sits in a local scale-stable regime:

- evaluates the Stage 32 echo Gram at `epsilon` and `2 epsilon`;
- verifies that echo norms scale linearly;
- verifies that Gram eigenvalues scale quadratically;
- checks that condition number and off-diagonal correlations remain stable;
- keeps base roundtrips and inverse-pulled link unitarity controlled;
- treats the result as a finite scale-stability audit, not as a continuum
  stability theorem.

Stage 33 verdict:

```text
QCA_SMV0_STAGE33_PHYSICAL_RIGHT_PRODUCTION_ECHO_GRAM_SCALE_PASS
```

Stage 34 extends the echo Gram to a finite-horizon audit:

- evaluates the Stage 32 echo Gram at one-tick and two-tick production
  trajectories;
- converts Gram eigenvalues into inverse-pulled echo gains;
- checks that gains stay finite, nonzero, and locally bounded across the two
  horizons;
- checks that condition number, off-diagonal correlations, base roundtrips, and
  inverse-pulled link unitarity remain controlled;
- treats the result as a finite-horizon diagnostic for the current discrete
  production map, not as a continuum Lyapunov theorem.

Stage 34 verdict:

```text
QCA_SMV0_STAGE34_PHYSICAL_RIGHT_PRODUCTION_ECHO_HORIZON_PASS
```

Stage 35 records the finite stencil envelope of the production tick:

- builds exact integer displacement sets for BCC transport, local Higgs terms,
  one-hop Higgs-gradient forces, and conservative two-hop plaquette/current
  forces;
- verifies the current one-tick production envelope has radius `2`;
- verifies the two-tick inverse echo envelope has radius `4`, with linear
  radius growth per tick;
- checks BCC inverse-hop closure and origin retention in the local tick
  stencil;
- treats the result as a finite-speed stencil audit for the implemented
  discrete map, not as a large-lattice spatial echo measurement or continuum
  causal-cone theorem.

Stage 35 verdict:

```text
QCA_SMV0_STAGE35_PHYSICAL_RIGHT_PRODUCTION_STENCIL_PASS
```

Stage 36 audits the dense production workload:

- computes the dense rollout-state storage per site from the implemented
  family, Higgs, SM-link, SM-momentum, and Higgs-link registers;
- verifies that the `3^3` certificate state is small enough in raw storage
  (`< 4 MiB`), so raw state size is not the spatial-echo blocker;
- computes the finite-difference Wilson-force coordinate count and global
  action-evaluation count;
- shows the current finite-difference plaquette workload scales quadratically
  in site count and is `5184x` larger than a local staple-force target on the
  `3^3` certificate lattice;
- treats the result as a scalability audit that selects the next engineering
  target, not as a performance benchmark or physics derivation.

Stage 36 verdict:

```text
QCA_SMV0_STAGE36_PHYSICAL_RIGHT_PRODUCTION_WORKLOAD_PASS
```

Stage 37 replaces the dense Wilson-force path with a local analytic staple:

- preserves the old centered finite-difference force as an explicit legacy
  oracle;
- makes `sm_left_wilson_force` dispatch to the local BCC plaquette-staple
  derivative;
- checks identity and pure-gauge zero force and nonflat curvature response;
- checks gauge covariance of the force coordinates;
- certifies representative coordinates against automatic differentiation of
  the exact Wilson action under left link updates;
- keeps a coarse finite-difference comparison as a sanity check while not
  using finite differences as the production path;
- reduces the Stage 37 certificate shape from `4608` finite-difference
  plaquette holonomies to `12` local plaquette holonomies (`384x`).

Stage 37 verdict:

```text
QCA_SMV0_STAGE37_PHYSICAL_RIGHT_PRODUCTION_LOCAL_FORCE_PASS
```

Stage 38 runs the actual production tick on a larger local-force certificate:

- builds the deterministic physical-right production state on a `2 x 2 x 1`
  lattice;
- advances one production tick using the Stage 37 local Wilson-force path;
- verifies the final state is finite and SM/Higgs links remain unitary;
- verifies the step is nontrivial in the Higgs field, SM links, and SM momenta;
- verifies changing the legacy `wilson_epsilon` keyword leaves the production
  step exactly unchanged, confirming the old finite-difference Wilson knob is
  no longer active in the tick;
- records a `768x` local-force plaquette-workload reduction relative to the
  legacy finite-difference path on the certificate lattice.

Stage 38 verdict:

```text
QCA_SMV0_STAGE38_PHYSICAL_RIGHT_PRODUCTION_LOCAL_ROLLOUT_PASS
```

Stage 39 runs a multi-step recorded rollout on the local-force path:

- uses the same `2 x 2 x 1` certificate lattice as Stage 38;
- records a two-step physical-right production rollout with both the Python
  loop runner and the `lax.scan` runner;
- verifies loop/scan final-state and observation agreement;
- verifies all recorded observations are finite, family norm drift remains
  bounded, and SM/Higgs link unitarity remains controlled;
- verifies Higgs, SM-link, and SM-momentum changes remain nontrivial over the
  recorded history;
- treats the result as a sparse-runner stability check, not as a large-lattice
  spatial echo measurement or performance benchmark.

Stage 39 verdict:

```text
QCA_SMV0_STAGE39_PHYSICAL_RIGHT_PRODUCTION_LOCAL_RECORDED_PASS
```

Stage 40 profiles the local-force production step:

- uses `sim.profile_callable` to produce a JSON-safe eager timing payload for
  one local-force production step on a `2 x 1 x 1` lattice;
- profiles a closed-over JIT version of the same step so `step_size` remains a
  static Python value;
- records separate JIT compile and cached-run timing fields;
- verifies eager and JIT outputs agree, the final state is finite, and
  SM/Higgs links remain unitary;
- treats all timing fields as machine-dependent diagnostics, not as strict
  performance claims.

Stage 40 verdict:

```text
QCA_SMV0_STAGE40_PHYSICAL_RIGHT_PRODUCTION_LOCAL_PROFILE_PASS
```

Stage 41 measures local Wilson-force spatial support numerically:

- perturbs one SM link on a `7 x 7 x 7` periodic lattice;
- computes the Stage 37 production local Wilson force;
- compares the measured support against the Stage 35 two-hop force envelope;
- finds support on `11` sites with radius distribution `(1, 7, 3)` over
  radii `(0, 1, 2)`;
- verifies the measured support radius is `2` and the detected norm outside
  radius `2` is zero;
- treats the result as a force-support measurement, not a full production-map
  large-lattice spatial echo or continuum causal-cone theorem.

Stage 41 verdict:

```text
QCA_SMV0_STAGE41_PHYSICAL_RIGHT_PRODUCTION_FORCE_SUPPORT_PASS
```

Stage 42 replaces the physical-right fermion current's production path with a
local analytic current:

- keeps the old finite-difference current as a one-site oracle;
- computes the production current as the local derivative of the same
  physical-right streaming bilinear;
- agrees with the finite-difference oracle at max residual `7.02e-6`;
- verifies the production current is independent of the old
  `fermion_current_epsilon` knob;
- runs a `2 x 2 x 1` production smoke tick with controlled family norm drift
  and SM/Higgs link unitarity;
- records an estimated `64x` dense-work reduction on that smoke lattice.

Stage 42 verdict:

```text
QCA_SMV0_STAGE42_PHYSICAL_RIGHT_PRODUCTION_LOCAL_CURRENT_PASS
```

Stage 43 measures one-tick spatial support of the assembled production map:

- perturbs family state, Higgs field, Higgs momentum, SM link, SM momentum, and
  Higgs link independently on a `7 x 7 x 7` periodic lattice;
- runs one full physical-right production tick for each perturbation;
- compares the combined per-site response against the Stage 35 one-tick
  production stencil radius `2`;
- measures maximum support radius `1`, with no detected support outside the
  predicted radius;
- records support counts `(8, 9, 1, 1, 1, 1)` for
  `(family, Higgs, Higgs momentum, SM link, SM momentum, Higgs link)`;
- treats the result as a one-tick finite-support audit, not as a multi-step
  spatial echo or continuum causal-cone theorem.

Stage 43 verdict:

```text
QCA_SMV0_STAGE43_PHYSICAL_RIGHT_PRODUCTION_SPATIAL_SUPPORT_PASS
```

Stage 44 measures finite-horizon family-cone growth of the assembled production
map:

- perturbs one family-state component at the center of a `7 x 7 x 7` periodic
  lattice;
- runs the full physical-right production map for one, two, and three ticks;
- measures support radii `(1, 2, 3)` for the three horizons;
- records support counts `(8, 22, 48)`;
- detects zero support outside the step-count cone at each horizon;
- treats the result as a finite-horizon discrete family-cone audit, not as a
  continuum light-cone theorem or all-sector propagation theorem.

Stage 44 verdict:

```text
QCA_SMV0_STAGE44_PHYSICAL_RIGHT_PRODUCTION_CONE_PASS
```

Stage 45 measures two-tick cone support for every production sector:

- perturbs family state, Higgs field, Higgs momentum, SM link, SM momentum, and
  Higgs link independently on a `5 x 5 x 5` periodic lattice;
- runs the full physical-right production map for two ticks for every
  perturbation;
- records support radii `(2, 1, 0, 0, 0, 1)` for
  `(family, Higgs, Higgs momentum, SM link, SM momentum, Higgs link)`;
- records support counts `(22, 9, 1, 1, 1, 2)`;
- detects zero support outside the two-step cone;
- treats the result as a finite-horizon all-sector cone audit, not as a
  continuum light-cone theorem, performance benchmark, or exact
  energy-conservation theorem.

Stage 45 verdict:

```text
QCA_SMV0_STAGE45_PHYSICAL_RIGHT_PRODUCTION_SECTOR_CONES_PASS
```

Stage 46 adds a momentum-only Gauss relaxation/projection precursor:

- treats the existing physical-right production Gauss residual as
  `G(P)=div_E(P)-rho` at fixed links and matter fields;
- computes the gradient of `0.5 ||G||^2` with respect to SM link momenta;
- chooses the exact one-dimensional least-squares line step along that
  gradient direction;
- leaves family state, Higgs field, Higgs momenta, SM links, and Higgs links
  unchanged;
- keeps the zero-source vacuum unchanged;
- reduces the deterministic Gauss norm from `5.939e-1` to `4.816e-1` in one
  relaxation step;
- treats the result as a constraint-solving precursor, not as a full nonlinear
  Gauss projection or Gauss-preserving production integrator.

Stage 46 verdict:

```text
QCA_SMV0_STAGE46_PHYSICAL_RIGHT_PRODUCTION_GAUSS_PROJECTION_PASS
```

Stage 47 iterates the momentum-only Gauss relaxation into a finite solver:

- repeats the Stage 46 exact-line relaxation for ten iterations;
- records Gauss norms, line steps, and gradient norms over the solver history;
- keeps the zero-source vacuum unchanged;
- reduces the deterministic Gauss norm monotonically from `5.939e-1` to
  `4.092e-1`;
- gives total reduction fraction `3.109e-1` with zero monotonicity violation;
- leaves family state, Higgs field, Higgs momenta, SM links, and Higgs links
  unchanged;
- treats the result as a finite frozen-state relaxation solver, not as a full
  nonlinear Gauss projection or Gauss-preserving production integrator.

Stage 47 verdict:

```text
QCA_SMV0_STAGE47_PHYSICAL_RIGHT_PRODUCTION_GAUSS_SOLVER_PASS
```

Stage 48 composes the current production tick with the Stage 47 finite
Gauss-relaxation solver:

- advances the deterministic state with the unprojected physical-right
  production tick;
- applies ten fixed-link, momentum-only Gauss relaxation iterations to the
  post-tick state;
- keeps the zero-source vacuum unchanged;
- reduces the post-tick deterministic Gauss norm from `5.938e-1` to
  `4.095e-1`;
- gives total reduction fraction `3.104e-1` with zero monotonicity violation;
- changes only SM link momenta relative to the unprojected production step;
- keeps SM and Higgs links unitary to float32 precision;
- treats the result as an explicit projected-step wrapper, not as a rewritten
  production tick, nonlinear gauge-orbit projection, or Gauss-preserving
  integrator.

Stage 48 verdict:

```text
QCA_SMV0_STAGE48_PHYSICAL_RIGHT_PRODUCTION_PROJECTED_STEP_PASS
```

Stage 49 iterates the projected production step over a short finite horizon:

- records a raw production rollout and a projected production rollout from the
  same deterministic initial state;
- applies ten fixed-link, momentum-only Gauss relaxation iterations after each
  projected tick;
- keeps the zero-source vacuum unchanged over the projected rollout;
- reduces the two-step deterministic final Gauss norm from raw `5.937e-1` to
  projected `3.787e-1`;
- gives final raw-vs-projected reduction fraction `3.621e-1`;
- each projected step has positive Gauss reduction and zero monotonicity
  violation inside its finite relaxation history;
- keeps projected SM and Higgs links unitary to float32 precision;
- treats the result as a finite projected-rollout audit, not as a
  Gauss-preserving integrator, exact energy-conservation theorem, or continuum
  stability theorem.

Stage 49 verdict:

```text
QCA_SMV0_STAGE49_PHYSICAL_RIGHT_PRODUCTION_PROJECTED_ROLLOUT_PASS
```

The charges, `lambda`, order-one coefficients, and center-power matrices are
simulator inputs, not BCC-bulk derivations. Quantized scalar/gauge registers,
boundary rules, microscopic derivation of the bridge, and derivation of the
simulator inputs are not implemented yet.

## Reuse Boundary

Allowed upstream infrastructure:

- `clifford_3plus2_d5.sim` for generic JAX state, links, scan runners, profiling,
  and persistence;
- `clifford_3plus2_d5.qca_smv0` local modules for the Stage 1/2 kernels.

This sidecar should not import from `spacetime_qca`, `lepton`, `cusp`, or other
theory sidecars.  If later work needs an older QCA kernel, copy/adapt the code
locally so this simulator can change orderings and performance boundaries
without changing the older sidecars.

Future sessions should keep imports narrow and run only the tests for the
current session.

## Run

```bash
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_01_bare_bcc_walk
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_02_static_sm_gauge_background
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_03_dynamic_sm_gauge
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_04_higgs_yukawa_collision
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_05_fn_recirculation
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_06_center_holonomy_cp
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_07_family_higgs_yukawa
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_08_higgs_dynamics
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_09_gauge_higgs_backreaction
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_10_fermion_higgs_backreaction
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_11_fermion_gauge_current
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_12_sourced_tick
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_13_family_gauge_current
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_14_family_sourced_tick
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_15_family_production_tick
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_16_gauge_convention_bridge
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_17_antiunitary_bridge
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_18_physical_right_transport
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_19_physical_right_current
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_20_physical_right_sourced_tick
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_21_physical_right_production_tick
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_22_physical_right_production_rollout
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_23_physical_right_production_gauss
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_24_physical_right_production_energy
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_25_physical_right_production_variational
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_26_physical_right_production_refinement
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_27_physical_right_production_adjoint
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_28_physical_right_production_inverse
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_29_physical_right_production_reversibility
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_30_physical_right_production_echo
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_31_physical_right_production_tangent
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_32_physical_right_production_echo_gram
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_33_physical_right_production_echo_scale
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_34_physical_right_production_echo_horizon
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_35_physical_right_production_stencil
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_36_physical_right_production_workload
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_37_physical_right_production_local_force
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_38_physical_right_production_local_rollout
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_39_physical_right_production_local_recorded
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_40_physical_right_production_local_profile
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_41_physical_right_production_force_support
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_42_physical_right_production_local_current
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_43_physical_right_production_spatial_support
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_44_physical_right_production_cone
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_45_physical_right_production_sector_cones
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_46_physical_right_production_gauss_projection
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_47_physical_right_production_gauss_solver
uv run pytest src/clifford_3plus2_d5/qca_smv0/tests -q
```
