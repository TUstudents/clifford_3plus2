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

Stage 20 implements the free BCC Weyl/Dirac bulk walk, static
Standard-Model gauge-background transport, pure dynamic SM gauge fields, and a
site-local Higgs/Yukawa collision, finite-path FN recirculation, and
center-holonomy CP coefficients, a three-family Higgs/Yukawa collision, and
dynamic Higgs-field evolution with scalar gauge and local fermion/Higgs
backreaction, plus BCC streaming fermion gauge currents, the merged
family-production tick, the gauge-convention bridge audit, and the
antiunitary singlet bridge, physical-right bridged transport, and the
physical-right bridged fermion current and sourced gauge tick:

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
- quark path lengths `n^u_ij=Q_i+U_j` and `n^d_ij=Q_i+D_j`;
- default charges `Q=(3,2,0)`, `U=(5,2,0)`, `D=(1,0,0)`;
- standard diagonal FN orders `lambda^8:lambda^4:1` and
  `lambda^4:lambda^2:1`;
- left-frame Wolfenstein scaling `lambda^abs(Q_i-Q_j)`;
- generated matrices `Y_ij=c_ij lambda^(Q_i+R_j)`;
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
- Stage 5/6 generated quark Yukawa matrices embedded in the local Higgs doors;
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

The charges, `lambda`, order-one coefficients, and center-power matrices are
simulator inputs, not BCC-bulk derivations. Quantized scalar/gauge registers,
boundary rules, microscopic derivation of the bridge, and rewriting the
production tick on the bridged carrier are not implemented yet.

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
uv run pytest src/clifford_3plus2_d5/qca_smv0/tests -q
```
