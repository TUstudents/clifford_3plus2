# QCA_SMv0 Status

## Verdict

```text
QCA_SMV0_STAGE12_SOURCED_SM_TICK_PASS
```

## Current State

Stage 12 coupled sourced SM gauge tick implemented on top of the Stage 1
free BCC Weyl/Dirac walk, Stage 2 static gauge transport, Stage 3 pure dynamic
gauge fields, Stage 4 local Higgs/Yukawa collision, Stage 5 FN recirculation,
Stage 6 center-holonomy CP, Stage 7 three-family Higgs/Yukawa collision, and
Stage 8 dynamic Higgs-field evolution, Stage 9 gauge-Higgs backreaction, and
Stage 10 local fermion/Higgs backreaction, and Stage 11 BCC streaming fermion
gauge current.

Implemented:

- package scaffold;
- documentation scaffold;
- `scripts/` and `tests/` directories;
- package metadata constants and free-walk exports;
- two-component BCC Weyl state layout `(nx, ny, nz, 2)`;
- Pauli/projector helpers;
- eight BCC hop matrices;
- pull-convention free BCC Weyl step;
- split-product Bloch symbol and hop-sum Bloch symbol;
- norm, hop-completeness, symbol-unitarity, small-k, and anisotropy-scaling
  diagnostics;
- four-component massless Dirac assembly from opposite-chirality Weyl blocks;
- Dirac hop-completeness, symbol-unitarity, norm, dispersion, and JIT tests;
- Session 01 script;
- local 32-component SM internal carrier;
- anti-Hermitian `SU(3)_c x SU(2)_L x U(1)_Y` generator basis;
- static BCC SM link fields with shape `(nx, ny, nz, 8, 32, 32)`;
- local state/link gauge transformations;
- static gauge-covariant Dirac BCC transport;
- identity-link reduction to the free internal Dirac step;
- Wilson plaquette traces over selected BCC parallelograms;
- weak-link linearization check for the continuum covariant derivative;
- Session 02 script;
- real SM link momenta with shape `(nx, ny, nz, 8, 12)`;
- algebra-coordinate projection and momentum gauge transforms;
- finite-difference left Wilson force;
- pure-gauge Hamiltonian density;
- compact momentum update and reversible leapfrog step;
- electric-divergence / Gauss diagnostic;
- pure-gauge zero-momentum Gauss-preservation audit;
- weak-field plaquette field-strength linearization;
- weak-field Wilson-action / Yang-Mills-density matching;
- no-backreaction fermion/gauge wrapper with spectator fermion transport;
- Session 03 script;
- local Higgs doublet fields with shape `(nx, ny, nz, 2)`;
- constant unitary-gauge Higgs helper `H=(0,v/sqrt(2))`;
- conjugate Higgs helper `H_tilde=i sigma_2 H^*`;
- Hermitian one-generation Yukawa matrix on the 32-component SM carrier;
- unitary-gauge SM door audit: `H_tilde` opens up/neutrino couplings and `H`
  opens down/electron couplings;
- exact site-local unitary collision `exp(-i step_size beta Y(H))`;
- zero-step and zero-Higgs identity controls;
- chirality-flip and massive-dispersion diagnostics;
- Session 04 script;
- family dimension `3`;
- explicit FN attenuation inputs for empirical Wolfenstein and shear-candidate
  modes;
- local two-state unitary beam splitter for one recirculation step;
- finite hidden-path unitary chains whose endpoint transfer is `lambda^n`;
- default simulator charges `Q=(3,2,0)`, `U=(5,2,0)`, `D=(1,0,0)`;
- quark path-length matrices `n^u_ij=Q_i+U_j`, `n^d_ij=Q_i+D_j`;
- diagonal FN scaling audits for `lambda^8:lambda^4:1` and
  `lambda^4:lambda^2:1`;
- left-frame Wolfenstein scaling matrix `lambda^abs(Q_i-Q_j)`;
- generated quark Yukawa matrices `Y_ij=c_ij lambda^(Q_i+R_j)`;
- singular masses and CKM-like left-frame mismatch from the same generated
  matrices;
- Session 05 script;
- `SU(3)_c` center phase helper `omega=exp(2 pi i / 3)`;
- explicit center-power matrices for up/down FN coefficient phases;
- center phase unit-modulus and `phase^3=1` audits;
- center-decorated FN coefficient matrices preserving order-one magnitudes;
- antiparticle Yukawas as complex conjugates;
- quark/antiquark singular-mass equality audit;
- nonzero CKM Jarlskog from center-decorated coefficients;
- all-real coefficient control with zero Jarlskog;
- nonzero CP-odd commutator trace for center-decorated Yukawas;
- Session 06 script;
- family-extended fermion state layout `(nx, ny, nz, 4, 32, 3)`;
- local internal/family dimension `96`;
- exact embedding of generated quark Yukawa matrices into local Higgs doors;
- explicit diagonal placeholder lepton matrices as simulator inputs;
- Hermitian three-family local Yukawa matrix;
- unitary-gauge quark block extraction and wrong-door controls;
- CKM-like left-frame mismatch consistency from embedded blocks;
- exact local three-family collision `exp(-i step_size beta Y_family(H))`;
- zero-step, zero-Higgs, norm, chirality-flip, and JIT audits;
- Session 07 script;
- Higgs electroweak `SU(2)_L x U(1)_Y` generator basis on the doublet;
- finite Higgs BCC links with shape `(nx, ny, nz, 8, 2, 2)`;
- Higgs field and momentum layout `(nx, ny, nz, 2)`;
- kinetic, gauge-covariant gradient, and quartic-potential energy densities;
- covariant Higgs force;
- unitary-gauge vacuum force audit;
- pure-gauge vacuum gradient-energy audit;
- force covariance and Hamiltonian gauge-invariance checks;
- reversible no-fermion Higgs leapfrog update;
- small-step Hamiltonian drift and JIT audits;
- Session 08 script;
- Higgs electroweak link momenta with shape `(nx, ny, nz, 8, 4)`;
- projection between Higgs algebra matrices and electroweak coordinates;
- target-site adjoint transforms for Higgs link momenta;
- Higgs link-momentum kinetic density;
- left-trivialized Higgs gauge force from the covariant-gradient energy;
- zero-force covariantly constant vacuum control;
- nonzero scalar gauge-current control for deterministic non-vacuum/nonflat
  fields;
- local Higgs charge density;
- Higgs Gauss diagnostic as electric divergence minus Higgs charge;
- covariance checks for the Higgs gauge force, charge density, and Gauss
  diagnostic;
- embedding of Higgs electroweak momenta into the full SM 12-generator momentum
  layout with zeros on color and components on `8..11`;
- coupled no-fermion Higgs/gauge leapfrog update;
- coupled link-unitarity, reversibility, small Hamiltonian-drift, and JIT
  audits;
- Session 09 script;
- local Yukawa energy density `E_Y=Re psi^dagger beta Y(H) psi` from the same
  Stage 7 three-family Yukawa matrix used by the collision;
- complex-field Higgs source force `-dE_Y/dH*` from real/imaginary automatic
  differentiation;
- deterministic fermion source state with nonzero Yukawa bilinears;
- zero-state and nonzero-source controls;
- explicit Yukawa-door electroweak gauge helper using physical right-handed
  hypercharges for the local Yukawa covariance audit;
- Yukawa energy gauge-invariance and Higgs-source covariance checks in that
  local door convention;
- reversible Higgs-momentum source kick;
- local collision-plus-kick wrapper;
- fermion norm, source-after-collision, and JIT audits;
- Session 10 script;
- BCC streaming bilinear `E_stream=Re sum_x,h psi[x]^dag D_h U_h[x]
  psi[x+h]`;
- left-trivialized fermion gauge current from the streaming bilinear;
- zero-state and nonzero-current controls;
- streaming-energy gauge-invariance audit;
- target-site adjoint covariance for the fermion current;
- local streaming fermion charge density;
- fermion Gauss diagnostic as electric divergence minus fermion charge;
- zero-state / zero-momentum Gauss control;
- reversible fermion-current momentum kick;
- kick-then-transport wrapper;
- kicked-link unitarity, spectator fermion norm, and JIT audits;
- Session 11 script;
- coupled sourced SM link force as the sum of Wilson force, embedded Higgs
  gauge force, and BCC streaming fermion current;
- side-by-side SM transport links and Higgs electroweak links updated from one
  12-coordinate SM momentum field;
- Higgs electroweak site-coordinate embedding into full SM coordinates on
  `8..11`;
- zero-source and nonzero-source controls for the sourced link force;
- shared electroweak covariance checks for the sourced force and sourced Gauss
  diagnostic;
- sourced Gauss diagnostic as electric divergence minus fermion charge minus
  embedded Higgs charge;
- zero-state/vacuum/zero-momentum Gauss control;
- reversible sourced SM momentum kick;
- coupled tick with fermion transport, Higgs-field advance, SM/Higgs link
  unitarity, fermion norm, and JIT audits;
- Session 12 script;
- focused tests for algebra, norm preservation, local gauge covariance, Wilson
  response, weak-link scaling, pure-gauge dynamics, Gauss covariance and
  preservation, weak-field Yang-Mills behavior, Higgs/Yukawa door structure,
  chirality flip, massive dispersion, FN recirculation powers, generated
  Yukawa matrices, center-holonomy CP, three-family Higgs/Yukawa embedding,
  dynamic Higgs evolution, gauge-Higgs backreaction, local fermion/Higgs
  backreaction, BCC streaming fermion gauge current, coupled sourced SM gauge
  tick, JIT compatibility, and small-momentum Weyl/Dirac behavior.

Not implemented:

- boundary rules;
- family-summed BCC streaming fermion current on gauge links;
- full merge of three-family Yukawa/Higgs source with the sourced gauge tick;
- quantized scalar registers;
- derivation of the FN charges, `lambda`, order-one coefficients, or
  center-power matrices from BCC bulk dynamics;
- derivation of Higgs potential parameters from BCC bulk dynamics;
- performance profiles beyond the small
  Session 01/02/03/04/05/06/07/08/09/10/11 diagnostics.

## Working Boundary

This sidecar is meant to focus the next simulator prototype around recent
sidecar work while reusing `sim` and local `qca_smv0` kernels only.  It does
not import from `spacetime_qca` or theory sidecars.

Run only focused `qca_smv0` tests while developing this sidecar.
