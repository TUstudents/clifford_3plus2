# QCA_SMv0 Status

## Verdict

```text
QCA_SMV0_STAGE34_PHYSICAL_RIGHT_PRODUCTION_ECHO_HORIZON_PASS
```

## Current State

Stage 34 physical-right production echo horizon implemented on top of the Stage 1
free BCC Weyl/Dirac walk, Stage 2 static gauge transport, Stage 3 pure dynamic
gauge fields, Stage 4 local Higgs/Yukawa collision, Stage 5 FN recirculation,
Stage 6 center-holonomy CP, Stage 7 three-family Higgs/Yukawa collision, and
Stage 8 dynamic Higgs-field evolution, Stage 9 gauge-Higgs backreaction, and
Stage 10 local fermion/Higgs backreaction, and Stage 11 BCC streaming fermion
gauge current, Stage 12 coupled sourced SM gauge tick, and Stage 13
family-summed BCC fermion gauge current, and Stage 14 family-sourced SM gauge
tick, Stage 15 full family production tick, and Stage 16 gauge-convention
bridge audit, Stage 17 antiunitary singlet bridge, and Stage 18
physical-right bridged transport, and Stage 19 physical-right bridged fermion
current, Stage 20 physical-right sourced gauge tick, and Stage 21
physical-right production tick, Stage 22 physical-right production rollout,
Stage 23 physical-right production Gauss monitor, Stage 24 physical-right
production energy monitor, Stage 25 variational force audit, Stage 26
refinement limitation audit, Stage 27 adjoint limitation audit, and Stage 28
explicit inverse helper, Stage 29 trajectory reversibility audit, and Stage 30
Loschmidt echo diagnostic, Stage 31 tangent echo audit, and Stage 32 echo Gram
audit, Stage 33 echo-Gram scale-stability audit, and Stage 34 finite-horizon
echo-spectrum audit.

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
- family-summed BCC streaming current on the Stage 7 family carrier
  `(nx, ny, nz, 4, 32, 3)`;
- family-blind SM link transport over the family register;
- single-family embedding/extraction helpers;
- one-family reduction controls against the Stage 11 energy, current, charge,
  and transport;
- family-summed local fermion charge density;
- family fermion Gauss diagnostic as electric divergence minus family-summed
  charge;
- family-current covariance, reversible momentum kick, kick-then-transport,
  link unitarity, family-state norm, and JIT audits;
- Session 13 script;
- family-sourced SM link force as the sum of Wilson force, embedded Higgs gauge
  force, and family-summed BCC fermion current;
- one-family reduction controls against the Stage 12 sourced force, sourced
  Gauss diagnostic, and tick;
- family-sourced Gauss diagnostic as electric divergence minus family-summed
  fermion charge minus embedded Higgs charge;
- shared electroweak covariance checks for family-sourced force and Gauss;
- reversible family-sourced SM momentum kick;
- family-sourced tick with family transport, Higgs-field advance, SM/Higgs link
  unitarity, family-state norm, and JIT audits;
- Session 14 script;
- full family production tick merging the Stage 10 local Yukawa/Higgs source
  and exact Stage 7 family Yukawa collision into the Stage 14 family-sourced
  tick;
- zero-Yukawa reduction controls against the Stage 14 family-sourced tick;
- production Higgs force as Higgs dynamics plus local Yukawa Higgs source;
- nonzero deterministic Yukawa-source and zero-state/vacuum-source controls;
- reversible production Higgs momentum kick for frozen fields;
- production tick with symmetric local half-collision, BCC family transport,
  Higgs-field advance, SM/Higgs link unitarity, family-state norm, and JIT
  audits;
- explicit status boundary preserving the separation between the Stage 10
  physical local Higgs-door gauge convention and the Stage 14 transport/current
  gauge convention;
- Session 15 script;
- explicit Stage 2 transport electroweak generators and Stage 10 physical
  Yukawa-door electroweak generators;
- finite exponentiation from the Stage 16 Yukawa-door generators reproduces the
  Stage 10 Yukawa-door helper;
- exact generator agreement on duplicated left doublet labels;
- exact zero `SU(2)_L` action on right singlet labels in both conventions;
- exact charge-conjugation relation for right-singlet hypercharge;
- nonzero full generator difference and nonzero sorted-hypercharge spectral
  mismatch proving there is no unitary similarity on the fixed 32-component
  carrier;
- physical Yukawa-door energy covariance and transport-convention
  non-invariance controls;
- JIT audits for the gauge-convention energy residuals;
- Session 16 script;
- physical-right full SM generators built from Stage 2 transport generators by
  left-linear / right-antilinear projector bridge;
- exact left-linear and right-antilinear generator residuals;
- electroweak slice matching the Stage 10/16 physical Yukawa-door generators;
- finite physical-right site gauge equal to the projected antiunitary finite
  bridge from the transport gauge;
- finite bridge unitarity audit;
- full physical-right gauge covariance for the local Yukawa energy and
  unbridged transport-gauge non-invariance control;
- JIT audits for the full bridge energy residuals;
- Session 17 script;
- physical-right finite BCC link bridge
  `U_phys=P_L U_transport P_L + P_R conj(U_transport) P_R`;
- identity-link bridge control;
- finite bridged-link unitarity audit;
- bridged-link covariance under physical-right site gauges;
- family BCC transport through physical-right bridged links;
- identity-link reduction to the existing family transport;
- physical-right family transport covariance and norm-preservation audits;
- nontrivial transport-kernel difference from the unbridged Stage 13 transport
  convention;
- JIT audit for the physical-right transport kernel;
- Session 18 script;
- physical-right streaming bilinear through bridged finite BCC links;
- left-trivialized finite-difference current in the same 12 transport
  coordinates used by the Stage 2 gauge links and Stage 3 momenta;
- zero-state and nonzero-current controls;
- nontrivial current difference from the unbridged Stage 13
  transport-convention family current;
- physical-right family charge density;
- physical-right electric-divergence / Gauss diagnostic;
- covariance checks for streaming energy, current, charge, and Gauss under the
  bridged physical-right site gauge;
- reversible physical-right current momentum kick;
- kick-then-physical-right-transport wrapper;
- kicked-link unitarity, spectator family-state norm, and JIT audits;
- Session 19 script;
- physical-right sourced link force as the sum of Wilson force, embedded Higgs
  gauge force, and physical-right family current;
- nontrivial force difference from the Stage 14 transport-convention
  family-sourced force;
- physical-right sourced Gauss diagnostic as physical-right electric
  divergence minus physical-right family charge minus embedded Higgs charge;
- covariance checks for the physical-right sourced force and Gauss diagnostic;
- zero-source and deterministic nonzero-source controls;
- reversible physical-right sourced SM momentum kick;
- coupled physical-right sourced tick with physical-right family transport,
  Higgs-field advance, SM/Higgs link unitarity, family-state norm, and JIT
  audits;
- nontrivial state difference from the Stage 14 transport-convention sourced
  tick;
- Session 20 script;
- physical-right production tick merging the Stage 10 local Yukawa/Higgs source
  and exact Stage 7 family Yukawa collision into the Stage 20 physical-right
  sourced tick;
- zero-Yukawa reduction controls against the Stage 20 physical-right sourced
  tick;
- production Higgs force as Higgs dynamics plus local Yukawa Higgs source;
- nonzero deterministic Yukawa-source and zero-state/vacuum-source controls;
- reversible production Higgs momentum kick for frozen fields;
- production tick with symmetric local half-collision, physical-right BCC
  family transport, Higgs-field advance, SM/Higgs link unitarity, family-state
  norm, and JIT audits;
- nontrivial difference from the Stage 15 transport-convention production tick;
- Session 21 script;
- physical-right production rollout state carrying family fermions, Higgs
  fields/momenta, SM links and momenta, and Higgs-representation links;
- direct final-state rollout using the Stage 21 physical-right production tick;
- sparse recorded rollout using the shared `sim.runner` loop/scan interface;
- rollout observables for family norm, Higgs norm, Higgs momentum norm, SM
  momentum norm, and SM/Higgs link unitarity;
- one-step direct-tick agreement and loop/scan agreement audits;
- multi-tick family-norm drift and link-unitarity controls;
- zero-Yukawa rollout difference control proving production remains active
  under iteration;
- Session 22 script;
- physical-right production Gauss monitor reading the existing Stage 20
  physical-right sourced Gauss diagnostic on Stage 21/22 production states;
- zero-source vacuum rollout control with zero Gauss norm throughout;
- deterministic production Gauss history with finite nonzero Gauss signal;
- final-Gauss contrast between default production and zero-Yukawa rollout;
- monitored rollout family-norm and SM/Higgs link-unitarity controls;
- Session 23 script;
- physical-right production energy monitor combining pure SM gauge Hamiltonian,
  Higgs Hamiltonian, physical-right streaming bilinear, and local
  three-family Yukawa energy;
- exact vacuum-zero monitored-total-energy control;
- deterministic component-energy history with finite nonzero signal;
- default-vs-zero-Yukawa monitored-total-energy contrast;
- monitored rollout family-norm control;
- Session 24 script;
- repaired Higgs force normalization so `sm_higgs_force` is the
  `-dE/dphi*` force of the monitored Higgs Hamiltonian density;
- physical-right production variational audit for force provenance;
- exact sourced-link force decomposition into Wilson force, embedded Higgs
  gauge force, and physical-right family current;
- exact production-Higgs force decomposition into Higgs Hamiltonian force plus
  local Yukawa Higgs source;
- selected color-link central-difference check against Wilson plus
  physical-right streaming energy;
- selected complex Higgs-field central-difference check against Higgs plus
  Yukawa energy;
- zero-source vacuum force controls and deterministic nonzero-force controls;
- Session 25 script;
- fixed-time timestep refinement audit for the monitored physical-right
  production total energy;
- comparison of `dt` and `dt/2` rollouts at matched physical time;
- finite and controlled refined rollout despite non-improving monitored
  energy drift;
- zero-source vacuum refined rollout with zero monitored energy drift;
- explicit limitation verdict: the current hybrid production tick should not
  be claimed as an energy-convergent Hamiltonian integrator;
- Session 26 script;
- physical-right BCC family transport adjoint step;
- exact adjoint audit for the frozen production fermion substage;
- local family Yukawa collision inverse audit under `step_size -> -step_size`;
- frozen fermion substage norm-preservation audit;
- explicit limitation verdict: a full production tick followed by a
  negative-timestep production tick is not the inverse of the full map;
- Session 27 script;
- explicit inverse helper for the current physical-right production tick;
- reconstruction of half-step momenta from final production forces;
- backward sourced link update and Higgs-field rewind;
- Stage 27 frozen fermion-stage adjoint inside the full inverse;
- recovery of initial momenta from reconstructed first production forces;
- forward-then-inverse and inverse-then-forward roundtrip audits to float32
  precision;
- large improvement over the naive negative-timestep production tick;
- restored family-norm and SM/Higgs link-unitarity controls;
- JIT audit for the explicit inverse helper;
- explicit boundary: the inverse is for the current discrete map and is not an
  energy-conservation, timestep-convergence, boundary, Gauss-projection,
  quantized-register, performance, or microscopic-input derivation claim;
- Session 28 script;
- multi-step inverse rollout by repeated Stage 28 inverse steps;
- trajectory roundtrip audit: forward rollout then inverse rollout restores
  the initial state;
- trajectory replay audit: restored initial state, advanced again, reproduces
  the stored final state;
- intermediate path-restore audit against the stored forward trajectory;
- multi-step naive negative-timestep control and improvement ratio;
- forward/inverse family-norm and SM/Higgs link-unitarity controls;
- fixed-step JIT audit for the inverse rollout;
- explicit boundary: trajectory reversibility is for the current discrete map
  and is not an energy-convergent Hamiltonian-integrator theorem;
- Session 29 script;
- Loschmidt echo diagnostic using the Stage 29 inverse trajectory;
- local final-time SM momentum perturbation at the trajectory apex;
- finite nonzero initial-surface echo after inverse rollback;
- doubled-perturbation control with locally linear doubled echo;
- unperturbed base roundtrip and perturbed inverse link-unitarity controls;
- explicit boundary: the echo is a stability diagnostic for the current
  discrete production map, not a new dynamics rule or energy-convergence
  theorem;
- Session 30 script;
- finite tangent-response echo diagnostic using two independent final-time
  perturbations;
- SM-momentum and Higgs-momentum final-time kicks pulled back separately by
  the explicit inverse trajectory;
- combined-kick pullback compared against the sum of the two separate echoes;
- finite nonzero echo norms and small superposition residual;
- combined perturbed inverse link-unitarity controls;
- explicit boundary: the tangent echo is a local finite-difference diagnostic
  for the current discrete production map, not a new dynamics rule or
  conservation theorem;
- Session 31 script;
- local echo Gram matrix built from three independent final-time perturbations
  pulled back through the inverse trajectory;
- finite nonzero echo norms, exact Gram symmetry, positive minimum eigenvalue,
  finite condition number, and bounded off-diagonal correlations;
- inverse-pulled echo-state SM/Higgs link-unitarity controls;
- explicit boundary: the echo Gram is a finite local tangent-metric diagnostic
  for the current discrete production map, not a continuum stability theorem;
- Session 32 script;
- echo-Gram scale-stability audit comparing `epsilon` and `2 epsilon`;
- linear scaling check for echo norms;
- quadratic scaling check for Gram eigenvalues;
- dimensionless stability checks for Gram condition number and off-diagonal
  correlations;
- base-roundtrip and inverse-pulled link-unitarity controls at both scales;
- explicit boundary: the scale audit is finite-difference evidence for the
  current diagnostic regime, not a continuum stability theorem;
- Session 33 script;
- finite-horizon echo-spectrum audit comparing one-tick and two-tick production
  trajectories;
- echo gains derived from the Stage 32 Gram eigenvalues at both horizons;
- bounded finite-horizon gain growth on the certificate trajectory;
- dimensionless stability checks for Gram condition number and off-diagonal
  correlations across horizons;
- base-roundtrip and inverse-pulled link-unitarity controls at both horizons;
- explicit boundary: the horizon audit is a finite diagnostic for the current
  discrete production map, not a continuum Lyapunov theorem;
- Session 34 script;
- focused tests for algebra, norm preservation, local gauge covariance, Wilson
  response, weak-link scaling, pure-gauge dynamics, Gauss covariance and
  preservation, weak-field Yang-Mills behavior, Higgs/Yukawa door structure,
  chirality flip, massive dispersion, FN recirculation powers, generated
  Yukawa matrices, center-holonomy CP, three-family Higgs/Yukawa embedding,
  dynamic Higgs evolution, gauge-Higgs backreaction, local fermion/Higgs
  backreaction, BCC streaming fermion gauge current, coupled sourced SM gauge
  tick, family-summed BCC fermion gauge current, family-sourced SM gauge tick,
  full family production tick, gauge-convention bridge audit, antiunitary
  singlet bridge, physical-right bridged transport, physical-right bridged
  fermion current, physical-right sourced gauge tick, physical-right production
  tick, physical-right production rollout, physical-right production Gauss
  monitor, physical-right production energy monitor, physical-right production
  variational audit, physical-right production refinement limitation audit, JIT
  compatibility, physical-right production adjoint limitation audit,
  physical-right production explicit inverse audit, physical-right production
  trajectory reversibility audit, physical-right production Loschmidt echo
  audit, physical-right production tangent echo audit, physical-right
  production echo Gram audit, physical-right production echo-Gram scale audit,
  physical-right production echo horizon audit, and
  small-momentum Weyl/Dirac behavior.

Not implemented:

- boundary rules;
- quantized scalar registers;
- microscopic BCC derivation of the antiunitary bridge;
- derivation of the FN charges, `lambda`, order-one coefficients, or
  center-power matrices from BCC bulk dynamics;
- derivation of Higgs potential parameters from BCC bulk dynamics;
- Gauss projection / constraint-solving dynamics;
- exact conservation claim for the full hybrid production energy monitor;
- timestep-refined energy-convergence claim for the current hybrid production
  tick;
- energy-conserving or timestep-convergent reversible integrator claim for the
  current hybrid production tick beyond the explicit inverse helper;
- performance profiles beyond the small
  Session 01/02/03/04/05/06/07/08/09/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34 diagnostics.

## Working Boundary

This sidecar is meant to focus the next simulator prototype around recent
sidecar work while reusing `sim` and local `qca_smv0` kernels only.  It does
not import from `spacetime_qca` or theory sidecars.

Run only focused `qca_smv0` tests while developing this sidecar.
