# QCA_SMv0 Status

## Verdict

```text
QCA_SMV0_STAGE7_FAMILY_HIGGS_YUKAWA_PASS
```

## Current State

Stage 7 three-family Higgs/Yukawa collision implemented on top of the Stage 1
free BCC Weyl/Dirac walk, Stage 2 static gauge transport, Stage 3 pure dynamic
gauge fields, Stage 4 local Higgs/Yukawa collision, Stage 5 FN recirculation,
and Stage 6 center-holonomy CP.

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
- focused tests for algebra, norm preservation, local gauge covariance, Wilson
  response, weak-link scaling, pure-gauge dynamics, Gauss covariance and
  preservation, weak-field Yang-Mills behavior, Higgs/Yukawa door structure,
  chirality flip, massive dispersion, FN recirculation powers, generated
  Yukawa matrices, center-holonomy CP, three-family Higgs/Yukawa embedding,
  JIT compatibility, and small-momentum Weyl/Dirac behavior.

Not implemented:

- boundary rules;
- matter backreaction into gauge momenta;
- dynamic Higgs-field evolution / Higgs potential;
- derivation of the FN charges, `lambda`, order-one coefficients, or
  center-power matrices from BCC bulk dynamics;
- performance profiles beyond the small Session 01/02/03/04/05/06/07 diagnostics.

## Working Boundary

This sidecar is meant to focus the next simulator prototype around recent
sidecar work while reusing `sim` and local `qca_smv0` kernels only.  It does
not import from `spacetime_qca` or theory sidecars.

Run only focused `qca_smv0` tests while developing this sidecar.
