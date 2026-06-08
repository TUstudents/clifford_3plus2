# cusp - Status

**Status**: CUSP-PLAN Targets A-D implemented.  MicroCUSP Sessions A-H now
derive q-stiffness, no-incoming retarded boundary, weak `Z2` branch parity,
color `Z3` center holonomy, independent SM center axes, the Schur semigroup,
`lambda_rec`, the Target-C module, and the Target-D topology inside the current
BCC boundary-register model.

## Verdict

```text
CUSP_TARGETS_A_D_MICRO_BOUNDARY_PASS
```

## Implemented

- [CUSP-PLAN.md](CUSP-PLAN.md) is the source theory/plan.
- Target A: the local BCC edge q-clock has two q-preserving same-normal
  branches and two q-changing leakage branches.  Session 03 imports the exact
  BB q=0/leakage norm split, `I/2 + I/2 = I`, and derives the weak/BCC source
  from the same-normal branch parity: order `2`, channel count `2`, one-step
  forbidden.  Session 04 stacks the same/mixed BB outgoing blocks into an
  `8 x 2` isometry, certifies finite unitary completion, makes mixed-normal
  feedback vanish under the q-mismatch hard-gap/outgoing condition, and rejects
  recurrent leakage because it has a nonzero two-step visible return.  The
  Session 05 origin audit derives `p(q)=g q^2` as the unique lowest-degree
  q-local, q-reflection-even positive penalty that vanishes on q=0; linear and
  constant controls fail.  It also selects the outgoing closure from the
  no-incoming causal rule and keeps recurrent leakage as the rejected control.
  The
  color `SU(3)` center supplies primitive closure `3`.
  Session 02 replaces fixed-length return paths with local center-charge
  recirculation automata: one tick advances center charge, and visible readout
  is allowed only at charge `0`.  The weak/BCC automaton returns first at `2`,
  the color automaton returns first at `3`, and the generated graph gives two
  length-2 weak returns and three length-3 color returns.  Closed-walk
  enumeration recovers primitive lengths
  `(2,3)`, `C[t^2,t^3]`, first
  valuations `(0,2,3)`, and gap `1`; `C[t]`, `C[t^2]`, `C[t^3]`, one-step-loop,
  weak-only, color-only, and wrong-color-length finite-graph controls are
  rejected.
- Target B: normalized color/weak shear is the unique solution and stable
  minimum of the squared mismatch functional for `(1+t)sqrt(2)=sqrt(3)`, giving
  `lambda_rec=sqrt(3/2)-1`.  Session 07 upgrades this from a formula
  certificate to a finite one-sided matching theorem: the ordinary reflection
  coefficient solves the two-sided scattering control
  `(1+r)sqrt(2)=(1-r)sqrt(3)`, but leaves a nonzero residual in the retarded
  recirculation equation, so it is dynamically excluded by the one-sided
  boundary readout.  Count-ratio and inverse-amplitude shears fail the same
  one-sided norm-matching equation.  No CKM data are used.
- Target C: SM hypercharge conservation forces the up Yukawa to use `H_tilde`
  and the down Yukawa to use `H`; swapped doors are rejected.  Given that door
  orientation, the semigroup `S=<2,3>` derives conductor `c=2` and Frobenius
  gap `1`.  Session 08 selects the down right charges as the conductor-ideal
  residue `D=max(Q-c,0)=(1,0,0)` and selects the up lift factor from the
  weak/BCC primitive closure order `2`, giving `U=(5,2,0)`.  Wrong-conductor,
  trivial-lift, and color-order-lift controls miss the exponent skeleton.  A
  finite solver confirms these are the unique nonnegative right charges for the
  CUSP-PLAN diagonal exponent targets, but the solver is now a consistency
  check rather than the source of the rule.  The resulting diagonal FN powers
  are `(8,4,0)` and `(4,2,0)`, with CKM powers `V_us=lambda`,
  `V_cb=lambda^2`, `V_ub=lambda^3`.
- Target D: coefficients are finite path sums
  `c_ij=sum_gamma A_gamma Omega_gamma` with `Omega_gamma` in the color center.
  Session 10 selects the up center powers as geodesic distances on the
  non-cyclic length-3 cusp flag and the down center powers as the unit bilinear
  pairing on `F3` center labels.  The rule-pair witness gives nonzero
  `Im tr([YuYu^dagger,YdYd^dagger]^3)`; the all-real, one-sector, and separable
  row/column controls give zero, and a full field-rephasing check with
  common-left/up-right/down-right center phases leaves the invariant unchanged.
  Session 09 derives the positive amplitude weights from cusp-module path
  counts, `A_ij=max(1,# decompositions of q_i+r_j in <2,3>)`, treating the gap
  exponent `1` as one irreducible conductor-module/contact path.  This derived
  measure remains CP-active, while the all-real control with the same
  amplitudes remains CP-zero.  The older fixed non-unit positive deformation is
  kept only as a robustness control.
- MicroCUSP Session A: the q-local stiffness assumption is upgraded to a
  local two-normal material model with stiffness matrix
  `2g [[1,-1],[-1,1]]`.  Its microscopic action is `g(r1-r2)^2`, giving
  effective mismatch action `gq^2`, zero trace-mode energy, no linear term,
  positive q-curvature `2g`, and mixed-normal gap `4g`.  Zero-stiffness,
  q-odd linear, and nonanalytic `g|q|` controls are rejected.
- MicroCUSP Session B: the no-incoming retarded asymptotic condition is
  upgraded to a local outgoing-boundary audit.  The hidden mixed-normal return
  matrix is `R=0`, so the local Schur correction `G R M` vanishes and visible
  powers match the q=0 survival powers.  Recurrent leakage (`R=I`), hard-wall
  reflection (`R=-I`), and incoming/outgoing symmetric closure (`R=I/2`) all
  produce nonzero visible mixed-normal feedback and are rejected.
- MicroCUSP Session C: the weak/BCC `Z2` automaton is derived from same-normal
  branch parity.  The common signs `s=sigma1=sigma2=(+1,-1)` form a local
  `Z2` group; one primitive tick is charged, visible readout first returns at
  two ticks, and trivial/order-one/weak-only controls are rejected.
- MicroCUSP Session D: the color `Z3` automaton is derived from closed
  `SU(3)` center holonomy.  One- and two-tick paths carry nontrivial center
  phase, the three-tick holonomy is neutral, and wrong-length, spectator-color,
  and gauged-away-open-phase controls are rejected.
- MicroCUSP Session 15: the SM global quotient gate compares independent
  boundary center axes `(1,0),(0,1)` against a correlated quotient-diagonal
  tick `(1,1)`.  Independent axes give primitive lengths `(2,3)` and basis
  `1,t^2,t^3`; the diagonal `Z6` control gives `(0,6,12)` and the
  U(1)-collapsed control gives `(0,1,2)`, so both controls are rejected.
- MicroCUSP Session 16: microscopic Schur return moments satisfy `M1=0`,
  `M2!=0`, and `M3!=0`, recovering primitive semigroup `(2,3)`, algebra
  `C[t^2,t^3]`, module basis `1,t^2,t^3`, and low valuations `(0,2,3)`;
  neighboring semigroup controls are rejected.
- MicroCUSP Session 17: the weak/color return moments `2` and `3` select
  `lambda_rec=sqrt(3/2)-1` through the one-sided retarded matching equation;
  ordinary reflection, count-ratio shear, and inverse-amplitude shear fail.
- MicroCUSP Session 18: Target C is recovered from microscopic modules, not
  exponent targets: `(0,2,3)` gives `Q=(3,2,0)`, conductor `c=2` gives
  `D=(1,0,0)`, and the weak double cover gives `U=(5,2,0)`.  Wrong-conductor,
  trivial-lift, color-order-lift, diagonal-target, and mass-fit controls are
  rejected.
- MicroCUSP Session 19: Target D topology is recovered from microscopic
  topology: up center powers are flag geodesic distances on the non-cyclic
  cusp flag, down powers are the unit bilinear pairing of `F3` color-center
  labels, the derived CP invariant is nonzero, and all-real / one-sector /
  separable controls are zero.

## Honest Boundary

- Target A/B are exact certificates inside the minimal recirculation model.
- Target C gives a microscopic module audit for the right charges inside the
  current boundary-register model; it does not use diagonal target exponents as
  inputs.
- Target D now has microscopic topology selection and a finite cusp-module
  amplitude measure inside the current boundary-register model.
- The q-local positive q-reflection stiffness, no-incoming retarded
  asymptotics, weak `Z2` branch parity, color `Z3` center holonomy, and SM
  global quotient behavior now have local microscopic audits.  The
  conductor/weak-cover dynamics and Target-D center topology are also derived
  by MicroCUSP Sessions 18 and 19.  No MicroCUSP A-H gates remain open.
