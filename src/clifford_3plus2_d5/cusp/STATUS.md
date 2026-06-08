# cusp - Status

**Status**: CUSP-PLAN Targets A-D implemented at finite model level.  Session
06 remains open as a post-target microscopic boundary-material gate.

## Verdict

```text
CUSP_TARGETS_A_D_FINITE_PASS_MICRO_GATES_OPEN
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

## Honest Boundary

- Target A/B are exact certificates inside the minimal recirculation model.
- Target C gives a finite conductor-module / weak-double-cover origin audit for
  the right charges; the deeper microscopic BCC/SM origin of those boundary
  dynamics is still pending.
- Target D now has a finite center-topology selection and a finite cusp-module
  amplitude measure.  The remaining gap is the microscopic BCC boundary-material
  derivation of that finite topology.
- The finite material-origin audit still assumes two physical boundary axioms:
  q-local positive q-reflection stiffness and no-incoming retarded asymptotics.
  Deriving those from deeper BB/QCA boundary material dynamics remains the main
  post-target theorem gate.
