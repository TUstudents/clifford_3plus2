# cusp Plan

## Purpose

Implement [CUSP-PLAN.md](CUSP-PLAN.md): a Froggatt-Nielsen flavor program where
the hierarchy comes from the valuation filtration of a `(2,3)` recirculation
cusp.

## Session 01 - CUSP-PLAN Targets A-D

Pass only if:

- Target A produces first valuations `(0,2,3)` from primitive closures `(2,3)`;
- Target A ties closure length `2` to the local BCC edge q-clock's two
  q-preserving same-normal branches and closure length `3` to the color
  `SU(3)` center-neutral return order;
- Target A enumerates closed walks in a finite graph with two weak length-2
  returns and three color length-3 returns;
- the one-step-return control collapses to `(0,1,2)`;
- `C[t]`, `C[t^2]`, and `C[t^3]` controls are distinct from `C[t^2,t^3]`;
- finite neighboring graph controls with a one-step loop, weak-only graph,
  color-only graph, or wrong color closure length miss `(0,2,3)`;
- Target B solves `(1+t)sqrt(2)=sqrt(3)`, certifies it as the stable minimum of
  the squared mismatch functional, and rejects the ordinary reflection
  coefficient, count-ratio shear, and inverse-amplitude shear because they fail
  this norm-matching equation;
- Target C supplies a finite conductor/orientation candidate rule for the right
  charges, derives the conductor from `S=<2,3>`, and confirms the selected
  charges with wrong-conductor, wrong-lift, and nonnegative-solver controls;
- Target D builds coefficients as finite center-holonomy path sums, gives a
  nonzero CP invariant, zero all-real / one-sector / separable-phase controls,
  a positive full field-rephasing-invariance check, and initially records the
  center topology as a witness to be upgraded by the later coefficient-measure
  and topology-selection sessions.

Verdict:

```text
CUSP_TARGETS_A_D_MICRO_BOUNDARY_PASS
```

## Closure

The current sidecar verdict is recorded in [CLOSURE.md](CLOSURE.md):

```text
CUSP_TARGETS_A_D_MICRO_BOUNDARY_PASS
```

Targets A-D are implemented and verified inside the current CUSP finite model,
and MicroCUSP Sessions A-H now derive the q-local stiffness, no-incoming
retarded boundary, weak `Z2`, color `Z3`, independent SM center axes, Schur
semigroup, `lambda_rec`, Target-C module, and Target-D topology from the
current microscopic BCC boundary-register model.

## Session 02 - Local Center-Charge Source

Replaces the abstract fixed-length weak/color return paths in Target A with a
local center-charge recirculation automaton.

Pass only if:

- one local tick advances the fundamental center charge;
- visible readout is allowed only at charge `0`;
- weak/BCC center order `2` gives charge table `(0,1,0,1,...)` and primitive
  return `2`;
- color center order `3` gives charge table `(0,1,2,0,...)` and primitive
  return `3`;
- the generated automaton graph has closed-walk counts
  `(1,0,2,3,4,12,17)` through length `6` and primitive generators `(2,3)`;
- the one-step, weak-only, color-only, and wrong-color-length controls still
  miss `(0,2,3)`.

Verdict remains:

```text
CUSP_TARGETS_A_D_MICRO_BOUNDARY_PASS
```

Session 02 is a finite local source model, not the full microscopic BB
boundary-material theorem.

## Session 03 - Boundary-Material Source

Ties the finite source model to the currently available exact BB edge data.

Pass only if:

- the exact BB edge blocks split norm as same-normal `I/2` and mixed-normal
  `I/2`;
- the weak/BCC source is derived from the two q-preserving same-normal branches,
  with order `2`, channel count `2`, and forbidden one-step readout;
- the color source remains the `SU(3)` center clock, with order `3` and channel
  count `3`;
- these material-selected sources reproduce the Session 02 automata and Target
  A closed-walk data.

Verdict remains:

```text
CUSP_TARGETS_A_D_MICRO_BOUNDARY_PASS
```

Session 03 is a finite material-source certificate.  It still does not derive
the locking field, outgoing leakage condition, or full local unitary boundary
dilation.

## Session 04 - Boundary Dilation

Closes the finite local dilation audit for the material-source model.

Pass only if:

- the exact local BB same/mixed outgoing channel stack is an isometry;
- the isometry has finite unitary completion;
- q-changing leakage has scalar finite-gap feedback and zero hard-gap feedback;
- retarded/outgoing closure preserves q=0 visible powers;
- recurrent mixed-normal closure has a nonzero visible return and is rejected.

Verdict remains:

```text
CUSP_TARGETS_A_D_MICRO_BOUNDARY_PASS
```

Session 04 is still conditional on two material inputs: the single-clock
locking field and outgoing mixed-normal asymptotic leads.

## Session 05 - Boundary Material Origin

Derives the finite locking/outgoing rules from explicit local material axioms.

Pass only if:

- the lowest-degree q-local, q-reflection-even, positive leakage penalty
  vanishing on q=0 is uniquely `p(q)=g q^2`;
- the `q=+-2` leakage gap is `4g`;
- linear and constant controls fail for clear reasons;
- the no-incoming causal rule selects the retarded/outgoing closure;
- recurrent mixed-normal closure remains the rejected nonzero-return control.

Verdict remains:

```text
CUSP_TARGETS_A_D_MICRO_BOUNDARY_PASS
```

Session 05 still assumes the deeper physical axioms: q-local positive
q-reflection stiffness and no-incoming retarded asymptotics.

## Session 07 - Shear Matching Principle

Promotes Target B from a formula certificate to a finite one-sided matching
theorem.  Session 06 remains open; this session uses the retarded
recirculation readout already selected in the finite boundary audits.

Pass only if:

- the normalized amplitude shear is selected by the one-sided outgoing
  boundary condition `(1+t)sqrt(N_w)=sqrt(N_c)`;
- `t=sqrt(3/2)-1` is the positive oriented cusp-uniformizer branch;
- the ordinary reflection coefficient solves the two-sided scattering control
  `(1+r)sqrt(N_w)=(1-r)sqrt(N_c)`, but fails the one-sided recirculation
  equation;
- the derivation uses only weak/color channel counts and the one-sided
  matching rule, not CKM data.

Verdict remains:

```text
CUSP_TARGETS_A_D_MICRO_BOUNDARY_PASS
```

## Session 08 - Right-Handed Charge Origin

Promotes Target C from a diagonal-fit candidate to a finite boundary-module
origin audit.  The selected right charges are now computed from the
conductor/weak-order rule, and the diagonal-exponent solver is retained only as
a consistency check.

Pass only if:

- the down right charges are the conductor-ideal residue
  `r_d=max(q-c,0)`, with `c=2` derived from `S=<2,3>`;
- the up lift factor is the weak/BCC primitive closure order `2`, not an
  adjustable exponent multiplier;
- the selected charges are `D=(1,0,0)` and `U=(5,2,0)`;
- trivial lift and color-order lift controls miss the up exponent skeleton;
- all selected right charges are nonnegative;
- no fitted mass data are used to select the charges.

Verdict remains:

```text
CUSP_TARGETS_A_D_MICRO_BOUNDARY_PASS
```

Session 08 left the deeper microscopic BCC/SM origin of the conductor-module
and weak-double-cover boundary dynamics open; Session 18 closes that gate.

## Next Gates

### Session 06 - Boundary Material Microphysics

Derive the Session 05 physical axioms from an actual BCC boundary-material
construction.

Pass only if:

- q-local positive q-reflection stiffness is produced by a microscopic boundary
  material, not imposed;
- no-incoming retarded asymptotics are produced by the boundary exterior, not
  imposed;
- the resulting boundary model preserves the `(2,3)` cusp graph and its
  controls.

### Session 11 / MicroCUSP A - q-Stiffness

Implement the first MicroCUSP microscopic boundary-material gate.

Pass only if:

- a local two-normal BCC boundary material produces a rank-one positive
  stiffness matrix for the mismatch mode `q=r1-r2`;
- the microscopic action is `g(r1-r2)^2`, hence `gq^2` on the q-coordinate;
- the trace/center normal mode remains ungapped;
- q-reflection evenness removes the linear term and the q-mode curvature is
  positive;
- the adjacent mixed-normal gap is `4g`, matching Session 05;
- the hard-gap limit still suppresses mixed-normal feedback;
- zero-stiffness, q-odd linear, and nonanalytic `g|q|` controls are rejected.

Status: implemented as a local harmonic material audit.  This closes the
q-stiffness part of Session 06, but not the no-incoming retarded asymptotics or
the microscopic center-recirculation/topology gates.

### Session 12 / MicroCUSP B - Retarded Asymptotics

Implement the no-incoming outgoing-boundary gate.

Pass only if:

- mixed-normal outgoing clock-error channels are modeled as genuine hidden
  asymptotic channels in a local unitary-completable BB boundary isometry;
- the no-incoming condition is encoded as zero hidden return matrix `R=0`;
- the local Schur correction `G R M` from mixed-normal channels back to the
  visible sheet vanishes;
- visible powers under the no-incoming closure match the q=0 survival powers;
- recurrent leakage, hard-wall reflection, and incoming/outgoing symmetric
  closure all produce nonzero visible feedback and are rejected;
- the finite cusp graph preserves primitive closures `(2,3)`, first valuations
  `(0,2,3)`, and one-step gap `1`.

Status: implemented as a local outgoing-boundary audit.  This closed the
no-incoming retarded-asymptotic part of Session 06 and left the microscopic
`Z2/Z3` center recirculation, SM global quotient, and Target C/D topology gates
for subsequent sessions.

### Session 13 / MicroCUSP C - Weak Z2 Recirculation

Implement the weak/BCC center-charge automaton from same-normal branch parity.

Pass only if:

- the q-preserving same-normal BCC branches carry common signs
  `s=sigma1=sigma2=(+1,-1)`;
- those signs form the local `Z2` branch-parity group;
- one primitive weak/BCC tick is center charged and visible readout is allowed
  only after two ticks;
- the finite automaton charge table is `(0,1,0,1,0,1,0)`;
- primitive return length and channel count are both `2`;
- trivial weak charge, order-one weak closure, and weak-only semigroup controls
  miss the cusp module.

Status: implemented as a microscopic same-normal branch-parity audit.

### Session 14 / MicroCUSP D - Color Z3 Recirculation

Implement the color-center automaton as a closed `SU(3)` center Wilson
holonomy.

Pass only if:

- one primitive color tick carries the nontrivial center phase
  `omega=-1/2+i sqrt(3)/2`;
- one- and two-tick paths are center charged, while the three-tick closed
  holonomy is neutral;
- the finite automaton charge table is `(0,1,2,0,1,2,0)`;
- primitive return length and channel count are both `3`;
- combining weak `Z2` and color `Z3` preserves primitive closures `(2,3)`,
  first valuations `(0,2,3)`, and gap `1`;
- wrong color length `2`, wrong color length `4`, spectator color, and
  gauged-away open-phase controls miss the cusp module.

Status: implemented as a microscopic color-center holonomy audit.  The next
gate is the SM global quotient audit: independent primitive `Z2,Z3` closures
versus a correlated `Z6` boundary rule.

### Session 15 / MicroCUSP Global Gate - SM Quotient

Audit independent nonabelian center closures against a quotient-correlated
`Z6` boundary rule.

Pass only if:

- the weak and color boundary ticks are independent axes `(1,0)` and `(0,1)`
  in `Z2 x Z3`;
- their primitive neutral lengths are `(2,3)`;
- the resulting low valuations and family basis are `(0,2,3)` and
  `1,t^2,t^3`;
- the quotient-diagonal tick `(1,1)` closes only at length `6` and gives low
  valuations `(0,6,12)`;
- a U(1)-collapsed one-step control gives `(0,1,2)`;
- both quotient controls are rejected.

Status: implemented.  This left the Target C/D boundary topology:
conductor-module / weak-cover origin and center-holonomy topology.

### Session 16 / MicroCUSP E - Schur Semigroup

Recover `S=<2,3>` from microscopic Schur moments.

Pass only if:

- the visible return moments satisfy `M1=0`, `M2!=0`, and `M3!=0`;
- the primitive return semigroup is `(2,3)`;
- the recirculation algebra is `C[t^2,t^3]`;
- the family module basis is `1,t^2,t^3`;
- controls `S=<1>`, `S=<2>`, `S=<3>`, `S=<2,4>`, and `S=<3,4>` miss
  `(0,2,3)`.

Status: implemented.

### Session 17 / MicroCUSP F - Lambda Rec

Recover `lambda_rec=sqrt(3/2)-1` from microscopic return moments and the
one-sided retarded matching rule.

Pass only if:

- weak/color moments give channel counts `2` and `3`;
- `(1+t)sqrt(2)=sqrt(3)` has the unique positive solution
  `t=sqrt(3/2)-1`;
- ordinary reflection, count-ratio shear, and inverse-amplitude shear fail the
  one-sided equation.

Status: implemented.

### Session 18 / MicroCUSP G - Target C Module

Recover Target C from modules, not exponent targets.

Pass only if:

- microscopic module valuations `(0,2,3)` give `Q=(3,2,0)` in light-to-heavy
  order;
- conductor `c=2` comes from `S=<2,3>`;
- `D=max(Q-c,0)=(1,0,0)`;
- the weak/BCC double-cover factor `2` gives `U=(5,2,0)`;
- wrong-conductor, trivial-lift, color-order-lift, diagonal-target, and
  mass-fit controls are rejected.

Status: implemented.

### Session 19 / MicroCUSP H - Target D Topology

Recover Target D from microscopic topology.

Pass only if:

- up center powers are geodesic distances on the microscopic non-cyclic cusp
  flag;
- down center powers are the unit bilinear pairing of microscopic `F3`
  color-center labels;
- cusp-module path-count amplitudes are retained;
- the CP invariant is nonzero, all-real / one-sector / separable controls are
  zero, and field-rephasing invariance holds.

Status: implemented.  No MicroCUSP A-H gates remain open.

### Session 09 - Center-Holonomy Coefficient Measure

Replace the arbitrary Target D positive-amplitude deformation by a finite
cusp-module path-count measure.

Pass only if:

- coefficients are sums over allowed paths with center holonomies;
- the amplitude weight is derived from the FN exponent as
  `A_ij=max(1,# decompositions of q_i+r_j in <2,3>)`;
- the gap exponent `1` is treated as one irreducible conductor-module/contact
  path, not as a zero coefficient;
- field rephasings are accounted for;
- the commutator invariant remains nonzero for the derived measure.

Verdict remains:

```text
CUSP_TARGETS_A_D_MICRO_BOUNDARY_PASS
```

Session 09 derives the positive amplitude measure inside the finite
cusp-module model.  Session 19 later derives the up flag-winding / down
bilinear-linking center-power topology from microscopic boundary data.

### Session 10 - Center-Holonomy Topology

Replace the declared Target D center-power topology by a finite topology
selection inside the cusp flag / center-label model.

Pass only if:

- the up center powers are the geodesic distances on the non-cyclic length-3
  cusp flag;
- cyclic group-difference, complete-graph distance, and all-trivial up
  controls differ from the selected flag distance;
- the down center powers are the unit bilinear pairing on `F3` center labels;
- all-trivial, additive, and cyclic-difference down controls differ from the
  selected bilinear pairing;
- direct row/column down links are trivial and the `(1,1)` unit pairing is
  normalized to `1`;
- the result is marked as a finite topology selection, not a microscopic BCC
  boundary-material derivation.

Verdict remains:

```text
CUSP_TARGETS_A_D_MICRO_BOUNDARY_PASS
```
