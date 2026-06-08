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
CUSP_TARGETS_A_D_FINITE_PASS_MICRO_GATES_OPEN
```

## Closure

The current sidecar verdict is recorded in [CLOSURE.md](CLOSURE.md):

```text
CUSP_TARGETS_A_D_FINITE_PASS_MICRO_GATES_OPEN
```

Targets A-D are implemented and verified at the finite recirculation-model
level.  The remaining work is explicitly post-target: derive the q-local
positive stiffness/no-incoming retarded boundary material, the microscopic
BCC/SM realization of the Target C conductor-module / weak-double-cover
dynamics, and the microscopic BCC realization of the Target D finite center
topology.

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
CUSP_TARGETS_A_D_FINITE_PASS_MICRO_GATES_OPEN
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
CUSP_TARGETS_A_D_FINITE_PASS_MICRO_GATES_OPEN
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
CUSP_TARGETS_A_D_FINITE_PASS_MICRO_GATES_OPEN
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
CUSP_TARGETS_A_D_FINITE_PASS_MICRO_GATES_OPEN
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
CUSP_TARGETS_A_D_FINITE_PASS_MICRO_GATES_OPEN
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
CUSP_TARGETS_A_D_FINITE_PASS_MICRO_GATES_OPEN
```

Session 08 still leaves the deeper microscopic BCC/SM origin of the
conductor-module and weak-double-cover boundary dynamics open.

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
CUSP_TARGETS_A_D_FINITE_PASS_MICRO_GATES_OPEN
```

Session 09 derives the positive amplitude measure inside the finite
cusp-module model.  It still does not derive the up flag-winding / down
bilinear-linking center-power topology from microscopic boundary dynamics.

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
CUSP_TARGETS_A_D_FINITE_PASS_MICRO_GATES_OPEN
```
