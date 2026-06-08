# cusp Closure

## Verdict

```text
CUSP_TARGETS_A_D_FINITE_PASS_MICRO_GATES_OPEN
```

The CUSP-PLAN Targets A-D are implemented and verified at the finite
recirculation-model level.  This is not a claim that the deeper BCC boundary
material theorem is closed; it records that the target sidecar gates now have
finite certificates or finite audits, with the remaining microscopic gates
named explicitly.

## Target A

Target A derives the `(2,3)` cusp source inside a finite local boundary model.
The local BCC edge q-clock has two q-preserving same-normal branches and two
q-changing leakage branches.  The exact BB edge blocks provide the norm split
`same-normal = I/2`, `mixed-normal = I/2`, and the stacked outgoing map is an
`8 x 2` isometry with finite unitary completion.

The q-mismatch material audit selects `p(q)=g q^2` as the unique lowest-degree
q-local, q-reflection-even positive penalty that vanishes on `q=0`; linear and
constant controls fail.  The no-incoming retarded closure kills mixed-normal
feedback in the hard-gap limit, while recurrent leakage has a nonzero visible
return and is rejected.

The weak/BCC source is the order-2 q-preserving branch parity, and the color
source is the order-3 `SU(3)` center return.  Local center-charge automata
advance the fundamental charge each tick and allow visible readout only at
charge `0`, forcing primitive returns `2` and `3`.  Closed-walk enumeration
gives `C[t^2,t^3]`, first valuations `(0,2,3)`, and gap `1`; controls `C[t]`,
`C[t^2]`, `C[t^3]`, one-step loop, weak-only, color-only, and wrong-color
finite graphs miss the target.

## Target B

Target B derives the Cusp-FN shear inside the one-sided retarded
recirculation readout:

```text
(1 + t) sqrt(2) = sqrt(3)
```

The selected branch is `lambda_rec = sqrt(3/2) - 1`, and it is the stable
minimum of the squared mismatch functional.  The ordinary reflection
coefficient solves the different two-sided scattering control and leaves a
nonzero one-sided residual, so it is rejected by the cusp boundary condition.
Count-ratio and inverse-amplitude shears fail the same one-sided equation.

## Target C

Target C closes the finite right-charge origin audit.  SM hypercharge
conservation forces `H_tilde` for up and `H` for down; swapped doors are
rejected.  Given that door orientation, `S=<2,3>` derives conductor `c=2` and
Frobenius gap `1`.

The down right charges are the conductor-ideal residue
`D=max(Q-c,0)=(1,0,0)`.  The up lift factor is the weak/BCC primitive closure
order `2`, giving `U=(5,2,0)`.  Wrong-conductor, trivial-lift, and
color-order-lift controls miss the exponent skeleton.  The nonnegative solver
is retained only as a consistency check for the diagonal powers `(8,4,0)` and
`(4,2,0)`.

## Target D

Target D now has both a finite center-topology selection and a finite positive
amplitude measure.  The coefficients are path sums
`c_ij=sum_gamma A_gamma Omega_gamma`, with `Omega_gamma` in the color center.
The up center powers are selected as geodesic distances on the non-cyclic
length-3 cusp flag, while the down powers are selected as the unit bilinear
pairing on `F3` center labels.  Cyclic-difference, complete-graph,
all-trivial, additive, and separable controls fail as appropriate.

The amplitude weights are derived from the cusp-module count rule
`A_ij=max(1,# decompositions of q_i+r_j in <2,3>)`, with exponent `1` treated
as one irreducible conductor-module/contact path.  The resulting rule pair has
nonzero `Im tr([YuYu^dagger,YdYd^dagger]^3)`, all-real controls are CP-zero,
and full common-left/up-right/down-right field rephasings leave the invariant
unchanged.

## Open Post-Target Gates

- Derive q-local positive q-reflection stiffness and no-incoming retarded
  asymptotics from deeper BCC boundary material, rather than taking them as
  finite material axioms.
- Derive the microscopic BCC/SM realization of the conductor-module and
  weak-double-cover dynamics behind Target C.
- Derive why the microscopic BCC boundary realizes the finite center topology
  selected in Target D.

## Focused Verification

```bash
uv run python -m clifford_3plus2_d5.cusp.scripts.session_01_targets
uv run python -m clifford_3plus2_d5.cusp.scripts.session_10_center_topology
uv run pytest src/clifford_3plus2_d5/cusp/tests -q
uv run ruff check src/clifford_3plus2_d5/cusp
git diff --check
```
