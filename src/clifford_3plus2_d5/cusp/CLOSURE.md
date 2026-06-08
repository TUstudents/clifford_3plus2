# cusp Closure

## Verdict

```text
CUSP_TARGETS_A_D_MICRO_BOUNDARY_PASS
```

The CUSP-PLAN Targets A-D are implemented and verified, and MicroCUSP Sessions
A-H close the microscopic boundary-material gate inside the current BCC/SM
boundary-register model.  This is not a claim that the Standard Model gauge
group alone proves three generations; it records that the CUSP sidecar's
declared microscopic gates have finite certificates or audits.

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

MicroCUSP Session G recovers this Target-C module from microscopic Schur
moments rather than diagonal targets: `(0,2,3)` gives `Q=(3,2,0)`, conductor
`c=2` gives `D=(1,0,0)`, and the weak/BCC double-cover factor gives
`U=(5,2,0)`.  Wrong-conductor, trivial-lift, color-order-lift, diagonal-target,
and mass-fit controls are rejected.

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

MicroCUSP Session H recovers this Target-D topology from microscopic boundary
data: the up powers are geodesic distances on the non-cyclic cusp flag, the
down powers are the unit bilinear pairing of `F3` color-center labels, the
cusp-module path-count amplitudes are retained, and the all-real / one-sector /
separable controls remain zero.

## Closed MicroCUSP Gates

- Session A: q-local positive q-reflection stiffness.
- Session B: no-incoming retarded asymptotics.
- Session C: weak `Z2` same-normal branch parity.
- Session D: color `Z3` center holonomy.
- Global gate: independent weak/color center axes rather than correlated `Z6`.
- Session E: Schur moments recover `C[t^2,t^3]`.
- Session F: one-sided matching recovers `lambda_rec=sqrt(3/2)-1`.
- Session G: Target C recovered from microscopic modules.
- Session H: Target D recovered from microscopic topology.

## Focused Verification

```bash
uv run python -m clifford_3plus2_d5.cusp.scripts.session_01_targets
uv run python -m clifford_3plus2_d5.cusp.scripts.session_10_center_topology
uv run python -m clifford_3plus2_d5.cusp.scripts.session_11_micro_q_stiffness
uv run python -m clifford_3plus2_d5.cusp.scripts.session_12_micro_retarded_asymptotics
uv run python -m clifford_3plus2_d5.cusp.scripts.session_13_micro_weak_z2
uv run python -m clifford_3plus2_d5.cusp.scripts.session_14_micro_color_z3
uv run python -m clifford_3plus2_d5.cusp.scripts.session_15_sm_global_quotient
uv run python -m clifford_3plus2_d5.cusp.scripts.session_16_micro_schur_semigroup
uv run python -m clifford_3plus2_d5.cusp.scripts.session_17_micro_lambda_rec
uv run python -m clifford_3plus2_d5.cusp.scripts.session_18_micro_target_c_module
uv run python -m clifford_3plus2_d5.cusp.scripts.session_19_micro_target_d_topology
uv run pytest src/clifford_3plus2_d5/cusp/tests -q
uv run ruff check src/clifford_3plus2_d5/cusp
git diff --check
```
