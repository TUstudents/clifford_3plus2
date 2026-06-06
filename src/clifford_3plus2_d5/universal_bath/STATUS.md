# universal_bath - Status

**Status**: PARKED after Session 24.  See `CLOSURE.md`.

The sidecar is closed for now because the quark route has reached a genuine
dynamical blocker: the available symmetry/current/flag data do not derive the
Higgs-door orientation coupling.  Session 24 shows that endpoint reflection
maps the flag to `N.T`, not to the Hermitian closure `Delta_N`; the swapped
assignment remains allowed.

## Current theorem-level status

Session 01 proves the common bath spine:

```text
finite Lanczos head + period-one retarded tail
```

with the universal tail

```text
t(z) = (z - sqrt(z^2 - 4)) / 2
t(2 sqrt(2)) = sqrt(2) - 1
```

The silver value is inherited from the BCC/BB band-edge theorem. The sidecar
does not yet prove that any physical sector source has the required finite
head or nonzero coupling to the universal tail.

Session 01 also fixes the reduction taxonomy:

```text
positive one-sided       -> scalar Jacobi
real non-positive shell  -> indefinite look-ahead Jacobi
chiral/unitary phase     -> CMV/OPUC
```

The CMV/OPUC free-tail criterion is `alpha_n = 0` after the finite head.

## Session 02 source dictionary

Session 02 upgrades the source-dictionary scaffold into a write-once source
ledger.  The BB same-normal first-hop identity is:

```text
B_+^* B_+ + B_-^* B_- = I/2
```

Hence every normalized frozen $q=0$ source has radial first-hop survival weight

```text
1/2
```

The frozen anchors are:

```text
neutrino_collective_u       u, depth 1, scalar Jacobi
neutrino_edge_b             b, depth 0, scalar Jacobi
charged_lepton_active_e1    e1, depth 2, CMV/OPUC
```

The charged-lepton port is exactly:

```text
e1 = sqrt(2/3) a + 1/sqrt(3) u
```

with no `b` component.

The quark source records are present but deliberately unresolved:

```text
up_quark_boundary_source
down_quark_boundary_source
```

Their SM charge anchors are known, but the universal-bath BCC source vectors and
normal-depth placements are not yet derived.  Session 02 therefore refuses to
freeze them.

## Session 03 neutrino product bath

Session 03 applies the universal bath spine to the frozen neutrino anchors:

```text
neutrino_collective_u = u, depth 1
neutrino_edge_b       = b, depth 0
```

with the product hidden bath

```text
H_Q = H_chain tensor I_family
```

For the product states `|0> tensor u` and `|0> tensor b`, the certificate
checks:

```text
<u|H_Q^k|b> = 0,      k = 0..4
<u|H_Q^k|u> = <b|H_Q^k|b>,      k = 0..4
```

The semi-infinite chain Weyl readout is the universal tail:

```text
m(z) = t(z) = (z - sqrt(z^2 - 4)) / 2
```

so the normalized neutrino response is:

```text
Sigma_hat_nu(z) = m(z)^2 P_u + P_b
```

At `z = 2 sqrt(2)`:

```text
m(z) = epsilon = sqrt(2) - 1
Sigma_hat_nu = epsilon^2 P_u + P_b
m2/m3 = epsilon^2
Delta m^2_21 / Delta m^2_31 = epsilon^4
```

Rank-one, wrong-source, and alternate-tail controls are rejected.

## Session 04 charged-lepton CMV head

Session 04 applies the CMV/OPUC reduction to the frozen charged-lepton anchor:

```text
charged_lepton_active_e1 = e1, depth 2
```

with

```text
e1 = sqrt(2/3) a + 1/sqrt(3) u
```

The two-step universal-tail leakage gives:

```text
sin(theta_e) = sqrt(3/2) epsilon^2
sin^2(theta_e) = (3/2) epsilon^4
```

The upstream primitive boundary holonomy supplies the phase:

```text
SCHUR_RETURN -> PARENT_A3 -> RESIDUAL_A2
phase = exp(-5 pi i / 12)
```

The finite Verblunsky coefficient is:

```text
alpha_e = sin(theta_e) exp(-5 pi i / 12)
```

and the two-state CMV/Givens head is exactly unitary.  After the finite head,
the CMV/OPUC tail is free:

```text
alpha_n = 0
```

Depth-one, depth-three, b-leakage, and holonomy subword controls are rejected.

## Session 05 charged-lepton 2/9 torsion gate

Session 05 answers the specific torsion question using the same frozen source:

```text
e1 = sqrt(2/3) a + 1/sqrt(3) u
```

The occupation weights are:

```text
p_a = 2/3
p_u = 1/3
p_b = 0
```

Therefore the incoherent source-transition weight is:

```text
p_a p_u = 2/9
```

The coherent amplitude control is:

```text
sqrt(2)/3
```

and is rejected as the torsion value. Equal-weight and one-port controls are
also rejected. Thus `2/9` is derived as a source-geometry occupation moment,
not as the full CMV phase or charged-lepton mass angle.

## Session 06 up-quark nilpotent CMV head

Session 06 proves the conditional up finite-head algebra. The up source is
still deliberately unresolved in the write-once dictionary:

```text
up_quark_boundary_source: unresolved_not_frozen
```

The BB same-normal survival identity gives:

```text
B_+^* B_+ + B_-^* B_- = I/2
```

so the injection amplitude is:

```text
x = sqrt(1/2) = 1/sqrt(2)
```

The length-3 nilpotent Taylor head gives:

```text
exp(xN) = I + xN + x^2 N^2 / 2
(x^2/2, x, 1) = (1/4, 1/sqrt(2), 1)
```

The geometric control at the same injection gives `(1/2,1/sqrt(2),1)` and is
rejected. The old `sqrt(2)` charm control is also rejected. The finite CMV head
`(1/sqrt(2),1/4)` lies inside the unit disk and is followed by the free tail
`alpha_n=0`.

## Session 07 down-quark indefinite Jacobi head

Session 07 implements the count-level real/symmetric down head and keeps the
source unresolved:

```text
down_quark_boundary_source: unresolved_not_frozen
```

The three-port residual shell gives:

```text
(3,1,2)/3 -> (1, 1/sqrt(3), sqrt(2/3))
```

The six-element regular S3 shell gives the baseline:

```text
(6,2,4)/6 -> (1, 1/sqrt(3), sqrt(2/3))
```

and can host the candidate:

```text
(6,2,5)/6 -> (1, 1/sqrt(3), sqrt(5/6))
```

The three-port shell cannot host the candidate because `5/6 * 3 = 5/2` is not
an integer rank. The regular shell can host it, but S3 does not force it:
rank 2 requires defect polarization and rank 5 is not unique. The bottom
`4 -> 5` line remains an open selection theorem.

## Session 08A quark height-door audit

Session 08A separates the exact SM charge door from the repair-mode premise.
The electroweak charge checks are:

```text
-Y(Q_L) + Y(H_tilde) + Y(u_R) = -1/6 - 1/2 + 2/3 = 0
-Y(Q_L) + Y(H)       + Y(d_R) = -1/6 + 1/2 - 1/3 = 0
```

and both neutral Higgs components obey:

```text
Q_em = Y + T3 = 0
```

The declared height assignment is:

```text
up   -> oriented length-3 nilpotent N
down -> Hermitian closure Delta_N
```

where `N^3=0`, `N^2 != 0`, and

```text
Spec(Delta_N) = {0,1,3}
```

The negative control is important: swapping the repair modes is still
hypercharge-allowed. Thus the coherent-up / Hermitian-down split is not derived
from electroweak charges alone; it remains a named height-dynamics premise.

## Session 08B quark color-lift audit

Session 08B separates visible color covariance from hidden color-return
multiplicity.

The fixed-color control

```text
diag(1,0,0)
```

does not commute with `SU(3)_c` and is rejected.

The color-scalar spectator embedding preserves visible color but stays on the
three-port shell:

```text
(3,1,2)/3 -> (1, 1/sqrt(3), sqrt(2/3))
```

The active hidden color-return lift also preserves visible color, while its
hidden shell reaches:

```text
1_direct + 2_BCC + 3_color
```

with breakdown:

```text
even_direct = 1
bcc_odd     = 2
color_odd   = 3
odd_total   = 5
total       = 6
```

It reproduces the regular-S3 baseline and makes the rank-five candidate
available:

```text
(6,2,4)/6 -> (1, 1/sqrt(3), sqrt(2/3))
(6,2,5)/6 -> (1, 1/sqrt(3), sqrt(5/6))
```

Gauge covariance alone rejects fixed color but does not select active over
spectator. The microscopic active-return selection rule remains open.

## Session 09 neutrino BCC moment audit

Session 09 audits the strongest upgrade gate for the neutrino sector.  Session
03 checks

```text
H_Q = H_chain tensor I_family
```

so its `u/b` cross moments vanish by the product family factor.  Session 09
asks whether the currently modeled microscopic BB edge update can instead
compute

```text
<u|H_BCC^k|b>
```

as a BCC walk-counting observable.

The exact microscopic edge data pass:

```text
B_+^* B_+ + B_-^* B_- = I/2
M_+2^* M_+2 + M_-2^* M_-2 = I/2
```

and therefore the local same-normal/leakage split is exact:

```text
I/2 + I/2 = I
```

However, the current microscopic edge graph has only:

```text
spinor
q0_same_normal
q_plus2_leakage
q_minus2_leakage
```

It does not yet contain:

```text
family_port_u
family_port_b
```

Therefore the true microscopic family moments are not yet defined.  Adding
`I_family` would return to the Session 03 product ansatz.  The rank-one
no-family control still has a nonzero `u/b` cross return, so the product result
is sensitive but not microscopic.

## Session 10 selected neutrino family-port graph

Session 10 supplies the family-port graph that Session 09 identified as
missing.  The graph uses the residual basis:

```text
u = (1,1,1)/sqrt(3)
a = (2,-1,-1)/sqrt(6)
b = (0,1,-1)/sqrt(2)
```

and splits the selected boundary family space as:

```text
P_act = P_u + P_b
P_rad = P_a
P_act + P_rad = I
```

with ranks:

```text
rank(P_act) = 2
rank(P_rad) = 1
```

The finite graph is:

```text
H_fam = H_chain tensor P_act + I tensor Lambda P_rad
```

It is not the full product graph `H_chain tensor I_family`, because the radial
`a` mode is not propagated as a third active copy.  Direct finite graph
moments give:

```text
<u|H_fam^k|b> = 0,          k = 0..4
<u|H_fam^k|u> = <b|H_fam^k|b>,  k = 0..4
<a|H_fam^k|u> = 0,          k = 0..4
```

Closing the active plane with the universal retarded tail gives:

```text
Sigma_hat_nu = epsilon^2 P_u + P_b
```

Controls:

- the full residual `K3` graph fails equal `u/b` diagonal returns;
- the full product identity graph propagates the radial `a` mode;
- the rank-one no-family control has nonzero `u/b` cross return;
- the alternate-tail control fails.

This completes the internal neutrino family-port graph.  The remaining physical
gate is deriving the selected active-plane boundary condition
`P_act=P_u+P_b` from the microscopic BB edge update rather than imposing it.

## Session 11 selected active-plane incidence

Session 11 derives the Session 10 active projector from selected residual-port
incidence.  The selected residual port is

```text
e1 = (1,0,0)
```

and in the residual basis it decomposes as:

```text
e1 = sqrt(2/3) a + u/sqrt(3)
```

with no `b` component.  Removing the collective trace gives the selected
traceless radial line:

```text
e1 - <u,e1> u = sqrt(2/3) a
```

Therefore:

```text
P_rad = P_a
P_act = I - P_a = P_u + P_b
```

Equivalently, the active incidence channels are the collective channel `u` and
the unique current perpendicular to both `u` and `a`, namely:

```text
b = (0,1,-1)/sqrt(2)
```

Controls:

- selected `S2` symmetry alone is too weak, because `P_e1` is selected-`S2`
  invariant but mixes the radial `a` and active `u` lines;
- the raw selected-port line control is rejected, since `I-P_e1 != P_u+P_b`;
- full `S3` invariance is rejected, as required by the neutrino splitting.

Thus the active plane is no longer an arbitrary product-family ansatz.  It is
the selected-port incidence plane after detracing the collective mode.  The
remaining physical BB/QCA gate is now sharper:

```text
bb q-mismatch penalizes the detraced selected-port radial line a
retarded outgoing boundary condition closes only the active u/b incidence plane
```

## Session 12 q-mismatch hard gap and retarded closure

Session 12 connects the Session 11 active-plane incidence result to the exact
BB edge blocks under the single-clock/outgoing-boundary model.

At a BCC edge:

```text
q = r1 - r2
Delta q = sigma1 - sigma2
```

The eight body-diagonal hops split into:

```text
sigma1 = sigma2      -> Delta q = 0      (4 same-normal directions)
sigma1 = -sigma2     -> Delta q = +-2    (4 mixed-normal directions)
```

The exact BB blocks satisfy:

```text
B_+^* B_+ + B_-^* B_- = I/2
M_+2^* M_+2 + M_-2^* M_-2 = I/2
```

The single-clock mismatch model uses:

```text
H_lock = g q^2
H_leak(q=+-2) = 4g I
```

With `M=(M_+2,M_-2)^T`, mixed-normal Schur feedback is:

```text
Sigma_mix(z,g) = M^* (z-4g)^-1 M = I/[2(z-4g)]
```

so:

```text
lim_{g->infty} Sigma_mix(z,g) = 0
```

The outgoing half-line Weyl branch satisfies:

```text
m = 1/(z-m),    m(z) ~ 1/z
```

and retarded clock-error closure has block-triangular form:

```text
T_R = [[A_N, 0],
       [E,   S_chi]]
```

Therefore visible powers equal q=0 survival powers:

```text
P_vis T_R^t iota_vis = A_N^t
```

The recurrent wedge control is rejected. Its local two-step return is:

```text
M_-2 M_+2 + M_+2 M_-2
= diag(-(1+i)/4, -(1-i)/4) != 0
```

Thus the Session 10 family graph is now microscopic inside the
single-clock/outgoing-boundary model:

```text
H_fam = H_chain tensor (P_u+P_b) + I tensor Lambda P_a
```

Remaining physical inputs:

```text
single_clock_locking_field_is_realized_by_boundary_material
mixed_normal_clock_error_ports_are_outgoing_asymptotic_leads
```

## Session 13 boundary-material origin audit

Session 13 asks whether those two remaining inputs are already forced by bare
BB block algebra.  The answer is no.

Derived:

```text
K = a r1 + b r2
K(r,r)=0  ->  a+b=0
K proportional to r1-r2 = q
```

Thus the local linear single-clock mismatch coordinate is unique.  If the
boundary admits this constraint field, then:

```text
K^T K = q^2
```

so the positive penalty has the required form.

Not derived:

```text
g or Lambda stiffness parameter
outgoing/no-incoming clock-error asymptotic condition
```

The bare BB blocks contain no stiffness symbol and no asymptotic boundary
condition.  Retarded and recurrent completions use the same local mixed-normal
emission norm, but the recurrent wedge has nonzero two-step return and changes
visible powers.  Therefore the conditional Session 12 graph is not promoted to
a from-bare-BB material theorem.

The next required object is a boundary-material model deriving:

```text
positive single-clock locking
outgoing clock-error asymptotics
```

## Session 14 charged-lepton minimal boundary graph

Session 14 builds the minimal colorless active charged-lepton family-port graph.
Because charged leptons are Dirac/Yukawa states, the correct object is a
two-sided chiral Schur kernel:

```text
B_e(z) = V_R^T (z-H_Q,e)^-1 V_L
```

not a one-sided positive self-energy.

Use:

```text
Q_e = span(t_+, t_-, p_a, p_b)
P = span(u, a, b)
```

with left coupling:

```text
V_L =
[[1/sqrt(2), 0, 0],
 [1/sqrt(2), 0, 0],
 [0, 1, 0],
 [0, 0, 1]]
```

and right coupling:

```text
V_R =
[[1, 0, 0],
 [1, 0, 0],
 [0, cos(theta), sin(theta)],
 [0, -sin(theta), cos(theta)]]
```

Then:

```text
V_R^T V_L = sqrt(2) P_u + R_theta P_perp
```

exactly.  The angle is assembled from existing gates:

```text
theta = -2*pi/3 - 2/9
```

Acting on:

```text
e1 = u/sqrt(3) + sqrt(2/3) a
```

gives:

```text
B_e e1 = sqrt(2/3) [u + cos(theta) a + sin(theta) b]
```

and after normalization:

```text
w_hat = [u + cos(theta) a + sin(theta) b]/sqrt(2)
```

Therefore:

```text
trace weight = 1/2
traceless weight = 1/2
K = 2/3
```

Controls:

- one trace path fails trace/traceless equipartition;
- a one-sided Hermitian self-energy cannot produce the nontrivial plane
  rotation;
- the `2/9` torsion is still not derived by the graph.

Remaining charged-lepton microscopic inputs:

```text
microscopic_colorless_bcc_higgs_boundary_derives_two_coherent_trace_paths
active_cmv_torsion_angle_2_over_9_is_generated_by_boundary_dynamics
charged_lepton_overall_scale_rho_or_mu_e
```

## Session 15 quark source assembly audit

Session 15 assembles the current quark-side certificates into one source-freeze
ledger.  The quark source must live in

```text
C^3_color tensor C^2_weak tensor span(u,a,b)
```

and the write-once dictionary still reports:

```text
up_quark_boundary_source:
  port_vector = None
  residual_components = {}
  normal_depth = None

down_quark_boundary_source:
  port_vector = None
  residual_components = {}
  normal_depth = None
```

Available:

```text
Session 11 common residual incidence basis
Session 08A SM H_tilde/H charge doors
Session 06 conditional up profile (1/4,1/sqrt(2),1)
Session 07/08B conditional down baseline (1,1/sqrt(3),sqrt(2/3))
Session 07/08B available down candidate (1,1/sqrt(3),sqrt(5/6))
```

Not derived:

```text
height_dynamics_selects_up_nilpotent_down_hermitian
microscopic_active_hidden_color_return_selects_regular_s3_shell
boundary_dynamics_selects_or_kills_down_rank_five_line
quark_normal_depth_placements_on_bcc_scar_are_frozen
```

Thus the verdict is:

```text
QUARK_SOURCE_FREEZE_NOT_DERIVED_AUDIT
```

The conditional quark heads are assembled, but `V_u,V_d` remain unfrozen.  The
sidecar must not promote the conditional up/down heads to complete microscopic
BCC family-port boundary graphs until those four inputs are derived or killed.

## Session 16 quark normal-depth placement audit

Session 16 checks whether the existing `depth_scar` theorem already closes the
quark normal-depth source-placement blocker.  It does not.

Depth scar supplies:

```text
h(u,a,b) = (0,1,2)
N = |u><a| + |a><b|
Spec(2 Delta(P3)) = {0,2,6}
```

The path-scar depth operator in the port basis is:

```text
[[ 2, -2,  0],
 [-2,  4, -2],
 [ 0, -2,  2]]
```

It is not the hand-written port diagonal:

```text
diag(0,2,6)
```

Also:

```text
2 h(u,a,b) = (0,2,4) != {0,2,6}
```

Thus the graph-depth theorem is exact but does not define:

```text
V_u.normal_depth
V_d.normal_depth
```

The current dictionary still has both values set to `None`.  The verdict is:

```text
QUARK_NORMAL_DEPTH_PLACEMENT_NOT_DERIVED_AUDIT
```

The normal-depth blocker is now sharpened to a source-placement theorem, not a
missing proof of the depth-scar algebra.

## Session 17 quark active color-return microcanonical audit

Session 17 reduces the active hidden color-return blocker using the upstream
primitive-shell microcanonical theorem.

The primitive quark shell is:

```text
1_even + 5_odd = 1_direct + 2_BCC + 3_color
```

The active hidden color lift reaches exactly that shell:

```text
even_direct = 1
bcc_odd     = 2
color_odd   = 3
odd_total   = 5
total       = 6
```

Equal-degeneracy microcanonical reduction gives:

```text
rho_label = I_6 / 6
weights = (1/6,1/6,1/6,1/6,1/6,1/6)
```

The spectator branch is a compressed 3-port control.  Compressed macrochannel
counting gives:

```text
r = 1/sqrt(5)
phase = pi/4
```

and is rejected by the upstream control.  Thus the verdict is:

```text
QUARK_ACTIVE_COLOR_RETURN_MICROCANONICAL_CONDITIONAL_PASS
```

This does not make active return a gauge theorem.  It reduces the blocker to:

```text
equal_boundary_degeneracy_or_max_entropy_prior
```

plus the upstream `vacuum_framing` and `transfer_probe` inputs of the
primitive-shell theorem.

## Session 18 down odd-shell rank-five audit

Session 18 reduces the down rank-five blocker using the active primitive shell.

Inside

```text
1_even + 5_odd = 1_direct + 2_BCC + 3_color
```

the down candidate counts are:

```text
d: full primitive shell = 6
s: BCC odd doublet      = 2
b: full odd shell       = 5
```

Thus:

```text
(6,2,5)/6 -> (1,1/sqrt(3),sqrt(5/6))
```

The rank-five line is the primitive odd shell:

```text
odd shell = regular primitive shell - even direct line
```

This removes the S3 rank-five ambiguity inside the primitive-shell model.  The
color-only middle control and compressed parity control are rejected.  The
verdict is:

```text
QUARK_DOWN_ODD_SHELL_RANK_FIVE_CONDITIONAL_PASS
```

The remaining premise is:

```text
down_bottom_reads_full_primitive_odd_shell
```

## Session 19 charged-lepton trace/torsion origin audit

Session 19 audits the two microscopic charged-lepton inputs left by Session 14:

```text
microscopic_colorless_bcc_higgs_boundary_derives_two_coherent_trace_paths
active_cmv_torsion_angle_2_over_9_is_generated_by_boundary_dynamics
```

For `n` coherent trace-return paths, the selected source

```text
e1 = sqrt(2/3) a + u/sqrt(3)
```

gives the exact weights

```text
w_trace(n) = n/(n+2)
w_perp(n)  = 2/(n+2)
```

Therefore trace/traceless equipartition uniquely selects:

```text
n = 2
```

The Session 14 minimal graph supplies exactly two trace-only pole rows.  The
one-trace and three-trace controls fail.  Thus the trace count is internally
sharp, but its microscopic BCC/Higgs origin is not derived.

The torsion side is similarly sharpened.  Session 05 derived:

```text
p_a p_u = 2/9
```

as a source-occupation moment, and Session 14 uses exactly that number as a
real plane-rotation angle:

```text
theta = -2*pi/3 - 2/9
```

Session 19 confirms the insertion but does not derive an occupation-to-angle
dynamics.  The CMV holonomy phase word is distinct from the `2/9` torsion.

The verdict is:

```text
CHARGED_LEPTON_TRACE_TORSION_ORIGIN_NOT_DERIVED_AUDIT
```

## Session 20 quark height-orientation bridge audit

Session 20 reduces the height-door premise using the depth-scar successor
certificate.  The certified active successors are:

```text
a -> u
b -> a
```

so the oriented repair flag is

```text
N = |u><a| + |a><b|
```

The declared up repair is exactly this nilpotent flag.  The declared down
repair is exactly the Hermitian flag-Laplacian closure:

```text
Delta_N = N N^T + N^T N - (N + N^T)
```

Thus the two quark repair objects are not independent assumptions:

```text
up repair   = retarded/oriented readout of N
down repair = Hermitian closure readout of N
```

What still does not follow is the coupling of the SM Higgs doors to those
readouts.  Hypercharge forces:

```text
up   -> H_tilde
down -> H
```

but the swapped repair assignment is still hypercharge-allowed.  The remaining
premise is now:

```text
higgs_door_orientation_couples_H_tilde_to_retarded_flag_and_H_to_flag_closure
```

The verdict is:

```text
QUARK_HEIGHT_ORIENTATION_BRIDGE_NOT_DERIVED_AUDIT
```

## Session 21 quark active-current readout

Session 21 implements the new quark-source ansatz:

```text
quark source = colored active current line b
```

The selected-incidence geometry already supplies:

```text
P_rad = P_a
P_act = P_u + P_b
```

Inside that active plane, `u` is the scalar trace and `b` is the unique
non-scalar current line.  In residual coordinates `(u,a,b)`:

```text
b = (0,0,1)
```

The certified oriented flag gives:

```text
N^0 b = b
N b   = a
N^2 b = u
```

so the first-passage orders are:

```text
n(u,a,b) = (2,1,0)
```

This replaces the failed idea of reading source depth from the eigenvalues of
`2 Delta(P3)`.  The closure types give:

```text
k_up   = 3 n      = (6,3,0)
k_down = 2(n + 1) = (6,4,2)
```

The coherent up readout is:

```text
exp(N/sqrt(2)) b = (1/4) u + (1/sqrt(2)) a + b
```

so the up profile is:

```text
(1/4,1/sqrt(2),1)
```

The down readout is not frozen as a scalar `b` vector.  It is kept as a
Hermitian current covariance / word-shell measure, with both shell measures
visible:

```text
(6,2,4)/6 -> (1,1/sqrt(3),sqrt(2/3))
(6,2,5)/6 -> (1,1/sqrt(3),sqrt(5/6))
```

The odd-shell candidate is selected only if one-tick retarded causality vetoes
the identity/direct return.  That veto is not derived here.

The verdict is:

```text
QUARK_ACTIVE_CURRENT_READOUT_CONDITIONAL_PASS
```

## Session 22 quark current-parity selector

Session 22 reduces the Session 21 current-source condition using the selected
residual symmetry.  Once the first family port is selected, the residual
symmetry preserving it is the swap of the two unselected ports:

```text
e2 <-> e3
```

In residual coordinates `(u,a,b)`, this selected `S2` acts as:

```text
diag(+,+,-)
```

Therefore:

```text
P_even = P_u + P_a
P_odd  = P_b
```

The oriented current across the unselected pair is:

```text
J_23 = (e2-e3)/sqrt(2) = b
```

Session 11 supplies the active incidence plane:

```text
P_act = P_u + P_b
```

The active plane alone is insufficient because it still contains the even
scalar line `u`.  Intersecting active incidence with the selected-odd current
line gives:

```text
P_act P_odd = P_b
```

Controls:

```text
selected scalar e1 has no b component
u is active but even/scalar
a is even and radial/gapped
active plane alone is a plane, not a current line
```

The verdict is:

```text
QUARK_CURRENT_PARITY_SELECTOR_PASS
```

This does not derive the full quark source from microscopic BCC dynamics.  It
reduces the old current-source premise to:

```text
colored_quark_mass_source_is_selected_S2_odd_boundary_current
```

## Session 23 down identity-return veto

Session 23 decides the down bottom fork inside the retarded-current model.
The primitive quark shell is:

```text
1_even + 5_odd = 1_direct + 2_BCC + 3_color
```

The finite retarded-current criterion is:

```text
down mass event leaves the visible sheet before returning
```

The only zero-excursion return is:

```text
direct_even_return
```

so the criterion rejects it.  The allowed retarded returns are exactly the five
odd current channels:

```text
2_BCC + 3_color
```

Therefore the selected retarded down profile is:

```text
(6,2,5)/6 -> (1,1/sqrt(3),sqrt(5/6))
```

The contact/S3 baseline remains visible as a control:

```text
(6,2,4)/6 -> (1,1/sqrt(3),sqrt(2/3))
```

It fails the nonidentity retarded-current predicate because its bottom count
is not the five hidden odd returns.

The verdict is:

```text
DOWN_IDENTITY_RETURN_VETO_RANK_FIVE_CONDITIONAL_PASS
```

This proves the rank-five coefficient inside the retarded-current model.  It
does not derive the non-contact criterion from bare BB block algebra.

## Session 24 quark Higgs-door orientation coupling

Session 24 tests the remaining quark keystone:

```text
H_tilde -> retarded/oriented flag N
H       -> Hermitian closure Delta_N
```

The available ingredients pass:

```text
SM hypercharge forces H_tilde/H doors
selected-S2 odd current selects b
depth-scar successor certificate supplies N
Hermitian closure supplies Delta_N
```

The endpoint reflection in residual order `(u,a,b)` is:

```text
R = [[0,0,1],
     [0,1,0],
     [1,0,0]]
```

It obeys:

```text
R N R = N.T
```

so Higgs conjugation/reversal supplies the reverse flag.  The down operator is
the paired Hermitian closure:

```text
Delta_N = N N.T + N.T N - (N + N.T)
```

and it is reflection-invariant:

```text
R Delta_N R = Delta_N
```

but:

```text
R N R != Delta_N
```

Thus conjugation/reversal does not produce the Hermitian closure.  It produces
the reverse oriented flag, while the down readout needs an additional pairing
operation.

The declared assignment and the swapped assignment both survive the currently
encoded gauge, selected-current, and flag constraints:

```text
declared: H_tilde -> N,       H -> Delta_N
swapped:  H_tilde -> Delta_N, H -> N
```

The verdict is:

```text
QUARK_HIGGS_ORIENTATION_COUPLING_NOT_DERIVED_AUDIT
```

The remaining premise is still:

```text
higgs_door_orientation_couples_H_tilde_to_retarded_flag_and_H_to_flag_closure
```

but it is now sharper: the missing physics is the boundary dynamics that makes
the direct Higgs door read a paired Hermitian current covariance rather than
the reverse flag.

## Verdict summary

- Session 01 verdict: **UNIVERSAL_BATH_SPINE_PASS**.
- Session 02 verdict: **SOURCE_DICTIONARY_CORE_PASS**.
- Session 03 verdict: **NEUTRINO_PRODUCT_BATH_INTERNAL_PASS**.
- Session 04 verdict: **CHARGED_LEPTON_CMV_HEAD_PACKAGING_PASS**.
- Session 05 verdict: **CHARGED_LEPTON_2_OVER_9_OCCUPATION_PASS**.
- Session 06 verdict: **UP_NILPOTENT_HEAD_CONDITIONAL_PASS**.
- Session 07 verdict: **DOWN_HEAD_FORK_LOCALIZED_PASS**.
- Session 08A verdict: **QUARK_HEIGHT_DOOR_NO_DERIVATION_AUDIT**.
- Session 08B verdict: **QUARK_COLOR_LIFT_NO_SELECTION_AUDIT**.
- Session 09 verdict: **NEUTRINO_BCC_MOMENT_GRAPH_NOT_DERIVED_AUDIT**.
- Session 10 verdict: **NEUTRINO_FAMILY_PORT_GRAPH_INTERNAL_PASS**.
- Session 11 verdict: **NEUTRINO_ACTIVE_PLANE_INCIDENCE_PASS**.
- Session 12 verdict: **NEUTRINO_Q_MISMATCH_RETARDED_COMPRESSION_PASS**.
- Session 13 verdict: **NEUTRINO_BOUNDARY_MATERIAL_ORIGIN_NOT_DERIVED_AUDIT**.
- Session 14 verdict: **CHARGED_LEPTON_MINIMAL_BOUNDARY_GRAPH_PASS**.
- Session 15 verdict: **QUARK_SOURCE_FREEZE_NOT_DERIVED_AUDIT**.
- Session 16 verdict: **QUARK_NORMAL_DEPTH_PLACEMENT_NOT_DERIVED_AUDIT**.
- Session 17 verdict: **QUARK_ACTIVE_COLOR_RETURN_MICROCANONICAL_CONDITIONAL_PASS**.
- Session 18 verdict: **QUARK_DOWN_ODD_SHELL_RANK_FIVE_CONDITIONAL_PASS**.
- Session 19 verdict: **CHARGED_LEPTON_TRACE_TORSION_ORIGIN_NOT_DERIVED_AUDIT**.
- Session 20 verdict: **QUARK_HEIGHT_ORIENTATION_BRIDGE_NOT_DERIVED_AUDIT**.
- Session 21 verdict: **QUARK_ACTIVE_CURRENT_READOUT_CONDITIONAL_PASS**.
- Session 22 verdict: **QUARK_CURRENT_PARITY_SELECTOR_PASS**.
- Session 23 verdict: **DOWN_IDENTITY_RETURN_VETO_RANK_FIVE_CONDITIONAL_PASS**.
- Session 24 verdict: **QUARK_HIGGS_ORIENTATION_COUPLING_NOT_DERIVED_AUDIT**.

## Meaning

The sidecar has converted "universal silver tail" into exact Schur/Jacobi/CMV
algebra, frozen the lepton-side source anchors that are supported upstream,
certified the neutrino core inside the product half-line bath, audited that the
current microscopic BB edge update still lacks the family-port graph needed to
upgrade that product result, supplied the selected internal family-port graph
whose direct moments give the required `u/b` decoupling, built the
selected-port incidence derivation of the active plane,
connected that active plane to the exact BB q-mismatch hard-gap and retarded
compression model,
audited that the deeper boundary-material origin of locking/outgoing
asymptotics is not derived from bare BB blocks,
packaged the charged-lepton finite CMV head,
derived the charged-lepton `2/9` torsion as an occupation moment,
constructed the exact minimal charged-lepton two-sided boundary graph and its
Koide equipartition residue,
audited the charged-lepton trace/torsion origin and shown that equipartition
forces exactly two coherent trace paths inside the minimal graph while their
microscopic BCC/Higgs origin and the `2/9` occupation-to-angle dynamics remain
open,
reduced the quark height-door premise by showing that the up nilpotent and down
Hermitian repair are two readouts of the same certified depth-scar successor
flag,
implemented the colored active-current quark ansatz in which the source line is
`b`, normal depth is first-passage order from `b`, the up profile follows from
`exp(N/sqrt(2))b`, and the down sector is treated as a Hermitian current
covariance over shell measures rather than a scalar vector,
reduced the current-source selector by showing that the selected-port `S2`
odd current is exactly `b`,
decided the down bottom fork inside the retarded-current model by vetoing the
direct identity/contact return and selecting the five-channel odd shell,
audited the Higgs-door orientation coupling and showed that endpoint
reflection gives `N.T`, not the Hermitian closure `Delta_N`,
implemented the conditional up nilpotent finite head,
implemented the conditional down count-level Jacobi head,
and assembled the quark source dependency graph while refusing to freeze
`V_u,V_d` because the four microscopic source prerequisites remain open. It
has now also audited the normal-depth prerequisite against the exact
depth-scar theorem and shown that graph normal-mode depths do not by themselves
freeze source placements. It has now reduced the active hidden color-return
bit to the primitive-shell equal-degeneracy / max-entropy prior. The isolated
quark-source preconditions are: the Higgs-door orientation-coupling rule, that
microcanonical active-return prior, the non-contact retarded down-current rule,
and the normal-depth placements.
Session 21 supplies a better candidate for the normal-depth placements:
first-passage order from the colored active-current endpoint `b`.  Session 22
reduces the source condition to a selected-`S2` odd-current statement; it
remains conditional until that odd-current dynamics is derived from the
microscopic colored boundary event.

The parked gates are now sharper:

- build the microscopic BCC family-port graph and compute
  `<u|H_BCC^k|b>` without inserting `I_family` (**internal selected graph
  supplied in Session 10; incidence projector supplied in Session 11;
  q-mismatch/retarded closure supplied in Session 12 under the
  single-clock/outgoing-boundary model; deeper boundary-material origin still
  open**);
- derive, or audit as irreducible, the deeper boundary-material origin of the
  single-clock locking field and outgoing clock-error asymptotics (**audited
  in Session 13 as not derived from bare BB blocks; a new boundary-material
  model would be required**);
- derive the charged-lepton holonomy selection dynamics beyond the `2/9`
  occupation moment (**Session 14 realizes the graph once the angle is supplied;
  Session 19 proves that equipartition forces two coherent trace paths inside
  the minimal graph, but microscopic trace-path origin and occupation-to-angle
  dynamics remain open**);
- freeze the up/down quark BCC source vectors without flavor data (**Session
  15 assembles the exact dependency graph and records this as not yet
  derived; Session 16 shows the depth-scar spectrum alone does not provide
  normal-depth source placements; Session 21 proposes first-passage order from
  the active current endpoint `b` as the replacement source-depth theorem**);
- derive or replace the Higgs-door orientation-coupling rule that maps
  `H_tilde` to the retarded/oriented flag readout and `H` to the Hermitian
  closure (**Session 20 shows both repair objects come from one certified
  successor flag, but hypercharge still permits the swapped assignment;
  Session 24 shows Higgs conjugation/reversal gives `N.T`, not `Delta_N`,
  so the missing operation is a dynamical pairing/Hermitian closure rule**);
- derive or replace the active hidden color-return rule that selects the
  regular six-channel shell over the spectator three-port shell (**Session 17
  reduces this to the equal-degeneracy / max-entropy primitive-shell prior**);
- derive or replace the non-contact retarded down-current rule (**Session 18
  reduces rank five to the full primitive odd shell; Session 21 reframes this
  as an identity/direct-return veto; Session 23 proves the veto selects
  `(6,2,5)` inside the retarded-current model, but does not derive the
  criterion from bare BB dynamics**);
- derive that a colored quark boundary mass event is a selected-`S2` odd
  boundary current (**Session 22 then forces the line to be `b`**);

CKM/PMNS mixing from Krylov/CMV overlaps is deliberately parked.  It should not
be the next automatic session until at least one of the dynamical gates above
is replaced by a physical principle rather than another conditional audit.
