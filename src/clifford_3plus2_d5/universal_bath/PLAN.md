# universal_bath Plan

## Purpose

Turn the bath idea into a computable sidecar:

```text
sector response = finite Lanczos head + universal silver tail
```

The sidecar tests whether flavor sectors differ only in the first few floors of
their boundary spectral measure while sharing one retarded BCC/BB tail.

## Session 01 - Bath spine

Pass only if:

- the period-one tail satisfies `t = 1 / (z - t)`;
- `t(2 sqrt(2)) = sqrt(2) - 1`;
- finite scalar moments round-trip through a Jacobi head;
- finite-head Schur response equals the continued fraction;
- replacing the silver tail changes the response;
- the reduction taxonomy separates positive Jacobi, indefinite look-ahead
  Jacobi, and CMV/OPUC sectors;
- the CMV/OPUC free-tail criterion is `alpha_n = 0` after the finite head.

Verdict target:

```text
UNIVERSAL_BATH_SPINE_PASS
```

Session 01 proves the common algebraic spine. It does not derive any sector
masses.

## Session 02 - Write-once source dictionary

Freeze sector source anchors before computing any masses:

```text
V_f = {charge slot, residual port geometry, normal-depth placement}
```

Pass only if:

- every frozen source is fixed without using flavor data;
- every frozen source has a declared charge anchor;
- every frozen source has a declared residual BCC boundary port vector;
- every frozen source has a declared normal-depth placement;
- the radial first-hop survival weight is universal, inherited from the BB
  same-normal identity.

Session 02 freezes only anchors that are supported by existing sidecars:

- `neutrino_collective_u`: $u$, depth 1;
- `neutrino_edge_b`: $b$, depth 0;
- `charged_lepton_active_e1`: $e_1$, depth 2.

It records up/down quark charge anchors but leaves their BCC source vectors
unfrozen.  That is the intended anti-circularity behavior: do not invent a quark
source to make the universal-bath program look complete.

Verdict target:

```text
SOURCE_DICTIONARY_CORE_PASS
```

Abort only if a frozen source fails normalization, uses flavor data, or does not
share the BB survival branch.

## Session 03 - Neutrino product bath

Compute cross-return moments in the framed sterile source model:

```text
<u|H_Q^k|b>, k = 1,2,3,4
```

The product bath becomes a theorem only if the cross returns vanish from the
graph/source structure rather than from an inserted tensor ansatz.

Session 03 uses the frozen Session 02 neutrino anchors and the product bath

```text
H_Q = H_chain tensor I_family
```

to verify:

- diagonal returns for `u` and `b` are equal;
- cross-return moments vanish for `k = 0..4`;
- the normalized Weyl-tail response is `t(z)^2 P_u + P_b`;
- at `z = 2 sqrt(2)`, the response is `epsilon^2 P_u + P_b`;
- rank-one, wrong-source, and alternate-tail controls fail.

Verdict target:

```text
NEUTRINO_PRODUCT_BATH_INTERNAL_PASS
```

This is `C:9` inside the product half-line bath. Session 09 blocks the physical
upgrade until a microscopic BCC family-port graph defines the `u/b` cross
moments without inserting `I_family`.

## Session 04 - Charged-lepton CMV/OPUC head

Use CMV/OPUC seeded by the charged-lepton source. The charged-lepton holonomy
should appear as a finite-head Schur/Verblunsky coefficient, not as an inserted
rotation.

Session 04 uses:

- frozen source `charged_lepton_active_e1`;
- decomposition `e1 = sqrt(2/3) a + 1/sqrt(3) u`;
- source depth 2 and the universal tail value `epsilon`;
- primitive holonomy word `SCHUR_RETURN -> PARENT_A3 -> RESIDUAL_A2`.

It verifies:

```text
sin(theta_e) = sqrt(3/2) epsilon^2
alpha_e = sin(theta_e) exp(-5 pi i / 12)
|alpha_e|^2 < 1
```

and builds an exact two-state CMV/Givens head followed by the free OPUC tail
`alpha_n = 0`.

Verdict target:

```text
CHARGED_LEPTON_CMV_HEAD_PACKAGING_PASS
```

This session does not derive charged-lepton masses and does not assemble PMNS.

## Session 05 - Charged-lepton 2/9 torsion gate

Decide whether the charged-lepton torsion value is a real source moment or a
post-hoc arithmetic identity.

Pass only if the frozen source

```text
e1 = sqrt(2/3) a + 1/sqrt(3) u
```

has exact occupations

```text
p_a = 2/3, p_u = 1/3, p_b = 0
```

so the incoherent transition weight is

```text
p_a p_u = 2/9.
```

The coherent amplitude control `sqrt(2)/3`, equal-weight control `1/4`, and
one-port controls must be rejected. This session does not rederive the CMV
phase or charged-lepton masses.

Verdict target:

```text
CHARGED_LEPTON_2_OVER_9_OCCUPATION_PASS
```

## Session 06 - Up-quark nilpotent CMV head

Model the finite nilpotent head and test whether silver-tail injection forces
the coherent entry value:

```text
x = 1 / sqrt(2)
```

The pass verdict is conditional because the up-quark BCC source vector remains
unfrozen in the Session 02 dictionary.

Verdict target:

```text
UP_NILPOTENT_HEAD_CONDITIONAL_PASS
```

## Session 07 - Down-quark indefinite symmetric head

Compare the 3-port permutation graph with the 6-element `S_3` Cayley graph.
The physical down bath is whichever graph produces the correct first weights
without inserting the answer. Use look-ahead indefinite Lanczos if a real
signature breakdown occurs; do not clamp negative $b_n^2$.

The implemented count-level gate separates the clean `(6,2,4)` baseline from
the available-but-unforced `(6,2,5)` candidate. It does not select the bottom
rank-5 line.

Verdict target:

```text
DOWN_HEAD_FORK_LOCALIZED_PASS
```

## Session 08A - Quark height-door audit

Before freezing quark sources, separate the electroweak charge theorem from the
height-dynamics premise.

Pass only if:

- hypercharge forces `H_tilde` for up and `H` for down;
- both neutral Higgs components have `Q_em = Y + T3 = 0`;
- the declared up repair is the oriented length-3 nilpotent;
- the declared down repair is the Hermitian path closure with spectrum
  `{0,1,3}`;
- swapping repair modes is still hypercharge-allowed, proving the repair split
  does not come from electroweak charges alone.

Verdict target:

```text
QUARK_HEIGHT_DOOR_NO_DERIVATION_AUDIT
```

This session does not freeze `V_u` or `V_d`.

## Session 08B - Quark color-lift audit

Audit color as visible gauge covariance versus hidden return multiplicity.

Pass only if:

- a fixed rank-one visible color source is rejected by `SU(3)_c` covariance;
- a color-scalar spectator embedding preserves visible color but stays on the
  three-port shell;
- an active hidden color-return embedding also preserves visible color while
  reaching the six-channel shell `1_direct + 2_BCC + 3_color`;
- the active lift makes `(6,2,5)` available but not forced;
- gauge covariance alone does not select active over spectator.

Verdict target:

```text
QUARK_COLOR_LIFT_NO_SELECTION_AUDIT
```

## Session 09 - Neutrino microscopic BCC moment audit

Attack the strongest product-bath premise before assembling more sectors.
Session 03 vanishing cross moments are internal to

```text
H_Q = H_chain tensor I_family
```

and are automatic from the inserted family identity.  The microscopic gate is:

```text
<u|H_BCC^k|b>, k = 0,1,2,3,4
```

computed from a BCC boundary graph carrying the `u` and `b` family ports.

The implemented Session 09 audit checks the currently available microscopic BB
edge update:

- exact q=0 same-normal norm `I/2`;
- exact q=+-2 mixed-normal leakage norm `I/2`;
- total edge norm `I`;
- Session 03 product-bath cross moments still vanish internally;
- the rank-one no-family control still has cross return;
- the current microscopic BB edge graph has spinor/q-depth labels but no `u,b`
  family-port labels.

Verdict target:

```text
NEUTRINO_BCC_MOMENT_GRAPH_NOT_DERIVED_AUDIT
```

This is not a pass of the neutrino crown. It records that the exact BB edge
update exists, but the BCC family-port graph needed to define
`<u|H_BCC^k|b>` has not been built.

## Session 10 - Selected neutrino family-port graph

Build the missing family-port graph without using flavor data:

```text
H_fam = H_chain tensor (P_u + P_b) + I tensor Lambda P_a
```

The selected radial mode `a` is separated. The active neutrino plane `span(u,b)`
gets two isomorphic radial scar fibers.

Pass only if:

- `P_u + P_b + P_a = I`;
- the graph differs from `H_chain tensor I_family`;
- direct graph moments give `<u|H_fam^k|b> = 0` for checked finite powers;
- direct graph moments give equal `u` and `b` diagonal returns;
- radial-active moments such as `<a|H_fam^k|u>` vanish;
- the full residual `K3` graph control fails equal `u/b` returns;
- the full product identity control propagates the radial `a` mode and is
  therefore not the selected graph;
- closing the active plane with the universal retarded tail gives
  `epsilon^2 P_u + P_b`.

Verdict target:

```text
NEUTRINO_FAMILY_PORT_GRAPH_INTERNAL_PASS
```

This completes the internal family-port graph needed by Session 09. It does not
yet prove that the selected active-plane condition is forced by the microscopic
BB edge update.

## Session 11 - Selected active-plane incidence

Derive

```text
P_act = P_u + P_b, P_rad = P_a
```

from selected boundary incidence:

```text
e1 = sqrt(2/3) a + u/sqrt(3)
detrace(e1) = a
active plane = a^perp = span(u,b)
```

Pass only if selected `S2` symmetry alone is shown not to be sufficient.  The
verdict is:

```text
NEUTRINO_ACTIVE_PLANE_INCIDENCE_PASS
```

This proves the projector from selected-port incidence but still leaves the
physical BB/QCA dynamics open.

## Session 12 - Q-mismatch radial penalty and retarded closure

Derive the conditional microscopic model that penalizes the detraced
selected-port line `a` and closes the universal outgoing tail only on the
active `u,b` plane:

```text
bb q-mismatch -> Lambda P_a
retarded outgoing bath -> H_chain tensor (P_u + P_b)
```

The certificate combines:

- exact BB same-normal/mixed-normal split by `Delta q`;
- exact `1/2 + 1/2` BB norm split;
- `g q^2` hard-gap Schur feedback `I/[2(z-4g)] -> 0`;
- retarded half-line Weyl branch;
- block-triangular retarded closure;
- recurrent wedge control rejection.

Verdict:

```text
NEUTRINO_Q_MISMATCH_RETARDED_COMPRESSION_PASS
```

This closes the gate inside the single-clock/outgoing-boundary model. It does
not derive the deeper boundary material that realizes that model.

## Session 13 - Boundary-material origin audit

Audit the two remaining physical inputs:

```text
single_clock_locking_field_is_realized_by_boundary_material
mixed_normal_clock_error_ports_are_outgoing_asymptotic_leads
```

The audit derives the local mismatch coordinate but rejects a full derivation
from bare BB blocks:

```text
q = r1-r2 is unique among local linear single-clock mismatches
K^T K = q^2 if the constraint field is admitted
bare BB blocks contain no stiffness/gap parameter
bare BB blocks do not select outgoing over recurrent asymptotics
```

Verdict:

```text
NEUTRINO_BOUNDARY_MATERIAL_ORIGIN_NOT_DERIVED_AUDIT
```

The conditional neutrino graph is now complete inside the
single-clock/outgoing-boundary model; the deeper material origin remains a
named premise.

## Session 14 - Charged-lepton minimal family-port boundary graph

Build the minimal two-sided colorless active boundary graph. The known source
facts are:

```text
e1 = sqrt(2/3) a + u/sqrt(3)
2/9 = p_a p_u source occupation moment
```

The exact graph is:

```text
Q_e = span(t_+, t_-, p_a, p_b)
B_e = V_R^T (z-H_Q)^-1 V_L
Res B_e = sqrt(2) P_u + R_theta P_perp
theta = -2*pi/3 - 2/9
```

Pass only if:

- the two-sided residue matches the target;
- acting on `e1` gives trace/traceless equipartition;
- Koide `K=2/3` is exact;
- one trace path fails equipartition;
- a one-sided Hermitian self-energy cannot produce the nontrivial rotation;
- the `2/9` torsion is kept as an input, not rederived.

Verdict:

```text
CHARGED_LEPTON_MINIMAL_BOUNDARY_GRAPH_PASS
```

This is a minimal graph realization.  It does not derive the microscopic
BCC/Higgs origin of the two trace paths or the active `2/9` torsion dynamics.

## Session 15 - Quark source assembly and freeze audit

Assemble the conditional quark heads and the source-freeze prerequisites in one
ledger.  The goal is not to fit quark masses.  The gate is:

```text
Can V_u,V_d be frozen as microscopic BCC family-port sources without flavor data?
```

Pass the audit only if:

- the common residual family incidence basis from Session 11 is available;
- SM Higgs charge doors from Session 08A are available;
- the conditional up nilpotent head from Session 06 is assembled;
- the conditional down real-symmetric heads from Session 07 and color lift from
  Session 08B are assembled;
- unresolved source fields are explicitly reported;
- the audit refuses to freeze quark sources while these inputs remain open:

```text
height_dynamics_selects_up_nilpotent_down_hermitian
microscopic_active_hidden_color_return_selects_regular_s3_shell
boundary_dynamics_selects_or_kills_down_rank_five_line
quark_normal_depth_placements_on_bcc_scar_are_frozen
```

Verdict target:

```text
QUARK_SOURCE_FREEZE_NOT_DERIVED_AUDIT
```

## Session 16 - Quark normal-depth placement audit

Audit whether the existing depth-scar theorem already freezes quark source
normal-depth placements.

Pass the audit only if:

- the Session 15 quark source assembly ledger is available;
- the nilpotent depth-scar flag `N=|u><a|+|a><b|` passes;
- the microscopic-locality theorem supplies the conditional filtration
  `h(u,a,b)=(0,1,2)`;
- the normal-mode depth spectrum is `{0,2,6}`;
- the audit shows this does **not** imply `V_u.normal_depth` or
  `V_d.normal_depth`, because graph depths are normal-mode data, not source
  placements.

Verdict target:

```text
QUARK_NORMAL_DEPTH_PLACEMENT_NOT_DERIVED_AUDIT
```

## Session 17 - Quark active color-return microcanonical audit

Reduce the active hidden color-return blocker using the upstream primitive-shell
microcanonical theorem.

Pass only if:

- the active hidden color lift reaches the full primitive shell
  `1_direct + 2_BCC + 3_color`;
- the spectator shell is a compressed 3-port control;
- equal-degeneracy microcanonical reduction gives the uniform six-label
  density `I_6/6`;
- compressed macrochannel counting gives the rejected `r=1/sqrt(5)`,
  `phase=pi/4` branch;
- the audit keeps the equal-degeneracy / max-entropy prior as an input rather
  than a gauge theorem.

Verdict target:

```text
QUARK_ACTIVE_COLOR_RETURN_MICROCANONICAL_CONDITIONAL_PASS
```

## Session 18 - Down odd-shell rank-five audit

Reduce the down rank-five blocker using the active primitive shell.

Pass only if:

- the active primitive shell from Session 17 is available;
- the full shell, BCC odd doublet, and full odd shell have counts `(6,2,5)`;
- the rank-five bottom line is the primitive odd shell, i.e. the complement of
  the even direct line;
- the BCC odd doublet, not the color triplet, supplies the middle count;
- the audit keeps the physical bottom readout as a remaining premise.

Verdict target:

```text
QUARK_DOWN_ODD_SHELL_RANK_FIVE_CONDITIONAL_PASS
```

## Session 19 - Charged-lepton trace-path and torsion origin audit

Try to derive, or honestly audit as irreducible, the two charged-lepton
microscopic inputs left by Session 14:

```text
microscopic_colorless_bcc_higgs_boundary_derives_two_coherent_trace_paths
active_cmv_torsion_angle_2_over_9_is_generated_by_boundary_dynamics
```

Pass the audit only if:

- the Session 14 minimal graph is available;
- the Session 05 torsion occupation gate is available;
- the trace/traceless weight formula for `n` coherent trace paths is derived:

```text
w_trace(n) = n/(n+2),   w_perp(n) = 2/(n+2)
```

- trace/traceless equipartition uniquely selects `n=2`;
- the Session 14 graph supplies exactly two trace-only pole rows;
- one-trace and three-trace controls fail;
- the audit keeps the BCC/Higgs origin of those two rows open;
- the audit confirms that `2/9` is inserted as a rotation angle while no
  occupation-to-angle boundary dynamics is derived.

Verdict target:

```text
CHARGED_LEPTON_TRACE_TORSION_ORIGIN_NOT_DERIVED_AUDIT
```

## Session 20 - Quark height-orientation bridge audit

Reduce the height-door premise using the depth-scar successor certificate.

Pass only if:

- Session 08A still shows that hypercharge forces `H_tilde/H` doors but not
  repair modes;
- the depth-scar successor certificate supplies the oriented flag
  `a -> u`, `b -> a`;
- the up repair is exactly the oriented nilpotent flag;
- the down repair is exactly the Hermitian flag-Laplacian closure of the same
  flag;
- the audit keeps the Higgs-door orientation-coupling rule open rather than
  pretending it follows from hypercharge.

Verdict target:

```text
QUARK_HEIGHT_ORIENTATION_BRIDGE_NOT_DERIVED_AUDIT
```

## Session 21 - Quark active-current readout

Test the source-freeze hypothesis:

```text
quark source = colored active current line b
```

rather than the selected scalar port `e1`.

Pass only if:

- Session 11 supplies `P_rad=P_a` and `P_act=P_u+P_b`;
- the non-scalar active current line is uniquely `b`;
- first-passage orders from `b` under the certified flag are `(2,1,0)` in
  light-to-heavy order;
- coherent up readout `exp(N/sqrt(2)) b` reproduces
  `(1/4,1/sqrt(2),1)`;
- the geometric up control is rejected;
- down is treated as a Hermitian current covariance / word-shell measure, with
  both `(6,2,4)` and `(6,2,5)` visible;
- the identity-word veto remains an open microscopic rule, not an inserted
  data preference.

Verdict target:

```text
QUARK_ACTIVE_CURRENT_READOUT_CONDITIONAL_PASS
```

## Session 22 - Quark current-parity selector

Reduce the Session 21 current-source ansatz using the selected-port residual
symmetry.

Pass only if:

- the selected `S2` swap of the two unselected ports acts on `(u,a,b)` as
  `diag(+,+,-)`;
- the even projector is `P_u+P_a`;
- the odd projector is `P_b`;
- the oriented current across the unselected pair is
  `(e2-e3)/sqrt(2)=b`;
- Session 11 active incidence still gives `P_act=P_u+P_b`;
- active incidence alone is shown insufficient because it also contains the
  even scalar line `u`;
- intersecting the active plane with the selected-odd current line selects
  `b`;
- scalar `u`, radial `a`, and selected scalar port `e1` controls are rejected.

Verdict target:

```text
QUARK_CURRENT_PARITY_SELECTOR_PASS
```

This reduces the old Session 21 premise
`colored_quark_current_selects_active_non_scalar_b_line` to the sharper
physical premise
`colored_quark_mass_source_is_selected_S2_odd_boundary_current`.  It does not
derive the Higgs-door orientation coupling, the down Hermitian covariance
readout, or the identity-return veto.

## Session 23 - Down identity-return veto

Decide the down bottom fork inside the retarded-current model:

```text
sqrt(2/3)  contact/S3 baseline
sqrt(5/6)  nonidentity odd-current shell
```

Pass only if:

- Session 21 supplies the Hermitian down current-covariance readout frame;
- Session 22 supplies the selected-`S2` odd current line;
- Session 18 supplies the active primitive shell;
- the primitive shell has a unique identity/direct return line;
- the finite retarded-current criterion rejects exactly that zero-excursion
  identity/direct line;
- the allowed down current returns are exactly the five odd channels
  `2_BCC + 3_color`;
- the strange count remains the BCC odd doublet;
- the retarded count vector is `(6,2,5)`;
- the contact/S3 baseline `(6,2,4)` is kept as a rejected control under the
  retarded predicate, not erased.

Verdict target:

```text
DOWN_IDENTITY_RETURN_VETO_RANK_FIVE_CONDITIONAL_PASS
```

This decides the bottom coefficient inside the retarded-current model.  It does
not derive the physical non-contact criterion from bare BB block algebra.

## Session 24 - Quark Higgs-door orientation coupling

Try to derive, or honestly audit as irreducible, the assignment:

```text
H_tilde -> retarded/oriented flag N
H       -> Hermitian closure Delta_N
```

Pass as a theorem only if the available gauge/current/flag geometry rejects the
swapped assignment.  Pass as an audit if the obstruction is exact and the
swapped assignment remains allowed.

Checks:

- SM hypercharge still forces the `H_tilde/H` doors;
- Session 22 supplies the selected-`S2` odd current source;
- Session 20 supplies one certified flag and its Hermitian closure;
- endpoint reflection satisfies `R N R = N.T`;
- `Delta_N` is reflection-invariant;
- endpoint reflection does not map `N` to `Delta_N`;
- the declared assignment is constructible;
- the swapped assignment is also constructible under the same available
  constraints.

Verdict target:

```text
QUARK_HIGGS_ORIENTATION_COUPLING_NOT_DERIVED_AUDIT
```

This would show the missing operation is the dynamical pairing/Hermitian
closure selected by the direct Higgs door, not Higgs conjugation alone.

## Parked - No automatic Session 25

The sidecar is parked after Session 24.  Do not proceed to CKM/PMNS Krylov
overlaps as the next automatic session.  Mixing requires quark kernels whose
door-to-readout assignments are not just conditional.

Reopen only if a new physical principle derives or replaces at least one
parked dynamical gate, especially:

```text
H_tilde -> retarded/oriented flag N
H       -> paired Hermitian current covariance Delta_N
```

Without that, CKM/PMNS assembly would mostly stack conditional overlaps on top
of the unresolved Higgs-door orientation premise.
