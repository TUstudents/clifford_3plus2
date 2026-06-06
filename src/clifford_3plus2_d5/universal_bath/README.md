# universal_bath

Universal bath sidecar for the flavor and mass theory.

The sidecar studies the hypothesis

$$
\text{sector bath}
=
\text{finite source-dependent Lanczos head}
+
\text{universal retarded silver tail}.
$$

The finite head contains sector geometry: source placement, family ports,
color, weak charge, and left/right coupling. The tail is universal and is
inherited from the BCC/BB marginal-stability theorem:

$$
t(z)=\frac{z-\sqrt{z^2-4}}2,
\qquad
t(2\sqrt2)=\sqrt2-1.
$$

The sidecar is not a new fit of masses. It is the proposed forward
spectral-density principle replacing inverse reconstruction in
`radial_response`.

## Closure Axiom

The physical axiom of the sidecar is:

$$
\boxed{
\text{after its finite forced head, every sector closes on the BB band-edge
silver tail.}
}
$$

This replaces many sector prefactor choices by one falsifiable closure. It is
not hidden as a theorem from nothing. The BB tail itself is `C:9` from the
band-edge sidecar; the claim that every physical sector reaches this same tail
is `C:6` until the source moments are computed.

## Reduction Taxonomy

Different sectors require different reduction tools:

| Sector class | Tool | Reason |
|---|---|---|
| positive one-sided responses | scalar Jacobi | genuine positive measure |
| real non-positive shells | symmetric indefinite look-ahead Jacobi | signature may force $b_n^2<0$ or serial breakdown |
| chiral/unitary sectors and phases | CMV/OPUC | phases live in Verblunsky coefficients on the disk |

In CMV/OPUC language the universal free tail is

$$
\alpha_n=0
$$

after the finite head. This is the circle analogue of
$(a_\infty,b_\infty)=(0,1)$.

## Upstream Inputs

- `synthesis/BAND_EDGE_SELECTION_THEOREM.md`: proves the regular-retarded BB
  marginal saddle and the silver tail value.
- `boundary_response`: supplies the framed sterile prototype and transfer root.
- `radial_response`: proves that finite spectral data alone reconstruct masses
  but do not select them.
- `scalar_clebsch`: supplies the current quark coefficient targets.
- `synthesis/NEUTRINO_PROTOTYPE.md`: prototype sector for a clean Schur/seesaw
  bath.

## Session Discipline

Each session should add:

1. a Markdown report `SESSION_XX_*.md`;
2. a Python certificate under `scripts/`;
3. pytest invariants for exact algebra;
4. an updated `STATUS.md` verdict.

Exact finite identities are `C:9`. Sector universality is conditional until the
sector source moments are computed.

## Current Certificates

```bash
uv run python -m clifford_3plus2_d5.universal_bath.scripts.session_01_bath_spine
uv run python -m clifford_3plus2_d5.universal_bath.scripts.session_02_source_dictionary
uv run python -m clifford_3plus2_d5.universal_bath.scripts.session_03_neutrino_product_bath
uv run python -m clifford_3plus2_d5.universal_bath.scripts.session_04_charged_lepton_cmv_head
uv run python -m clifford_3plus2_d5.universal_bath.scripts.session_05_charged_lepton_torsion
uv run python -m clifford_3plus2_d5.universal_bath.scripts.session_06_up_quark_nilpotent_cmv_head
uv run python -m clifford_3plus2_d5.universal_bath.scripts.session_07_down_quark_indefinite_jacobi_head
uv run python -m clifford_3plus2_d5.universal_bath.scripts.session_08a_quark_height_door_audit
uv run python -m clifford_3plus2_d5.universal_bath.scripts.session_08b_quark_color_lift_audit
uv run python -m clifford_3plus2_d5.universal_bath.scripts.session_09_neutrino_bcc_moment_audit
uv run python -m clifford_3plus2_d5.universal_bath.scripts.session_10_neutrino_family_port_graph
uv run python -m clifford_3plus2_d5.universal_bath.scripts.session_11_neutrino_active_plane
uv run python -m clifford_3plus2_d5.universal_bath.scripts.session_12_neutrino_q_mismatch_retarded
uv run python -m clifford_3plus2_d5.universal_bath.scripts.session_13_neutrino_boundary_material_audit
uv run python -m clifford_3plus2_d5.universal_bath.scripts.session_14_charged_lepton_boundary_graph
uv run python -m clifford_3plus2_d5.universal_bath.scripts.session_15_quark_source_assembly
uv run python -m clifford_3plus2_d5.universal_bath.scripts.session_16_quark_normal_depth_audit
uv run python -m clifford_3plus2_d5.universal_bath.scripts.session_17_quark_active_color_microcanonical
uv run python -m clifford_3plus2_d5.universal_bath.scripts.session_18_quark_down_odd_shell
```

Current verdicts:

- `UNIVERSAL_BATH_SPINE_PASS`
- `SOURCE_DICTIONARY_CORE_PASS`
- `NEUTRINO_PRODUCT_BATH_INTERNAL_PASS`
- `CHARGED_LEPTON_CMV_HEAD_PACKAGING_PASS`
- `CHARGED_LEPTON_2_OVER_9_OCCUPATION_PASS`
- `UP_NILPOTENT_HEAD_CONDITIONAL_PASS`
- `DOWN_HEAD_FORK_LOCALIZED_PASS`
- `QUARK_HEIGHT_DOOR_NO_DERIVATION_AUDIT`
- `QUARK_COLOR_LIFT_NO_SELECTION_AUDIT`
- `NEUTRINO_BCC_MOMENT_GRAPH_NOT_DERIVED_AUDIT`
- `NEUTRINO_FAMILY_PORT_GRAPH_INTERNAL_PASS`
- `NEUTRINO_ACTIVE_PLANE_INCIDENCE_PASS`
- `NEUTRINO_Q_MISMATCH_RETARDED_COMPRESSION_PASS`
- `NEUTRINO_BOUNDARY_MATERIAL_ORIGIN_NOT_DERIVED_AUDIT`
- `CHARGED_LEPTON_MINIMAL_BOUNDARY_GRAPH_PASS`
- `QUARK_SOURCE_FREEZE_NOT_DERIVED_AUDIT`
- `QUARK_NORMAL_DEPTH_PLACEMENT_NOT_DERIVED_AUDIT`
- `QUARK_ACTIVE_COLOR_RETURN_MICROCANONICAL_CONDITIONAL_PASS`
- `QUARK_DOWN_ODD_SHELL_RANK_FIVE_CONDITIONAL_PASS`

Session 02 freezes the supported lepton-side anchors and keeps quark sources
explicitly unresolved until their BCC source vectors are derived without flavor
data. Session 03 certifies the neutrino core inside the product half-line bath,
with the physical product-factorization premise still named. Session 04 builds
the charged-lepton finite CMV head and keeps PMNS assembly parked. Session 05
derives `2/9` as the frozen charged-lepton source occupation transition
`p_a p_u`, while rejecting the coherent-amplitude interpretation. Session 06
closes the conditional up nilpotent CMV head with `x=1/sqrt(2)` from the BB
survival branch. Session 07 compares the down 3-port and regular-S3 real
symmetric heads, showing the `(6,2,5)` rank-five candidate is available but not
selected by S3 alone. Session 08A proves the SM Higgs doors but shows the
coherent-up/Hermitian-down height split is not derived from hypercharge alone.
Session 08B rejects a fixed visible color vector, then separates color-scalar
spectator return from active hidden color return; the latter reaches the
regular six-channel shell but is not selected by gauge covariance alone.
Session 09 audits the strongest neutrino upgrade gate: the exact microscopic
BB edge update supplies the q=0 scar and leakage blocks, but it has no `u,b`
family-port graph.  Thus the Session 03 `epsilon^4` result remains internal to
the product bath until the BCC family-port moment graph is built.
Session 10 supplies the minimal selected family-port graph: the radial `a`
mode is separated, the active `u,b` plane carries isomorphic radial scar
fibers, direct graph moments give zero `u/b` cross returns and equal diagonal
returns, and the universal tail readout gives `epsilon^2 P_u + P_b`.  The
remaining physical gate is deriving that selected active-plane condition from
the microscopic BB edge update.
Session 11 derives that active plane at the incidence level: detracing the
selected residual port `e1` against the collective channel gives the radial
line `a`, and the orthogonal active plane is exactly `P_u + P_b`.  The
remaining physical gate is now sharper: derive that BB q-mismatch penalizes
this detraced `a` line and that the retarded outgoing boundary closes only on
the active incidence plane.
Session 12 closes that gate inside the single-clock/outgoing-boundary model:
the BB edge directions split into same-normal `q=0` and mixed-normal `q=+-2`
blocks with exact `1/2+1/2` norm split; a `g q^2` hard gap makes mixed-normal
Schur feedback vanish; and retarded clock-error leads make visible powers equal
the q=0 survival powers.  The remaining boundary-material question is why the
physical defect realizes the single-clock locking field and outgoing
asymptotics.
Session 13 audits that last question and records a no-derivation result: the
local mismatch coordinate `q=r1-r2` is unique, and `K^T K` gives a positive
`q^2` penalty if the constraint field is admitted, but the bare BB blocks
contain no stiffness parameter and no condition selecting outgoing clock-error
leads over recurrent wedge return.
Session 14 builds the minimal colorless active charged-lepton family-port graph.
The two-sided chiral pole residue is exactly `sqrt(2) P_u + R_theta P_perp`,
and acting on `e1` gives exact trace/traceless equipartition and Koide
`K=2/3`.  The microscopic origin of the two trace paths and the active `2/9`
torsion dynamics remain open.
Session 15 assembles the quark source dependency graph.  The common residual
family incidence basis, SM charge doors, conditional up nilpotent head, and
conditional down real-symmetric heads are all available, but the source freeze
is not derived.  The open microscopic inputs are the height-dynamics rule,
active hidden color-return selection, the down rank-five decision, and actual
normal-depth placements for `V_u,V_d`.
Session 16 audits the normal-depth blocker against the depth-scar theorem.  It
confirms the exact nilpotent flag and `{0,2,6}` normal-mode spectrum, but shows
that these graph depths are not source placements: the path-scar operator is
not a diagonal port-depth assignment, doubled port heights are `(0,2,4)`, and
the quark dictionary still has `normal_depth=None` for both source anchors.
Session 17 reduces the active hidden color-return blocker: inside the primitive
six-label shell, equal-degeneracy microcanonical reduction gives `I_6/6` and
selects the active `1_direct + 2_BCC + 3_color` shell over the compressed
three-port spectator control.  The remaining input is the equal-degeneracy /
max-entropy prior, not gauge covariance.
Session 18 reduces the down rank-five blocker: in the active primitive shell,
the bottom candidate is the full odd shell, the complement of the even direct
line, and the strange middle channel is the BCC odd doublet.  The remaining
premise is the physical down-head readout that assigns bottom to the full odd
shell.
