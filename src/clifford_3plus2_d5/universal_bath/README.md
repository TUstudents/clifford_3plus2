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
