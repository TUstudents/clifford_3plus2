# Spatial 1D Sidecar Report

Status: sidecar Route-2 local-QCA prototype implemented.

This report records the exact 1D local-QCA sidecar implemented in
[`src/clifford_3plus2_d5/qca/spatial_1d.py`](../../src/clifford_3plus2_d5/qca/spatial_1d.py)
and exposed by
[`scripts/spatial_1d_alpha_search.py`](../../scripts/spatial_1d_alpha_search.py).

## Purpose

The noncommuting on-site Floquet-alpha route reduced the continuous compatible
`J` family to four finite block-sign choices, but did not tie the alpha and
eta signs together. This sidecar tests the Route-2 idea that a genuine spatial
cycle can supply that missing orientation constraint.

It is not a replacement for `rule_to_verdict.py`, and it is not a load-bearing
QCA bridge claim.

## Prototype

The prototype uses a period-12 Brillouin cycle:

```text
alpha winding = 4
eta winding = 3
gcd(4,3) = 1
lcm(4,3) = 12
```

The sidecar now includes an explicit finite-radius local QCA layer. Its
Laurent symbol is reconstructed from two local hopping terms:

```text
hop shift 4 on the three alpha mode-pairs
hop shift 3 on the two eta mode-pairs
mode_windings = [4,4,4,3,3]
T(z) = P_alpha z^4 + P_eta z^3
```

The layer is checked as a real locality-preserving QCA by exact Laurent
coefficient identities, not only by sampled roots. At each root `zeta_12^n`,
the reconstructed transfer matrix agrees exactly with the root-of-unity
transfer. The alpha sector has rank `6`; the eta sector has rank `4`.

The sidecar then applies the tested transport hypothesis: coprime alpha/eta
windings around one shared spatial cycle allow only sign choices with the same
alpha and eta orientation.

## Command

```bash
uv run python scripts/spatial_1d_alpha_search.py --check
```

Current output:

```text
candidate_count: 1
unitary_candidates: 1
coarse_6_4_band_candidates: 1
period: 12
alpha_winding: 4
eta_winding: 3
winding_gcd: 1
winding_lcm: 12
locality_radius: 4
sample_count: 12
transfer_unitary_on_samples: true
alpha_projector_rank: 6
eta_projector_rank: 4
orientation_choices_before_transport: 4
orientation_choices_after_transport: 2
sign_coupled_to_global_pm: true
strict_bridge_candidates: 0
route_label: spatial_signs_coupled_to_global_pm
local_hopping_term_count: 2
local_hopping_shifts: [3, 4]
local_hopping_mode_windings: [4, 4, 4, 3, 3]
local_hopping_reconstructs_transfer_on_samples: true
local_hopping_orientation_choices_after_transport: 2
local_hopping_route_label: spatial_local_hopping_signs_coupled
local_qca_layer_name: spatial_1d_alpha_projector_shift_qca
local_qca_term_count: 2
local_qca_shifts: [3, 4]
local_qca_locality_radius: 4
local_qca_finite_radius: true
local_qca_laurent_orthogonal: true
local_qca_symbol_reconstructs_transfer_on_samples: true
local_qca_symbol_unitary_on_samples: true
local_qca_coefficient_algebra_dimension: 2
local_qca_coefficient_center_dimension: 2
local_qca_central_idempotent_ranks: [0, 4, 6, 10]
local_qca_lower_rank_central_idempotents: 0
local_qca_orientation_choices_after_transport: 2
local_qca_route_label: spatial_local_qca_signs_coupled_not_load_bearing
load_bearing_qca_bridge: false
```

Allowed orientation orbits:

```text
alpha=+1, eta=+1
alpha=-1, eta=-1
```

Rejected orientation orbits:

```text
alpha=+1, eta=-1
alpha=-1, eta=+1
```

## Interpretation

This is the first sidecar calculation that directly targets the remaining
block-sign obstruction with an actual finite-radius local QCA layer. In the
prototype, exact Laurent data computes the alpha/eta windings, and spatial
transport reduces the four on-site choices to global `±J`.

The result should be read narrowly. It shows that the Route-2 mechanism has
the right combinatorial shape and that the projector-shift layer is a genuine
locality-preserving QCA. It does not prove that the band projectors and `J` are
forced by microscopic local gates, because the coarse `P_alpha/P_eta`
projectors still enter as layer coefficients.

The next useful Route-2 step is to factor or replace this projector-shift QCA
with microscopic local gates that do not seed the coarse split.
