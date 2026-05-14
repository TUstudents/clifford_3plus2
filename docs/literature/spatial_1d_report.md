# Spatial 1D Sidecar Report

Status: sidecar Route-2 prototype implemented.

This report records the exact 1D transfer diagnostic implemented in
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

At each root `zeta_12^n`, the transfer matrix is a diagonal exact
root-of-unity matrix on the ten real carrier coordinates. The alpha sector has
rank `6`; the eta sector has rank `4`.

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
block-sign obstruction. In the prototype, spatial transport reduces the four
on-site choices to global `±J`.

The result should be read narrowly. It shows that the Route-2 mechanism has
the right combinatorial shape; it does not prove that a finite-depth real QCA
generates the transfer rule, nor that the band projectors and `J` are forced by
microscopic local gates.

The next useful Route-2 step is to replace this diagonal root-of-unity transfer
diagnostic with an explicit local hopping construction whose transfer matrix
has the same sign-coupling invariant.
