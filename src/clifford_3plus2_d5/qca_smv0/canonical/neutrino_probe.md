# QCA_SMv0 Neutrino Carrier Probe

This canonical lepton-sector probe promotes the best existing neutrino result
to a simulator input.  It uses direct diagonal effective neutrino Yukawas
`(0, eta, 1)`, with `eta=(sqrt(2)-1)^2`, while charged-lepton Yukawas are set
to zero.  The Schur/seesaw input mode is recorded as the next backend, not used
in this direct probe.

## Inputs

- scale label: `benchmark`
- lattice shape: `(1, 1, 1)`
- steps: `1`
- neutrino Yukawas: `(0.0, 0.17157287525381, 1.0)`
- charged-lepton Yukawas: `(0.0, 0.0, 0.0)`
- source: `L(weak=0)`
- target: `nu_c`

## Neutrino Metadata

- hierarchy: `normal`
- lightest massless: `True`
- epsilon: `0.414213562373`
- eta: `0.171572875254`
- Delta m2 ratio: `0.0294372515229`
- expected epsilon^4: `0.0294372515229`
- ratio error: `3.47e-18`

## Physics Tests

- overall passed: `True`

| check | passed |
|---|---:|
| normal_ordering | True |
| massless_lightest | True |
| ratio_check | True |
| norm_conservation | True |
| carrier_transfer | True |
| no_e_c_leakage | True |
| production_contract | True |

## Family Response

| family | input Yukawa | nu_c population | e_c leakage | norm drift |
|---|---:|---:|---:|---:|
| nu1 | 0 | 0 | 0 | 0 |
| nu2 | 0.171572875 | 0.000147179016 | 0 | 5.96e-08 |
| nu3 | 1 | 0.00499167154 | 0 | 5.96e-08 |

## Interpretation

The probe verifies the direct neutrino baseline on the production fibre:
`L(weak=0)` routes into `nu_c`, zero charged-lepton Yukawas prevent `e_c`
leakage, the direct input has normal ordering with one massless lightest state,
and the mass-squared ratio is the silver prediction `epsilon^4`.
