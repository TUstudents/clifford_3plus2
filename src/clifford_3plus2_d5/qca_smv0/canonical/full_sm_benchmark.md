# QCA_SMv0 Full SM Benchmark

This canonical short rollout activates quark FN effective Yukawas, charged
lepton Yukawas, and PMNS neutrino Yukawas on the compressed production fibre.
It reports the field rollout, quark carrier response, lepton carrier response,
PMNS response, Schur spectrum check, norm conservation, and memory footprint.

## Inputs

- mode: `schur`
- scale label: `benchmark`
- lattice shape: `(2, 1, 1)`
- steps: `2`
- lepton production input: `schur_reduced_pmns_effective_yukawa`

## Rollout

- collision mode: `effective_yukawa`
- norm drift: `1.19e-07`
- extended norm drift: `1.19e-07`
- runtime array bytes: `7296`
- runtime array bytes/site: `3648`

## Physics Tests

- overall passed: `True`

| check | passed |
|---|---:|
| quark_fit | True |
| quark_carrier_transfer | True |
| lepton_carrier_transfer | True |
| pmns_response | True |
| schur_spectrum | True |
| norm_conservation | True |
| no_cross_lepton_leakage | True |
| memory_footprint | True |
| cp_phase_sensitivity | True |
| pmns_ckm_frame_mismatch | True |
| production_contract | True |

## Quark Response

- max up transfer population: `0.0001951`
- max down transfer population: `0.000156101`
- up one-tick residual: `9.1e-11`
- down one-tick residual: `1.3e-10`

## Lepton Response

- charged hierarchy input: `(0.00028758537532501155, 0.059463534268316014, 1.0)`
- neutrino masses: `(0.0, 0.17157287525381, 1.0)`
- PMNS ratio error: `3.47e-18`
- Schur spectrum residual: `4.47e-08`
- PMNS/CKM abs-frame mismatch norm: `1.19804`
- PMNS Jarlskog abs: `0.00857777`
