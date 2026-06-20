# QCA_SMv0 Lepton Phenomenology

This report is produced by the production-facing lepton command.  It converts
charged-lepton masses, neutrino masses/splittings, and PMNS angles into
effective lepton Yukawa matrices, then runs charged and neutrino carrier probes
on the lean QCA production fibre.

## Inputs

- mode: `direct`
- scale label: `benchmark`
- lattice shape: `(1, 1, 1)`
- steps: `1`
- charged-lepton masses: `(0.00051099895, 0.1056583755, 1.77686)`
- charged-lepton Yukawas: `(0.00028758537532501155, 0.059463534268316014, 1.0)`
- neutrino masses: `(0.0, 0.17157287525381, 1.0)`
- PMNS angles: `(0.5836381018669037, 0.8587019919812102, 0.14957417646704585, 3.4033920413889422)`
- production input: `direct_pmns_effective_yukawa`

## Diagnostics

- PMNS unitarity residual: `1.19e-07`
- Delta m2 ratio: `0.0294372515229`
- expected epsilon^4: `0.0294372515229`
- Schur spectrum residual: `4.47e-08`

## Physics Tests

- overall passed: `True`

| check | passed |
|---|---:|
| charged_hierarchy | True |
| charged_carrier_transfer | True |
| charged_no_nu_c_leakage | True |
| normal_ordering | True |
| massless_lightest | True |
| ratio_check | True |
| pmns_unitarity | True |
| pmns_mixing | True |
| neutrino_carrier_transfer | True |
| neutrino_no_e_c_leakage | True |
| schur_spectrum | True |
| norm_conservation | True |
| production_contract | True |

## Charged-Lepton Carrier Response

| family | Yukawa | e_c population | nu_c leakage | norm drift |
|---|---:|---:|---:|---:|
| e | 0.000287585375 | 1.65410637e-11 | 0 | 1.19e-07 |
| mu | 0.0594635343 | 7.07182039e-07 | 0 | 0 |
| tau | 1 | 0.000199986607 | 0 | 1.79e-07 |

## Neutrino PMNS Carrier Response

| source flavor | nu_c population | e_c leakage | PMNS weight error | norm drift |
|---|---:|---:|---:|---:|
| e-flavor | 6.18906188e-06 | 0 | 1.31e-05 | 2.38e-07 |
| mu-flavor | 0.000114214417 | 0 | 1.2e-06 | 0 |
| tau-flavor | 8.54706232e-05 | 0 | 1.47e-06 | 9.54e-07 |
