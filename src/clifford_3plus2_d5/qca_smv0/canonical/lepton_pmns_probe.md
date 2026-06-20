# QCA_SMv0 Lepton PMNS Probe

This canonical lepton-sector probe tests mixing rather than only diagonal
transfer.  The neutrino door receives `Y_nu = U_PMNS diag(0, eta, 1)`, charged
lepton Yukawas are zero, and the target-family populations are compared with
the PMNS-weighted transfer distribution.  A type-I Schur/seesaw backend is
constructed in the same payload and checked against the same low-energy
spectrum.

## Inputs

- scale label: `benchmark`
- lattice shape: `(1, 1, 1)`
- steps: `1`
- neutrino Yukawas: `(0.0, 0.17157287525381, 1.0)`
- PMNS angles: `(0.5836381018669037, 0.8587019919812102, 0.14957417646704585, 3.4033920413889422)`
- source: `L(weak=0)`
- target: `nu_c`

## Neutrino And Schur Metadata

- hierarchy: `normal`
- lightest massless: `True`
- Delta m2 ratio: `0.0294372515229`
- expected epsilon^4: `0.0294372515229`
- Schur recovered masses: `(6.207423020043734e-09, 0.17157292366027832, 1.0)`
- Schur spectrum residual: `4.47e-08`

## Physics Tests

- overall passed: `True`

| check | passed |
|---|---:|
| normal_ordering | True |
| massless_lightest | True |
| ratio_check | True |
| pmns_unitarity | True |
| pmns_mixing | True |
| schur_spectrum | True |
| norm_conservation | True |
| carrier_transfer | True |
| no_e_c_leakage | True |
| production_contract | True |

## Flavor Response

| source flavor | nu_c population | e_c leakage | PMNS weight error | norm drift |
|---|---:|---:|---:|---:|
| e-flavor | 6.18906142e-06 | 0 | 1.3e-05 | 3.58e-07 |
| mu-flavor | 0.000114214417 | 0 | 1.2e-06 | 1.19e-07 |
| tau-flavor | 8.54706232e-05 | 0 | 1.47e-06 | 7.15e-07 |

## Interpretation

The probe verifies that the lepton production door can consume a non-diagonal
PMNS neutrino input, produce mixed `nu_c` target-family populations, conserve
norm, avoid charged-lepton leakage, and represent the same `(0, eta, 1)`
low-energy neutrino spectrum through an explicit heavy-sterile Schur backend.
