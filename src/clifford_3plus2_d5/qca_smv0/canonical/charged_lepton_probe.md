# QCA_SMv0 Charged-Lepton Carrier Probe

This is the first canonical lepton-sector production probe.  It is a simulator
door test: diagonal charged-lepton Yukawas are supplied as inputs, neutrino
Yukawas are set to zero, and clean `L(weak=1)` carrier states are evolved
through the compressed production collision.

## Inputs

- scale label: `benchmark`
- lattice shape: `(1, 1, 1)`
- steps: `1`
- charged-lepton Yukawas: `(0.1, 0.2, 0.5)`
- neutrino Yukawas: `(0.0, 0.0, 0.0)`
- source: `L(weak=1)`
- target: `e_c`

## Physics Tests

- overall passed: `True`

| check | passed |
|---|---:|
| norm_conservation | True |
| correct_carrier_direction | True |
| no_nu_c_leakage | True |
| monotonic_family_response | True |
| production_contract | True |

## Family Response

| family | input Yukawa | e_c population | nu_c leakage | norm drift |
|---|---:|---:|---:|---:|
| e | 0.1 | 4.99991656e-05 | 0 | 0 |
| mu | 0.2 | 0.00019998668 | 0 | 1.79e-07 |
| tau | 0.5 | 0.00124947913 | 0 | 1.19e-07 |

## Interpretation

The probe verifies the current lepton carrier contract on the production fibre:
`L(weak=1)` routes into `e_c`, zero neutrino Yukawas prevent `nu_c` leakage,
the response is monotonic across the supplied diagonal Yukawas, and the
production contract remains lean with no exact hidden FN path memory.
