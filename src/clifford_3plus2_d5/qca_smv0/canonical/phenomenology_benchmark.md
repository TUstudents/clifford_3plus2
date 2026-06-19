# QCA_SMv0 Canonical Phenomenology Benchmark

This is the standard constructive benchmark for the QCA_SMv0 front door. It is
a calibrated simulator run, not a derivation of the FN charges, lambda, or
order-one coefficients.

## Inputs

- scale label: `benchmark`
- rollout steps: `4`
- lambda: `0.22501`
- charges Q/U/D: `(3, 2, 0)` / `(5, 2, 0)` / `(1, 0, 0)`
- collision mode: `effective_yukawa`
- stream mode: `split_axis`
- Yukawa strategy: `fast`
- lepton mode: `zero`

## Calibration Verdict

- selected center-CP texture: `wilson_flux_rule`
- status: `pass`
- passed: `True`
- objective: `0.00137818`
- up/down mass log RMS: `0.00194976` / `0.00222662`
- CKM absolute residual: `0.00195287`
- Jarlskog candidate/target: `3.11831e-05` / `3.117e-05`
- Jarlskog relative residual: `0.000418303`

## Physics Tests

- overall passed: `True`

| check | passed |
|---|---:|
| center_cp_quark_fit | True |
| order_one_coefficients | True |
| norm_conservation | True |
| quark_carrier_transfer | True |
| no_hidden_fn_path_memory | True |
| memory_accounting | True |
| production_contract | True |

## Coefficient Diagnostics

- magnitude min/max/mean: `0.295292` / `3.96102` / `1.11843`
- coefficient residual: `3.96102`
- phase residual: `2.17651`

### Up Magnitudes

| row | 1 | 2 | 3 |
|---:|---:|---:|---:|
| 1 | 1.27265 | 0.78576 | 1.85219 |
| 2 | 0.78576 | 1.27265 | 0.458693 |
| 3 | 3.96102 | 3.4183 | 0.987706 |

### Down Magnitudes

| row | 1 | 2 | 3 |
|---:|---:|---:|---:|
| 1 | 0.35199 | 0.316821 | 0.601507 |
| 2 | 0.574091 | 0.565377 | 0.295292 |
| 3 | 1.40412 | 0.883492 | 0.344245 |

### Up Center Powers

| row | 1 | 2 | 3 |
|---:|---:|---:|---:|
| 1 | 2 | 1 | 1 |
| 2 | 1 | 0 | 0 |
| 3 | 0 | 2 | 0 |

### Down Center Powers

| row | 1 | 2 | 3 |
|---:|---:|---:|---:|
| 1 | 1 | 1 | 1 |
| 2 | 2 | 0 | 0 |
| 3 | 1 | 2 | 0 |

## Field Rollout

- entrypoint: `calibrated_production_api`
- norm drift: `1.07288e-06`
- extended norm drift: `1.07288e-06`
- max density initial/final: `1.32812512` / `1.32812595`
- used Higgs/FN collision: `True`
- used exact FN dilation: `False`

| sector | initial | final | drift |
|---|---:|---:|---:|
| Q | 1.26562512 | 1.26536 | -0.000265121 |
| u_c | 0.265625 | 0.265891194 | 0.000266194 |
| d_c | 0 | 0 | 0 |
| L | 0 | 0 | 0 |
| e_c | 0 | 0 | 0 |
| nu_c | 0.26562503 | 0.26562503 | 0 |

## Memory And Contract

- visible complex elements: `768`
- hidden FN aux complex elements: `0`
- runtime array bytes: `7296`
- runtime array bytes per site: `3648`
- production contract: `gauge_links_present=False, higgs_field_present=False, lean_effective_yukawa=True, raw_readout_arrays_present=False, raw_yukawa_arrays_present=False, state_only=True, structured_collision_cache_present=True, uses_production_api=True`

## Interpretation

The benchmark demonstrates the current constructive simulator path: physical
quark inputs calibrate a center-CP FN texture, the compressed effective-Yukawa
collision runs on the `4 x 32 x 3` field fibre, and the output exposes both
coefficient diagnostics and field observables without carrying exact hidden FN
path memory in production.
