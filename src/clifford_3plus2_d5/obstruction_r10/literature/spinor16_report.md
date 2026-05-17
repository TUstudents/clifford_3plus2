# Spinor 16 Report

Phase 6 reconstructs the standard even spinor space from the previously
declared real carrier, complex structure, and `3+2` split:

```text
S^+ = Lambda^even(C^5)
    = Lambda^0(C^5) plus Lambda^2(C^5) plus Lambda^4(C^5).
```

This phase is representation reconstruction only. It does not prove that QCA
rule data force `J`, `P_3/P_2`, or the finite-depth update.

## Provenance Guard

The checker verifies that reconstruction uses prior data:

```text
uses_phase_1_real_carrier: true
uses_phase_1_j: true
uses_phase_1_projectors: true
uses_phase_3_split_candidate: true
introduces_new_complex_structure: false
introduces_new_3plus2_split: false
```

## Exact Hypercharge Table

The hypercharge formula is

```text
Y = -1/3 N_3 + 1/2 N_2.
```

The reconstructed sector table is:

```text
N_3=0, N_2=0, multiplicity=1, Y=0, label=nu^c
N_3=0, N_2=2, multiplicity=1, Y=1, label=e^c
N_3=1, N_2=1, multiplicity=6, Y=1/6, label=Q
N_3=2, N_2=0, multiplicity=3, Y=-2/3, label=u^c
N_3=2, N_2=2, multiplicity=3, Y=1/3, label=d^c
N_3=3, N_2=1, multiplicity=2, Y=-1/2, label=L
```

## Current Status

```text
spinor16_dimension: 16
degree_dimensions: {'0': 1, '2': 10, '4': 5}
hypercharge_check_passed: true
branching_table_check_passed: true
uses_existing_j_and_split: true
introduces_new_complex_structure: false
introduces_new_3plus2_split: false
spinor16_verdict: candidate_only
load_bearing_qca_bridge: false
```

The checker confirms the guarded representation table. Since the prior
geometric data remain candidate-only, the spinor reconstruction also remains
candidate-only.
