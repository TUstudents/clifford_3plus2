# texture_provenance — design (roadmap gate A3b)

## Load-bearing question

> Of the CKM/PMNS texture factors that A3a left as inputs, which are **derived**
> from the chiral-16 quantum numbers / BCC geometry, and which are **free** —
> and is the resulting count predictive or numerology?

This is the "derive or count" step (roadmap A3 core): the place where theory vs
numerology is decided.

## Finding from exploration (the ledger is mixed — and that is the honest result)

The texture's *group-theoretic / geometric* factors are derived; the *generation
hierarchy* is free input.

- **Derived** (functions of the chiral-16 quantum numbers / BCC geometry):
  `C_F = sum_A T^A T^A = 4/3`; the coin base `sqrt(5)` from `Gamma_q^2 = 5I`
  with `5 = 2_BCC + 3_color`; BCC Clebsches `sqrt(2)`, `1/sqrt(2)`; the
  charged-lepton `sqrt(3/2)` from the residual port; the V10 leptonic phase word;
  the quark CP phase `atan(sqrt(5))`.
- **Free input** (4): the quark depth embedding `{0,2,6}` (fit to CKM hierarchy,
  even+additive checked); the charged-lepton two-step depth; the flat-coin `r=1`
  ergodicity prior; the CP-phase branch.

So A3b's realistic, honest outcome is a precise *parameter ledger*: the framework
predicts texture **structure** (CP phases, PMNS angle structure, group-theory
factors) while the **hierarchy** rides on the free depth embedding. The gate
reports `N_free` vs `N_observables` and **kills** if `N_free >= N_observables`.

## Gates

- **B1** (`b1_derived_factors.py`). For each derived factor store
  `(value, source, check_passes)` and verify: `C_F` via the SU(3) contraction
  `== (4/3)I`; coin base via `Gamma_q^2 == 5I` and `5 == 2_BCC + 3_color`; BCC
  `sqrt(2)`/`1/sqrt(2)`; charged-lepton `sqrt(3/2)` from the residual port; the
  V10 leptonic phase word via the holonomy verdict
  `LEPTONIC_PHASE_WORD_DERIVED_PASS`. Verdict `DERIVED_FACTORS_CATALOGUED`.

- **B2** (`b2_free_inputs.py`). Enumerate the 4 free inputs with sources and a
  one-line "what is assigned"; check the depth embedding is even+additive.
  Verdict `FREE_INPUTS_ENUMERATED`.

- **B3** (`b3_parameter_count.py`). Count `N_free = 4` against the flavor
  observables (CKM 3 angles + 1 phase, PMNS 3 angles + 1 phase = 8). Report
  `surplus = N_observables - N_free` and separate the genuine derived predictions.
  Verdict `TEXTURE_PREDICTIVE` if `surplus > 0`, else `TEXTURE_NUMEROLOGY_KILL`.

- **B4** (`texture_provenance_audit.py`). Aggregate. Pass
  `TEXTURE_STRUCTURE_DERIVED_HIERARCHY_INPUT`; record the remaining input
  `generation_depth_embedding_derived`.

## Acceptance standard

The gate **can kill but does not prove** the full flavor pattern. A pass means the
group-theoretic / geometric factors and CP phases are derived, the hierarchy rides
on the free depth embedding, and `N_free < N_observables` (not numerology).
Deriving the depth embedding `{0,2,6}` is a generation mechanism (open; `N=3`
empirical per the closed `triality/broken_triality/exceptional/topology` kills)
and is recorded as the remaining input, **not attempted** here.

## Reuse (all via `reuse.py`)

- `boundary_response.quark_clebsch_factors` — `color_return_contraction`, `color_return_factor`, `bcc_symmetric_two_path_factor`, `bcc_antisymmetric_projection_factor`.
- `boundary_response.quark_boundary_shell` — `quark_gamma_sum`, `quark_shell_dimension_breakdown`, `quark_boundary_phase_angle`.
- `boundary_response.quark_transfer_hierarchy` — `quark_family_depths`, `quark_transition_depths`.
- `boundary_response.charged_lepton_leakage` — `selected_port_residual_components`, `charged_lepton_rotation_sine`.
- `boundary_response.leptonic_boundary_holonomy` — `leptonic_boundary_holonomy_audit_payload` (V10 derived word).
- `boundary_response.local_boundary_fiber` — `REMAINING_DECLARED_INPUTS_AFTER_LOCAL_FIBER` (ergodicity prior).
