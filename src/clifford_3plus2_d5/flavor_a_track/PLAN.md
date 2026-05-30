# flavor_a_track — design (roadmap Track A: A2, A3a, A3b)

## Load-bearing question

> Is the dimensionless SM flavor pattern the spectral data of **one** boundary
> operator `H_Q`, read through four quantum-number projections
> `Sigma_f(z) = V_f^dagger (z - H_Q)^{-1} V_f` — with all sector differences
> carried by the coupling `V_f`?

The A-track is one block answering this in three kill-disciplined phases. It can
KILL cheaply at any phase and never overclaims: every factor is tagged `derived`
(machine-checked) or `free input`, and the final count is reported.

## Phase A2 — universality (necessary conditions) — `universality_audit`

The literal "one `H_Q` reproduces all `Sigma_f`" is not a single confirmable
computation: leptons use the 3-channel K3 residual (no color), quarks a
`1 + (2_BCC + 3_color)` Cl_5 shell (with color) — different representation spaces.
A2 audits the **necessary conditions**, which are cheap and can KILL.

- **U1** (`u1_shared_transfer.py`). Every sector's `epsilon`-power equals
  `rho^power` for the single residual K3 root `rho = sqrt(2)-1`. Controls:
  K2/K4 graph-tracking, independent-epsilon negative control.
  Kill `INDEPENDENT_EPSILON_KILL`.
- **U2** (`u2_color_label_partition.py`). The quark non-color core `1_direct +
  2_BCC` matches the lepton K3 residual `1 + 2`; quarks add exactly 3 color ports;
  conserved labels split 3 color / 3 non-color. Kill `SHELLS_INDEPENDENT_KILL`.
- **U3** (`u3_coupling_catalog.py`). Each field's chiral-16 multiplicity factors as
  `color x weak`; hypercharges match; quarks triplet, leptons singlet. The
  explicit 32x32 projector build is deferred. Kill
  `COUPLINGS_NOT_QUANTUM_NUMBER_DETERMINED_KILL`.
- **U4** (`universality_audit.py`). Aggregate. Pass
  `UNIVERSAL_BOUNDARY_NECESSARY_CONDITIONS_PASS`, else `SECTOR_DEPENDENT_BOUNDARY_KILL`.

## Phase A3a — unified transfer boundary — `unified_boundary_audit`

Decisive reframe: the quark sector has no Schur self-energy (CKM is assembled
combinatorially); only the neutrino core `K_nu = eps^2 P_u + P_b` is a genuine
`Sigma = V^dagger (z - H_Q)^{-1} V`. A3a puts the quark *transfer* sector on the
same Schur footing using the *same* `H_Q` — the quark amplitudes are powers of the
same sterile-chain Weyl factor that drives the neutrino core.

- **A3-1** (`a31_common_hq.py`). One common sterile chain; transfer factor
  `sqrt(2)-1`; normalized Schur self-energy `= K_nu`. Verdict
  `LEPTON_SIGMA_FROM_COMMON_HQ`.
- **A3-2** (`a32_quark_transfer_schur.py`). Quark transfer amplitudes equal
  `(common chain factor)^depth` for depths `{2,4,6}`; finite-shell Green ratios
  converge. Controls: odd depth, scaled chain (K2/golden). Verdict
  `QUARK_TRANSFER_IS_COMMON_CHAIN_SCHUR` / `TRANSFER_NOT_UNIFIABLE_KILL`.
- **A3-3** (`a33_coupling_structure.py`). Remaining quark structure is `V_f` data:
  `sum_A T^A T^A = (4/3)I`, BCC `sqrt(2)`/`1/sqrt(2)`, coin `atan(sqrt(5))`;
  lepton singlet, quark triplet. Verdict `SECTOR_STRUCTURE_IN_COUPLINGS`.
- **A3-4** (`unified_boundary_audit.py`). Aggregate. Pass
  `UNIFIED_TRANSFER_BOUNDARY_PASS`.

## Phase A3b — texture provenance ("derive or count") — `texture_provenance_audit`

The honest ledger of the factors A3a left as inputs — where theory vs numerology
is decided. The ledger is mixed, and that is the result.

- **B1** (`b1_derived_factors.py`). Derived factors, each with a machine-checked
  source: `C_F` via `sum_A T^A T^A == (4/3)I`; coin base via `Gamma_q^2 == 5I` and
  `5 == 2_BCC + 3_color`; BCC `sqrt(2)`/`1/sqrt(2)`; charged-lepton `sqrt(3/2)`;
  V10 leptonic phase word via the holonomy verdict. Verdict
  `DERIVED_FACTORS_CATALOGUED`.
- **B2** (`b2_free_inputs.py`). The 4 free inputs (depth embedding `{0,2,6}` —
  even+additive checked and load-bearing; charged-lepton depth; `r=1` ergodicity
  prior; CP branch). Verdict `FREE_INPUTS_ENUMERATED` /
  `FREE_INPUTS_NOT_ENUMERATED_KILL`.
- **B3** (`b3_parameter_count.py`). `N_free = 4` vs `N_observables = 8` (CKM/PMNS
  angles + phases). Verdict `TEXTURE_PREDICTIVE` if surplus > 0, else
  `TEXTURE_NUMEROLOGY_KILL`.
- **B4** (`texture_provenance_audit.py`). Aggregate (B2 participates). Pass
  `TEXTURE_STRUCTURE_DERIVED_HIERARCHY_INPUT`; record
  `generation_depth_embedding_derived`.

## Acceptance standard

Each phase **can kill but does not over-confirm**. A2 establishes necessary
conditions (cannot confirm universality). A3a unifies the transfer boundary
(does not prove the full flavor pattern). A3b predicts texture *structure*
conditional on the free depth embedding (does not derive the hierarchy). Deriving
`{0,2,6}` is a generation mechanism (`N=3` empirical per the closed
`triality/broken_triality/exceptional/topology` kills), the one terminal input.

## Reuse (no duplicated physics)

External borrowings (`boundary_response`, `lepton`) go through `reuse.py`:
- `transfer` — `epsilon`, `epsilon_fourth`; `residual_graph_transfer` —
  `residual_graph_decaying_factor`; `weyl_sterile` —
  `semi_infinite_weyl_function`, `weyl_product_sterile_normalized_response`;
  `explicit_hq` — `green_transfer_amplitude`, `transfer_probe`.
- `residual_basis` — `k_nu_operator`, `residual_vectors`; `conserved_label_partition`
  — `primitive_conserved_labels`, `conserved_label_partition_is_complete`.
- `quark_boundary_shell` — `quark_boundary_phase_angle`, `quark_gamma_sum`,
  `quark_shell_dimension_breakdown`; `quark_clebsch_factors` — `color_return_*`,
  BCC factors; `quark_transfer_hierarchy` — `CKM_TRANSITIONS`,
  `quark_family_depths`, `quark_transition_amplitude`, `quark_transition_depths`.
- `charged_lepton_leakage` — `charged_lepton_leakage_depth_amplitude`,
  `selected_port_residual_components`; `leptonic_boundary_holonomy` —
  `leptonic_boundary_holonomy_audit_payload`; `local_boundary_fiber` —
  `REMAINING_DECLARED_INPUTS_AFTER_LOCAL_FIBER`.
- `lepton.sm_hypercharge` — `sm_field_multiplicity_table`.

Intra-A-track references (e.g. A3-3 reading U3's `color_split_is_quark_triplet_
lepton_singlet`) are ordinary sibling imports, not via `reuse` — so `reuse`
imports nothing from `flavor_a_track` and there is no import cycle.

## Verification

```bash
uv run pytest src/clifford_3plus2_d5/flavor_a_track/tests -q   # 57 passing
uv run ruff check src/clifford_3plus2_d5/flavor_a_track/        # clean
```
