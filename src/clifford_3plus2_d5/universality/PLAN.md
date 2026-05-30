# universality — design (roadmap gate A2)

## Load-bearing question

> Do flavor differences between the sectors (nu, e, u, d) come entirely from the
> couplings `V_f` — which SM quantum numbers connect to the boundary — rather
> than from different boundary operators `H_Q`?

If yes, the whole dimensionless flavor pattern is the spectral data of **one**
`H_Q` read through four projections `Sigma_f(z) = V_f^dagger (z - H_Q)^{-1} V_f`.

## Why this is not a single confirmable computation

The current code does not use one `H_Q`: leptons use the 3-channel K3 residual
`(u, a, b)` (no color); quarks use a `1 + (2_BCC + 3_color)` Cl_5 shell (4x4
spinor rep, with color). These live in **different representation spaces**, so a
literal "one operator reproduces all `Sigma_f`" check requires embedding both in
lepton's chiral-16 carrier — the full flavor program (roadmap A3). This sidecar
instead audits the **necessary conditions**, which are cheap and can KILL.

## Gates

- **U1 — shared transfer invariant** (`u1_shared_transfer.py`). Every sector's
  `epsilon`-power equals `rho^power` for the single residual K3 graph root
  `rho = residual_graph_decaying_factor(3) = sqrt(2)-1`: nu ratio `rho^4`,
  charged-lepton leakage `rho^2`, quark transitions `rho^2, rho^4, rho^6`.
  Controls: graph-tracking (K2/K4 roots differ from K3) and a negative control
  (a K4-rooted sector is detected as non-universal). Kill: `INDEPENDENT_EPSILON_KILL`.

- **U2 — sector difference = color label** (`u2_color_label_partition.py`). The
  quark shell's non-color core `1_direct + 2_BCC` matches the lepton K3 residual
  `1 (trivial u) + 2 (doublet a,b)`; the quark shell is that core plus exactly 3
  color ports; the conserved-label partition `(parity, bcc_index, color_index)`
  splits 3 color / 3 non-color. Kill: `SHELLS_INDEPENDENT_KILL`.

- **U3 — coupling catalog** (`u3_coupling_catalog.py`). Each one-generation
  field's chiral-16 multiplicity factors exactly as `color x weak`
  (Q=3x2, u^c=3x1, d^c=3x1, L=1x2, e^c=1x1, nu^c=1x1), hypercharges match
  lepton's field table, quarks color-triplet, leptons color-singlet — so `V_f`
  is the SM quantum-number projection. The explicit 32x32 chiral-16 projector
  build is deferred to A3. Kill: `COUPLINGS_NOT_QUANTUM_NUMBER_DETERMINED_KILL`.

- **U4 — combined verdict** (`universality_audit.py`). Aggregates U1-U3.
  Pass: `UNIVERSAL_BOUNDARY_NECESSARY_CONDITIONS_PASS` with the A3 deferral
  recorded. Kill: `SECTOR_DEPENDENT_BOUNDARY_KILL`.

## Acceptance standard

The gate **can kill but cannot confirm** universality. A pass means the necessary
conditions hold and the unified-`H_Q` construction (A3) is warranted; it is not a
claim that universality is proven.

## Reuse (no duplicated physics; all via `reuse.py`)

- `boundary_response.transfer` — `epsilon`, `epsilon_fourth`.
- `boundary_response.residual_graph_transfer` — `residual_graph_decaying_factor`.
- `boundary_response.charged_lepton_leakage` — `charged_lepton_leakage_depth_amplitude`.
- `boundary_response.quark_transfer_hierarchy` — `quark_transition_amplitude`.
- `boundary_response.quark_boundary_shell` — `quark_primitive_channels`, `quark_shell_dimension_breakdown`.
- `boundary_response.conserved_label_partition` — `primitive_conserved_labels`, `conserved_label_partition_is_complete`.
- `boundary_response.residual_basis` — `residual_vectors`.
- `lepton.sm_hypercharge` — `sm_field_multiplicity_table`.
