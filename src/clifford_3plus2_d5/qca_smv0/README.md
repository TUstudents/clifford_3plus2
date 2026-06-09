# QCA_SMv0

Simulator sidecar for focused Standard-Model QCA experiments.

This sidecar is the place to build a smaller, recent-sidecar-informed simulator
without changing the shared `sim` package or the broader `spacetime_qca`
production prototype.

## Scope

`QCA_SMv0` should be used for:

- focused QCA simulator experiments tied to the recent boundary/flavor sidecars;
- small, JAX-native kernels that can later graduate into `spacetime_qca`;
- explicit session notes, local scripts, and focused tests.

It should not be used for:

- broad regression bookkeeping;
- importing theory claims before a session states them;
- replacing the shared `sim` infrastructure.

## Current State

Stage 4 implements the free BCC Weyl/Dirac bulk walk, static
Standard-Model gauge-background transport, pure dynamic SM gauge fields, and a
site-local Higgs/Yukawa collision:

```text
qca_smv0/
  README.md
  PLAN.md
  STATUS.md
  __init__.py
  bulk_bcc.py
  sm_gauge.py
  sm_dynamics.py
  sm_higgs.py
  scripts/
    session_01_bare_bcc_walk.py
    session_02_static_sm_gauge_background.py
    session_03_dynamic_sm_gauge.py
    session_04_higgs_yukawa_collision.py
  tests/
    test_bulk_bcc.py
    test_sm_gauge.py
    test_sm_dynamics.py
    test_sm_higgs.py
```

The implemented Weyl kernel is a two-component periodic BCC bulk walk:

```text
psi_next[x] = sum_h A_h psi[x + h]
```

with eight BCC source offsets `h in {+-1}^3` and

```text
A_h = P_x^{h_x} P_y^{h_y} P_z^{h_z}.
```

With this pull convention the Bloch symbol is exactly

```text
U(k)=S_x(k_x) S_y(k_y) S_z(k_z),
S_j(k_j)=P_j^+ exp(i k_j)+P_j^- exp(-i k_j).
```

Verdict:

```text
QCA_SMV0_STAGE1_FREE_BCC_PASS
```

Stage 1 also includes:

- exact hop-completeness checks for Weyl and Dirac hops;
- periodic norm-conservation tests;
- split-product versus hop-sum Bloch-symbol consistency;
- small-momentum Weyl/Dirac dispersion checks;
- directional anisotropy scaling under momentum halving;
- massless Dirac assembly from opposite-chirality Weyl blocks;
- JIT compatibility checks.

Stage 2 adds a static background gauge layer:

- a local 32-component internal SM carrier: one left-handed chiral-16 label set
  duplicated across the four-component Dirac spin block;
- local anti-Hermitian `SU(3)_c x SU(2)_L x U(1)_Y` generators;
- finite BCC edge links with shape `(nx, ny, nz, 8, 32, 32)`;
- a gauge-covariant Dirac BCC step through those static links;
- site-local gauge transformations of states and links;
- selected BCC plaquette Wilson traces;
- weak-link linearization as the simulator covariant-derivative check;
- JIT compatibility for the gauged transport step.

Stage 2 verdict:

```text
QCA_SMV0_STAGE2_STATIC_SM_GAUGE_PASS
```

Stage 3 adds a pure dynamic gauge layer:

- real link momenta with shape `(nx, ny, nz, 8, 12)` in the SM generator basis;
- projection between algebra matrices and SM generator coordinates;
- target-site adjoint momentum gauge transforms;
- Wilson-force finite differences under left link perturbations;
- Hamiltonian density `K(P) + beta S_W(U)`;
- reversible pure-gauge leapfrog update;
- electric divergence / Gauss diagnostic for the pure gauge field;
- pure-gauge zero-momentum Gauss-preservation audit;
- weak-field plaquette field-strength linearization;
- Wilson action matching to the weak-field Yang-Mills density;
- no-backreaction fermion/gauge wrapper: evolve links and momenta, then
  transport spectator fermions through the updated links;
- JIT compatibility for the leapfrog update.

Stage 3 verdict:

```text
QCA_SMV0_STAGE3_DYNAMIC_SM_GAUGE_PASS
```

Stage 4 adds a local Higgs/Yukawa collision:

- local Higgs doublet fields with shape `(nx, ny, nz, 2)`;
- the constant unitary-gauge Higgs helper `H=(0,v/sqrt(2))`;
- `H_tilde=i sigma_2 H^*`;
- a Hermitian one-generation Yukawa matrix on the local SM carrier;
- unitary-gauge doors: `H_tilde` opens up/neutrino couplings while `H` opens
  down/electron couplings;
- exact site-local rotation `exp(-i step_size beta Y(H))`;
- zero-step and zero-Higgs identity controls;
- norm preservation, chirality-flip, massive-dispersion, and JIT audits.

Stage 4 verdict:

```text
QCA_SMV0_STAGE4_HIGGS_YUKAWA_PASS
```

Matter backreaction, dynamic Higgs-field evolution, boundary rules, flavor
registers, FN recirculation, and center-holonomy CP are not implemented yet.

## Reuse Boundary

Allowed upstream infrastructure:

- `clifford_3plus2_d5.sim` for generic JAX state, links, scan runners, profiling,
  and persistence;
- `clifford_3plus2_d5.qca_smv0` local modules for the Stage 1/2 kernels.

This sidecar should not import from `spacetime_qca`, `lepton`, `cusp`, or other
theory sidecars.  If later work needs an older QCA kernel, copy/adapt the code
locally so this simulator can change orderings and performance boundaries
without changing the older sidecars.

Future sessions should keep imports narrow and run only the tests for the
current session.

## Run

```bash
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_01_bare_bcc_walk
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_02_static_sm_gauge_background
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_03_dynamic_sm_gauge
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_04_higgs_yukawa_collision
uv run pytest src/clifford_3plus2_d5/qca_smv0/tests -q
```
