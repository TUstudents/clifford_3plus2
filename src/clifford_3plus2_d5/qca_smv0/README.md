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

Stage 1 implements the free BCC Weyl/Dirac bulk walk:

```text
qca_smv0/
  README.md
  PLAN.md
  STATUS.md
  __init__.py
  bulk_bcc.py
  scripts/
    session_01_bare_bcc_walk.py
  tests/
    test_bulk_bcc.py
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

No boundary, gauge links, dynamic gauge fields, Higgs field, Standard Model
carrier beyond the free Dirac spin block, flavor register, or recirculation
module is implemented yet.

## Reuse Boundary

Allowed upstream infrastructure:

- `clifford_3plus2_d5.sim` for generic JAX state, links, scan runners, profiling,
  and persistence;
- `clifford_3plus2_d5.spacetime_qca` for existing BCC/BB/QCA kernels when a
  session explicitly imports them.

Future sessions should keep imports narrow and run only the tests for the
current session.

## Run

```bash
uv run python -m clifford_3plus2_d5.qca_smv0.scripts.session_01_bare_bcc_walk
uv run pytest src/clifford_3plus2_d5/qca_smv0/tests -q
```
