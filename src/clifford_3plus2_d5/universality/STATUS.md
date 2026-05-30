# universality — Status

**Status**: necessary conditions audited.
Verdict: **UNIVERSAL_BOUNDARY_NECESSARY_CONDITIONS_PASS**.

Roadmap gate A2. The strong claim under test: flavor differences between sectors
(nu, e, u, d) come entirely from the couplings `V_f` (which SM quantum numbers
connect to the boundary), not from different boundaries `H_Q` — so the whole
dimensionless flavor pattern is the spectral data of one `H_Q` read through four
projections `Sigma_f(z) = V_f^dagger (z - H_Q)^{-1} V_f`.

**Honest scope:** this sidecar tests the *necessary conditions* and can return a
clean KILL. It does **not** confirm universality. The full numerical
reproduction of every `Sigma_f` from one `H_Q` on lepton's chiral-16 carrier is
the flavor program (roadmap A3) and is recorded as the one remaining declared
input: `unified_H_Q_on_chiral16_reproduces_all_Sigma_f`.

## Gate status

| Gate | Status | Verdict |
|---|---|---|
| U1 — shared transfer invariant | done | **SHARED_TRANSFER_INVARIANT** — all sectors are powers of the single residual K3 root `sqrt(2)-1`; K2/K4 graph-tracking + independent-epsilon negative controls pass |
| U2 — sector difference = color label | done | **SECTOR_DIFFERENCE_IS_COLOR_LABEL** — quark & lepton shells share the `1+2` non-color core; quarks add exactly 3 color ports; conserved labels split 3 color / 3 non-color |
| U3 — coupling catalog | done | **COUPLINGS_ARE_QUANTUM_NUMBER_PROJECTIONS** — each field's chiral-16 multiplicity factors as `color x weak`; quarks color-triplet, leptons color-singlet |
| U4 — combined verdict | done | **UNIVERSAL_BOUNDARY_NECESSARY_CONDITIONS_PASS**; full `Sigma_f` reproduction deferred to A3 |

## What this does and does not establish

- **Does:** the transfer invariant is genuinely shared (not three coincidences);
  the quark/lepton boundaries differ by *exactly* the color quantum number; the
  per-sector couplings are SM quantum-number projections. These are *necessary*
  for one-boundary universality, and any failure would KILL the claim cheaply.
- **Does not:** prove that one explicit `H_Q` on the chiral-16 carrier reproduces
  all four `Sigma_f` numerically (the quark Cl_5 shell and lepton K3 residual
  currently live in different representation spaces; unifying them is A3).

## Cross-module dependency

First sidecar bridging `boundary_response` (transfer invariant, sector shells,
conserved labels) and `lepton` (chiral-16 SM quantum-number content). All imports
go through `reuse.py`.

## Test command

```bash
uv run pytest src/clifford_3plus2_d5/universality/tests -q
```

Expected: 18 passing. A decisive negative control
(`test_gate_can_actually_fail_decisive_negative_control`) confirms the gate is
sensitive: the real sectors match the K3 root but not the K4 root, so a genuinely
sector-dependent `epsilon` would force `INDEPENDENT_EPSILON_KILL`.
