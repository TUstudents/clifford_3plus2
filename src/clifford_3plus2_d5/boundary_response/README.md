# boundary_response

Boundary-response sidecar for the BCC-QCA flavor and vacuum-selector program.

The sidecar audits one idea:

```text
observed flavor data = Schur / Green-function response of unresolved boundary modes
```

It is intentionally theorem-gated.  Each proposed mechanism either produces a
named pass verdict or is left as a declared input.

## Current result

The selector sector is closed through V43 modulo one explicit intermediate
axiom:

```text
positive_quartic_backreaction_bounds_selector_radius
```

The proven chain is:

```text
single-Weyl BB walk
  -> real helicity-locked A2u selector term
  -> accepted tetrahedral branch selected
  -> free BB even energy destabilizes r = 0
  -> positive quartic backreaction bounds the radius
  -> finite nonzero selector vacuum
```

The positive quartic coefficient is not microscopically derived in this
sidecar.  It is named as an intermediate axiom so the result can be polished
without hiding the remaining physical input.

## Core results

| Gate | Result |
|---|---|
| V1 | unbroken `K_3` tail cannot produce `K_nu`; framing is required |
| V5-V6 | product sterile half-line derives `epsilon^2 P_u + P_b` exactly |
| V7-V10 | charged-lepton leakage and leptonic phase word gates pass |
| V11-V15 | quark primitive shell and flat-coin phase are reduced to rigidity/entropy gates |
| V20-V24 | primitive partition and degeneracy assumptions are reduced to conservation/microcanonical statements |
| V35-V37 | BB filled-band eigenphase contains the real chiral selector structure |
| V38 | free BB walk selects branch but does not radially stabilize alone |
| V39-V42 | Higgs/backreaction radial closure and analytic radial theorem pass |
| V43 | selector sidecar closed with one explicit quartic axiom |

## What is not claimed

- No microscopic derivation of the positive quartic coefficient.
- No full dynamical gauge/Higgs simulation.
- PMNS and CKM textures remain gated by their boundary-shell assumptions.
- The full Standard Model dynamics are not derived here.

## Main docs

- `STATUS.md` is the chronological session/status record.
- `PLAN.md` is the historical implementation plan and verdict standard.
- `parameter_ledger.md` lists exact values, operators, and remaining inputs.
- `SESSION_BOUNDARY_CORE.md` records the original boundary-core session.

## Focused tests

For the closed selector stack:

```bash
uv run pytest \
  src/clifford_3plus2_d5/boundary_response/tests/test_vacuum_selector_bb_induced_breaking.py \
  src/clifford_3plus2_d5/boundary_response/tests/test_vacuum_selector_radial_theorem.py \
  src/clifford_3plus2_d5/boundary_response/tests/test_vacuum_selector_closure.py \
  -q
```

Full sidecar suite:

```bash
uv run pytest src/clifford_3plus2_d5/boundary_response/tests -q
```
