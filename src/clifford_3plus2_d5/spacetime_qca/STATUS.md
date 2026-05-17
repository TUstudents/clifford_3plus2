# spacetime_qca — Status

**Status**: in progress. Session 20 scaffolding present; full audit not yet
complete.

This module builds the 3D spatial side of the QCA: a BCC Weyl walk
(Bialynicki-Birula 1994) and its chiral assembly into a 4D Dirac carrier,
with the eventual tensor lift against `lepton`'s chiral-16 internal carrier.

## What exists

- `bcc_geometry.py` — 8 body-diagonal BCC vectors, Brillouin-zone sample points.
- `pauli.py` — 2×2 Pauli matrices.
- `bcc_weyl.py` — Bialynicki-Birula hop matrices and 2×2 Weyl Bloch symbol.
- `dirac.py` — chiral-basis Dirac assembly.
- `hypercube_control.py` — naive central-difference cubic control Hamiltonian.
- `continuum.py` — first-order expansion utilities.
- `gauge_lift.py` — tensor lift placeholders.
- `audit.py` — result payloads for the report.
- 20 passing tests.

## What is open

- Complete audit comparing BCC Weyl continuum limit to α·k.
- Verify hypercube control exhibits 8 BZ-corner doublers (Nielsen-Ninomiya
  doubling control).
- Tensor lift to `lepton`'s chiral-16 internal carrier (C^4 ⊗ C^16 = C^64 per site).
- Background SU(N) gauge covariance audit on the BCC walk.

## Sessions ahead

- Session 20: complete BCC Weyl + Dirac assembly + tensor lift + background gauge audit.
- Session 21: mass / Yukawa layer.
- Session 22: real-space finite-lattice instantiation.

See [PLAN.md](PLAN.md) for the detailed Session 20 plan and
[SESSION_20_BCC_DIRAC.md](SESSION_20_BCC_DIRAC.md) for the running report.

## Cross-module dependency

No cross-module dependencies outside the shared `algebra/` utilities.
The tensor lift against `lepton`'s internal carrier is the only future
coupling, and it is currently a placeholder.

## Tests

```bash
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/ -q
```

Expected: 20 tests green.
