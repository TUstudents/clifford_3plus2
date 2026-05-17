# spacetime_qca — Status

**Status**: in progress. Session 20 BCC Dirac audit complete at the
Bloch-symbol / sampled-corner level.

This module builds the 3D spatial side of the QCA: a BCC Weyl walk
(Bialynicki-Birula 1994) and its chiral assembly into a 4D Dirac carrier,
with a constant-background tensor lift against `lepton`'s chiral-16 internal
carrier.

## What exists

- `bcc_geometry.py` — 8 body-diagonal BCC vectors, Brillouin-zone sample points.
- `pauli.py` — 2×2 Pauli matrices.
- `bcc_weyl.py` — Bialynicki-Birula hop matrices and 2×2 Weyl Bloch symbol.
- `dirac.py` — chiral-basis Dirac assembly.
- `hypercube_control.py` — naive central-difference cubic control Hamiltonian.
- `continuum.py` — first-order expansion utilities.
- `gauge_lift.py` — constant-background internal gauge tensor lift.
- `audit.py` — result payloads for the report.
- `SESSION_20_BCC_DIRAC.md` — Session 20 result report.
- 20 passing tests.

## Session 20 result

- Bialynicki-Birula hop matrices are pinned to Phys. Rev. D 49, 6920
  (1994), Section II.
- Right Weyl block has first-order Hamiltonian `H_R(k) = sigma . k`.
- Left Weyl block has first-order Hamiltonian `H_L(k) = -sigma . k`.
- Chiral Dirac assembly has `H_D(k) = alpha . k`.
- Hypercube control has 8 literal cubic Brillouin-zone corner doublers.
- BCC cubic-corner gapless representatives are reciprocal-lattice origin
  equivalents for the body-diagonal lattice.
- Constant background gauge lift gives
  `H_eff = alpha.k x I_internal + I_spacetime x iA`.
- The gauge lift is tested with a real 32×32 Pati-Salam `SU(2)_L` generator
  from `lepton`.

## What is open

- Full fundamental-BCC-Brillouin-zone no-doubling proof.
- Finite real-space BCC `step(state, links)` implementation.
- Position-dependent gauge links and site-local gauge covariance.
- Mass / Yukawa layer.
- Dynamical gauge fields.
- Lorentz boost recovery beyond the `alpha . k` continuum precursor.

## Sessions ahead

- Session 21: mass / Yukawa layer.
- Session 22: real-space finite-lattice instantiation.

See [PLAN.md](PLAN.md) for the detailed Session 20 plan and
[SESSION_20_BCC_DIRAC.md](SESSION_20_BCC_DIRAC.md) for the running report.

## Cross-module dependency

The Session 20 gauge-lift test imports one real Pati-Salam background
generator from `lepton`.  The spacetime pieces remain factored, and future
coupling should continue through explicit tensor-lift interfaces.

## Tests

```bash
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/ -q
```

Expected: 20 tests green.
