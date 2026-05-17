# spacetime_qca — Status

**Status**: in progress. Sessions 20-23 complete through finite real-space
BCC stepping and representation-level Higgs/Yukawa audit.

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
- `mass.py` — Dirac mass-layer and gauge-compatibility helpers.
- `yukawa.py` — representation-level Higgs/Yukawa charge-shift audit.
- `lattice.py`, `state.py`, `step.py` — finite periodic real-space BCC step.
- `audit.py` — result payloads for the report.
- `SESSION_20_BCC_DIRAC.md` — Session 20 result report.
- `SESSION_21_MASS_LAYER.md` — Session 21 result report.
- `SESSION_22_REAL_SPACE_STEP.md` — Session 22 result report.
- `SESSION_23_YUKAWA_REPRESENTATION.md` — Session 23 result report.
- 45 passing tests.

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

## Session 22 result

- Periodic 3D lattice wrapper implemented.
- Pull-form Weyl step implemented:
  `out[x] = sum_h W(h) psi[x+h]`.
- Dirac step implemented as blockwise right/left Weyl steps.
- Plane-wave states match the BCC Weyl/Dirac Bloch symbols exactly.
- Weyl and Dirac steps preserve norm on deterministic finite-lattice
  plane-wave states.
- Constant identity-link tensor step matches ungauged Dirac stepping.

## What is open

- Full fundamental-BCC-Brillouin-zone no-doubling proof.
- Position-dependent gauge links and site-local gauge covariance.
- Higgs / Yukawa representation audit.
- Dynamical gauge fields.
- Lorentz boost recovery beyond the `alpha . k` continuum precursor.

## Sessions ahead

- Session 24: position-dependent links and finite-lattice gauge covariance, or
  Hermitian/dynamical Higgs-Yukawa layer.

See [PLAN.md](PLAN.md) for the detailed Session 20 plan and
[SESSION_20_BCC_DIRAC.md](SESSION_20_BCC_DIRAC.md) for the running report.

## Cross-module dependency

The Session 20 gauge-lift test and Session 21 mass-compatibility tests import
real Pati-Salam / SM generators from `lepton`.  The spacetime pieces remain
factored, and future coupling should continue through explicit tensor-lift
interfaces.

## Tests

```bash
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/ -q
```

Expected: 45 tests green.

## Session 21 result

- `beta = gamma^0` satisfies `beta^2 = I`.
- `beta` anticommutes with all three `alpha_i`.
- Scalar mass control obeys
  `H_m(k)^2 = (k_x^2 + k_y^2 + k_z^2 + m^2) I`.
- At `k = 0`, the spectrum is `+m` and `-m` with the expected
  Dirac/internal multiplicities.
- Scalar internal mass commutes with Pati-Salam and SM gauge generators from
  `lepton`.
- A non-scalar projector mass breaks at least one SM generator, verifying the
  compatibility test is sensitive.

Interpretation: scalar mass validates the Dirac mass slot
`beta x M_internal`, but it is not an SM Yukawa/Higgs mass spectrum.

## Session 23 result

- `beta = gamma^0` is off-diagonal between spacetime chiralities.
- The existing lepton `(Y, T3_L)` joint table is used as the SM charge
  decomposition.
- Universal scalar mass preserves all SM sectors but is not Higgs-like.
- A non-scalar projector control breaks at least one SM sector.
- An exact color-singlet internal map exists with
  `Delta Y = +1/2` and `Delta T3_L = +1/2`.
- Its transpose gives the conjugate component with
  `Delta Y = -1/2` and `Delta T3_L = -1/2`.
- The Higgs-like charge-shift solution space has real dimension `4`.

Interpretation: the chiral-16 internal carrier contains a representation-level
Higgs-like map slot.  This is not yet a dynamical Higgs/Yukawa QCA layer.
