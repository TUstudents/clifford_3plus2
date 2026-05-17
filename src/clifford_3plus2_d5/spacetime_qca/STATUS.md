# spacetime_qca — Status

**Status**: in progress. Sessions 20-25 complete through finite real-space
BCC stepping, representation-level Higgs/Yukawa audit, position-dependent
background gauge covariance, BCC plaquette holonomy geometry, and a static
Higgs/Yukawa map-layer audit.

This module builds the 3D spatial side of the QCA: a BCC Weyl walk
(Bialynicki-Birula 1994) and its chiral assembly into a 4D Dirac carrier,
with a constant-background tensor lift against `lepton`'s chiral-16 internal
carrier.

Carrier convention: the internal chiral-16 is represented in the exact real
`R^32` basis supplied by `lepton`, with its compatible `J` carrying the
`C^16` interpretation.  Current tensor-lift matrices therefore use
`C^4_Dirac x R^32_internal` and have size `128 x 128` over the Bloch complex
scalars.  A compressed explicit `C^16_internal` basis is not implemented yet.

## What exists

- `bcc_geometry.py` — 8 body-diagonal BCC vectors, Brillouin-zone sample points.
- `pauli.py` — 2×2 Pauli matrices.
- `bcc_weyl.py` — Bialynicki-Birula hop matrices and 2×2 Weyl Bloch symbol.
- `dirac.py` — chiral-basis Dirac assembly.
- `hypercube_control.py` — naive central-difference cubic control Hamiltonian.
- `continuum.py` — first-order expansion utilities.
- `gauge_lift.py` — constant-background internal gauge tensor lift.
- `links.py` — finite-lattice position-dependent internal link fields.
- `plaquette.py` — BCC elementary plaquette shapes and holonomy covariance.
- `mass.py` — Dirac mass-layer and gauge-compatibility helpers.
- `yukawa.py` — representation-level Higgs/Yukawa charge-shift and static
  control audits.
- `lattice.py`, `state.py`, `step.py` — finite periodic real-space BCC step.
- `audit.py` — result payloads for the report.
- `SESSION_20_BCC_DIRAC.md` — Session 20 result report.
- `SESSION_21_MASS_LAYER.md` — Session 21 result report.
- `SESSION_22_REAL_SPACE_STEP.md` — Session 22 result report.
- `SESSION_23_YUKAWA_REPRESENTATION.md` — Session 23 result report.
- `SESSION_24_GAUGE_COVARIANCE.md` — Session 24 result report.
- `SESSION_24B_PLAQUETTE_HOLONOMY.md` — Session 24b result report.
- `SESSION_25_STATIC_YUKAWA.md` — Session 25 result report.
- 63 passing tests.

## Session 20 result

- Bialynicki-Birula hop matrices are pinned to Phys. Rev. D 49, 6920
  (1994), Section II.
- Right Weyl block has first-order Hamiltonian `H_R(k) = sigma . k`.
- Left Weyl block has first-order Hamiltonian `H_L(k) = -sigma . k`.
- Chiral Dirac assembly has `H_D(k) = alpha . k`.
- Hypercube control has 8 literal cubic Brillouin-zone corner doublers.
- BCC cubic-corner gapless representatives are reciprocal-lattice origin
  equivalents for the body-diagonal lattice.
- BCC Bloch-symbol unitarity is sample-checked; the all-momentum unitarity
  claim is inherited from the pinned Bialynicki-Birula source convention.
- The hypercube control is Hamiltonian-form, not a unitary cubic walk; it is
  used as the exact naive-lattice doubling diagnostic.
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
- Full symbolic all-momentum BCC Bloch-symbol unitarity proof.
- Explicit `J`-adapted `C^16` internal basis, if we want code dimensions to
  match the compressed complex carrier rather than the current `R^32` real
  form.
- Plaquette action / Wilson observable normalization.
- Dynamic Higgs-Yukawa layer.
- Dynamical gauge fields.
- Lorentz boost recovery beyond the `alpha . k` continuum precursor.

## Sessions ahead

- Session 20b: full symbolic BCC unitarity and no-doubling hardening.
- Session 26: Wilson observable normalization / plaquette action audit.

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

Expected: 63 tests green.

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
Higgs-like map slot.  The transpose result is a real-linear charge-shift
conjugate, not a complex Hermitian-conjugate Higgs field.  The 4-real-dim
solution space has not yet been decomposed as an `SU(2)_L` Higgs doublet.
This is not yet a dynamical Higgs/Yukawa QCA layer.

## Session 25 result

- The upper Higgs-like map space has real dimension `4` and charge shift
  `(Delta Y, Delta T3_L) = (+1/2, +1/2)`.
- Acting with the non-Cartan `SU(2)_L` generators produces the lower
  component, also real dimension `4`, with
  `(Delta Y, Delta T3_L) = (+1/2, -1/2)`.
- The combined upper/lower map module has real dimension `8`.
- Static real-form controls built as `M + M.T` are real symmetric, and
  `beta x Y_static` is Hermitian.
- The neutral lower-component VEV control preserves color and
  `Q_em = Y + T3_L` while breaking `Y` and `T3_L` separately.
- The default static controls are rank `2` with nullity `30`, so they are
  low-rank background probes, not realistic mass matrices.

Interpretation: the module now has the exact static Higgs-doublet charge
structure and neutral-VEV breaking pattern
`SU(2)_L x U(1)_Y -> U(1)_em`.  It is still not a dynamical Higgs field or a
full Yukawa mass spectrum.

## Session 24 result

- Position-dependent internal link fields are keyed by
  `(target_site, displacement)` in the pull convention
  `target_site <- target_site + displacement`.
- `dirac_step_with_link_field` implements
  `out[x] = sum_h (W_h x U[x <- x+h]) psi[x+h]`.
- Site-local gauge transforms act as
  `psi[x] -> (I_space x G[x]) psi[x]` and
  `U[x <- y] -> G[x] U[x <- y] G[y]^-1`.
- Exact finite-lattice covariance is tested:
  `step(G psi, GUG^-1) = G step(psi, U)`.
- Identity and constant link fields regress to the previous Session 22 paths.
- Pure-gauge links preserve norm on the deterministic plane-wave test state.

Interpretation: this is a background-link gauge-covariant finite QCA rule.
It is not yet a plaquette/curvature construction or a dynamical gauge field.

## Session 24b result

- Elementary BCC plaquettes are four-hop non-backtracking parallelograms
  `(a, b, -a, -b)` built from body-diagonal displacements.
- Modulo cyclic rotation and orientation reversal, there are six canonical
  unoriented elementary plaquette shapes.
- Plaquette holonomy uses the pull convention:
  `H[x0] = U[x0 <- x1] U[x1 <- x2] U[x2 <- x3] U[x3 <- x0]`.
- Holonomy transforms by base-site conjugation:
  `H[x0] -> G[x0] H[x0] G[x0]^-1`.
- Identity and pure-gauge link fields have identity holonomy.

Interpretation: the module now has the BCC loop geometry needed for curvature
observables.  This is not yet a plaquette action or dynamical gauge-field
update rule.
