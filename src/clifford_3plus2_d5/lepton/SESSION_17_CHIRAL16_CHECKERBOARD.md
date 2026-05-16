# Session 17 - Cl(0,10) Chiral-16 Checkerboard

## Goal

Lift the Session 16 Cl(0,8) checkerboard walk from the chiral octonion
block `R^8_+` to the Cl(0,10) chiral-16 carrier. The test is deliberately
1D and background-gauge only: verify that the existing checkerboard
continuum result survives on the larger carrier, and audit whether the
Cl(0,2) factor supplies electroweak `SU(2)_L` by itself.

## What Was Built

- `clifford_chiral16.py`
  - Constructs exact real `Cl(0,10)` gamma matrices on a 64-dimensional
    real module.
  - Builds a real chirality involution whose projectors have ranks
    `32 + 32`.
  - Extracts the positive chiral block, real dimension 32, corresponding to
    complex dimension 16 after choosing a commuting Cl(0,2) complex
    structure.
  - Lifts the Session 15 octonionic `su(3)` basis into the chiral-16 block.

- `checkerboard_chiral16.py`
  - Reuses the Session 16 1D checkerboard walk with internal dimension 32.
  - Verifies the massless continuum Hamiltonian
    `H(k) = k sigma_z tensor I_32`.
  - Adds a first-order background `SU(3)` link on the chiral-16 internal
    sector.
  - Checks exact finite gauge covariance under the lifted rigid `SU(3)`
    automorphism.

## Exact Results

- `Cl(0,10)` relations pass for 10 exact 64x64 real gamma matrices.
- The raw 10-volume squares to `-I`; multiplying by the commuting Cl(0,2)
  complex structure gives a chirality involution.
- Chiral projectors have ranks `(32, 32)`.
- Odd gamma words swap chirality; even words preserve it.
- The Cl(0,2) factor supplies two chirality-preserving complex structures
  on the selected chiral block.
- The chosen commuting `J` squares to `-I`, is orthogonal, and commutes with
  the lifted `su(3)` generators.
- Lifted `su(3)` has span dimension 8 on the chiral-16 carrier.
- The massless checkerboard Hamiltonian has eigenvalues `+k` and `-k`,
  each with multiplicity 32.
- The sampled exact Floquet spectrum is gapless only at `k = 0`.
- The background `SU(3)` connection enters in covariant-derivative form:
  `H(k, A) = k sigma_z tensor I_32 + I_2 tensor iA`.

## Electroweak Audit

The Cl(0,2) factor supplies a natural commuting complex structure for the
chiral-16 block, but it does not supply `SU(2)_L` by itself. Its spin group
is `Spin(0,2) = U(1)`, so electroweak `SU(2)_L` still requires additional
structure beyond this Cl(0,8) tensor Cl(0,2) split.

This is the main negative finding of the session.

## Interpretation

The Cl(0,10) lift preserves the Session 16 dynamics-level positive result:
the checkerboard QCA has the correct massless Dirac/Weyl continuum form,
exact background `SU(3)` covariance, and no sampled lattice doubling.

The lift does not yet produce the full Standard Model gauge sector. The
color piece carries through cleanly from the octonion `G2 superset SU(3)`
chain; the electroweak sector remains open.

## Verification

Focused tests:

```bash
uv run pytest \
  src/clifford_3plus2_d5/lepton/tests/test_clifford_chiral16.py \
  src/clifford_3plus2_d5/lepton/tests/test_checkerboard_chiral16.py -q
```

Result:

```text
13 passed
```
