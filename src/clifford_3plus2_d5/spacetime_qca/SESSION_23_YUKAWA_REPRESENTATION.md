# Session 23 - Higgs / Yukawa Representation Audit

Status: implemented as a representation-level audit.

## Target

Session 21 verified the scalar Dirac mass slot:

```text
H_m(k) = alpha.k x I_internal + beta x M_internal.
```

The scalar control `M_internal = m I` is a valid Dirac mass control, but it is
not the Standard Model Yukawa sector.  Session 23 asks a narrower question:

> Does the internal chiral-16 carrier contain exact static maps with the charge
> profile expected of a Higgs/Yukawa insertion?

This is not a dynamical Higgs field implementation.

## Charge-Shift Criterion

For a charge observable `Q`, an internal map `M` has charge shift `delta` when

```text
Q M - M Q = delta M.
```

The implemented Higgs-like candidate is required to be:

```text
color singlet
Delta Y      = +1/2
Delta T3_L   = +1/2
```

The opposite Higgs component is represented by the corresponding sign-flipped
constraint.  In the real form used here, the transpose of a `(+1/2, +1/2)`
solution is a `(-1/2, -1/2)` solution because the charge observables are real
symmetric.

## Construction

The candidate is not guessed.  The code solves an exact linear system over the
32-real-dimensional chiral-16 internal carrier:

```text
[g_color, M] = 0                  for all su(3)_c generators
[Q_Y, M] = (+1/2) M
[Q_T3L, M] = (+1/2) M
```

The solution space has real dimension `4`.  The first basis element is used as
a representative static positive-shift Higgs-like map, and its transpose is
used as the conjugate negative-shift component.  This avoids a second
nullspace solve while still checking both charge directions exactly.

## Audit Results

- `beta = gamma^0` is off-diagonal between spacetime chiralities.
- The lepton module's joint `(Y, T3_L)` table matches the SM one-generation
  table.
- Universal scalar `I_32`:
  - preserves color;
  - preserves `SU(2)_L`;
  - preserves hypercharge;
  - is not Higgs-like.
- Projector control:
  - breaks at least one SM sector;
  - is not Higgs-like.
- Higgs-like charge-shift candidate:
  - preserves color;
  - does not commute with `SU(2)_L`;
  - does not commute with hypercharge;
  - satisfies `Delta Y = +1/2`;
  - satisfies `Delta T3_L = +1/2`;
  - is therefore a representation-level Higgs-like insertion.
- Conjugate Higgs-like component:
  - is the transpose of the positive-shift map;
  - preserves color;
  - satisfies `Delta Y = -1/2`;
  - satisfies `Delta T3_L = -1/2`.

## Interpretation

This establishes that the internal carrier contains the correct kind of
static charge-shifting maps for a Higgs/Yukawa insertion.  It does not yet
provide:

- a dynamical Higgs field;
- Yukawa coupling constants;
- a Hermitian full Yukawa Hamiltonian built from both Higgs components;
- a finite real-space QCA layer;
- spontaneous symmetry breaking.

The result should be read as:

```text
The chiral-16 internal representation has the correct Higgs-like map slot.
```

not as:

```text
The Standard Model Higgs/Yukawa dynamics has been derived.
```

## Validation

Scoped commands:

```bash
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests -q
uv run ruff check src/clifford_3plus2_d5/spacetime_qca
```

Current expected result: 45 tests green.

## Still Open

- Hermitian Yukawa Hamiltonian using both Higgs charge components.
- Dynamic Higgs field and position-dependent scalar background.
- Coupling constants and mass hierarchy.
- Finite real-space BCC `step(state, links)`.
- Site-local gauge covariance with position-dependent links.
