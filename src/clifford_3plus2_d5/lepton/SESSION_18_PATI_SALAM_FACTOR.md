# Session 18 - Pati-Salam Clifford Factorization

## Goal

Session 17 showed that `Cl(0,8) tensor Cl(0,2)` lifts the checkerboard
dynamics to chiral-16 and supplies a compatible complex structure `J`, but
only gives `Spin(0,2) = U(1)` on the electroweak side.

Session 18 keeps the same real chiral-16 target and asks whether the
Pati-Salam-aligned factorization

```text
Cl(0,10) = Cl(0,6) tensor Cl(0,4)
```

exposes the full Pati-Salam gauge algebra:

```text
Spin(0,6) x Spin(0,4) ~= SU(4) x SU(2)_L x SU(2)_R.
```

## What Was Built

- `clifford_patisalam.py`
  - Exact 8x8 real `Cl(0,6)` gamma matrices.
  - Exact 8x8 real `Cl(0,4)` gamma matrices.
  - Exact 64x64 real `Cl(0,10)` tensor product representation.
  - Chiral projectors with ranks `(32, 32)`.
  - Chiral-16 blocks for:
    - `spin06 ~= su4`
    - `spin04`
    - self-dual `su2_l`
    - anti-self-dual `su2_r`
  - A right-quaternionic `Cl(0,4)` commutant complex structure `J`.
  - A comparison audit for a simple Spin(4) bivector `J`.

- `checkerboard_patisalam.py`
  - Reuses the 1D checkerboard continuum machinery on the chiral-16
    internal carrier.
  - Tests background `SU(4)`, `SU(2)_L`, and `SU(2)_R` generators.

## Exact Clifford Results

```text
Cl(0,6) relations:        pass
Cl(0,4) relations:        pass
Cl(0,10) relations:       pass
Chirality ranks:          (32, 32)
spin06 / su4 dimension:   15
spin04 dimension:         6
su2_l dimension:          3
su2_r dimension:          3
[su2_l, su2_r]:           0
```

This confirms that the Pati-Salam factorization exposes
`SU(4) x SU(2)_L x SU(2)_R` on the same chiral-16 carrier.

## Complex Structure Finding

The compatible default `J` is a right-quaternionic unit in the commutant of
the `Cl(0,4)` action.

It satisfies:

```text
J^2 = -I
J^T J = I
[J, su4] = 0
[J, su2_l] = 0
[J, su2_r] = 0
```

So this factorization supplies a `J` compatible with the full Pati-Salam
algebra. It does not by itself break `SU(2)_R`.

The comparison audit is important: a simple Spin(4) bivector also squares to
`-I`, but it commutes with only one generator in each `SU(2)` factor. That
choice breaks both `SU(2)` factors to Cartans, so it is not the correct
default chiral-16 complex structure.

## Checkerboard Dynamics

For background generators from `su4`, `su2_l`, and `su2_r`, the continuum
Hamiltonian has the same form as Session 17:

```text
H(k, A) = k sigma_z tensor I_32 + I_2 tensor iA.
```

The sampled exact Floquet spectrum remains gapless only at `k = 0`.

```text
Internal real dimension:       32
Massless Floquet shape:        (64, 64)
Massless eigenvalues:          {k: 32, -k: 32}
Gapless sampled momenta:       (0,)
```

## Interpretation

Session 18 closes the electroweak gap identified in Session 17 at the
Pati-Salam level:

```text
Cl(0,8) tensor Cl(0,2):  SU(3) + J + U(1), but no SU(2)_L.
Cl(0,6) tensor Cl(0,4):  SU(4) x SU(2)_L x SU(2)_R + compatible J.
```

This is a stronger gauge-source result than Session 17. The full Standard
Model subgroup has not yet been extracted, because that requires the
Pati-Salam breaking choice:

```text
SU(4) -> SU(3)_c x U(1)_{B-L}
Y = T3_R + (B-L)/2.
```

That is the natural Session 19 target.

## Verification

Focused tests:

```bash
uv run pytest \
  src/clifford_3plus2_d5/lepton/tests/test_clifford_patisalam.py \
  src/clifford_3plus2_d5/lepton/tests/test_checkerboard_patisalam.py -q
```

Result:

```text
14 passed
```
