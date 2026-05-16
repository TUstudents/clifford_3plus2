# Session 19a - SM Gauge Algebra Extraction

## Goal

Extract the Standard Model gauge algebra from the Session 18 Pati-Salam
factorization on the same real chiral-16 carrier:

```text
SU(4) x SU(2)_L x SU(2)_R
  -> SU(3)_c x SU(2)_L x U(1)_Y
```

with the Pati-Salam hypercharge convention

```text
Y_raw = T3_R + (B-L)/2.
```

This session is about the algebra and background-gauge checkerboard
dynamics. Charge normalization and the named field table are left to Session
19b.

## What Was Built

- `patisalam_sm.py`
  - Selects a `B-L` direction in `su4` as the sum of the three commuting
    `Spin(0,6)` Cartan bivectors.
  - Computes the centralizer of `B-L` inside `su4`.
  - Extracts `su3_c` as the trace-orthogonal centralizer slice.
  - Selects `T3_R` from `su2_r`.
  - Builds raw hypercharge `Y_raw = T3_R + (B-L)/2`.
  - Audits dimensions, commutators, closure, and `J` compatibility.

- `checkerboard_sm.py`
  - Reuses the Session 18 checkerboard dynamics for representative
    `su3_c`, `su2_l`, and `u1_y` background generators.

## Exact Algebra Results

```text
su4 dimension:                    15
B-L centralizer in su4:            9
su3_c dimension:                   8
su3_c closes:                      True
su2_l dimension:                   3
u1_y dimension:                    1
SM total dimension:                12
[su3_c, su2_l]:                    0
[su3_c, Y]:                        0
[su2_l, Y]:                        0
chosen J commutes with SM:         True
Y nonzero:                         True
Y skew-symmetric:                  True
Y independent from su3_c + su2_l:  True
```

The extracted Standard Model algebra has the expected dimension:

```text
dim SU(3)c + dim SU(2)L + dim U(1)Y = 8 + 3 + 1 = 12.
```

## Checkerboard Dynamics

For representative background generators from `su3_c`, `su2_l`, and
`u1_y`, the continuum Hamiltonian remains:

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

Session 19a establishes the full Standard Model gauge algebra inside the
`Cl(0,10) = Cl(0,6) tensor Cl(0,4)` Pati-Salam chiral-16 carrier, with the
same compatible right-quaternionic `J` found in Session 18.

This closes the gauge-algebra extraction step:

```text
Session 18: SU(4) x SU(2)_L x SU(2)_R + compatible J
Session 19a: SU(3)_c x SU(2)_L x U(1)_Y + compatible J
```

The remaining work is normalization and representation labeling. Session 19b
should compute the hypercharge eigenspectrum/table and compare it to the
textbook one-generation chiral-16 decomposition.

## Verification

Focused tests:

```bash
uv run pytest \
  src/clifford_3plus2_d5/lepton/tests/test_patisalam_sm.py \
  src/clifford_3plus2_d5/lepton/tests/test_checkerboard_sm.py -q
```

Result:

```text
11 passed
```
