# Session 16 — 1D Cl(8) Checkerboard Continuum Limit

Status: implemented as the first dynamics-level test of the `Cl(8)` pivot.

## Question

Does a 1D checkerboard/Floquet walk on the positive chiral `R8_+` octonion
block have the expected massless Dirac/Weyl continuum limit, and do exact
background `SU(3)` links enter as a gauge connection?

This session is intentionally narrow:

- 1D lattice only;
- massless walk only;
- internal carrier `R8_+`;
- background `SU(3)` links only;
- no mass term, no dynamical gauge fields, no 3+1D, no full chiral-16.

## Conventions

The momentum `k` is dimensionful. The lattice Bloch phase over one lattice
step is:

```text
exp(± i k epsilon)
```

The continuum limit is `epsilon -> 0` with `k` fixed.

We keep two generators distinct:

- `K` is the QCA infinitesimal generator:
  `U = I + epsilon K + O(epsilon^2)`.
- `H = i K` is the Hermitian Bloch Hamiltonian used for Dirac comparison.

The Dirac statement is made at the `H` level.

## Massless Walk

The exact massless Bloch walk is:

```text
U(k, epsilon) = exp(-i k epsilon sigma_z) tensor I_8
```

First order:

```text
K(k) = -i k sigma_z tensor I_8
H(k) =  k sigma_z tensor I_8
```

The dispersion eigenvalues are exactly:

```text
{+k: multiplicity 8, -k: multiplicity 8}
```

This verifies the expected 1D massless checkerboard/Weyl continuum generator
tensored with the `R8_+` internal block.

## Background SU(3) Link

For a real-skew `A in su(3) subset g2 subset so(8)`, the linked right-moving
block is:

```text
U_R(k, epsilon) = exp(-i k epsilon) exp(epsilon A)
```

For the v1 exact symbolic calculation we use the first-order link
`exp(epsilon A) = I + epsilon A + O(epsilon^2)`.

Since `A` commutes with the scalar Bloch phase:

```text
K_R = A - i k I_8
K_L = A + i k I_8

K = I_2 tensor A - i k sigma_z tensor I_8
H = I_2 tensor iA + k sigma_z tensor I_8
```

So the background `SU(3)` link appears in the continuum Hamiltonian as the
internal Hermitian gauge-potential term `iA`.

## Gauge Covariance

The finite gauge covariance check uses exact rigid `SU(3)`-fixing
automorphisms from the Session 14 Clifford-word audit, avoiding symbolic
matrix exponentials.

For an edge link:

```text
link' = G_left link G_right^-1
```

with real orthogonal `G`, `G^-1 = G^T`. The tested identity is:

```text
link' G_right = G_left link
```

This passes as an exact matrix identity.

## Doubling Sample

The doubling check is performed on the exact lattice Floquet spectrum, not on
the continuum Hamiltonian. For sample points:

```text
0, pi/(2 epsilon), pi/epsilon, -pi/(2 epsilon)
```

only `k = 0` has Floquet eigenvalue `1`. At `k = pi/epsilon`, the eigenvalue is
`-1`, the Brillouin-zone edge, not a gapless mode.

## Failure Modes

The implementation names the following failure modes conceptually:

- F1: `H_eff` is not linear in `k`.
- F2: `H_eff(k=0)` is nonzero in the massless case.
- F3: exact Floquet eigenvalue `1` occurs at sampled `k != 0`.
- F4: gauge link contribution has the wrong block structure.
- F5: finite `SU(3)` gauge covariance identity fails.

None of these occur in the v1 tests.

## Result

Session 16 gives the first dynamics-level positive result of the `Cl(8)` pivot:

```text
H_massless(k) = k sigma_z tensor I_8
H_gauge(k,A) = k sigma_z tensor I_8 + I_2 tensor iA
```

with exact finite background-gauge covariance and no sampled lattice doubling
for the massless 1D walk.

This is still only a 1D background-gauge result. It does not yet include a mass
term, dynamical Yang-Mills fields, 3+1D dynamics, or the full chiral-16.
