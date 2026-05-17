# Session 20 - BCC Dirac Infrastructure

Status: implemented as a scoped spacetime-QCA audit.

## Source Convention

The BCC Weyl hop matrices are pinned to:

```text
I. Bialynicki-Birula,
"Weyl, Dirac, and Maxwell equations on a lattice as unitary cellular automata,"
Phys. Rev. D 49, 6920 (1994), Section II.
```

The code implements the paper's source ordering:

```text
W_{+++}, W_{++-}, W_{+-+}, W_{+--},
W_{-++}, W_{-+-}, W_{--+}, W_{---}
```

with `q_+ = (1+i)/4`, `q_- = (1-i)/4`, and the four rank-one matrices
`P_1, ..., P_4` from the paper.

## Implemented Objects

- BCC body-diagonal directions.
- Bialynicki-Birula right-helicity Weyl hops.
- Opposite-helicity hops by the paper's parity rule `W(h) -> W(-h)`.
- Weyl Bloch symbols.
- Massless Dirac assembly in chiral basis.
- Naive hypercube control Hamiltonian.
- Constant-background internal gauge lift.

## Continuum Result

With Bloch phase convention `exp(-i epsilon k.h)`, the sourced hop matrices
recover:

```text
H_R(k) =  sigma . k
H_L(k) = -sigma . k
H_D(k) =  alpha . k
```

where `alpha_i = diag(sigma_i, -sigma_i)` in chiral basis.

This is the Hamiltonian-form audit.  The code intentionally compares against
`alpha . k`, not against the Lagrangian operator `gamma_mu k^mu`.

## Unitarity Scope

The package currently sample-checks the BCC Bloch symbol's unitarity at the
origin and three single-axis cubic corners.  It does not yet prove the full
symbolic identity `S(k)^dagger S(k) = I` for all momenta.  The all-momentum
unitarity claim is therefore inherited from the pinned Bialynicki-Birula
source convention, while this package verifies representative samples and the
first-order continuum Hamiltonian.

## Hypercube Control

The naive cubic control is a Hamiltonian-form lattice fermion, not a unitary
walk.  It uses:

```text
H_cube(k) =
  sin(k_x epsilon) alpha_x / epsilon
+ sin(k_y epsilon) alpha_y / epsilon
+ sin(k_z epsilon) alpha_z / epsilon
```

At the 8 cubic Brillouin-zone corners this Hamiltonian is zero, so the control
has the expected 8 sampled doublers.

This is an apples-to-purpose comparison: both BCC and the control target the
same `alpha . k` continuum Hamiltonian, but the control is included only as
the exact naive-lattice doubling diagnostic.  A stricter cubic unitary-walk
control is future work.

## BCC Corner Caveat

For the body-diagonal BCC walk, the same cubic-corner sample has eigenvalue
`1` at four corners, not only at the literal `(0,0,0)` corner.  This is not
encoded as four independent doublers in the audit: those four corners are
reciprocal-lattice origin representatives for the body-diagonal lattice.

The parity rule is explicit.  For a cubic corner
`k = (pi/epsilon) (n_x,n_y,n_z)` and a body-diagonal hop
`h = epsilon (s_x,s_y,s_z)` with `s_i = +/-1`, the Bloch phase is

```text
exp(i k.h) = exp(i pi (n_x s_x + n_y s_y + n_z s_z))
           = (-1)^(n_x + n_y + n_z).
```

So even-parity cubic corners have phase `1` for every body-diagonal hop and
are reciprocal-origin representatives for this sampled BCC lattice.

The implemented check is therefore:

```text
all sampled BCC cubic-corner gapless points are reciprocal-origin equivalents
```

A full fundamental-BZ no-doubling proof remains future work.  This report does
not claim more than the implemented sampled/reciprocal audit.

## Gauge Lift

Session 20 uses constant background links:

```text
U_link = I + epsilon A + O(epsilon^2)
```

with real-skew internal `A`.

The BCC Dirac symbol lifts as:

```text
U_total(k) = U_BCC_Dirac(k) x U_link
```

and the first-order Hamiltonian is:

```text
H_eff(k, A) = alpha . k x I_internal + I_spacetime x iA
```

This was tested with both a small 2x2 control generator and a real 32x32
Pati-Salam `SU(2)_L` generator from the lepton module.

## Validation

Scoped validation commands:

```bash
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests -q
uv run ruff check src/clifford_3plus2_d5/spacetime_qca
```

No full-suite test is required for this package unless shared modules are
touched.

## Not Yet Done

- Fundamental BCC Brillouin-zone no-doubling proof.
- Finite real-space BCC `step(state, links)`.
- Position-dependent gauge links.
- Mass/Yukawa layer.
- Dynamical gauge fields.
- Lorentz boost recovery beyond the `alpha . k` continuum precursor.
