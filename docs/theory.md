# Theory

Spin(10) branching is textbook. The load-bearing question is whether QCA data
derive the real carrier, complex structure `J`, structural `3+2` split, and
safe gate algebra before any Spin(10) representation theory is invoked.

## J-First Bridge

The active theorem target is carrier-first:

```text
K_x ~= R^10
J in SO(K_x)
J^2 = -I
K_x = K_3 ⊕ K_2
dim_R K_3 = 6
dim_R K_2 = 4
J K_3 = K_3
J K_2 = K_2
```

This turns the real carrier into a complex space:

```text
W := (K_x, J) ~= C^5 = C^3 ⊕ C^2.
```

The proposed minimal algebraic ansatz is:

```text
K_x = R^2_clock ⊗ R^5_mode
J = epsilon ⊗ I_5
P_3 = I_2 ⊗ diag(1,1,1,0,0)
P_2 = I_2 ⊗ diag(0,0,0,1,1)
```

This ansatz is not a bridge proof by itself. It becomes meaningful only if QCA
rules force `J`, `P_3`, and `P_2`.

Phase 1 now verifies the exact identities for this ansatz. The check is
documented in [Real Carrier Report](literature/real_carrier_report.md).

Phase 2 verifies that a declared exact gate word can produce this `J`. The
check is documented in [Forced J Report](literature/forced_j_report.md). It
does not yet prove that microscopic QCA rule data force the word.

## Textbook Part

For a fixed complex vector-space split

```text
V = C^3 ⊕ C^2
```

the even exterior algebra has dimension:

```text
S_+ = Lambda^even(V)
dim S_+ = 16
```

Counting selected basis directions in each block gives occupation numbers
`N_3` and `N_2`. Solving

```text
Y(0,0)=0
Y(0,2)=1
Y(1,1)=1/6
```

for `Y = A + B N_3 + C N_2` gives:

```text
Y = -1/3 N_3 + 1/2 N_2
```

The implemented branching table is:

| `(N_3, N_2)` | Multiplicity | `Y` | Label |
| --- | ---: | ---: | --- |
| `(0, 0)` | 1 | `0` | `nu^c` |
| `(0, 2)` | 1 | `1` | `e^c` |
| `(1, 1)` | 6 | `1/6` | `Q` |
| `(2, 0)` | 3 | `-2/3` | `u^c` |
| `(2, 2)` | 3 | `1/3` | `d^c` |
| `(3, 1)` | 2 | `-1/2` | `L` |

This is a representation identity. It is not evidence that QCA geometry
supplies the split or `J`.

## No-Locking Gate Algebra

The geometric QCA gate algebra must lie in the Standard Model commutant.
On

```text
W = C^3 ⊕ C^2
```

the safe complex-linear one-particle endomorphisms are:

```text
End_{SU(3) x SU(2) x U(1)}(W) = C P_3 ⊕ C P_2.
```

So safe geometric gates are block scalars. The QCA may distinguish the whole
`3` block from the whole `2` block, but it may not resolve individual axes
inside either block.

Forbidden:

```text
Hom(C^3, C^2)
Hom(C^2, C^3)
rank-one color projectors
rank-one weak projectors
direction-conditioned internal projectors
```

Block diagonal is not enough. A rank-one projector inside `C^3` is
block-diagonal and still breaks `SU(3)`.

## Load-Bearing Conditions

The bridge becomes nontrivial only if QCA data supplies all of the following:

1. A real ten-dimensional local carrier or ten Majoranas.
2. A local finite-depth or micromotion-derived `J` with `J^2 = -I`.
3. A structural `J`-invariant `6+4` real split.
4. Projectors `P_3,P_2` but no smaller rank-one addressability.
5. A geometric gate algebra contained in `C P_3 ⊕ C P_2`.
6. Only then, the standard `Lambda^even(C^5)` spinor reconstruction.

Until those conditions are met, the verdict remains `notation_only`.
