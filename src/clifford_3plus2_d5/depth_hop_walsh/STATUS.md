# depth_hop_walsh — Status

**Status**: Claim A probe run. Verdict: **DEPTH_HOP_WALSH_SUPPORT_KILL_T2G_PRESENT**.

This probe tests **Claim A** of the depth mechanism
(`docs/depth_hierarchy_mechanism_review.md`): does the genuine Bialynicki-Birula
BCC Weyl hop source carry the primitive `[111]` parity-graded cube tower
`A₁g(0) ⊕ T₁u(1) ⊕ A₂u(3)` with **no** degree-2 even `T₂g` quadrupole? It
decomposes the eight 2×2 hop matrices `H_v` in their **coefficient-Walsh** modes
(NOT the Taylor degree of `h(k)`, NOT the BCH effective Hamiltonian).

**Result — the cube/parity mechanism is killed under both lenses.**

*Coefficient-Walsh (W2):* the hop source carries `A₁g`, `T₁u[111]`, **and** `A₂u`
(`±i/8·I`, depth-6 present) — **but also a nonzero degree-2 even `T₂g[111]`
singlet**, the quadrupole the parity rule must remove → `KILL_T2G_PRESENT` (both
helicities). The lattice symbol is **not C₃-covariant** (`covariance_check =
False`).

*Escape hatch (W4, covariant `O`-decomposition):* the `T₂g` block is matrix-valued,
so under a cubic rotation the Paulis transform too. The full covariant decomposition
gives `A₁=1, A₂=1/3, E=2/3, T₁=0, T₂=0` (reconstruction exact; group validated).
So the coefficient `T₂g` **does** reassemble (covariant `T₂=0` — the escape hatch's
mechanism is real), **but** a forbidden **`E` quadrupole is present (2/3)** and the
depth-2 **`T₁` vector is absent (0)**. The source is `A₁ ⊕ A₂ ⊕ E`, not the clean
`A₁ ⊕ T₁ ⊕ A₂` tower → `COVARIANT_KILL_FORBIDDEN_QUADRUPOLE`. The escape hatch is
**closed**.

Both lenses kill. Moreover `depth = 2 × Walsh-degree` is not even a covariant
label. The `{0,2,6}` ladder is **not** realized as a (parity- or covariantly-)
selected cube hop source; the depth embedding remains an honest free fit.

**Core obstruction (W5, the closure theorem).** Underneath every `{0,2,6}`
mechanism is Schur's lemma. The residual three-port family space is `3 = 1 ⊕ 2`
under `S₃`, so any `S₃`-**invariant** depth operator commutes with the rep and has
spectrum `{α, β, β}` (commutant dim 2; ≤ 2 distinct eigenvalues). The residual
`K₃` Laplacian — the graph that supplies `ε = √2−1` — has spectrum `{0, 3, 3}` →
doubled `{0, 6}`, **never** `{0, 2, 6}`. The depth operator `diag(0,2,6)` has
**three distinct** eigenvalues, so it is necessarily an `S₃`-breaking spurion:
invariant part `(8/3)I`, breaking spurion `diag(−8/3, −2/3, 10/3) ∼ (−4,−1,5)` in
the doublet sector. **Therefore deriving `{0,2,6}` is equivalent to deriving the
family-symmetry-breaking spurion — the same closed-negative generation problem**
(`triality`/`broken_triality`/`exceptional`). The depth hierarchy is the
generation problem in transfer-depth language, not a separable BCC consequence.

This also kills, in one line, the whole post-cube family of candidate mechanisms
(flag dimension, positive-root count, Coxeter length, `K_{n+1}` bonds, Clifford
bivectors, `P₃` Laplacian, rotor `N(N+1)`): they all reproduce `{0,2,6} = n(n+1)`
but each needs a *per-generation growing* internal structure (`SU(1)⊂SU(2)⊂SU(3)`
or nested sub-cliques) that the construction does not contain — color `SU(3)` and
weak `SU(2)` are fixed single copies across generations (F1 = wrong-sector), and
the three distinct depths require the `S₃`-breaking the kills cannot derive.

## Gate status

| Gate | Verdict |
|---|---|
| W1 — coefficient-Walsh decomposition | computed — all four irreps `A₁g, T₁u, T₂g, A₂u` present; `covariance_check = False` (lattice non-covariance) |
| W2 — `[111]`-singlet support (PRIMARY) | **DEPTH_HOP_WALSH_SUPPORT_KILL_T2G_PRESENT** (both helicities) |
| W4 — covariant `O`-decomposition (escape hatch) | **COVARIANT_KILL_FORBIDDEN_QUADRUPOLE** — `A₁=1, A₂=1/3, E=2/3, T₁=T₂=0`; covariant `T₂` reassembles to 0 but `E` present, `T₁` absent; escape hatch **closed** |
| W5 — `S₃`/Schur obstruction (closure theorem) | **DEPTH_HIERARCHY_REQUIRES_S3_BREAKING** — commutant dim 2; `K₃` Laplacian `{0,3,3}`→doubled `{0,6}`; `diag(0,2,6)` has 3 distinct eigenvalues ⇒ `S₃`-breaking spurion `(−4,−1,5)` |
| W3 — effective-Hamiltonian diagnostic | **DEPTH_EFFECTIVE_HAMILTONIAN_DIAGNOSTIC** (`A₂u(H_eff)=0`, `H⁽¹⁾≠0`; `diagnostic_only=True`) |

## What this does and does not establish

- **Does:** decisively falsify the cube/parity selection rule for the genuine BB
  Weyl source — the raw hop shell carries the degree-2 even `T₂g[111]` mode the
  mechanism needs absent, and is not C₃-covariant at the lattice level. A cheap,
  clean KILL of Claim A.
- **Does not:** say anything about the depth values being *wrong* — `{0,2,6}` may
  still be a good fit; it simply is **not** explained by a parity-selected cube
  hop source. The depths revert to a declared free input. The bridge
  `d_radial = 2·deg_Walsh` (Claim B) and `√5` compatibility (Claim C) are moot for
  this source.

## Note on the diagnostic (W3)

The strong-CP result `A₂u(H_eff)=0` (no effective Lorentz θ-term) is **diagnostic
only** and does not drive this verdict. It concerns the low-energy Lorentz grammar,
not the primitive hop-shell alphabet. Conflating them would be a category error;
the kill here is from the primitive `T₂g`, computed independently.

## Cross-module dependency

Reuses `spacetime_qca.bcc_weyl` (the genuine hop source), `topology.bcc_z3_rotation`
(`C₃`/spinor lift), `koide.koide_geometry` (the `[111]` axis), and — diagnostic
only — `strongcp.higher_order_parity`. All via `reuse.py`.

## Test command

```bash
uv run pytest src/clifford_3plus2_d5/depth_hop_walsh/tests -q
```

Expected: 27 passing. The Walsh transform is cross-checked against
`bialynicki_birula_s_matrices` (the `T₁u` block); the verdict helpers cover the
full taxonomies; the real verdicts (`KILL_T2G_PRESENT`,
`COVARIANT_KILL_FORBIDDEN_QUADRUPOLE`, `DEPTH_HIERARCHY_REQUIRES_S3_BREAKING`), the
covariant norms (`A₁=1, A₂=1/3, E=2/3, T₁=T₂=0`), the Schur facts (commutant dim 2,
`K₃`→`{0,3,3}`, spurion `(−8/3,−2/3,10/3)`), `covariance_check = False`, and the W4
group/reconstruction self-validations are locked as regression anchors.

## Closing ledger (CLOSED)

```
even_depth_unit_from_bcc_bipartiteness         = DERIVED
epsilon_from_residual_K3                       = DERIVED
depths_0_2_6_from_cube_parity (BB source)      = KILLED (W2/W4: T2g / E present)
depths_0_2_6_from_pronic/flag/root/Coxeter     = KILLED (F1: wrong-sector, not instantiated)
depths_0_2_6_from_residual_K3_subcliques       = KILLED (W5: S3 -> {0,d,d}, never {0,2,6})
generation_depth_embedding_derived             = FALSE
depth_hierarchy_requires_family_symmetry_break = TRUE  (W5; ≡ the closed N=3 problem)
```

**Status: closed.** What is derived: the `ε²` unit (BCC bipartiteness) and `ε`
itself (residual `K₃`). What is not: the sequence of *how many units* each
generation gets — `{0,1,3}` requires family-symmetry breaking, which is the
empirical/closed-negative generation problem. The depths stand as an honest
declared input with a now-precise reason.
