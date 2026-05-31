# depth_hop_walsh Parameter Ledger

Exact-symbolic throughout (with a numeric Frobenius-norm fallback for support).
No continuous fitting parameters.

## Inherited (read-only, via `reuse.py`)

From `spacetime_qca/`:
1. The 8 BCC body-diagonal directions `v ∈ {±1}³` in BB order
   (`bcc_weyl.bialynicki_birula_directions`).
2. The 8 right-handed Weyl hop matrices `H_v = q_± P_{1..4}` with
   `q_± = (1±i)/4` (`bcc_weyl.bialynicki_birula_hops`); the left-handed
   `H_v^L = H_{-v}` (`opposite_helicity_hops`).
3. The first-order source matrices `s_i = Σ_v v_i H_v`
   (`bcc_weyl.bialynicki_birula_s_matrices`) — used to cross-check the `T₁u`
   Walsh block (`s_i = 8 Ĥ_i`).

From `topology/`:
4. The body-diagonal `C₃` rotation `R: (x,y,z)→(y,z,x)` and the SU(2) spinor lift
   `U₃ = exp(−i(2π/3)(n·σ)/2)`, `n = (1,1,1)/√3` (`bcc_z3_rotation`).

From `koide/`:
5. The `[111]` axis `n̂ = (1,1,1)/√3` (`koide_geometry.trace_direction`).

From `strongcp/` (diagnostic only):
6. `A₂u(H⁽²⁾)=0` (`higher_order_parity.h2_a2u_component_is_zero`) and the
   degree-2 effective Hamiltonian `H⁽¹⁾` (`effective_hamiltonian_first_correction`).

## Computed Walsh content (right-handed; left = parity-signed)

| Walsh mode | `O_h` | depth `2|S|` | value (right) |
|---|---|---|---|
| `Ĥ_∅` | `A₁g` | 0 | `(1/8) I` |
| `Ĥ_x, Ĥ_y, Ĥ_z` | `T₁u` | 2 | `(1/8) σ_i` |
| `Ĥ_xy, Ĥ_yz, Ĥ_zx` | `T₂g` | 4 | `(i/8) σ_z, (i/8) σ_x, −(i/8) σ_y` |
| `Ĥ_xyz` | `A₂u` | 6 | `(i/8) I` |

`[111]` singlets: `A₁g`, `T₁u[111]`, `A₂u` **present**; `T₂g[111] =
(i/8√3)(σ_x − σ_y + σ_z)` **present (nonzero)**. Left-handed: same magnitudes,
parity signs flipped on odd `|S|` (`A₂u^L = −A₂u^R`).

## Covariant `O`-decomposition (W4 — the escape hatch)

Decomposing the *matrix-valued* source under the full octahedral rotation action
`(g·H)_v = U_g H_{R_g^{-1}v} U_g†` (24 elements, `R=Ad(U)`, character projectors;
reconstruction exact, total norm² = 2):

| covariant irrep | norm² | allowed? |
|---|---|---|
| `A₁` | 1 | yes (baseline) |
| `A₂` | 1/3 | yes (pseudoscalar) |
| `E` | 2/3 | **no — forbidden quadrupole** |
| `T₁` | 0 | (vector absent) |
| `T₂` | 0 | yes-absent |

So the coefficient-Walsh `T₂g` reassembles to covariant `T₂=0` (spinor conjugation
matters), but a forbidden `E` survives and `T₁` is absent. Both lenses kill;
`depth = 2·Walsh-degree` is not a covariant label.

## `S₃`/Schur obstruction (W5 — closure theorem)

Residual three-port space `3 = 1 ⊕ 2` under `S₃`. Schur: the commutant is
2-dimensional (`span{P₁, P₂}`), so any `S₃`-invariant depth operator has spectrum
`{α, β, β}` (≤ 2 distinct). Consequences:

| object | spectrum | note |
|---|---|---|
| `K₃` Laplacian `3I − J` | `{0, 3, 3}` | the residual graph that supplies `ε` |
| `2 · L(K₃)` | `{0, 6, 6}` | the best an unbroken `K₃` can do — **not** `{0,2,6}` |
| `diag(0,2,6)` | `{0, 2, 6}` (3 distinct) | **cannot** be `S₃`-invariant |

`diag(0,2,6)` decomposition: invariant part `(8/3) I`; breaking spurion
`diag(−8/3, −2/3, 10/3) ∼ (−4,−1,5)`, traceless ⇒ lives in the `S₃` doublet sector.
So `{0,2,6}` **is** a family-symmetry-breaking spurion; deriving it ≡ deriving the
generation-symmetry breaking (closed-negative: `triality`/`broken_triality`/
`exceptional`).

## Closing ledger

```
even_depth_unit_from_bcc_bipartiteness         = DERIVED
epsilon_from_residual_K3                       = DERIVED
depths_0_2_6_from_cube_parity (BB source)      = KILLED (W2/W4)
depths_0_2_6_from_pronic/flag/root/Coxeter     = KILLED (F1: wrong-sector)
depths_0_2_6_from_residual_K3_subcliques       = KILLED (W5: S3 -> {0,d,d})
generation_depth_embedding_derived             = FALSE
depth_hierarchy_requires_family_symmetry_break = TRUE  (W5; ≡ closed N=3 problem)
```

## Choices

- Support test: exact symbolic zero preferred; `Tr(X†X)` fallback.
- `[111]`-singlet (minimal) criterion for the verdict; the full `T₂g`-triplet
  vanishing is recorded but informational.
- Per-helicity verdicts; combined `PASS` only if both pass, else `HELICITY_SPLIT`
  or the shared kill reason.

## Remaining declared inputs

This probe moves **Claim A only**. The verdict is a KILL of the cube/parity
mechanism for the BB source, so the following remain (the depths revert to a free
fit):

- `radial_depth_equals_twice_walsh_degree` — the bridge (Claim B), untested here.
- `cube_geometry_compatible_with_sqrt5_coin` — Claim C, untested.
- `generation_depth_embedding_derived` — the standing flavor depth input
  (`flavor_a_track`), now confirmed **not** explained by a parity-selected cube
  hop source.
