# depth_hop_walsh — design (Claim A of the depth mechanism)

## Load-bearing question

> Does the genuine BCC Weyl **hop source** carry the primitive `[111]`
> parity-graded cube tower `A₁g(0) ⊕ T₁u(1) ⊕ A₂u(3)` with **no** degree-2 even
> `T₂g` quadrupole — the angular selection rule the `{0,2,6}` depth ladder needs?

This is **Claim A** of `docs/depth_hierarchy_mechanism_review.md`, the only part
testable now. A PASS establishes the angular `0,1,3` source ladder; it does **not**
derive the depths (Claims B and C remain open).

## What is decomposed (and what is not)

The object is the **coefficient-Walsh transform of the eight 2×2 hop matrices**
`H_v`, `v ∈ {±1}³`:

```
Ĥ_S = (1/8) Σ_v χ_S(v) H_v,   χ_S(v) = Π_{i∈S} v_i,
O_h by Walsh degree |S|: A₁g(0), T₁u(1), T₂g(2), A₂u(3),  depth = 2|S|.
```

NOT the Taylor degree of `h(k)` (the exponential makes a scalar source generate
harmless quadratic Taylor descendants), and NOT the BCH effective Hamiltonian
(that is the low-energy Lorentz grammar — W3 diagnostic only).

## Gates

- **W1** (`hop_walsh_decomposition.py`). Walsh coefficients per helicity; `[111]`
  singlets `A₁g = Ĥ_∅`, `T₁u[111] = (Ĥ_x+Ĥ_y+Ĥ_z)/√3`, `T₂g[111] =
  (Ĥ_xy+Ĥ_yz+Ĥ_zx)/√3`, `A₂u = Ĥ_xyz`; two-tier zero (`is_zero_symbolic`
  preferred, `frobenius_norm_squared` fallback). **Coefficient-Walsh ≠ covariant
  O_h** — `is_c3_covariant` checks whether the source is C₃-equivariant about
  `[111]` so the labels lift.
- **W2** (`hop_walsh_support_audit.py`) — PRIMARY. Per-helicity verdict
  (`right/left/combined`), support by nonzero norm (not cross-helicity equality;
  `A₂u` is parity-odd). Taxonomy: `PASS`, `KILL_MISSING_A2U`, `KILL_T2G_PRESENT`,
  `KILL_MISSING_T1U`; combined `PASS` iff both pass, `HELICITY_SPLIT` if exactly
  one passes.
- **W4** (`covariant_o_decomposition.py`) — the escape-hatch resolution. The `T₂g`
  coefficient block is matrix-valued, so under a cubic rotation the Paulis also
  transform. W4 builds the octahedral rotation group `O` (24 elements, `R = Ad(U)`,
  self-validated) and decomposes the source into genuine covariant irreps
  `{A₁,A₂,E,T₁,T₂}` via character projectors (reconstruction-checked). Allowed =
  `A₁,A₂,T₁`; forbidden = `E,T₂`. Verdict `COVARIANT_SUPPORT_PASS` /
  `COVARIANT_KILL_FORBIDDEN_QUADRUPOLE` / `..._MISSING_VECTOR` / etc.
- **W5** (`s3_schur_obstruction.py`) — the closure theorem. Schur's lemma on the
  residual `3 = 1 ⊕ 2`: any `S₃`-invariant depth operator has spectrum `{α,β,β}`
  (commutant dim 2). The `K₃` Laplacian gives `{0,3,3}`→doubled `{0,6}`; `diag(0,2,6)`
  has 3 distinct eigenvalues ⇒ an `S₃`-breaking spurion `(−4,−1,5)`. Verdict
  `DEPTH_HIERARCHY_REQUIRES_S3_BREAKING`. This kills the entire post-cube family of
  candidate mechanisms (flag/root/Coxeter/`K_{n+1}`/bivector/`P₃`/rotor) in one
  step, since none is instantiated per-generation (F1 = wrong-sector) and all need
  the `S₃`-breaking the closed kills cannot derive.
- **W3** (`effective_hamiltonian_diagnostic.py`) — DIAGNOSTIC ONLY. Reports
  `A₂u(H_eff)=0`, `H⁽¹⁾≠0` with `diagnostic_only=True`; cannot pass or kill.
- **Aggregate** (`depth_hop_walsh_audit.py`). Named primary = W2; W4 is the
  escape-hatch resolution; W3 diagnostic-only. `claim_a_killed_both_lenses` is True
  iff W2 and W4 both kill. Records the standing inputs (Claims B, C + the flavor
  depth).

## Acceptance standard

The gate **can kill cheaply** and a kill is a legitimate outcome. A PASS is
"Claim A holds — the raw BCC hop source carries the `[111]` `0,1,3` parity ladder,"
**not** "the depths are derived." A KILL retires the cube/parity story; depths
remain an honest free fit.

## Outcome (computed) — CLOSED

Killed under both lenses, with a representation-theory closure. **W2**
(coefficient-Walsh): `DEPTH_HOP_WALSH_SUPPORT_KILL_T2G_PRESENT` — nonzero degree-2
`T₂g[111]`; symbol not C₃-covariant. **W4** (covariant, the escape hatch):
`COVARIANT_KILL_FORBIDDEN_QUADRUPOLE` — `A₁=1, A₂=1/3, E=2/3, T₁=T₂=0`; the
coefficient `T₂g` reassembles to covariant `T₂=0`, but a forbidden `E` quadrupole
is present and the `T₁` vector is absent. **W5** (closure): `S₃`/Schur ⇒ any
invariant depth operator has spectrum `{α,β,β}`; `{0,2,6}` (3 distinct) is an
`S₃`-breaking spurion, so deriving it ≡ deriving the (closed-negative) generation
symmetry breaking. Derived: `ε²` unit + `ε`. Not derived: the per-generation count
`{0,1,3}`. The depth embedding stands as an honest declared input.

## Reuse (no new physics; all via `reuse.py`)

- `spacetime_qca.bcc_weyl` — `bialynicki_birula_directions`,
  `bialynicki_birula_hops`, `opposite_helicity_hops`, `bialynicki_birula_s_matrices`.
- `topology.bcc_z3_rotation` — `body_diagonal_rotation_matrix`,
  `apply_rotation_to_direction`, `dirac_spinor_lift`.
- `koide.koide_geometry` — `trace_direction` (the `[111]` axis convention).
- `strongcp.higher_order_parity` + `strongcp.reuse` — diagnostic only (W3).
