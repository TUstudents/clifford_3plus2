# Quark family depth hierarchy {0, 2, 6}: candidate mechanism and BCC topology

**Status:** research note for external review. **CLOSED.** The Claim-A kill-gate
was built and run (§7) including the matrix-valued covariant escape hatch — KILLED
under both lenses — and the post-cube mechanism family was settled by a Schur's-lemma
closure theorem (§7b): `{0,2,6}` is necessarily an `S₃`-breaking spurion, so deriving
it ≡ deriving the closed-negative generation symmetry breaking.
**Question:** is the quark family transfer-depth embedding `{gen1:0, gen2:2, gen3:6}`
a derivable consequence of BCC boundary topology, or an irreducible free input?
**Bottom line:** the candidate mechanism (an angular-momentum / cubic-harmonic
ladder with a parity selection rule) was sharp and testable, and the decisive cheap
test has now been run. The genuine BCC Weyl hop source carries the `A₁g`, `T₁u`,
**and `A₂u`** primitive `[111]` modes — but it **also** carries the degree-2 even
`T₂g[111]` quadrupole the parity rule must remove, and it is not C₃-covariant at the
lattice level. So **Claim A is falsified** (`depth_hop_walsh`,
`DEPTH_HOP_WALSH_SUPPORT_KILL_T2G_PRESENT`). The one serious escape hatch — that the
matrix-valued `T₂g` block reassembles under spinor conjugation — was checked
explicitly (W4, full covariant `O`-decomposition) and **closed**: covariant `T₂`
does reassemble to 0, but the source is `A₁ ⊕ A₂ ⊕ E` with a forbidden `E`
quadrupole present and the `T₁` vector absent — killed covariantly too. So `{0,2,6}`
is **not** a (parity- or covariantly-) selected cube hop source, and the depth
embedding reverts to an honest free fit. This is the intended outcome of a
kill-disciplined probe. This note lays out the argument, the
codebase reality, an explicit claim hierarchy with an honest certainty ledger, and
the computed result.

**Depth-scar sidecar update:** the follow-up `depth_scar` sidecar now treats the
depth hierarchy as a boundary repair-scar hypothesis rather than a cube-hop
projection.  V1-V3 show that an `S3 -> Z2` path repair scar has
`D_scar = 2 Delta(P3)` with spectrum `{0,2,6}` and can be selected by an
effective symmetric edge-weight potential.  V4 adds the clean CP split: the pure
path is a tree and has no intrinsic graph-holonomy phase, while restoring the
missing edge creates one loop with one gauge-invariant phase.  V5 derives the
same path Laplacian from a length-3 nilpotent repair flag
`N=|u><a|+|a><b|`.  This does not reverse the cube-hop kill; it records the
replacement mechanism and its remaining microscopic inputs.

---

## 1. Background: what "depth" is and what is currently assumed

In the flavor A-track (`src/clifford_3plus2_d5/flavor_a_track`, phase A3, built on
`boundary_response`), each quark generation `n` is assigned an integer **transfer
depth** `d_n` along the semi-infinite **sterile chain** — the radial fiber whose
Weyl transfer factor at the probe `z = 2√2` is

```
ε = √2 − 1   (the "silver ratio", the decaying root of the K3 residual graph).
```

Propagating `k` links along the chain costs amplitude `ε^k`. The CKM mixing
magnitudes are then read off as

```
|V_ij| ~ ε^{|d_i − d_j|},
```

with the embedding

```
d_1, d_2, d_3 = 0, 2, 6   ⟹   |V_12| ~ ε², |V_23| ~ ε⁴, |V_13| ~ ε⁶.
```

**Current provenance (audited).** `quark_transfer_hierarchy.quark_family_depths()`
returns `{1:0, 2:2, 3:6}` as "the minimal ordered quark boundary-depth embedding."
The audit (`flavor_a_track`, gate B2/V12) only *checks* three properties:

- **even:** all depths even;
- **additive:** `d_13 = d_12 + d_23` (the three families lie on a line);
- **CKM-ordered:** depths `{2,4,6}` match the observed hierarchy ordering.

with negative controls (odd-depth, non-additive, permuted-label embeddings are
rejected). The depths are **postulated, not derived** — this is the single
remaining declared input of the A-track, recorded as
`generation_depth_embedding_derived`.

### Numerical quality of the fit (for context)

```
ε²  = 3 − 2√2   ≈ 0.17157        |V_us| ≈ 0.2250    ratio 1.31
ε⁴  = 17 − 12√2 ≈ 0.029437       |V_cb| ≈ 0.0411    ratio 1.40
ε⁶  = 99 − 70√2 ≈ 0.0050506      |V_ub| ≈ 0.00369   ratio 0.73
```

The required O(1) coefficients are all near unity — the depth powers carry the
hierarchy, the Clebsches carry the rest. The Jarlskog invariant scales as
`J ~ ε^{2+4+6} sinδ = ε¹² sinδ ≈ 2.55×10⁻⁵ · sinδ`, vs observed
`J ≈ 3.1×10⁻⁵` — same order for an O(1) phase. The depths are a *good* fit; the
question is whether they are *derived*.

---

## 2. The candidate mechanism

### 2.1 The third regularity beyond even + additive

The gaps are `2, 4` (arithmetic, step 2), so the closed form is

```
d_n = n(n+1),   n = 0,1,2   →   0, 2, 6        (pronic numbers).
```

This is strictly stronger than "even": `n(n+1)` is a product of consecutive
integers, hence *automatically* even. And `n(n+1)` is the eigenvalue of the
**angular Laplacian / quadratic Casimir** — `ℓ(ℓ+1) = 0, 2, 6, 12` for
`ℓ = 0,1,2,3`. So the conjecture is:

> **family depth = boundary angular Casimir eigenvalue.**

### 2.2 Mechanism: a heat semigroup on the cube degree operator (not WKB)

A first instinct — "a centrifugal `ℓ(ℓ+1)/r²` barrier gives a tunnelling exponent
`∝ ℓ(ℓ+1)`" — is **wrong**. WKB through a `1/r²` barrier gives
`S ∝ √(ℓ(ℓ+1)) ∝ (ℓ+½)`, i.e. linear in `ℓ`, not quadratic. The correct sharp
mechanism is that the radial transfer operator *is* a **heat semigroup**, written
**microscopically on the discrete cube/cubic-harmonic degree operator** — the
continuum `S²` Laplacian is only its emergent mnemonic (see §2.4):

```
microscopic:  D_cube = L_{Q₃},   D_cube f_m = 2m·f_m   (Walsh / cubic-harmonic degree m)
              T_∂ = ε^{D_cube} = e^{−t D_cube},  t = −ln ε = ln(1+√2) = arcsinh(1) ≈ 0.881374
              ⟹  family-degree-m coupling = ε^{2m}

emergent:     2m ↔ ℓ(ℓ+1)  under  m = T_ℓ = ℓ(ℓ+1)/2,
              so ε^{2m} reads as the S² heat kernel e^{−t ℓ(ℓ+1)} — a continuum analogy only.
```

i.e. **one ε-step of radial transfer = one unit of cube-degree diffusion time.**
The silver ratio sets the diffusion time; the cube degree sets the suppression. The
proposition the construction must deliver is `T_∂ = ε^{L_{Q₃}}` — *the radial
transfer operator equals the exponential of the boundary cube-degree operator*. The
spherical-Laplacian form is the continuum mnemonic, not the fundamental object.

### 2.3 BCC realization and the parity selection rule

On a BCC lattice the 8 nearest neighbours of a site are the cube vertices
`(±1,±1,±1)`. Two BCC-intrinsic facts:

- **Bipartiteness ⟹ even depths.** The Weyl walk flips sublattice each step;
  returning to the coupling parity needs an even number of steps. Independent
  reason for evenness, consistent with `ℓ(ℓ+1)`.
- **The cube `O_h` harmonics give the ladder.** The cube-vertex function space
  decomposes as `A₁g ⊕ T₁u ⊕ T₂g ⊕ A₂u` (dims 1+3+3+1). Graded by polynomial
  degree (equivalently the hypercube graph-Laplacian eigenvalue `2·degree`):

  ```
  A₁g  degree 0   constant         parity +   eigenvalue 0
  T₁u  degree 1   x, y, z          parity −   eigenvalue 2
  T₂g  degree 2   xy, yz, zx       parity +   eigenvalue 4
  A₂u  degree 3   xyz              parity −   eigenvalue 6
  ```

Restricting to the `[111]` body-diagonal `C₃`-singlets (the axis already
identified as the Koide cone axis), the four singlets are `e₀=1, e₁=x+y+z,
e₂=xy+yz+zx, e₃=xyz` at depths `0,2,4,6`. The observed depths `{0,2,6}` are these
**minus the degree-2 quadrupole `e₂` (`T₂g`, depth 4)**.

**The selection rule (why skip `T₂g`).** A naked "parity-odd source" would also
kill the `e₀` baseline, which we need. The correct statement (due to the
collaborator) is

```
J_family = J_scalar + J_χ,   J_scalar ∈ A₁g,   P J_χ = −J_χ,
⟹  family sector = A₁g ⊕ (parity-odd C₃-singlets) = span{e₀, e₁, e₃},
⟹  depths {0, 2, 6}.
```

The orthogonality is then automatic: with the uniform cube inner product and
`P:(x,y,z)↦(−x,−y,−z)`,

```
⟨J_χ, e₂⟩ = ⟨P J_χ, P e₂⟩ = ⟨−J_χ, +e₂⟩ = −⟨J_χ, e₂⟩ = 0.
```

So the rule is **not** "parity-odd only" — that would also kill the `A₁g` baseline
we need. It is precisely **a trivial even scalar baseline (`A₁g`) plus the
nontrivial parity-odd chiral tower (`e₁, e₃`)**; the one thing removed is the
*nontrivial parity-even* mode `T₂g` (`e₂`). This also yields the count: the cube
`[111]`-singlet sector is 4-dim, its parity-odd part is 2-dim, plus the one even
baseline = **3 families**.

A useful **negative result**: the obvious source ansatz "a function of the
body-diagonal scalar `s = x+y+z`" does **not** work — `s² = 3 + 2(xy+yz+zx)`
regenerates `e₂`. So the source cannot merely be `F(x+y+z)`; it must be genuinely
parity-odd (only `e₁, e₃` plus the constant).

### 2.4 The rep-label reconciliation

There is an apparent contradiction: `ℓ(ℓ+1)=6` suggests `ℓ=2` (d-wave), but the
depth-6 cube mode is `xyz = A₂u`, a degree-3 ("`ℓ=3`-type") object. It dissolves
via `n(n+1) ≡ 2T_n` (`T_n` = triangular numbers `0,1,3`):

```
continuum:  ℓ(ℓ+1),  ℓ=0,1,2   = {0,2,6}
cube:        2·degree, degree=0,1,3 = {0,2,6}     (degree = T_ℓ)
```

They are the **same sequence** via `ℓ ↦ degree = T_ℓ` (`0,1,2 ↦ 0,1,3`). Since the
lattice (BCC) has no fundamental `S²`, the **cube/degree picture is the
microscopic one** and the continuum Casimir is the emergent interpretation. The
family is "`ℓ=2` by Casimir value, `A₂u`/degree-3 by representation."

---

## 3. Codebase reality check (the decisive findings)

This is the part that changes the recommended next step.

### 3.1 The flavor sector does NOT contain the cube

- `quark_family_depths()` returns the literal `{1:0,2:2,3:6}` — postulated. No
  cube, no graph Laplacian, no `{0,2,4,6}` spectrum, no `O_h` decomposition in the
  depth machinery.
- The flavor boundary coin is `Cl₅`, not a cube:
  `quark_boundary_shell` builds `1_direct + 2_BCC + 3_color` = 6 channels (5 odd),
  with `Γ_q² = 5I` and coin phase `atan(√5)`. The "`2_BCC`" is a **2-dimensional
  quadrature** (the `E`-doublet of the `[111]` frame), not the 8-vertex cube. The
  coin carries only the **vector (`T₁u`)** content, `C₃`-split as `1_direct +
  2_BCC`. `A₁g`, `T₂g`, `A₂u` are not present in the flavor coin.

**Consequence:** the depth is a *radial* (chain-step) quantity, postulated; the
coin is *angular* (`Cl₅`); the two are unconnected, and there is no operator in the
flavor sector with spectrum `{0,2,6}`. A test of the form "project the real `Cl₅`
flavor source onto cube `T₂g`" is **ill-posed** — `T₂g` is not in that source.

### 3.2 The required machinery DOES exist — in the strong-CP / spacetime sector

The exact `O_h` cubic-harmonic + parity engine the mechanism needs is built and
already exercised against the genuine BCC Weyl operator:

- `cp/cubic_harmonics.py` — degree-2 `O_h` projectors.
- `strongcp/cubic_harmonics_degree3.py` — degree-3 projectors `A₂u` (basis
  `k_x k_y k_z`, the pseudoscalar), `T₂u`, `T₁u`, with **parity selection already
  implemented**: every `g` (even-parity) irrep has a zero projector at odd degree
  "by parity selection". This is exactly the `g↔even`, `u↔odd` grading the
  mechanism needs.
- `strongcp/higher_order_parity.py` — takes `h²` of the real BCC Weyl Hamiltonian
  (`spacetime_qca.bcc_weyl.opposite_helicity_hops`), expands to degree 3 in
  momentum, decomposes into `A₂u/T₂u/T₁u`, and tests whether the parity-odd `A₂u`
  (the `xyz` θ-term shape) vanishes. **This is precisely the §-mechanism engine,
  already eating the real operator.**

### 3.3 The mechanism in the model's own objects

In the cubic-harmonic (momentum-polynomial) grading the `[111]`-aligned,
parity-graded tower is exactly the sharpened projector
`Π_fam = Π_{A₁g} ⊕ Π_{P=−1, [111]-singlet}`:

```
A₁g   degree 0   k-constant            parity +   → depth 0
T₁u   degree 1   k_x+k_y+k_z  ([111])  parity −   → depth 2
A₂u   degree 3   k_x k_y k_z           parity −   → depth 6
```

The model's own parity rule removes the *nontrivial* even sector (the degree-2
radial `A₁g`, `Eg`, `T₂g`), while the *trivial* degree-0 `A₁g` scalar is retained
as the boundary baseline — i.e. **scalar even baseline + nontrivial odd chiral
tower** (`T₁u`, `A₂u`), not "odd only". Depth `= 2 × degree`, degrees `{0,1,3}` →
`{0,2,6}`.

---

## 4. The real remaining gap

The honest conjecture is now sharp, and **different** from "project the flavor
source onto `T₂g`":

> **Bridge conjecture (NOT in the code):** a quark family's radial sterile-chain
> depth `= 2 × (cubic-harmonic degree of its [111] boundary mode)`, with the
> family modes the parity-graded `[111]`-singlets `A₁g(0), T₁u(1), A₂u(3)`.

Two obstacles stand between this and a derivation:

1. **Sector bridge (unasserted).** The cubic harmonics live in
   `strongcp`/`spacetime_qca` (momentum dispersion of the Weyl operator); the
   depths live in `boundary_response` (radial transfer). Nothing connects them.
   The bridge `radial_depth = 2 × angular_degree` is currently an unproven axiom.
2. **`√5` consistency hurdle (real).** The flavor coin's `√5 = √(2_BCC + 3_color)`
   is load-bearing and *derived* (it sets the coin phase `atan(√5)`), and it uses
   only the `T₁u` vector content. Any cube/cubic-harmonic depth story must be shown
   *consistent* with this `Cl₅` coin, not a competing geometry. This is not a
   formality.

---

## 5. Claim hierarchy and certainty ledger

Three claims must stay separate; a §7 pass proves **only Claim A**.

- **Claim A — angular selection (NOW TESTED → FALSIFIED, both lenses).** Whether the
  genuine BCC Weyl source supports `A₁g(0) ⊕ T₁u^[111](1) ⊕ A₂u(3)` with **no**
  degree-2 even support. Coefficient-Walsh (W2): `KILL_T2G_PRESENT` (`T₂g[111]`
  present, source not C₃-covariant). The matrix-valued escape hatch was then closed
  (W4, covariant `O`-decomposition): the source is `A₁ ⊕ A₂ ⊕ E` (`A₁=1, A₂=1/3,
  E=2/3, T₁=T₂=0`) — covariant `T₂` reassembles to 0, but a forbidden `E`
  quadrupole is present and the `T₁` vector is absent. Grade: **FALSIFIED for the
  BB source under both lenses.**
- **Claim B — radial-depth bridge.** `d_radial = 2 × deg_angular`. This is the real
  missing physical law; the cubic-harmonic decomposition does **not** test it.
  Grade: **C3.**
- **Claim C — flavor-sector embedding.** Quark generations actually occupy those
  three angular modes, compatibly with the `Cl₅` coin and `√5 = √(2_BCC+3_color)`.
  Grade: **C2–C3.**

A is necessary but not sufficient for B; A∧B necessary but not sufficient for C.
**The depths are *derived* only if all three hold.** A §7 pass alone moves Claim A,
nothing more.

| Claim / component | Grade | Basis |
|---|---|---|
| `{0,2,6} = n(n+1) = 2T_n`; even is a corollary | C9 | arithmetic identity |
| Cube `Q₃`/`O_h` spectrum `{0,2,4,6}`, parity grading | C9 | standard; implemented in `strongcp` |
| CKM order-of-magnitude from `ε^{2,4,6}`; `J ~ ε¹²` | C8 | numerical, O(1) coefficients |
| Heat semigroup `T_∂ = ε^{L_{Q₃}}` is the right form (vs WKB) | C7 | correct in principle; the operator must be exhibited |
| **Claim A** — angular `0,1,3` parity tower in the real Weyl source | **FALSIFIED (both lenses)** | W2 → `KILL_T2G_PRESENT`; W4 covariant → `A₁⊕A₂⊕E`, forbidden `E` present, `T₁` absent (escape hatch closed) |
| **Claim B** — bridge `d_radial = 2 × deg_angular` | C3 | unasserted; the real missing law |
| **Claim C** — flavor embedding + `√5` compatibility | C2–C3 | flavor coin is `Cl₅`, depths postulated; open obstruction |
| `N=3` from "scalar baseline + two parity-odd `[111]`-singlets" | C4 | new route; prior `N=3` kills don't cover it, but unproven |

The earlier working grade of "C6 for the mechanism" was for a structure not yet
confirmed to be instantiated. Correctly split, with §7 now run (incl. the covariant
escape hatch): **arithmetic/cube C9; angular selection FALSIFIED both lenses (W2
`T₂g`, W4 covariant `E` present + `T₁` absent); actual flavor-depth derivation
remains C3 — the depths are an honest free fit.**

---

## 6. Falsifiable predictions (if the ladder is real)

- A 4th generation would be `ℓ=3`/degree-3-next → depth `12` → mixing to gen-1
  `~ ε¹² ≈ 2.5×10⁻⁵`: hyper-suppressed, consistent with non-observation.
- The gen1↔gen3 mixing sits at the steepest depth `6` (`~ε⁶`), the `|V_ub|` corner.
- The depth ladder predicts the *functional form* `ε^{ℓ(ℓ+1)}`, not just three
  numbers; any future family must continue the pronic sequence `0,2,6,12,20,…`.

---

## 7. The kill-gate — and its result (BUILT: `depth_hop_walsh/`)

**The object decomposed is the raw BCC hop-shell, not the effective Hamiltonian.**
The gate computes the **coefficient-Walsh transform of the eight 2×2 hop matrices**
`H_v` (`spacetime_qca.bcc_weyl`), `Ĥ_S = (1/8) Σ_v χ_S(v) H_v`, and assigns `O_h`
irreps by **primitive Walsh degree** `|S|` (`A₁g`=0, `T₁u`=1, `T₂g`=2, `A₂u`=3,
depth `2|S|`). "Degree" must mean Walsh degree of the hop coefficients, **not** the
Taylor degree of `h(k)` — the exponential `e^{−iα k·v}` makes a scalar source
generate harmless quadratic Taylor descendants, which are not a `T₂g` family mode.
Restrict to `[111]`-singlets and test, **per helicity** (support = nonzero norm):
`A₁g`, `T₁u[111]`, `A₂u` present and `T₂g[111] = 0`.

> **Warning — the strong-CP `H_eff` result is a diagnostic, not the depth kill-gate.**
> The BCH/effective Hamiltonian (where `A₂u(H⁽²⁾)=0`) is the low-energy Lorentz
> grammar, a different object from the primitive hop-shell alphabet. A missing
> *effective* `A₂u` says nothing about a primitive hop-shell `A₂u` coefficient.
> Using `H_eff` as the depth kill would be a category error; it is kept as a
> separate `diagnostic_only` gate (W3). The correct, weaker conjecture is: *the
> family `A₂u` is a primitive BCC hop-shell component whose radial transfer depth
> is 6* — not "the family `A₂u` must survive as an effective Lorentz `A₂u` term."

**Verdicts** (`DEPTH_HOP_WALSH_SUPPORT_*`): `PASS`; `KILL_MISSING_A2U`;
`KILL_T2G_PRESENT`; `KILL_MISSING_T1U`; `HELICITY_SPLIT`. A `PASS` would establish
**Claim A only**; it does not derive the depths (Claims B, C remain open).

**Computed result — Claim A is KILLED: `DEPTH_HOP_WALSH_SUPPORT_KILL_T2G_PRESENT`.**
The genuine BB Weyl hop source carries the `A₁g` baseline, the `T₁u[111]` vector,
**and** the `A₂u` pseudoscalar (`Ĥ_xyz = ±i/8·I` — the depth-6 mode *is* present),
but **also a nonzero degree-2 even `T₂g[111]` singlet** (`Ĥ_xy=(i/8)σ_z`,
`Ĥ_yz=(i/8)σ_x`, `Ĥ_zx=−(i/8)σ_y`) — exactly the quadrupole the parity selection
rule must remove. Both helicities give the same kill. Moreover the lattice hop
symbol is **not C₃-covariant** about `[111]` (`covariance_check = False`; only the
IR limit `σ·k` restores rotation symmetry), so the coefficient-Walsh labels do not
even lift cleanly to covariant `O_h` irreps.

**The escape hatch — and its closure (W4, covariant `O`-decomposition).** The one
serious objection: the `T₂g` coefficient block is *matrix-valued*, so under a cubic
rotation the Paulis transform too; the relevant object may be the full covariant
`H_v ↦ U_R H_{R⁻¹v} U_R†`, not the coefficient-only Walsh expansion. This is real:
decomposing the source under the full octahedral rotation group `O` (24 elements,
`R = Ad(U)`, character projectors, reconstruction exact) gives

```
A₁ = 1,   A₂ = 1/3,   E = 2/3,   T₁ = 0,   T₂ = 0   (both helicities; Σ = 2 = total).
```

So the coefficient-Walsh `T₂g` **does** reassemble — covariant `T₂ = 0` (the escape
hatch's mechanism is genuine). **But** the covariant decomposition exposes its own
forbidden content: a nonzero **`E` quadrupole (2/3)** — the other `ℓ=2` even irrep —
and the depth-2 **`T₁` vector is absent (0)**. The source is `A₁ ⊕ A₂ ⊕ E`, not the
clean `A₁ ⊕ T₁ ⊕ A₂` tower. So **Claim A is killed under both lenses**; the precise
forbidden mode shifts (coefficient `T₂g` ↔ covariant `E`) but a forbidden even
quadrupole is always present, and covariantly the vector mode is missing too.
**Deeper:** `depth = 2 × Walsh-degree` is not even a covariant label (the directional
`T₁` ⊗ spin contracts to covariant `A₁`), so the cube depth-ladder framing is
undermined regardless of lens. **The cube/parity mechanism is falsified for the BB
source: `{0,2,6}` is not a (parity- or covariantly-) selected cube hop source, and
the depth embedding reverts to an honest free fit.** Intended cheap falsification,
not a failure of the gate.

**Location.** Built as `src/clifford_3plus2_d5/depth_hop_walsh/` (W1 decomposition,
W2 coefficient-support — named primary, W4 covariant `O`-decomposition — escape-hatch
resolution, W5 `S₃`/Schur obstruction — closure, W3 diagnostic, aggregate), in the
spacetime/BCC sector, reusing `spacetime_qca.bcc_weyl`, `topology.bcc_z3_rotation`,
`koide.koide_geometry`, and (diagnostic) `strongcp`. 27 tests pass; ruff clean.

---

## 7b. Post-cube mechanisms and the `S₃`/Schur closure (CLOSED)

After the cube route was killed, a family of new candidates for `{0,2,6}` was
considered — flag dimension `dim SU(n+1)/U(1)ⁿ`, positive-root count `2|Φ⁺(Aₙ)|`,
Coxeter length of `w₀(Sₙ₊₁)`, `K_{n+1}` edge count, Clifford bivector planes, `2·L(P₃)`,
rotor `N(N+1)`. **They are all the same arithmetic** `n(n+1) = 2·C(n+1,2) = 2Tₙ`;
reproducing the numbers is not the question. Three cheap reality-checks settle them:

- **F1 — instantiation: wrong-sector (KILL).** There is no generation-indexed
  `SU(1)⊂SU(2)⊂SU(3)` ladder in the construction. Color `SU(3)` and weak `SU(2)`
  are fixed single copies across generations; there is no positive-root / Weyl /
  flag machinery; the three generations are integer labels with postulated depths
  in a *fixed* family space `I₃` (`H_Q = H_chain ⊗ I₃`). Same wrong-sector failure
  as the cube — and it kills the whole family at once, since each needs the same
  nonexistent per-generation growing structure.
- **F2 — the `ε²` unit: the bipartite half is real.** Even-depth-from-bipartiteness
  and the `ε²` two-step return are genuinely derived; only the "per pair-closure"
  assignment is absent (the same unproven bridge).
- **F3 — `N=3` cutoff: closed-negative.** Three generations is empirical
  (`triality`/`broken_triality`/`exceptional` all KILL); color `SU(3)` does **not**
  supply a cutoff (one fixed copy; the exceptional-Jordan route gives one
  generation, not three).

**The closure theorem (W5, Schur's lemma).** The residual three-port family space
is `3 = 1 ⊕ 2` under `S₃`. By Schur, any `S₃`-**invariant** depth operator commutes
with the rep and has spectrum `{α, β, β}` (commutant dim 2, machine-checked). The
residual `K₃` Laplacian — the graph that supplies `ε` — has spectrum `{0, 3, 3}` →
doubled `{0, 6}`, never `{0, 2, 6}`. But `diag(0,2,6)` has **three distinct**
eigenvalues, so it is necessarily an `S₃`-breaking spurion: invariant part `(8/3)I`,
breaking spurion `diag(−8/3, −2/3, 10/3) ∼ (−4,−1,5)` in the doublet sector.

> **Therefore deriving `{0,2,6}` is equivalent to deriving the family-symmetry-breaking
> spurion — the same closed-negative generation problem.** The depth hierarchy is not
> a separable topological consequence of BCC; it is the `N=3`/generation problem in
> transfer-depth language. What is derived is the `ε²` unit and `ε`; the *number of
> units per generation* (`{0,1,3}`) requires the symmetry breaking the kills cannot
> supply.

This prevents the model from hiding an empirical family-breaking input behind
elegant arithmetic. The depths stand as an honest declared input with a precise
reason. If pursued further, the only non-illusory route is an **explicit `S₃`-doublet
spurion** — ideally the same one that breaks Koide/Yukawa alignment — declared unless
dynamically derived (and an `S₃`-invariant potential does not obviously select the
specific `(−4,−1,5)` direction without parameter choices).

---

## 8. Alternatives the reviewer should weigh

1. **Build the §7 kill-gate now (recommended).** Cheapest decisive test; can kill
   the whole picture immediately; needs no new physics, only wiring existing parts.
2. **Attack the bridge first.** Try to *derive* `radial_depth = 2 × angular_degree`
   from the sterile-chain ↔ boundary-shell relation before testing the angular
   selection. Higher value if it works, but no cheap kill; could sink time.
3. **Attack `√5` consistency first.** Show whether a cube geometry can coexist with
   `Cl₅ = 2_BCC + 3_color`. This is the obstruction most likely to *kill* the whole
   program; doing it first is the most ruthless ordering. Downside: harder to scope.
4. **Do nothing structural; keep depths as honest input.** The A-track is already
   internally consistent with depths declared. Defensible if the reviewer judges
   the mechanism too speculative (C3–C4) to invest in.

**Recommendation:** (1) then (3). The §7 probe is the cheapest gate that can kill
the angular half of the story; if it passes, the `√5` consistency check is the next
ruthless filter before any claim that the depths are derived. The bridge (2) is
only worth deriving if both survive.

**Final decision (one line).** The family depth ladder is **not derived yet** — but
there is now a falsifiable BCC-angular mechanism that, if it survives, would reduce
the depth input to a single bridge axiom `d_radial = 2 × deg_angular`. The immediate
experiment must not touch the flavor sidecar; it tests whether the real BCC Weyl
operator carries the `A₁g(0) ⊕ T₁u^[111](1) ⊕ A₂u(3)` tower with no degree-2 even
support. If it fails, the cube-depth story is killed and the depths remain an honest
free input. **The cube mechanism currently lives in the wrong sector; the next gate
tests whether it is a real BCC angular structure before any attempt to bridge it
into flavor.**

**Follow-up sidecar.** After the `depth_hop_walsh` kill, the next clean
operator-level route is `depth_scar`: replace the diagonal depth assignment by a
defected boundary repair Laplacian.  Its V1 theorem proves that an `S₃ -> Z₂`
path scar gives

```text
D_scar = 2 Delta(P3),        Spec(D_scar) = {0,2,6}.
```

This does not dynamically derive the scar, but it does upgrade the spectrum from
a hand-written diagonal spurion to a positive graph-native operator conditional
on the scar.  The remaining hard question is now sharply isolated: derive the
`S₃ -> Z₂` repair scar from a boundary condition, edge-weight potential,
monodromy, or microscopic update.

`depth_scar` V2 then separates the operator's built-in output from real
predictions: fixed `P₃` normal-mode families, the exact kernel
`T = P0 + ε²P2 + ε⁶P6`, a democratic rank-one leading response, CKM transfer
exponents `λ:λ²:λ³`, and the no-go that a pure path tree carries no intrinsic
graph-holonomy CP phase.  Mass exponents remain conditional on a later
left/right Yukawa assignment.

`depth_scar` V3 derives the path scar at the effective edge-weight level.  The
symmetric potential

```text
V=(S1-2)^2+(S2-1)^2+S3
```

over nonnegative repair weights has exactly the three missing-edge path scars
as zero-energy minima.  This is stronger than declaring a path, but weaker than
a microscopic QCA derivation of the potential.

---

## 9. Open questions for the reviewer

1. Is "depth = 2 × cubic-harmonic degree" the right bridge, or should depth map to
   the degree via a different (e.g. heat-kernel-time) relation? The whole ladder
   rests on this.
2. Is the `[111]` `C₃`-singlet restriction physically forced (Koide axis) or an
   assumption? If families need not be `[111]`-aligned, the count argument weakens.
3. Can the `Cl₅` coin (`2_BCC + 3_color`) and an 8-vertex-cube angular structure be
   two faces of one object, or are they genuinely incompatible geometries?
4. Does a pass on §7 actually reduce the A-track's declared inputs, or merely move
   `generation_depth_embedding_derived` into a (single) bridge axiom of equal
   weight? (i.e. is this real reduction or relabeling?)

---

## 10. References (repository files)

- Flavor depths (postulated): `boundary_response/quark_transfer_hierarchy.py`
  (`quark_family_depths`, `EXPECTED_TRANSITION_DEPTHS`).
- Flavor coin (`Cl₅`): `boundary_response/quark_boundary_shell.py`
  (`quark_odd_clifford_generators`, `quark_gamma_sum`, `quark_boundary_coin`).
- Cubic harmonics + parity engine: `cp/cubic_harmonics.py`,
  `strongcp/cubic_harmonics_degree3.py`, `strongcp/higher_order_parity.py`,
  `strongcp/bcc_centrosymmetry.py`.
- BCC Weyl operator / source: `spacetime_qca/bcc_weyl.py` (`opposite_helicity_hops`).
- `[111]` `Z₃` axis (Koide): `koide/bcc_z3_on_flavor.py`, `koide/koide_geometry.py`.
- ε disambiguation (silver ratio vs lattice spacing): `docs/epsilon_provenance.md`.
- A-track docs: `src/clifford_3plus2_d5/flavor_a_track/{STATUS,PLAN,parameter_ledger}.md`,
  `docs/flavor_a_track.html`.
```
