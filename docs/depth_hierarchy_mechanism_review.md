# Quark family depth hierarchy {0, 2, 6}: candidate mechanism and BCC topology

**Status:** research note for external review — *not* a result claim.
**Question:** is the quark family transfer-depth embedding `{gen1:0, gen2:2, gen3:6}`
a derivable consequence of BCC boundary topology, or an irreducible free input?
**Bottom line:** there is an elegant candidate mechanism (an angular-momentum /
cubic-harmonic ladder with a parity selection rule), the machinery to test it
already exists in the repository, but the mechanism is **not currently realized in
the flavor sector**, and bridging it carries a real consistency hurdle. We
recommend one cheap, decisive kill-test before any larger commitment. This note
lays out the argument, the codebase reality, an honest certainty ledger, and the
decision so a reviewer can confirm or redirect the next step.

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

### 2.2 Mechanism: angular heat-kernel (not WKB)

A first instinct — "a centrifugal `ℓ(ℓ+1)/r²` barrier gives a tunnelling exponent
`∝ ℓ(ℓ+1)`" — is **wrong**. WKB through a `1/r²` barrier gives
`S ∝ √(ℓ(ℓ+1)) ∝ (ℓ+½)`, i.e. linear in `ℓ`, not quadratic. The correct sharp
mechanism is that the radial transfer operator *is* the angular **heat semigroup**:

```
T_∂ = ε^{D_∂} = e^{−t D_∂},   t = −ln ε = ln(1+√2) = arcsinh(1) ≈ 0.881374,
D_∂ Y_ℓ = ℓ(ℓ+1) Y_ℓ   ⟹   family-ℓ coupling = e^{−t ℓ(ℓ+1)} = ε^{ℓ(ℓ+1)}.
```

i.e. **one ε-step of radial transfer = one unit of angular diffusion time.** The
silver ratio sets the diffusion time; angular momentum sets the suppression. This
is the proposition the construction must deliver: *the radial transfer operator
equals the exponential of the boundary angular Laplacian.*

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

So `T₂g` is removed because it is the *nontrivial parity-even* mode, and the only
parity-even family source is the trivial scalar baseline. This also yields the
count: the cube `[111]`-singlet sector is 4-dim, its parity-odd part is 2-dim,
plus the one even baseline = **3 families**.

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

The model's own parity rule removes the entire degree-2 even sector (radial `A₁g`,
`Eg`, `T₂g`), leaving exactly the scalar baseline plus two parity-odd singlets.
Depth `= 2 × degree`, degrees `{0,1,3}` → `{0,2,6}`.

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

## 5. Certainty ledger (honest)

| Claim | Grade | Basis |
|---|---|---|
| `{0,2,6} = n(n+1)`, `n=0,1,2`; even is a corollary | C9 | arithmetic identity |
| CKM order-of-magnitude from `ε^{2,4,6}`; `J ~ ε¹²` | C8 | numerical, O(1) coefficients |
| Cube `Q₃`/`O_h` spectrum `{0,2,4,6}`, parity grading | C9 | standard; implemented in `strongcp` |
| Heat-kernel form `T_∂ = ε^{D_∂}` is the right mechanism (vs WKB) | C7 | correct in principle; `D_∂` must be exhibited |
| Parity selection removes the even quadrupole `T₂g` | C8 | proof in §2.3; rule coded in `strongcp` |
| Cube depth ladder is realized in the **flavor** sector | C2 | **false as stated** — flavor coin is `Cl₅`, depths postulated |
| Bridge `radial_depth = 2 × angular_degree` | C3 | unasserted; right machinery, wrong sector |
| `√5 = √(2_BCC+3_color)` consistent with a cube geometry | C3 | open obstruction |
| `N=3` from "scalar + two parity-odd `[111]`-singlets" | C4 | new route; prior `N=3` kills (triality/exceptional/cobordism) don't cover it, but unproven |

The earlier working grade of "C6 for the mechanism" was for a structure not yet
confirmed to be instantiated; **as a derivation of the flavor depths it is C3–C4**.

---

## 6. Falsifiable predictions (if the ladder is real)

- A 4th generation would be `ℓ=3`/degree-3-next → depth `12` → mixing to gen-1
  `~ ε¹² ≈ 2.5×10⁻⁵`: hyper-suppressed, consistent with non-observation.
- The gen1↔gen3 mixing sits at the steepest depth `6` (`~ε⁶`), the `|V_ub|` corner.
- The depth ladder predicts the *functional form* `ε^{ℓ(ℓ+1)}`, not just three
  numbers; any future family must continue the pronic sequence `0,2,6,12,20,…`.

---

## 7. Proposed next step (the kill-gate)

**Do not** build the originally-proposed "project the flavor `Cl₅` source onto cube
`T₂g`" gate — its premise (a cube in the flavor source) is false. Instead, the one
cheap, decisive experiment that runs against objects that **exist**:

> **Decompose the genuine BCC-Weyl boundary source** (`spacetime_qca.bcc_weyl`,
> the same `opposite_helicity_hops` object the strong-CP audit uses) **in cubic
> harmonics through degree 3** (reusing `cp.cubic_harmonics` +
> `strongcp.cubic_harmonics_degree3` + `strongcp.higher_order_parity`), **restrict
> to `[111]`-singlets, and test:**
>
> - **support on `A₁g`(deg 0) ⊕ `T₁u`(deg 1) ⊕ `A₂u`(deg 3)**, and
> - **zero on the degree-2 even sector (`Eg`, `T₂g`).**

**Pass** (`DEPTH_DEGREE_PARITY_SUPPORT_PASS`): the model's own Weyl operator carries
the `{0,1,3}` parity-graded `[111]` tower. The only remaining input is then the
bridge `radial_depth = 2 × degree`, which would collapse
`generation_depth_embedding_derived` to that single axiom.

**Kill** (`DEPTH_DEGREE_PARITY_SUPPORT_KILL`): the source has degree-2 even
`[111]`-singlet support → the parity/cubic-harmonic story is wrong → depths remain
an honest fit, no harm done.

**What it establishes / does not.**
- *Establishes:* whether the parity selection rule is realized by the real BCC
  Weyl operator (the make-or-break for the cube picture).
- *Does NOT establish:* the bridge `radial_depth = 2×degree`, nor `√5` consistency,
  nor `N=3`. Even on a pass, the depths are not yet derived — only the angular
  selection rule is confirmed.

**Location.** This is a cross-sector probe and belongs in `strongcp`/`spacetime_qca`
(where the harmonics and the Weyl operator live), reusing the strong-CP parity
engine — **not** in the flavor sidecar. The roadmap flags cross-sector consistency
as the sharpest internal falsifier; this is one.

**Cost / risk.** Cheap (reuses existing projectors and the existing Weyl source;
a few symbolic projections). Main risk is **framing drift**: a pass must be
reported as "the angular selection rule holds," not "the depths are derived."

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
