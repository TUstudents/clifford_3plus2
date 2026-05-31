# ε Provenance — what "epsilon" means in each module

**Status:** authoritative cross-module definition note. Resolves a symbol
collision: the symbol `ε` is used for **two distinct mathematical objects** in
this workspace. This note pins the definitions, proves they are distinct,
records the genuine cross-sector correlations, and lists documentation that must
not blur the two.

_Verified 2026-05-30 against the live code (values reproduced by
`docs/scripts/verify_epsilon_provenance.py`)._

---

## 1. Catalog — the three appearances of `ε`

| site | object | dimension | value / role | source |
|---|---|---|---|---|
| `boundary_response` | **`ε_silver`** | **dimensionless** | `√2 − 1 ≈ 0.41421`; decaying root of the residual `K₃` transfer; sets the neutrino mass **ratio** `Δm²₂₁/Δm²₃₁ = ε⁴ = 17−12√2 ≈ 0.02944` | `boundary_response/transfer.py::epsilon`, `epsilon_fourth` |
| `cp` | **`ε_lattice`** | dimensionless **symbol** (BCH order parameter) | `sp.symbols("epsilon", positive=True)`; organizes the effective Hamiltonian as `H = H⁽⁰⁾ + ε H⁽¹⁾ + …`; the CP-odd correction `H⁽¹⁾` is the **O(ε)** term | `cp/continuum_cp.py::symbolic_momentum`, `effective_hamiltonian_first_correction` |
| `sme` | **`ε_lattice`** | **length (metres)** | the same lattice spacing given a physical value; `ε ≲ 1.97×10⁻³³ m`; `d⁽⁵⁾ = ε_lattice × (±1)` | `sme/epsilon_constraint.py::epsilon_constraint_payload` |

**Key point:** `cp`'s `ε` and `sme`'s `ε` are the **same object** — the BCC
lattice spacing (`cp` treats it as a formal expansion parameter; `sme` assigns it
a metres value and bounds it experimentally). Only `boundary_response`'s `ε` is a
*different* object: the dimensionless silver ratio. There are therefore **two**
ε's, not three:

```
  ε_silver   = √2 − 1            (dimensionless flavor invariant)   — boundary_response
  ε_lattice  = BCC lattice scale (length; CP order & Lorentz coeff) — cp = sme
```

---

## 2. Proof that `ε_silver ≠ ε_lattice`

Three independent arguments, any one sufficient:

1. **Dimensional.** `ε_silver` is a pure number (a root of `x² + 2x − 1 = 0`).
   `ε_lattice` carries dimension length — `sme/epsilon_constraint.py` states
   verbatim *"ε is the lattice scale in metres."* A dimensionless number cannot
   equal a length.
2. **Numerical.** `ε_silver = √2 − 1 ≈ 0.414`; `ε_lattice ≈ 1.97×10⁻³³ m`. They
   differ by ~**32 orders of magnitude** even ignoring units — they are not the
   same number under any natural identification.
3. **Role.** `ε_silver` is a fixed *output invariant* (it is what the K₃ recurrence
   *produces*). `ε_lattice` is a tunable *input scale* (a free expansion parameter,
   experimentally *bounded* but not predicted). One is derived; the other is an
   input.

**Consequence:** fixing the neutrino mass ratio (`ε_silver⁴`) does **not** fix the
Lorentz/CP coefficient (`ε_lattice`). Any claim of a single number tying flavor to
Lorentz/CP is a symbol collision, not a physical correlation.

---

## 3. The genuine correlations (what *is* real)

The symbol collision does **not** mean the three sectors are unrelated. Three
correlations survive scrutiny:

### 3a. CP ↔ Lorentz — one `ε_lattice` (numerical, real)
`cp`'s leading lattice correction `H⁽¹⁾(k)` is CP-odd and lives entirely in the
`T₂g` cubic-harmonic irrep (`‖H⁽¹⁾‖² = 12`, verified). `sme` maps that *same*
`H⁽¹⁾` to the dim-5 SME coefficient `d⁽⁵⁾ = ε_lattice × (±1)` with T₂g coefficients
`[+1, −1, +1]` (verified). So the CP-violation magnitude and the Lorentz-violation
coefficient are controlled by the **one** lattice `ε` — fixing it fixes both. This
is already embodied in code: `sme/reuse.py` imports `cp.continuum_cp`'s `H⁽¹⁾` and
T₂g machinery. **This is the only genuine *numerical* cross-sector correlation.**

### 3b. Shared BB-walk / BCC-symmetry origin (structural, all three)
All three sectors derive from the *same* Bialynicki-Birula BCC Weyl walk and BCC
centrosymmetry; the cubic point group `O_h` organizes them into irreps:
- `T₂g` — the CP-odd correction (`cp`) and the SME `d⁽⁵⁾` (`sme`);
- `A₂ᵤ` — the strong-CP θ-channel (`strongcp`) and the tetrahedral vacuum selector;
- `K₃` / `ε_silver` — the residual flavor transfer (`boundary_response`).

Code anchors: `boundary_response/vacuum_selector_chiral_bb.py` imports
`spacetime_qca.bcc_weyl`; `strongcp/bcc_centrosymmetry.py` and
`cp/walk_symmetries.py` establish the shared centrosymmetry / discrete-symmetry
pattern. The correlation here is *common origin and shared symmetry group*, not a
shared numerical value.

### 3c. Flavor ↔ CP via BB chirality (structural)
The leptonic CP phase (`boundary_response/leptonic_phase_word.py`, δ_CP) and `cp`'s
T₂g CP-odd correction both originate in the chiral coin `q_± = (1±i)/4`
(`spacetime_qca/bcc_weyl.py::bialynicki_birula_hops`). `cp/walk_symmetries.py`
shows the walk breaks CP/T while preserving CPT — i.e. the chirality is the common
CP-violation source. This is a shared *mechanism* (the complex coin), not a shared
number.

---

## 4. What is NOT a correlation (correction to the roadmap)

`claude review/FLAVOR_THEORY_AND_QSIM_ROADMAP.md` §6 (prediction #2) and §0 stated
a "single-ε correlation" under which *fixing the ν mass ratio fixes the Lorentz/CP
coefficients.* **That is invalid as stated** — it conflates `ε_silver` (the ν-ratio
invariant) with `ε_lattice` (the Lorentz/CP scale), which §2 proves are distinct.

The honest restatement:

> The flavor, CP, and Lorentz sectors share a **common origin** (the BB walk +
> BCC `O_h` symmetry) and the CP and Lorentz sectors share **one lattice ε**. But
> the flavor invariant `ε_silver = √2−1` is a *different object* from the lattice
> scale `ε_lattice`; the neutrino mass ratio and the Lorentz/CP magnitude are
> **independent**. There is no single number tying all three.

---

## 5. Decision: no dedicated sidecar

A kill-disciplined `epsilon_correlation` sidecar is **not warranted**:
- the only genuine *numerical* correlation (CP↔Lorentz, §3a) is already implemented
  (`sme` consumes `cp`);
- the others (§3b, §3c) are origin/symmetry facts, not new computable gates.

A new module would duplicate existing code and manufacture a "verdict" for a
definitional clarification. This note + the verifier (`docs/scripts/verify_epsilon_provenance.py`)
fully capture the result.

---

## 6. Corrections / disambiguation list

| file | finding | action |
|---|---|---|
| `claude review/FLAVOR_THEORY_AND_QSIM_ROADMAP.md` §0, §6 #2 | **conflates** ε_silver and ε_lattice ("single-ε correlation") | **corrected** — now points here and states the honest version |
| `docs/bcc_qca_boundary_response_research_note_v_2.md` | uses `ε = √2−1` (silver) throughout; §16 lists SME T₂g separately without claiming a shared ε | OK — no change needed; consistent with this note |
| `boundary_response/{README.md,STATUS.md,parameter_ledger.md}` | `ε` = silver ratio only; selector gates use BB filled-band energy, not a second epsilon | OK — current sidecar docs keep the meanings separate |
| `cp/parameter_ledger.md` | `ε` = "symbolic positive" (lattice/BCH expansion) | optionally relabel as `ε_lattice` / "lattice expansion parameter" |
| `sme/{STATUS.md,parameter_ledger.md}` | `ε` = lattice scale in metres | OK — optionally add a one-line pointer to this note |

The only mandatory correction is the roadmap. The module docs each use `ε` for a
single object internally and do not cross-link; the optional relabels are
preventive (they keep future readers from re-introducing the collision).

---

## 7. Verified facts (reproducible)

```
ε_silver        = √2 − 1          ≈ 0.41421     (dimensionless)
ε_silver⁴       = 17 − 12√2       ≈ 0.029437    (ν mass-ratio prediction)
ε_lattice bound ≈ 1.97×10⁻³³ m                  (length; CP/Lorentz scale)
magnitude gap   ≈ 32 orders                     (ε_silver vs ε_lattice)
‖H⁽¹⁾‖² (cp)     = 12                            (CP-odd, T₂g)
d⁽⁵⁾ T₂g coeffs  = [+1, −1, +1]                  (sme; d⁽⁵⁾ = ε_lattice × coeff)
```

Run `uv run python docs/scripts/verify_epsilon_provenance.py` to reproduce and
assert these (exits non-zero on any drift).
