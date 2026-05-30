# Phase A-3 — SME experimental bounds note

**Status**: literature scaffold with order-of-magnitude bounds.
Specific numerical entries marked **[Kostelecky-Russell entry id
required]** await verification against the current revision of
arXiv:0801.0287.  Order-of-magnitude bounds (rounded to 1-2
significant figures) are drawn from the principal review sources
listed below and from the body of the cited primary papers.

## Source references

| Source | Use |
|---|---|
| Kostelecky, V.A. and Russell, N. "Data Tables for Lorentz and CPT Violation." Rev. Mod. Phys. 83 (2011) 11, arXiv:0801.0287 (continuously updated; current revision used). | Comprehensive tabulated bounds across photon, fermion, and gravity sectors. |
| Kostelecky, V.A. and Mewes, M. "Lorentz and CPT violation in the Standard Model with a higher-derivative sector." Phys. Rev. D 88 (2013) 096006, arXiv:1308.4973. | Dim-5 non-minimal SME fermion-sector Lagrangian definitions and basis of operators. |
| Kostelecky, V.A. and Mewes, M. "Electrodynamics with Lorentz-violating operators of arbitrary dimension." Phys. Rev. D 80 (2009) 015020, arXiv:0905.0031. | Non-minimal SME photon sector for cross-checks. |
| Mattingly, D. "Modern Tests of Lorentz Invariance." Living Reviews in Relativity 8 (2005) 5. | Cross-check on the comparative scaling of experimental sensitivities. |
| Liberati, S. "Tests of Lorentz invariance: a 2013 update." Class. Quantum Grav. 30 (2013) 133001, arXiv:1304.5795. | Cross-check on cosmic-ray / GRB photon-dispersion bounds. |
| 2024-2025 atom-interferometry / optical-clock papers (see footnote below). | Cross-check for any tightened recent bounds on dim-5 fermion coefficients. |

**2024-2025 cross-check papers** (the note is incomplete without
verification against these — flagged for follow-up):

- Optical-clock comparison program at NIST, PTB, RIKEN; recent atom-
  interferometry results from the Stanford / Hannover / Bremen groups.
- These typically tighten dim-3 minimal SME bounds (a^μ, b^μ) by a
  small factor and may or may not constrain dim-5 axial-vector × 2
  derivatives.

## H^(1) target components (from Phase A-2)

The (CP-odd, T_{2g}) cell of H^(1) per chirality decomposes as

```text
H^(1)_chirality(k) = σ^x · k_y k_z  −  σ^y · k_x k_z  +  σ^z · k_x k_y
```

with three non-zero entries.  Mapped to dim-5 non-minimal SME fermion
sector (Kostelecky-Mewes 2013, axial-vector × two derivatives, CPT-
even, CP-odd), these correspond to three specific entries of the
``d^{(5)}_{αβγ}`` coefficient.  Using a spatial-index convention with
``a`` the spin axis and ``(i, j)`` the derivative-index pair, the
three non-zero components are:

| Spin axis ``a`` | Momentum pair ``(i, j)`` | Coefficient of ε |
| --- | --- | --- |
| x  | (y, z) | +1 |
| y  | (x, z) | −1 |
| z  | (x, y) | +1 |

The remaining 24 entries of the ``d^{(5)}_{αβγ}`` tensor vanish.

## Bounds for the identified components

Dim-5 non-minimal SME fermion-sector coefficients have units of
**inverse energy** (GeV⁻¹) in the standard normalization of
Kostelecky-Mewes 2013.

### Order-of-magnitude bounds (representative, pending Kostelecky-Russell verification)

| Component class | Bound on d^{(5)} entries | Origin | Note |
| --- | --- | --- | --- |
| Electron-sector spatial spin-tensor d^{(5)}_{aij}                       | ≲ 10⁻¹⁵ GeV⁻¹ | Atomic clock Hg/Cs comparison + Ne/He / Cs co-magnetometer | **[KR entry id required]** |
| Electron-sector spatial spin-tensor d^{(5)}_{aij}, sidereal modulation  | ≲ 10⁻¹⁷ GeV⁻¹ | Penning trap, optical clock co-rotation studies | **[KR entry id required]** |
| Neutron / proton sector dim-5 axial-vector                              | ≲ 10⁻¹⁵ GeV⁻¹ | Spin-precession (³He, ¹²⁹Xe, He/Xe co-magnetometer)        | **[KR entry id required]** |
| Atom-interferometry dim-5 fermion (2024-2025 update)                    | **[unverified]**            | Atom interferometer, gravimeter sidereal modulation         | **[recent paper id required]** |

The **tightest representative bound** to apply for the constraint on
ε is therefore

```text
|d^{(5)}|  ≲  10⁻¹⁷ GeV⁻¹       (electron-sector sidereal-modulation upper limit)
```

Cross-checks against Mattingly 2005 and Liberati 2013 show that the
photon-sector dim-5 (k_F^{(5)}) bound is significantly tighter
(~10⁻²⁹ GeV⁻¹ from GRB photon-dispersion), but the photon-sector
bound does NOT apply directly to ``d^{(5)}`` (which is a fermion-
sector coefficient).  We use the fermion-sector bound, treating the
photon-sector bound as orthogonal.

## Caveats and uncertainties

1. **Single-component vs. multi-component constraint**: H^(1) predicts
   three correlated d^{(5)} entries with specific relative signs.  If
   experimental analyses bound individual components separately, the
   tightest applicable bound is on the most-tightly-constrained
   component.  If a global analysis bounds an effective combination,
   the constraint on ε may be either looser or tighter depending on
   alignment with the three predicted entries.

2. **Sidereal modulation**: most tight atomic-clock bounds come from
   the sidereal modulation produced by Earth's rotation, picking out
   spatial-index-specific entries.  H^(1) predicts cubic-anisotropy
   along the BCC lattice axes, which may or may not coincide with
   Earth's spatial axes used in sidereal analysis.  This adds an
   O(1) coordinate-transformation uncertainty to the bound.

3. **2024-2025 atom-interferometry**: recent papers (yet to be
   verified) may improve dim-5 bounds by 1-2 orders of magnitude.  The
   present note assumes no such tightening; a follow-up should verify.

4. **Operator equivalence and field redefinitions**: the dim-5 d^{(5)}
   coefficient is, in some conventions, equivalent under field
   redefinitions to combinations of dim-3 b^μ coefficients.  If the
   H^(1) cubic-anisotropic structure falls in a field-redefinition-
   trivial direction, the bound on ε could be vacuous (failure mode
   F-sme-5 in the PLAN).  Phase A-5's audit checks for this.

## Verdict on the bound

The tightest applicable order-of-magnitude bound on ``d^{(5)}`` from
the standard SME literature is

```text
|d^{(5)}| ≲ 10⁻¹⁷ GeV⁻¹
```

In length units (using ``GeV⁻¹ = ℏc / (1 GeV) ≈ 1.97 × 10⁻¹⁶ m``):

```text
|d^{(5)}| ≲ 10⁻¹⁷ × 1.97 × 10⁻¹⁶ m  ≈  2 × 10⁻³³ m
```

The H^(1) prediction is ``d^{(5)} = ε × (1 unit)`` for each of the
three non-zero components, so

```text
ε  ≲  2 × 10⁻³³ m
```

This is **~ 100× the Planck length** (``ℓ_P ≈ 1.6 × 10⁻³⁵ m``), well
above the 10·ℓ_P boundary of PLANCK-CONSISTENT and below the 10⁻²⁵ m
observable threshold — i.e. in the **UNFALSIFIABLE PASS** verdict class
per the PLAN's classification.  See ``epsilon_constraint.py`` for the
symbolic constraint and final classification.

**Important**: the precise placement (PLANCK-CONSISTENT vs. UNFALSIFIABLE
PASS vs. tighter bound) depends on the Kostelecky-Russell entry id
verification.  This note assumes the conservative ~10⁻¹⁷ GeV⁻¹
fermion-sector bound; a verified bound 1-2 orders of magnitude tighter
would push ε into PLANCK-CONSISTENT or below.

## What this note does not address

- Photon-sector ``(k_F)^{(5)}`` bounds (orthogonal sector).
- Strong-CP / θ_QCD bound from the gauge sector (would need a separate
  audit; see brainstorm item 10).
- Higher-order ε³, ε⁴ corrections — these are not constrained at
  Phase A-3 because they live at higher dimension in the SME
  expansion.
- Specific 2024-2025 atom-interferometry results — flagged for
  follow-up verification.
