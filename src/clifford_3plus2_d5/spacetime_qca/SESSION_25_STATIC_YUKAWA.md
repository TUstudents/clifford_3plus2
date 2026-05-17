# Session 25 - Static Higgs / Yukawa Layer

Status: implemented as a static-background representation audit.

## Target

Session 23 found exact color-singlet internal maps with Higgs-like charge
shifts.  Session 25 asks the next narrower question:

> Do those maps assemble into an `SU(2)_L` Higgs-doublet charge structure, and
> can they form a Hermitian static Yukawa control with the Standard Model
> neutral-VEV breaking pattern?

This still does not implement a dynamical Higgs field or Yukawa couplings.

## Construction

The upper Higgs-like map space is the Session 23 solution space:

```text
Delta Y    = +1/2
Delta T3_L = +1/2
dim_R      = 4
```

Applying the non-Cartan `SU(2)_L` generators to that space produces the lower
component:

```text
Delta Y    = +1/2
Delta T3_L = -1/2
dim_R      = 4
```

The combined real map module has dimension `8`.  This is the real-form
Yukawa-map analogue of a Higgs doublet and its charge-shift action, not an
independent propagating scalar field.

Static Hermitian controls are built as real-form combinations
`M + M.T`.  The neutral static VEV uses the lower component only.  Since
`Q_em = Y + T3_L`, lower-component maps have `Delta Q_em = 0`.

## Audit Results

- Upper Higgs-like map space dimension: `4`.
- Lower Higgs-like map space dimension: `4`.
- Combined upper/lower map-space dimension: `8`.
- Both non-Cartan `SU(2)_L` generators span the same lower space.
- Upper maps satisfy `(Delta Y, Delta T3_L) = (+1/2, +1/2)`.
- Lower maps satisfy `(Delta Y, Delta T3_L) = (+1/2, -1/2)`.
- The individual map spaces are not internal-`J`-linear.
- Full static doublet control is real symmetric.
- `beta x Y_static` is Hermitian.
- The default full static control has rank `2` and nullity `30`.
- The neutral VEV control:
  - preserves color;
  - preserves `Q_em = Y + T3_L`;
  - breaks `Y`;
  - breaks `T3_L`;
  - has rank `2` and nullity `30`.

## Interpretation

Session 25 upgrades the Session 23 slot from "one Higgs-like charge-shift map"
to a static Higgs-doublet map module with the neutral-VEV charge pattern:

```text
SU(2)_L x U(1)_Y -> U(1)_em
```

at the level of exact internal charge observables.

The result is deliberately smaller than a Standard Model Higgs sector:

- no position-dependent scalar field;
- no kinetic term for the Higgs;
- no dynamical gauge-Higgs coupling;
- no Yukawa coupling constants or mass hierarchy;
- no full-rank fermion mass spectrum.

The default static controls are low-rank backgrounds.  They prove the
representation slot and breaking pattern exist; they do not yet produce a
realistic mass matrix.

## Validation

Scoped commands:

```bash
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests/test_yukawa.py -q
uv run pytest src/clifford_3plus2_d5/spacetime_qca/tests -q
uv run ruff check src/clifford_3plus2_d5/spacetime_qca
```

Current expected result after Session 25: 63 tests green.

## Still Open

- Promote the static Higgs maps to a site-local scalar field.
- Add a Higgs kinetic term.
- Build Yukawa coupling matrices instead of default rank-2 controls.
- Check electromagnetic preservation for arbitrary neutral VEV choices.
- Couple the Higgs field to position-dependent gauge links.
