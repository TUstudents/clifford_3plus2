# Spacetime QCA Package Plan

Status: new package scaffold for the BCC / STA dynamics direction.

This package is a deliberate split from `clifford_3plus2_d5.lepton`.  The
lepton package established the internal gauge-content side:

- chiral-16 internal carrier from `Cl(0,10)`,
- Pati-Salam factorization `Cl(0,6) x Cl(0,4)`,
- compatible complex structure `J`,
- SM breaking to `SU(3)_c x SU(2)_L x U(1)_Y`,
- correct one-generation hypercharge table,
- 1+1D checkerboard background-gauge continuum checks.

The new package asks the next question:

> Can the internal SM carrier be placed on a real 3+1D QCA spacetime rule
> whose continuum limit is a massless Dirac/Weyl Hamiltonian and whose lattice
> avoids the naive hypercubic doubling obstruction?

The first target is the Bialynicki-Birula BCC Weyl walk, then a massless Dirac
pair, then tensoring with the lepton package's real-form chiral-16 internal
carrier.  The internal factor is `R^32` equipped with the compatible `J`; this
is equivalent to `C^16`, but the current code keeps the exact real basis.

## Scope Boundary

This package owns spacetime dynamics:

- BCC lattice geometry and Bloch symbols.
- Minimal two-component Weyl walk.
- Four-component Dirac assembly in chiral basis.
- Hypercubic doubling control.
- Tensor lift to the internal chiral-16 carrier.
- Background gauge-link coupling in the spacetime walk.
- Later real-space finite-chain / finite-lattice stepping.

This package does not own the internal SM algebra:

- Pati-Salam generators stay in `lepton`.
- SM breaking and hypercharge tables stay in `lepton`.
- Internal `J` construction stays in `lepton`.

The packages meet only at tensor-lift interfaces.

## Architecture Target

Compressed mathematical carrier:

```text
spacetime factor: Cl(1,3) Dirac spinor
internal factor:  Cl(0,6) x Cl(0,4) chiral-16 from lepton
site carrier:     C^4_Dirac x C^16_internal = C^64 per site
```

This is the conceptual target after choosing a `J`-adapted complex basis for
the internal factor.  It is not the matrix size used by the current exact
implementation.

Implementation convention:

```text
spacetime Bloch factor: C^4_Dirac
internal exact basis:   R^32_internal with J, i.e. C^16 after choosing
                        a J-adapted complex basis
matrix size used now:   4 x 32 = 128 complex Bloch components
```

So the package currently audits the real-form tensor lift.  A future
J-adapted compression to explicit `C^16_internal` matrices would change the
implementation size to `C^64`, but that is not the present representation.

For implementation, build in layers:

```text
1. C^2 Weyl BCC walk
2. C^4 massless Dirac pair = Weyl_R direct_sum Weyl_L
3. C^4 x R^32 tensor lift, with R^32 interpreted as C^16 via J
4. background gauge-link lift
```

This order is mandatory.  Debugging should happen on the smallest block that
can expose the failure.

## Session 20 Goal

Implement and audit a 3D BCC massless Dirac walk with internal tensor lift.

The target result is:

```text
H_eff(k) = alpha . k x I_internal
```

where `alpha_i = gamma^0 gamma^i` are Hamiltonian Dirac matrices in chiral
basis.  The Bloch audit compares against `alpha . k`, not against the
Lagrangian operator `gamma_mu k^mu`.

With a background internal gauge generator `A`, the expected first-order
continuum form is:

```text
H_eff(k, A) = alpha . k x I_internal + I_spacetime x i A
```

where `A` is a real-skew generator supplied by the internal package, and `iA`
is the Hermitian charge-side Hamiltonian contribution.

## Load-Bearing Rule Definition

The BCC rule must be the actual Bialynicki-Birula Weyl automaton, not a generic
8-neighbor spinor walk.

Required before implementation:

- Pin the exact 8 BCC body-diagonal directions.
- Pin the exact 2x2 hopping matrices from a reliable source.
- Document the chosen convention and signs in the module docstring.

Do not implement the quaternionic hop pattern from memory.  A plausible but
wrong sign convention can still look reasonable while failing unitarity or
giving the wrong continuum Hamiltonian.

Primary source to pin:

```text
I. Bialynicki-Birula,
"Weyl, Dirac, and Maxwell equations on a lattice as unitary cellular automata,"
Phys. Rev. D 49, 6920 (1994), Section II, equations (8)-(11).
```

The implementation should copy the 8-hop Weyl convention from this source, or
explicitly document if a Meyer / D'Ariano-Perinotti equivalent convention is
chosen instead.  No physics verdict should be claimed until the source
convention is named in `bcc_weyl.py`.

The intended Weyl form is:

```text
psi(x, t + eps) = sum_v H_v psi(x + eps v, t)
```

where `v` ranges over the 8 BCC body diagonals and `H_v` are the specific
Bialynicki-Birula hop matrices.

The Dirac assembly uses chiral basis:

```text
psi = (psi_R, psi_L)^T
alpha_i = diag(sigma_i, -sigma_i)
H_free(k) = alpha . k
```

Mass is explicitly out of scope for Session 20.

## Hypercube Control

Session 20 must include a naive hypercubic control.  The point is not merely
to show that BCC works, but to show that BCC is structurally distinguished from
the naive lattice.

Control Hamiltonian:

```text
H_cube(k) = sin(k_x eps) alpha_x / eps
          + sin(k_y eps) alpha_y / eps
          + sin(k_z eps) alpha_z / eps
```

Expected:

- continuum near `k = 0` matches `alpha . k`;
- gapless points occur at all 8 Brillouin-zone corners;
- therefore the hypercube control has Nielsen-Ninomiya doublers.

This is a Hamiltonian-form control, not a unitary cubic walk.  It is included
as the minimal exact diagnostic for the naive-lattice doubling pattern under
the same `alpha . k` continuum target.  A stricter unitary cubic-walk control
is future work.

BCC expected for the Session 20 sampled corner audit:

- continuum near `k = 0` matches `alpha . k`;
- cubic-corner gapless samples are all reciprocal-lattice origin equivalents.

The Bialynicki-Birula body-diagonal lattice has reciprocal-identifications that
make some cubic-corner representatives equivalent to the origin.  Session 20
therefore does not claim a full fundamental-Brillouin-zone proof.  It claims
the sourced BCC walk has the correct continuum Hamiltonian and that the naive
hypercube has the expected 8 literal corner doublers.

The hypercube failure is part of the positive evidence for choosing BCC.

## Proposed Files

Initial package files:

```text
src/clifford_3plus2_d5/spacetime_qca/
  __init__.py
  PLAN.md
```

Session 20 implementation files:

```text
src/clifford_3plus2_d5/spacetime_qca/
  bcc_geometry.py
      BCC directions, Brillouin-zone sample points, vector utilities.

  pauli.py
      Small 2x2 Pauli matrices and exact identity helpers.

  bcc_weyl.py
      Bialynicki-Birula hop matrices and 2x2 Weyl Bloch symbol.

  dirac.py
      Chiral-basis Dirac assembly and alpha matrices.

  hypercube_control.py
      Naive cubic control Hamiltonian and gapless-point checks.

  continuum.py
      First-order expansion helpers for Bloch symbols.

  gauge_lift.py
      Tensor lift against internal generators from lepton.

  audit.py
      Small result payloads for Session 20 reports.

  SESSION_20_BCC_DIRAC.md
      Human-readable report.

  tests/
      test_bcc_geometry.py
      test_bcc_weyl.py
      test_dirac_assembly.py
      test_hypercube_control.py
      test_gauge_lift.py
```

Keep the small exact blocks independent.  Do not make the first tests depend
on the 128-dimensional real-form tensor lift.

## Session 20 Test Checklist

Build tests:

1. BCC directions are exactly the 8 body diagonals.
2. BCC Weyl hop matrices satisfy the chosen source convention.
3. Weyl Bloch symbol has the expected sampled unitarity/orthogonality
   property.  Full all-`k` symbolic unitarity is deferred; the all-`k` claim is
   sourced from the Bialynicki-Birula construction.
4. Dirac assembly in chiral basis gives `alpha_i = diag(sigma_i, -sigma_i)`.

Continuum tests:

5. Right Weyl block has the first-order expansion
   `U(k, eps) = I + eps(-i sigma . k) + O(eps^2)` or the documented
   sign convention.  Equivalently:
   `K_eff = -i sigma . k` and `H_eff = i K_eff = sigma . k`.
6. Left Weyl block expands to the opposite chirality.
7. Dirac pair expands to `alpha . k`.

Doubling tests:

8. Hypercube control has zero Hamiltonian eigenvalues at all 8 cubic
   Brillouin-zone corners.  Direct symbolic check:
   `H_cube(pi/eps * n_x, pi/eps * n_y, pi/eps * n_z) = 0` for
   `n_i in {0, 1}`.
9. BCC cubic-corner gapless samples are reciprocal-lattice origin
   representatives for the body-diagonal lattice.  A full fundamental-BZ
   no-doubling proof is future work.

Tensor/gauge tests:

10. Tensoring with the `I_32` real-form internal identity multiplies
    eigenvalue multiplicities only.  This is the exact-basis version of a
    `C^16` internal lift.
11. Constant background internal `A` gives
    `H_eff = alpha . k x I_32 + I_4 x iA`.
12. At least one finite internal gauge element from `lepton` satisfies the
    expected covariance identity after tensor lift.

## Gauge-Link Placement

Session 20 uses only constant background links.  Position-dependent gauge
fields are deferred to the finite real-space QCA session.

For a BCC hop along direction `v`, the background-link convention is:

```text
psi(x, t + eps)
  = sum_v (H_v x U_link(x <- x + eps v)) psi(x + eps v, t)
```

For the constant-background Session 20 audit:

```text
U_link = I + eps A + O(eps^2)
```

where `A` is a real-skew internal gauge generator.  This placement makes the
gauge link travel with the hop on the internal factor and gives the expected
first-order Hamiltonian:

```text
H_eff(k, A) = alpha . k x I_internal + I_spacetime x iA
```

Other placements, such as applying one gauge link after the entire hop sum, are
not the Session 20 convention and should not be mixed into the audit.

## Pass Criteria

Session 20 is a pass if:

- the exact Weyl/BCC block is verified against its pinned convention;
- the massless Dirac continuum Hamiltonian is `alpha . k`;
- BCC cubic-corner gapless representatives are reciprocal-origin equivalents
  while the hypercube control has 8 literal corner doublers;
- the internal tensor lift preserves the spacetime result by multiplicity;
- background gauge generators enter additively in covariant-derivative form.

Session 20 is not a pass if:

- the BCC hop matrices are not pinned from a reliable source;
- only a generic 8-neighbor walk is implemented;
- the audit compares against `gamma_mu k^mu` instead of `alpha . k`;
- no hypercube doubling control is present;
- the only verification happens after tensoring to the large internal carrier.

## Failure Modes

F1: BCC symbol is not unitary.

Diagnosis: hop matrix signs or normalization are wrong.

F2: Continuum expansion gives anisotropic coefficients.

Diagnosis: BCC hop convention or lattice-vector normalization is wrong.

F3: Continuum expansion gives `-alpha . k`.

Diagnosis: chirality/sign convention.  Acceptable only if documented and
consistent across R/L blocks.

F4: BCC shows sampled extra gapless points.

Diagnosis: wrong walk, wrong Brillouin-zone sampling, or a real doubling issue.

F5: Hypercube control does not show 8 doublers.

Diagnosis: the control is not the naive central-difference control or the
gapless test is wrong.

F6: Gauge lift breaks covariance.

Diagnosis: link placement or left/right action convention is wrong.

## What Session 20 Would Prove

If Session 20 passes, the project has its first 3+1D dynamics-level result:

- a real geometric 3D lattice, not just a 1D Bloch index;
- massless Dirac/Weyl continuum kinematics;
- BCC distinguished from hypercube by sampled doubling behavior;
- compatibility with the internal Pati-Salam/SM carrier by tensor lift;
- background gauge coupling in covariant Hamiltonian form.

This is still kinematics, not full SM dynamics.

## What Session 20 Does Not Prove

Still out of scope:

- mass/Yukawa layer;
- Higgs sector;
- dynamical gauge fields / Yang-Mills plaquettes;
- full Lorentz boost recovery beyond the `alpha . k` continuum precursor;
- anomaly checks;
- three generations;
- interactions beyond fixed background links.

## Session 21 Preview

If Session 20 passes, Session 21 should add the first mass layer:

```text
H_mass = beta x M_internal
```

where `beta = gamma^0` couples the Weyl_R and Weyl_L blocks.  The audit should
check:

- mass gap opens at `k = 0`;
- no new sampled doublers appear;
- mass term commutes with color where expected;
- allowed internal mass matrices match the SM gauge representation constraints.

## Session 22 Preview

Move from Bloch-symbol audits to an exact finite real-space QCA:

- finite BCC patch or periodic lattice;
- `step(state, links)` implementation;
- exact norm preservation;
- exact background gauge covariance under site-local gauge transforms;
- comparison to Bloch-symbol predictions on periodic lattices.

Only after this should the package approach dynamical gauge fields.

## Current Roadmap After Session 55

Sessions 20-55 have now completed the original Session 20-22 launch arc and
added the first compact gauge-dynamics and Higgs-field infrastructure stack.
The package has:

- finite BCC Dirac stepping;
- position-dependent background gauge covariance;
- Wilson plaquette observables and action densities;
- compact Wilson forces and leapfrog prototypes for SU(2), SU(3), SU(4), and
  Pati-Salam/SM sectors;
- static Higgs/Yukawa representation controls;
- a no-backreaction fermion/gauge wrapper;
- Gauss-law and matter-current backreaction prototypes;
- a site-local Higgs field with gauge-covariant BCC differences, kinetic and
  potential diagnostics, and a sitewise bridge back to `Y(Phi)`;
- a first coupled fermion/gauge/Higgs prototype with Higgs conjugate momentum,
  fixed-link Higgs leapfrog, and a first-order site-local Yukawa kick.
- physical `U(1)_Y` as the default SM sector convention, with exact
  one-generation anomaly diagnostics and explicit raw-hypercharge aliases for
  regression.
- finite-spacing BCC Dirac dispersion anisotropy diagnostics showing cubic
  Weyl anisotropy cancellation and quartic leading Dirac residual.
- tiny-lattice scaling diagnostics for neutral-vacuum density normalization,
  coupled one-step drift, and bounded step-size sweeps.
- optional exact local unitary Yukawa insertion for fixed `Phi`, with the
  first-order update kept as the default compatibility path.
- bounded multi-step tiny-lattice trajectory diagnostics and timing probes for
  the coupled prototype.
- a deterministic tiny-lattice simulation runner with observable histories,
  JSON summaries, and `.npz`/JSON output.
- a split simulator organization: prototype lab runner, generic shared `sim`
  loop/scan/io helpers, and a scan-backed main `spacetime_qca.simulator`
  interface.
- import-boundary tests and local package notes that pin the lab/main/sim
  split.
- bounded scan-backed simulator profiling with a JSON CLI and bottleneck
  report.
- warm kernel profiling with repeated timing payloads and focused kernel cases.
- full-SM coupled-step breakdown profiling for Yukawa kicks, no-backreaction
  fermion/gauge stepping, Higgs leapfrog, diagnostics, Gauss residual, gauge
  Hamiltonian density, and Wilson left-force.
- SM gauge-step microbreakdown cases for gauge leapfrog, Dirac transport,
  compact momentum update, and first/second finite-difference left-force
  probes.
- a batched finite-difference compact Wilson force path, exposed through
  Pati-Salam adapters, simulator/scaling configs, and explicit step-breakdown
  comparison cases.
- a force chunk comparison workflow showing `chunk_size=32` is currently best
  locally among batched finite-difference probes.
- an opt-in analytic staple-like compact Wilson force path for the current BCC
  plaquette convention, with focused SM profiles reducing the first/second
  left-force probes to `0.680 s` and `0.099 s`.

The key priority is now numerical credibility.  Gauge constraints, matter
current, Higgs dynamics, exact local Yukawa insertion, anomaly diagnostics,
Lorentz free-dispersion diagnostics, and a scaling harness are all in place.
The next priority is profiling the scan-backed no-matter and coupled simulator
steps with `method="analytic_staple"` enabled, then optimizing the new measured
bottleneck.  Scalar and batched finite differences stay as correctness oracles
for force changes.
