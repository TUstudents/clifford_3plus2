# radial_response

Theory sidecar for the mass-sector recirculation picture.

The core claim is:

```text
quark masses are boundary Green-function residues / pole shifts,
not direct finite-group outputs.
```

Finite boundary algebra still matters, but it supplies channels, projectors,
depth filters, and selection rules. The radial values come from:

```text
Sigma_f(z) = V_f^dagger (z - H_Q,f)^-1 V_f.
```

## Result

```text
RADIAL_RESPONSE_THEORY_PASS
```

## Implemented Gates

```text
R1 MASS_AS_BOUNDARY_RECIRCULATION_PASS
```

The Schur complement proves that a low-energy mass is a pole shift/residue from
repeated `P -> Q -> P` boundary returns.

```text
R2 UP_STACKING_LAW_EXPONENTIAL_FAVORED
```

The up-sector invariant distinguishes stacking laws:

```text
exponential / Poisson:  C_u C_t / C_c^2 = 1/2
geometric / resolvent: C_u C_t / C_c^2 = 1
```

So nilpotent Taylor earns the factorial relation, not the repair amplitude `x`.

```text
R3 LITERAL_NILPOTENT_YUKAWA_KILL
```

Literal `exp(xN)` is not the family-space Yukawa matrix. At `x=1`, its singular
values are `(2,1,1/2)`, and it has a non-diagonal left metric. The nilpotent is a
scalar-response jet, not the physical Yukawa matrix.

```text
R4 DOWN_DARK_LINE_AVAILABLE_NOT_DERIVED
```

The down bottom candidate `C_b^2 = 5/6` can be read as regular S3 minus one
forbidden one-dimensional line. But S3 alone does not choose the line, and the
middle standard copy also requires defect polarization.

```text
R5 TWO_CHANNEL_REPAIR_ISOMETRY_PASS
```

If the scalar repair state has exactly two symmetry-related one-tick successors
and no leakage, unitarity forces equal amplitudes:

```text
(1/sqrt(2), 1/sqrt(2)).
```

This derives the up repair amplitude `x = 1/sqrt(2)` only under the
microscopic successor/no-leakage hypotheses. The one-channel, three-channel,
leakage, and asymmetric controls do not give the required value.

```text
R6 MINIMAL_UNITARY_S3_DEFECT_FORM_PASS
```

The minimal exact Floquet form

```text
U = S C
```

uses an S3 regular shift and a Givens defect coin coupling the resolved scalar
port to an unresolved S3 shell. It is exactly unitary and its P-block resolvent
matches the unitary Schur self-energy:

```text
Sigma_U(z) = U_PQ (z I - U_QQ)^-1 U_QP.
```

Changing the coin angle or the S3 defect vector changes `Sigma_U`, so the form
does not by itself force the physical phase or radial values.

```text
R7 SCALAR_SUCCESSOR_NO_LEAKAGE_CERTIFICATE_PASS
```

The finite scalar successor certificate has one active source and exactly two
allowed one-tick repair outputs:

```text
triality_plus, triality_minus.
```

Every forbidden control has an explicit veto: same-state, wrong-height,
two-tick, external-leakage, asymmetric-sector, and third-successor candidates
are all rejected. This certifies R5's two-channel no-leakage hypothesis within
the modeled finite basis, but does not prove that the basis is the complete
microscopic QCA boundary Hilbert space.

```text
R8 SCALAR_S3_SHELL_COMPLETENESS_PASS
```

Inside the S3 regular shell, the scalar holomorphic one-tick repair sector is
exactly:

```text
A3 \ {e} = {triality_plus, triality_minus}.
```

The identity is rejected as same-state, and the three transpositions are
rejected as Hermitian/Z2 repair-sector elements. Every transposition conjugates
`triality_plus` to `triality_minus`, so the two allowed cyclic successors are a
single Z2-conjugate pair. This proves R7's successor pair is complete inside
S3, but still does not prove that the full QCA boundary has no scalar repair
outputs outside the S3 shell.

```text
R9 QCA_SCALAR_BOUNDARY_REDUCES_TO_S3_PASS
```

Vacuum-frame-preserving tetrahedral exit automorphisms have selected-exit
stabilizer S3 and induce the full residual S3 shell. Permutations that move the
selected exit are rejected as vacuum-frame breaking; non-automorphism controls
are rejected. Applying the scalar holomorphic restriction to the induced
residual S3 leaves exactly the same `triality_plus`, `triality_minus` pair as
R8.

This reduces the scalar repair problem to S3 under one named premise:

```text
one-tick scalar boundary repair is represented by a vacuum-frame-preserving
tetrahedral exit automorphism.
```

```text
R10 SCALAR_AUTOMORPHISM_PREMISE_CONDITIONAL_PASS
```

A declared one-tick scalar-local boundary map with deterministic tetrahedral
exit action is enough to imply the R9 automorphism premise. The only accepted
maps induce:

```text
triality_plus, triality_minus.
```

The bridge gate rejects selected-exit-moving maps, generic linear mixtures,
non-automorphisms, two-tick maps, spin-coupled/non-scalar maps, same-state
identity, and Hermitian/Z2 transpositions. This closes the premise only inside
the declared scalar-local map class; it still does not derive that class from
the full BB/QCA microscopic update.

```text
R11 RADIAL_SILVER_TRANSFER_INHERITANCE_PASS
```

The sidecar now inherits the already-derived transfer invariant rather than
recomputing it:

```text
epsilon = sqrt(2) - 1
eta = epsilon^2
r = epsilon^4
```

The residual K3 graph root, the sterile-chain Weyl function, the flavor A-track
shared-transfer audit, and the quark common-chain Schur audit all agree. An
independent fitted `eta`, K2/K4 alternate graph roots, and the claim that the
minimal `U=S C` toy alone forces radial values are rejected. The open mass
problem is therefore pole/residue rigidity, not silver-root derivation.

```text
R12 RADIAL_POLE_RESIDUE_RIGIDITY_NO_GO_PASS
```

The current finite data do not force mass pole values. A one-level triality-head
bath and a symmetric two-level tail bath preserve the same inherited transfer,
scalar successor labels, and normalized two-channel head coupling, but they give
different self-energies:

```text
Sigma_1(z) = 1/z
Sigma_2(z) = (z - 1)/(z^2 - z - 1)
```

Their poles and residues differ. Therefore radial masses need an additional
spectral-density or dynamical principle beyond finite S3 grammar and silver
transfer inheritance.

```text
R13 RADIAL_SPECTRAL_MEASURE_RECONSTRUCTION_ONLY
```

The desired up/down quark textures define positive finite Stieltjes measures:

```text
Sigma_f(z) = sum_i w_i / (z - lambda_i)
```

and each target measure reconstructs a unique finite Jacobi bath. This proves
that the textures are compatible with a boundary Green-function realization.
It does not prove that the QCA selects those measures.

The explicit negative controls are:

```text
R12 one-level bath
R12 two-level tail bath
P3 repair Jacobi bath
constant silver-tail Jacobi bath
minimal unitary S3 toy at z=2
```

None select the target measures. The mass sector therefore needs a forward
spectral-density principle, not another inverse encoding.

## Open Burdens

1. Derive the declared one-tick scalar-local deterministic exit-map class from
   the actual BB/QCA scalar boundary update.
2. Identify the forward spectral-density principle that selects the target
   measure rather than merely reconstructing a Jacobi bath from it.
3. Derive or kill the down dark-line selection rule.
4. Build a simulator-backed boundary spectral density `J_f(omega)` in a later
   pass.

## Tests

```bash
uv run pytest src/clifford_3plus2_d5/radial_response/tests -q
```
