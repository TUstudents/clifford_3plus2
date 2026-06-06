# radial_response - Status

**Status**: R1-R13 implemented.

## Verdict

```text
RADIAL_RESPONSE_THEORY_PASS
```

## R1 - Boundary Green-Function Mass Form

For a resolved/unresolved split:

```text
H = [[H_P, V^T],
     [V,   H_Q]]
```

the P-block resolvent is:

```text
G_P(z) = [z - H_P - Sigma(z)]^-1
Sigma(z) = V^T (z - H_Q)^-1 V.
```

Expanding `(z-H_Q)^-1` gives repeated boundary returns:

```text
P -> Q -> P
P -> Q -> Q -> P
P -> Q -> Q -> Q -> P
...
```

Verdict:

```text
MASS_AS_BOUNDARY_RECIRCULATION_PASS
```

## R2 - Up Stacking-Law Fork

Two exact radial stackings give different invariant predictions:

```text
exponential / Poisson:  (x^2/2, x, 1) -> C_u C_t / C_c^2 = 1/2
geometric / resolvent: (x^2,   x, 1) -> C_u C_t / C_c^2 = 1
```

Thus the up-sector `1/2!` relation is a stacking-law statement. It does not
derive `x`.

Verdict:

```text
UP_STACKING_LAW_EXPONENTIAL_FAVORED
```

## R3 - Literal Nilpotent Yukawa Kill

Literal `exp(N)` for the unit-shift nilpotent has singular values:

```text
(2, 1, 1/2)
```

and a non-diagonal left metric. Therefore it cannot be the physical Yukawa
matrix if the CKM matrix is to remain a boundary charged-current holonomy.

Verdict:

```text
LITERAL_NILPOTENT_YUKAWA_KILL
```

## R4 - Down Dark-Line Gate

The data-improved bottom coefficient:

```text
C_b^2 = 5/6
```

can be read as:

```text
regular S3 shell minus one forbidden one-dimensional line.
```

But the S3 projector audit shows this is available, not forced. There are two
rank-5 complements, and the rank-2 middle standard copy also needs defect
polarization.

Verdict:

```text
DOWN_DARK_LINE_AVAILABLE_NOT_DERIVED
```

## R5 - Two-Channel Repair Isometry

If an active scalar repair state has exactly two symmetry-related successors
and no leakage, norm preservation gives:

```text
|alpha_1|^2 + |alpha_2|^2 = 1
|alpha_1| = |alpha_2|
```

so:

```text
alpha_1 = alpha_2 = 1/sqrt(2)
```

up to phases. This supplies the repaired up-sector scalar amplitude only under
the finite successor/no-leakage condition. The controls show that one channel,
three channels, leakage, or asymmetric two-channel splitting do not force the
target value.

Verdict:

```text
TWO_CHANNEL_REPAIR_ISOMETRY_PASS
```

## R6 - Minimal Unitary S3 Defect Form

The minimal exact Floquet toy is:

```text
U = S C
```

where `S` is an S3 regular shift on the unresolved shell and `C` is a Givens
defect coin coupling the resolved scalar port to a normalized S3 defect vector.
The resulting matrix is exactly unitary.

For the unitary resolved/unresolved split, the self-energy is:

```text
Sigma_U(z) = U_PQ (z I - U_QQ)^-1 U_QP
```

and the Schur P-block agrees with the full resolvent P-block at the tested
regular point. Changing the coin angle or the defect vector changes the
self-energy. Therefore this gate proves a coherent exact QCA/Floquet form, not
the physical radial values or phase.

Verdict:

```text
MINIMAL_UNITARY_S3_DEFECT_FORM_PASS
```

## R7 - Scalar Successor / No-Leakage Certificate

R7 certifies the finite modeled scalar repair basis behind R5. There is one
active scalar source:

```text
scalar_repair_seed
```

and exactly two allowed one-tick repair successors:

```text
triality_plus
triality_minus
```

All controls are vetoed by exact selection rules:

```text
same_state              -> HEIGHT_LOWERING, ONE_TICK_REPAIR
wrong_height            -> HEIGHT_LOWERING
two_tick_repair         -> ONE_TICK_REPAIR
external_leakage        -> BOUNDARY_REPAIR_SECTOR
asymmetric_sector       -> SCALAR_SECTOR
third_repair_successor  -> Z2_CONJUGATE_PAIR
```

Therefore the finite certificate satisfies the condition used by R5:

```text
two Z2-conjugate scalar successors + no leakage -> x = 1/sqrt(2).
```

The gate remains conditional: it does not prove that this finite candidate
basis exhausts the full microscopic QCA boundary Hilbert space.

Verdict:

```text
SCALAR_SUCCESSOR_NO_LEAKAGE_CERTIFICATE_PASS
```

## R8 - S3 Scalar-Shell Completeness

R8 replaces the R7 finite successor pair with an exact S3 group census. The six
S3 elements split as:

```text
identity                 -> rejected as same-state / no repair
two non-identity cycles  -> scalar holomorphic repair successors
three transpositions     -> Hermitian/Z2 repair sector
```

Thus the scalar holomorphic one-tick sector inside the S3 shell is:

```text
A3 \ {e} = {triality_plus, triality_minus}.
```

Every transposition conjugates the two triality elements into each other:

```text
t triality_plus t^-1 = triality_minus
t triality_minus t^-1 = triality_plus
```

So the two allowed scalar successors are one Z2-conjugate pair. This proves R7's
successor certificate is complete **inside S3**. It still does not prove that
the full QCA boundary Hilbert space has no scalar repair outputs outside the S3
shell.

Verdict:

```text
SCALAR_S3_SHELL_COMPLETENESS_PASS
```

## R9 - QCA-to-S3 Scalar Boundary Reduction

R9 connects the S3 shell back to BCC/vacuum-framed geometry. The finite
tetrahedral automorphism census has:

```text
24 total tetrahedral exit automorphisms
6 selected-exit-preserving automorphisms
18 selected-exit-moving controls
```

The six selected-exit-preserving automorphisms induce the full residual S3
shell. The 18 selected-exit-moving automorphisms are rejected as vacuum-frame
breaking, and explicit non-automorphism controls are rejected as not
tetrahedral QCA boundary automorphisms.

Inside the selected-preserving residual S3:

```text
identity       -> same-state / no repair
transpositions -> Hermitian/Z2 sector
3-cycles       -> scalar holomorphic repair
```

Thus the scalar holomorphic residual outputs are exactly:

```text
triality_plus
triality_minus
```

R9 still depends on a named premise: actual one-tick scalar boundary repair is
represented by vacuum-frame-preserving tetrahedral exit automorphisms. That
premise is not derived from the full BB Weyl update.

Verdict:

```text
QCA_SCALAR_BOUNDARY_REDUCES_TO_S3_PASS
```

## R10 - Scalar Automorphism Premise Bridge

R10 narrows the R9 open premise. It does not claim to derive the scalar map
from the full BB update. Instead, it proves that once a one-tick scalar-local
boundary map is declared, the needed R9 automorphism premise follows exactly
if that map has deterministic tetrahedral exit action and preserves the vacuum
frame.

The accepted maps are precisely:

```text
triality_plus_automorphism
triality_minus_automorphism
```

and they induce:

```text
triality_plus
triality_minus
```

The controls are rejected as follows:

```text
selected_exit_moving_automorphism -> VACUUM_FRAME_PRESERVING
generic_linear_exit_mixture       -> DETERMINISTIC_EXIT_MAP
non_automorphism_exit_map         -> TETRAHEDRAL_AUTOMORPHISM
two_tick_triality_map             -> ONE_TICK_LOCALITY
spin_coupled_triality_map         -> SCALAR_LOCALITY
identity_same_state_map           -> SCALAR_HOLOMORPHIC_SECTOR
hermitian_z2_transposition_map    -> SCALAR_HOLOMORPHIC_SECTOR
```

Thus R10 derives:

```text
declared one-tick scalar-local deterministic exit map
-> vacuum-frame-preserving tetrahedral automorphism premise
-> R9 S3 reduction
```

It still does not derive the declared scalar-local map class from the full BB
Weyl/QCA update.

Verdict:

```text
SCALAR_AUTOMORPHISM_PREMISE_CONDITIONAL_PASS
```

## R11 - Silver Transfer Inheritance

R11 consolidates the sidecars instead of repeating the transfer calculation.
The silver transfer root is inherited from the established
`boundary_response` / `flavor_a_track` stack:

```text
epsilon = sqrt(2) - 1
eta = epsilon^2 = 3 - 2 sqrt(2)
r = epsilon^4 = 17 - 12 sqrt(2)
```

The inheritance checks all existing sources:

```text
residual K3 graph root              -> epsilon
semi-infinite sterile-chain Weyl m  -> epsilon
flavor A-track shared transfer      -> SHARED_TRANSFER_INVARIANT
quark common-chain Schur gate       -> QUARK_TRANSFER_IS_COMMON_CHAIN_SCHUR
```

Negative controls are explicit:

```text
independent fitted eta = 0.172089       -> rejected
K2 / K4 alternate transfer roots        -> rejected
minimal U=S C toy value-forcing claim   -> rejected
```

Thus the radial sidecar no longer treats the silver root as an open derivation
or a local fit. Its real open work is pole/residue rigidity and down-sector
selection.

Verdict:

```text
RADIAL_SILVER_TRANSFER_INHERITANCE_PASS
```

## R12 - Radial Pole/Residue Rigidity No-Go

R12 tests whether the inherited silver transfer, scalar S3 successor pair, and
two-channel no-leakage norm already force the radial mass poles. They do not.

Two exact baths preserve the same triality-head data:

```text
head amplitudes = (1/sqrt(2), 1/sqrt(2))
allowed successors = triality_plus, triality_minus
coupling norm = 1
```

The one-level bath has:

```text
Sigma_1(z) = 1/z
poles = {0}
residues = {1}
```

The symmetric two-level tail bath has:

```text
Sigma_2(z) = (z - 1)/(z^2 - z - 1)
poles = {(1 - sqrt(5))/2, (1 + sqrt(5))/2}
residues = {(5 + sqrt(5))/10, (5 - sqrt(5))/10}
```

Both baths preserve the current finite channel grammar, but their self-energies,
poles, and residues differ. Therefore the existing finite S3/silver-transfer
data do not force mass pole values. A further spectral-density or dynamical
principle is required.

Verdict:

```text
RADIAL_POLE_RESIDUE_RIGIDITY_NO_GO_PASS
```

## R13 - Spectral-Measure Selection Study

R13 turns the mass-texture question into a finite Stieltjes/Jacobi problem.
The target up, down-baseline, and down-candidate textures define positive
finite spectral measures:

```text
Sigma_f(z) = sum_i w_i / (z - lambda_i)
```

with inherited:

```text
eta = (sqrt(2) - 1)^2.
```

The up target is:

```text
poles   = {eta^6/4, eta^3/sqrt(2), 1}
weights = {1/25, 8/25, 16/25}
```

The down S3-baseline target is:

```text
poles   = {sqrt(3/2) eta^4, eta^2/sqrt(2), 1}
weights = {1/2, 1/6, 1/3}
```

The down odd-shell candidate is:

```text
poles   = {sqrt(6/5) eta^4, sqrt(2/5) eta^2, 1}
weights = {6/13, 2/13, 5/13}
```

Each measure reconstructs a unique three-site Jacobi bath by the finite inverse
spectral problem, and the reconstructed Jacobi moments round-trip against the
target moments.

The negative controls are the point of the study:

```text
R12 one-level bath              -> does not match target measures
R12 two-level tail bath         -> does not match target measures
P3 repair Jacobi bath           -> does not match target measures
constant silver-tail Jacobi bath -> does not match target measures
minimal unitary S3 toy at z=2   -> does not match the up target
```

Thus target quark textures can be encoded as boundary spectral measures, but
the existing simple S3/silver/QCA baths do not select those measures. This is
inverse reconstruction, not a derivation of quark masses.

Verdict:

```text
RADIAL_SPECTRAL_MEASURE_RECONSTRUCTION_ONLY
```

## Open Burdens

1. Derive the declared one-tick scalar-local deterministic exit-map class from
   the actual BB/QCA scalar boundary update.
2. Identify a forward spectral-density principle that selects the target
   measure rather than merely reconstructing a Jacobi bath from it.
3. Derive or kill the down one-dimensional dark-line selection rule.
4. Build a simulator-backed spectral density `J_f(omega)` and pole-shift audit
   later.
