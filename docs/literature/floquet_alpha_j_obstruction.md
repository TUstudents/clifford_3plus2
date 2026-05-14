# Floquet Alpha J Obstruction

Status: strict-standard obstruction, not a cosmetic sign choice.

Floquet-α and α+ now separate two notions of "forced `J`".

## Spectral-Polarization `J`

Given the mandatory Floquet operator `U` with real minimal-polynomial factors:

```text
f_alpha(x) = x^2 + x + 1
f_eta(x) = x^2 + 1
```

Bezout gives:

```text
-x f_alpha + (x + 1) f_eta = 1
```

so the rank-`6` and rank-`4` spectral projectors are:

```text
P_alpha = (U + I)(U^2 + I)
P_eta = I - P_alpha
```

The oriented Floquet branches define:

```text
K_alpha = (2U + I) P_alpha
K_alpha^2 = -3 P_alpha
J_eta = U P_eta
J_eta^2 = -P_eta
J_alpha = K_alpha / sqrt(3)
J = J_alpha + J_eta
```

The exact certificate is the scaled relation for `K_alpha`, over
`QQ(zeta_12)` with real entries in `QQ(sqrt(3))`. The normalized `J_alpha`
requires division by `sqrt(3)` and is recorded as a derived algebraic-field
object, not as something the rational-basis solver must recover.

It satisfies:

```text
J^2 = -I
J^T J = I
[J, P_alpha] = [J, P_eta] = 0
```

## Strict Compatible-Commutant `J`

The stricter test asks for uniqueness of compatible complex structures:

```text
J' in SO(10)
(J')^2 = -I
[J', U] = 0
```

and wants only:

```text
J' = ±J
```

The current exact solve does not certify that. More importantly, the failure is
not just the four choices `±J_alpha ± J_eta`.

For each Floquet-α pattern, the compatible centralizer of `U` is:

```text
alpha sector: rank 6 = three identical R(2 pi / 3) blocks
  Cent_U(alpha) ~= M_3(C), real dimension 18

eta sector: rank 4 = two identical R(pi / 2) blocks
  Cent_U(eta) ~= M_2(C), real dimension 8

total compatible_centralizer_dimension = 26
```

Inside that centralizer, the orthogonal compatible complex structures form a
continuous family. The standard orbit count gives:

```text
alpha sector: U(3) / O(3), real dimension 6
eta sector: U(2) / O(2), real dimension 3
total compatible_J_family_dimension ~= 9
```

So the spectral-vs-strict gap is a roughly nine-parameter family of valid
compatible `J'`, not a finite sign ambiguity. The checker reports:

```text
generated_j_moduli_dimension = 0
compatible_centralizer_dimension = 26
compatible_j_moduli_dimension = 9
locality_radius_bound = 0
local_compatible_operator_dimension = 4
local_compatible_j_moduli_dimension = 0
local_compatible_complex_structure_count = 4
strict_compatible_j_forced = false
pass_strict_rule_to_bridge = false
```

The strict checker now treats forcedness as a local moduli condition. A
length-two candidate list is only meaningful after the rule-generated local
`J` moduli dimension has collapsed to `0`; parametric families are never
accepted as forced. In Floquet-α, the rule-generated local center already
contains four compatible complex structures. This does not enumerate all
on-site local `J in M_10(R)`; it is enough to show that the rule's own algebra
has not singled out a unique `±J`.

## Decision

The project must choose which theorem standard it wants:

```text
spectral standard:
  J is accepted when it is a canonical oriented spectral polynomial in U.

strict commutant standard:
  J is accepted only when the full compatible-J variety collapses to ±J.
```

The current bridge remains non-load-bearing under the strict standard.
Under the spectral standard, Floquet-α solves the algebraic `J` and coarse
center problems but still needs a source-backed microscopic reason why the
oriented spectral branch is mandatory. Under the strict standard, the next
candidate must make the rule-generated local algebra select only `±J`, without
using the forbidden lower-rank central projectors.

## Commuting Second-Layer Dilemma

The literal second-layer move has now been checked: add a mandatory real
orthogonal layer `V` with `[U,V]=0`, a 3-cycle on the alpha modes, and a swap
on the eta modes. It collapses the compatible centralizer from `26` to `10`,
but fails the bridge:

```text
generated_algebra_dimension = 10
center_dimension = 10
compatible_centralizer_dimension = 10
explicit_lower_rank_projector_ranks = [2, 2, 2]
no_locking_guardrail_passed = false
pass_strict_rule_to_bridge = false
```

The obstruction is structural. If a commuting real-orthogonal layer is
non-scalar inside a spectral block of `U`, its spectral projectors are
polynomials in `V`; because `[U,V]=0`, those projectors are central in the
rule-generated algebra. Therefore any such layer that actually breaks the
within-block centralizer also creates lower-rank central idempotents inside the
coarse `6+4` blocks. If `V` avoids those lower projectors, it is scalar on the
blocks and cannot reduce the compatible centralizer.

So the strict route cannot be "add any commuting semisimple lock." The next
candidate must break the compatible-`J` ambiguity without making the lock's
within-block spectral projectors available as rule-generated central
idempotents. The formal proposition and proof are recorded in
[Theory](../theory.md).

## Active Noncommuting Route

The next checked route is no longer a commuting lock. The new
`floquet_alpha_noncommuting` family adds a block-preserving signed orientation
twist `U2` with `[U1,U2] != 0`. It preserves the alpha/eta spectral projectors
as subspaces, so the coarse `6+4` split remains available, but the generated
rule algebra is noncommutative and the commuting-layer no-locking proposition
does not apply.

Representative result:

```text
generated_algebra_dimension = 22
center_dimension = 3
central_idempotent_ranks = [0, 4, 6, 10]
compatible_centralizer_dimension = 6
compatible_j_moduli_dimension = 0
compatible_complex_structure_count = 4
local_compatible_operator_dimension = 3
local_compatible_complex_structure_count = 0
forced_j_found = false
pass_strict_rule_to_bridge = false
route_label = coarse_center_preserved_compatible_j_not_rule_generated
```

This is progress but not a bridge witness. It escapes the structural
commuting no-go and avoids lower-rank central projectors, while reducing the
compatible `J` variety to finite signs. The remaining gap is exactly the
missing microscopic primitive: the compatible `J` is not yet certified as a
rule-generated local element.

The extracted compatible `J` list makes the gap concrete. The four finite
solutions are pair-orientation choices:

```text
(+,+,-,+,-)
(+,+,-,-,+)
(-,-,+,+,-)
(-,-,+,-,+)
```

All four commute with `U1` and `U2`; all four square to `-I` and are
orthogonal. However:

```text
compatible_j_in_generated_algebra_count = 0
compatible_j_in_rule_local_center_count = 0
spectral_polarization_j_matched_count = 0
reason_for_forced_j_failure = compatible_j_finite_but_not_generated_or_rule_local
```

Thus the next microscopic input cannot merely collapse the compatible
centralizer. It must also produce one of these local pair-orientation
structures, or produce a different noncommuting mechanism whose finite
compatible `J` is already in the rule algebra.
