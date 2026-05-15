# Theory

Spin(10) branching is textbook. The load-bearing question is whether QCA data
derive the real carrier, complex structure `J`, structural `3+2` split, and
safe gate algebra before any Spin(10) representation theory is invoked.

## J-First Bridge

The active theorem target is carrier-first:

```text
K_x ~= R^10
J in SO(K_x)
J^2 = -I
K_x = K_3 ⊕ K_2
dim_R K_3 = 6
dim_R K_2 = 4
J K_3 = K_3
J K_2 = K_2
```

This turns the real carrier into a complex space:

```text
W := (K_x, J) ~= C^5 = C^3 ⊕ C^2.
```

The proposed minimal algebraic ansatz is:

```text
K_x = R^2_clock ⊗ R^5_mode
J = epsilon ⊗ I_5
P_3 = I_2 ⊗ diag(1,1,1,0,0)
P_2 = I_2 ⊗ diag(0,0,0,1,1)
```

This ansatz is not a bridge proof by itself. It becomes meaningful only if QCA
rules force `J`, `P_3`, and `P_2`.

Phase 1 now verifies the exact identities for this ansatz. The check is
documented in [Real Carrier Report](literature/real_carrier_report.md).

Phase 2 verifies that a declared exact gate word can produce this `J`. The
check is documented in [Forced J Report](literature/forced_j_report.md). It
does not yet prove that microscopic QCA rule data force the word.

Phase 3 verifies that the declared `P_3/P_2` candidate is an exact
`J`-compatible projector split and that rank-one color or weak addressability
is falsifying. The check is documented in
[Projector Lattice Report](literature/projector_lattice_report.md). It does
not yet prove that microscopic QCA rule data force the split.

## Textbook Part

For a fixed complex vector-space split

```text
V = C^3 ⊕ C^2
```

the even exterior algebra has dimension:

```text
S_+ = Lambda^even(V)
dim S_+ = 16
```

Counting selected basis directions in each block gives occupation numbers
`N_3` and `N_2`. Solving

```text
Y(0,0)=0
Y(0,2)=1
Y(1,1)=1/6
```

for `Y = A + B N_3 + C N_2` gives:

```text
Y = -1/3 N_3 + 1/2 N_2
```

The implemented branching table is:

| `(N_3, N_2)` | Multiplicity | `Y` | Label |
| --- | ---: | ---: | --- |
| `(0, 0)` | 1 | `0` | `nu^c` |
| `(0, 2)` | 1 | `1` | `e^c` |
| `(1, 1)` | 6 | `1/6` | `Q` |
| `(2, 0)` | 3 | `-2/3` | `u^c` |
| `(2, 2)` | 3 | `1/3` | `d^c` |
| `(3, 1)` | 2 | `-1/2` | `L` |

This is a representation identity. It is not evidence that QCA geometry
supplies the split or `J`.

## No-Locking Gate Algebra

The geometric QCA gate algebra must lie in the Standard Model commutant.
On

```text
W = C^3 ⊕ C^2
```

the safe complex-linear one-particle endomorphisms are:

```text
End_{SU(3) x SU(2) x U(1)}(W) = C P_3 ⊕ C P_2.
```

So safe geometric gates are block scalars. The QCA may distinguish the whole
`3` block from the whole `2` block, but it may not resolve individual axes
inside either block.

Forbidden:

```text
Hom(C^3, C^2)
Hom(C^2, C^3)
rank-one color projectors
rank-one weak projectors
direction-conditioned internal projectors
```

Block diagonal is not enough. A rank-one projector inside `C^3` is
block-diagonal and still breaks `SU(3)`.

Phase 4 implements this as an exact classifier. The check is documented in
[Gate Classification Report](literature/gate_classification_report.md). It
proves the oracle behavior on canonical safe and unsafe gates, but it does not
prove that microscopic QCA rule data supply only safe geometric gates.

Phase 5 packages the period-four micromotion candidate as an exact finite-depth
update certificate. The check is documented in
[QCA Update Certificate](literature/qca_update_certificate.md). It verifies the
candidate identities `U(T/4)=J`, `U(T/2)=-I`, and `U(T)=I`, but it does not
prove that microscopic QCA rule data force the update.

Phase 6 reconstructs `Lambda^even(C^5)` using the prior `J` and `P_3/P_2`
candidates. The check is documented in
[Spinor 16 Report](literature/spinor16_report.md). It verifies the exact
16-state table and hypercharges, but it does not introduce a new complex
structure or a new `3+2` split.

Phase 7 computes normalizer and forcedness proxies for the candidate data. The
checks are documented in [Normalizer Report](literature/normalizer_report.md)
and [Forcedness Certificate](literature/forcedness_certificate.md). The current
normalizer preserves the declared `P_3/P_2` split, but it does not force the
candidate `J`, and no source-backed microscopic rule data are present.

Phase 8A starts the fallback route with a stronger real-QCA-first branch
checker. The check is documented in
[Real-QCA Branch Report](literature/real_qca_branch_report.md). It composes the
gate-word, structural split, and normalizer certificates for finite-depth real
gate-word rule spaces. The current default branch still generates only a
candidate `J` and split; it does not force them from microscopic data.

E1 starts actual bounded rule-space exploration. The report is
[Rule-Space Exploration Report](literature/rule_space_exploration_report.md).
The first sprint scans 170 exact words across 10 primitive sets and finds 73
`J`/period-four hits, but zero forced surviving bridge candidates.

E2 removes the seeded `P_3/P_2` controls from the default projector search.
The report is
[Unseeded Projector Discovery Report](literature/unseeded_projector_discovery_report.md).
The bounded unseeded pass scans 6 primitive sets and 1924 exact algebra
elements. It finds no complementary rank-`6+4` projector pair, while recording
three unsafe rank-2 projectors as falsifier pressure.

Together E1 and E2 close the current coarse primitive search class.

Proposition (E1/E2 coarse primitive obstruction). In the bounded primitive
class generated by global clocks, whole-block reflections, and within-block
mode permutations, a rule either preserves too much symmetry to derive a new
coarse `6+4` center, or the coarse `P_3/P_2` split has already entered as
generating data. If rank-one distinguishers are added, the no-locking
guardrail rejects the rule.

Proof. The E1/E2 searches exhaust the declared bounded primitive class. The
SO(6) x SO(4)-symmetric primitives do not produce an unseeded complementary
`6+4` central idempotent pair. The primitives that distinguish individual
axes produce lower-rank central idempotents, including unsafe rank-2
projectors, and are rejected by the no-locking checks. Therefore deeper search
inside this primitive class is not the next physics input.

The next physics input must therefore create `3` versus `2` asymmetry before
projector discovery, for example through Floquet polarization or defect
monodromy.

The collapsed rule-to-verdict checker is documented in
[Rule-To-Verdict Report](literature/rule_to_verdict_report.md). It evaluates
finite-depth layer data directly: Floquet spectrum, generated algebra center,
central idempotent lattice, rule-generated complex structures, and the single
bridge-candidate boolean. This is the main interface for new microscopic
candidate families.

The first replacement primitive family is Floquet-α, documented in
[Floquet Alpha Report](literature/floquet_alpha_report.md). It uses one
mandatory quantized resonance layer with phases `2 pi / 3` on three mode pairs
and `pi / 2` on two mode pairs. All ten resonance patterns produce the coarse
`[0,4,6,10]` central idempotent lattice without rank-one centers. Alpha-plus
extracts a canonical spectral-polarization `J` as a polynomial in the Floquet
operator for all ten patterns. The strict compatible-commutant equations still
do not force `J` uniquely: the compatible centralizer is
`M_3(C) plus M_2(C)`, real dimension `26`, and the compatible orthogonal
complex structures contain a 9-dimensional `U(3)/O(3) x U(2)/O(2)` family.

The theorem-standard decision is documented in
[Floquet Alpha J Obstruction](literature/floquet_alpha_j_obstruction.md). The
project must distinguish a canonical spectral/monodromy `J` from strict
uniqueness of the full compatible-`J` variety.
In the checker, strict forcedness is therefore a local moduli condition:
`local_compatible_j_moduli_dimension` must be `0` before finite candidate
matching can prove `forced_j_found`. The `local_compatible_*` fields search
the rule-generated local center, not all on-site local operators. For α this
rule-generated local center already contains four compatible `J` choices
rather than `±J`, so the rule does not single out a unique complex structure.
The Route-3 time-reversal sidecar adds a declared real involution `K` with
`KUK = U^-1` and `K J K = -J`. It is exact, but not generated by the Floquet
rule algebra. It reduces the estimated full compatible-`J` moduli dimension
from `9` to `3`, while the rule-local center still contains four `K`-compatible
`J` choices. Thus declared time reversal is useful pressure, not a strict
bridge witness.

The literal commuting second-layer repair has also been checked for Floquet-α:
a mandatory real-orthogonal layer that cycles the three alpha modes and swaps
the two eta modes does commute with `U` and reduces the compatible centralizer
to dimension `10`, but it generates rank-2 central projectors.

Proposition (commuting second-layer no-locking obstruction). Let
`U, V in O(10)` with `[U,V] = 0`, and let `P_alpha, P_eta` be the spectral
projectors of `U`. If `V` restricted to `P_alpha R^10` is non-scalar in the
compatible centralizer of `U` on that block, then `V` has a spectral projector
`Q` with `Q <= P_alpha` and `rank(Q) < 6`, and `Q` is central in the algebra
generated by `U` and `V`.

Proof. Since `[U,V] = 0`, the generated algebra `R[U,V]` is commutative.
The real spectral projectors of `V` are polynomials in `V`, by Bezout
decomposition of the real minimal polynomial of `V`. Hence those projectors
lie in `R[U,V]`, and because `R[U,V]` is commutative they are central in the
generated algebra. If `V` is non-scalar on the alpha block, its restriction has
at least two distinct real-irreducible spectral components. The projector onto
one such component satisfies `Q <= P_alpha` and has rank strictly smaller than
`rank(P_alpha) = 6`.

Corollary. No commuting real-orthogonal second layer that breaks the
within-block compatible-`J` ambiguity preserves the no-locking guardrail. If
the layer is scalar on each coarse block it cannot collapse the compatible
centralizer; if it is non-scalar, the proposition produces a lower-rank
central idempotent inside a coarse block.

The active escape from this proposition is a noncommuting, block-preserving
second layer. The first exact representative is the Floquet-alpha signed twist:
`U2` is a signed alpha 3-cycle plus eta swap with hidden orientation signs.
It satisfies `[U1,U2] != 0` while preserving `P_alpha` and `P_eta`, so the
generated algebra is noncommutative and the commuting no-locking proof does
not apply. The checker reports central idempotent ranks `[0,4,6,10]`, no
lower-rank central idempotents, compatible centralizer dimension `6`, and a
zero-dimensional compatible-`J` set. It still fails the strict bridge because
the compatible `J` is not certified as rule-generated/local. This is the
current finite-dimensional laboratory for the missing microscopic primitive.
The extracted finite `J` candidates are exactly four pair-orientation sign
patterns, `(+,+,-,+,-)`, `(+,+,-,-,+)`, `(-,-,+,+,-)`, and `(-,-,+,-,+)`.
None lies in the generated algebra or in the rule-local center, and none
matches the spectral-polarization `J`. The remaining theorem target is
therefore not centralizer collapse; it is microscopic production of the local
orientation structure.
The minimal completion experiment declares one such orientation as a third
layer. That does not generate lower-rank central idempotents, but it still
leaves four rule-local compatible complex structures. The residual obstruction
is the independent alpha/eta block sign: the strict bridge needs a mechanism
that reduces those four choices to global `±J`.

Proposition (implemented route obstruction). In the implemented finite-route
families checked so far, the strict bridge failure has been reduced to
rule-generation/locality of `J`. More precisely:

1. The time-reversal sidecar reduces the full compatible-`J` moduli estimate
   from `9` to `3`, but `K` is declared rather than generated and the
   compatible-`J` family is still positive-dimensional.
2. The noncommuting signed twist preserves the coarse `[0,4,6,10]` central
   idempotent lattice, creates no lower-rank central idempotents, and reduces
   compatible `J` to a finite four-element set, but none of those four `J`
   candidates lies in the generated algebra or rule-local center.
3. The completion experiment declares one finite compatible `J` as a third
   layer and still creates no lower-rank central idempotents, but the
   rule-local center continues to admit four compatible `J` choices rather
   than one global `±J` orbit.

Therefore the current obstruction is not the coarse `6+4` center and, in the
noncommuting route, not continuous compatible-`J` moduli. The missing
microscopic primitive must produce the compatible orientation as rule-local
data and must reduce the remaining four block-sign choices to global `±J`.

Proof. Each statement is the exact output of the implemented route diagnostics:
`scripts/floquet_alpha.py --variant time-reversal`,
`scripts/floquet_alpha.py --variant noncommuting`,
`scripts/floquet_alpha.py --variant noncommuting-gap`, and
`scripts/floquet_alpha.py --variant noncommuting-completion`.

Route 2 is now represented by a spatial 1D sidecar. It does not alter the
finite-depth on-site verdict checker. Instead it tests the remaining sign
obstruction with a finite-radius local QCA layer whose Laurent symbol is
`T(z) = P_alpha z^4 + P_eta z^3`. The exact Laurent coefficient identities
prove unitarity as a locality-preserving real QCA, and the coefficient algebra
has central idempotent ranks `[0,4,6,10]` with no lower central locking. Under
the sidecar's shared spatial-orientation transport constraint, the four
alpha/eta block-sign choices reduce to the two global choices `(+,+)` and
`(-,-)`. This has the right theorem shape for the missing topological
mechanism, but it is not yet load-bearing because the coarse projectors appear
as input coefficients rather than being generated microscopically.
The first unseeded Route-2 scan makes that boundary explicit: block-blind
finite-radius layers do not produce a bridge candidate, and the working
projector-shift layer is rejected by the seeded-coefficient guardrail.

The second physical family is Defect-β, documented in
[Defect Beta Report](literature/defect_beta_report.md). It computes
round-trip monodromy from two distinct orientation-reversing wall-cycle
transition functions. All ten defect charge patterns produce the same coarse
center and canonical monodromy `J`, but that monodromy is exactly the matching
Floquet-α operator. The checker now evaluates β on the actual noncommuting
transition pair `(T_entry, T_exit)`, which has generated algebra dimension `8`
and compatible centralizer dimension `13`, but still does not force `J`.
It is parked as a load-bearing route until rebuilt as a genuine
higher-dimensional defect calculation.

## Load-Bearing Conditions

The bridge becomes nontrivial only if QCA data supplies all of the following:

1. A real ten-dimensional local carrier or ten Majoranas.
2. A local finite-depth or micromotion-derived `J` with `J^2 = -I`.
3. A structural `J`-invariant `6+4` real split.
4. Projectors `P_3,P_2` but no smaller rank-one addressability.
5. A geometric gate algebra contained in `C P_3 ⊕ C P_2`.
6. Only then, the standard `Lambda^even(C^5)` spinor reconstruction.

Until those conditions are met, the verdict remains `notation_only`.
