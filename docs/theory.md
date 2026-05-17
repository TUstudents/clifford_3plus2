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

Proposition 1 (E1/E2 coarse primitive obstruction). In the bounded primitive
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

Proposition 2 (commuting second-layer no-locking obstruction). Let
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

Corollary 2.1. No commuting real-orthogonal second layer that breaks the
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
Equivalently, the obstruction is the centralizer gap. For this representative,
the joint compatible centralizer has dimension `6`, while
`A_R ∩ Cent_R = Z(A_R)` has dimension `3`. The four compatible `J` candidates
are finite points in `Cent_R` outside `A_R` and outside the rule-local center.
The next primitive must therefore add rule-local access to an operator in this
3-dimensional quotient `Cent_R / Z(A_R)`.
The minimal completion experiment declares one such orientation as a third
layer. That does not generate lower-rank central idempotents, but it still
leaves four rule-local compatible complex structures. The residual obstruction
is the independent alpha/eta block sign: the strict bridge needs a mechanism
that reduces those four choices to global `±J`.

Proposition 3 (Floquet-alpha block-preserving noncommuting locking
obstruction). Let `U1, U2 in O(10)` be block-preserving real-orthogonal
layers, `[Ui, P_alpha] = [Ui, P_eta] = 0`, and assume `[U1,U2] != 0`.
Assume also that `U1` is Floquet-alpha type: after passing to the natural
spectral complex structure, `U1|_{P_alpha}` is one complex scalar on
`C^3` and `U1|_{P_eta}` is one complex scalar on `C^2` (equivalently, one
irreducible real quadratic spectral factor per coarse block). Let
`A = R<U1,U2>`. If `A` contains a central complex structure `J in Z(A)` with
`J^2 = -I`, then at least one of the following holds:

1. `A` contains a central idempotent strictly inside `P_alpha` or `P_eta`,
   so the no-locking guardrail fails.
2. On a coarse block, `A` acts as the full internal addressability algebra,
   for example `A|_{P_alpha} = M_3(C)` or `A|_{P_eta} = M_2(C)`, so the
   geometric gate algebra is larger than `C P_alpha ⊕ C P_eta`.
3. The supposedly noncommuting layers are block scalars and commute on the
   coarse blocks, contradicting the intended noncommuting escape.

In particular, no Floquet-alpha-type block-preserving on-site rule on a single
ten-dimensional carrier can satisfy both a rule-generated central compatible
`J` and the no-locking guardrail while keeping the geometric gate algebra in
the Standard Model commutant.

Proof. Since `J in Z(A)` and `J^2 = -I`, restriction to the alpha block gives
`J_alpha in Z(A|_{P_alpha})` with `J_alpha^2 = -P_alpha`; hence
`Z(A|_{P_alpha})` contains a copy of `C`. By Wedderburn, the finite-dimensional
real algebra `A|_{P_alpha}` decomposes into central simple summands
`⊕_i M_{n_i}(D_i)`. In the block-preserving signed-twist class under study,
acting faithfully on the real six-dimensional space `P_alpha R^10 ~= C^3`
leaves only two admissible shapes if no lower central idempotent is allowed.

If there is more than one summand, the center contains a nontrivial central
idempotent below `P_alpha`, with rank strictly between `0` and `6`. This is
case 1. If there is one summand, the center containing `C` forces a complex
central simple action. On `C^3`, the options are scalar `C` or full
`M_3(C)`: the scalar case makes `U2|_{P_alpha}` a complex block scalar, while
the full case is exactly internal addressability on the alpha block. The same
argument applies on the eta block, with scalar `C` or `M_2(C)`. If both blocks
are scalar, the layers commute blockwise, giving case 3. If either block is
full matrix algebra, the SM commutant condition fails, giving case 2. Thus any
attempt to make `J` central and rule-generated in a block-preserving on-site
noncommuting rule falls into central locking, full addressability, or
commutativity.

The exhaustive discrete signed-twist run is the implemented finite witness for
this proposition's pressure point. It scans `3840` block-preserving twists,
reduces them to `96` exact symmetry classes, and finds `720`
generated-compatible-`J` hits, including `240` minimal four-`J` hits. None of
those hits occurs in the no-locking shape:

```text
no_locking_shape_candidates = 240
no_locking_shape_j_in_generated_algebra_candidates = 0
bridge_candidate_count = 0
```

Corollary 3.1 (on-site bridge closure). On a single ten-dimensional on-site
carrier, the Floquet-alpha block-preserving class cannot simultaneously
satisfy all three strict bridge requirements:

1. a coarse rank-`(6,4)` center with no lower-rank refinement;
2. a geometric gate algebra contained in `C P_alpha ⊕ C P_eta`;
3. a compatible complex structure `J` generated by the rule, unique up to
   global `±J` gauge.

Proof. The E1/E2 obstruction closes the coarse symmetric primitive class. The
commuting second-layer proposition closes commuting non-scalar locks, because
they generate lower-rank central projectors. The block-preserving
noncommuting proposition above closes the Floquet-alpha-type on-site
noncommuting escape: a rule-generated central `J` forces lower central
locking, full internal addressability, or effective commutativity. This closes
the on-site block-preserving classes actually used by the Floquet-alpha search
and exhaustive signed-twist witness; it does not claim to classify arbitrary
block-preserving on-site rules outside the one-quadratic-factor-per-block
hypothesis.

Proposition 5 (Route-1 SM-inequivalence of compatible `J` orbits). The four
compatible orthogonal complex structures of the Route-1 noncommuting
signed-twist algebra are not one Standard-Model-equivalence class under the
fixed `SU(3) x SU(2) x U(1)` data. They split into two global `±J` classes
under independent alpha/eta block-sign flips. The eta-block flip sends
`N_2 -> 2 - N_2` and does not preserve the fixed hypercharge table
`Y = -N_3/3 + N_2/2`. The alpha-block flip sends `N_3 -> 3 - N_3` and changes
even chirality. Therefore the Route-1 compatible `J` set cannot be accepted as
a single gauge-equivalence class, and the strict global `{±J}` standard cannot
be relaxed for this family.

Proof. The four Route-1 compatible pair-orientation signs are
`(+,+,-,+,-)`, `(+,+,-,-,+)`, `(-,-,+,+,-)`, and `(-,-,+,-,+)`. Modulo the
global replacement `J -> -J`, these form two classes: the block-aligned class
`(alpha, eta) = (1,1)` and the non-global class `(alpha, eta) = (1,-1)`. If
each `J` is allowed to relabel its own holomorphic directions, the textbook
`Lambda^even(C^3 plus C^2)` branching table is recovered in every case; that
is only intrinsic bookkeeping. With the already-fixed Standard Model
identification, the eta flip applies the Hodge complement on the weak block,
so a sector with occupation `N_2` is mapped to one with occupation `2 - N_2`.
The fixed hypercharge changes from `-N_3/3 + N_2/2` to
`-N_3/3 + 1 - N_2/2`, which is not the same table. The alpha flip applies a
Hodge complement on a three-dimensional block and reverses even/odd chirality.
The implemented certificate
`uv run python scripts/gauge_equivalence_check.py --check` records
`compatible_j_count = 4`, `global_pm_orbit_count = 2`,
`intrinsic_branching_tables_match = true`, and
`fixed_sm_branching_tables_match_mod_global_pm = false`. No rule-generated
normalizer orbit connecting the two global classes is certified. Hence the
fixed-`SU(3) x SU(2) x U(1)` gauge-equivalence relaxation fails for Route 1.
The load-bearing negative is the hypercharge mismatch. A broader relaxation
using a microscopic charge-conjugation or outer Spin(10) automorphism is not
checked here and would require a separate rule-generated normalizer
certificate.

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
The combined Route-1/Route-2 sidecar tests the direct synthesis
`T(z) = (U2 U1)(P_alpha z^4 + P_eta z^3)`. It reaches the topological shape:
the Route-1 finite four-`J` set is reduced by spatial transport to two global
signs, and those two signs commute with the sampled Bloch symbols. It still
does not satisfy strict forcedness because the joint rule algebra generates
zero transported `J` elements.
The first unseeded Route-2 scan makes that boundary explicit: block-blind
finite-radius layers do not produce a bridge candidate, and the working
projector-shift layer is rejected by the seeded-coefficient guardrail.
The Bloch Path-A checker is the first version of the correct off-site verdict
shape. It evaluates sampled root-of-unity Bloch symbols and rejects candidates
whose coefficient algebra already generates `P_alpha` or `P_eta`. Its mixed
candidate panel still reports seeded topological shape only, while its default
projector-free headline now reaches the dim-34 joint-algebra result. The old
cap-16 projector-free mode is retained only as a regression check.
The main `rule_to_verdict` checker now has the same Bloch-period mode:
with `bloch_period = n`, it uses the joint algebra
`R<T(zeta^0),...,T(zeta^(n-1))>` as the generated algebra. The stepwise
projector-free combined Route-1/Route-2 candidates replace
`P_alpha z^4 + P_eta z^3` with partial monomial hops carrying source-mode
shifts and on-site update `U2 U1`; their raw coefficients are not the coarse
projectors. The completed census covers all `10 * 24 * 10 = 2400`
projector-free monomial-hop candidates: ten Floquet-alpha patterns, twenty-four
five-cycles, and ten source-shift assignments. The only center-dimension-`4`
survivors occur at generated-algebra dimensions `26` and `34`.

Proposition 4a (projector-free monomial-hop Bloch obstruction, finite census).
Let `T(z)` range over the full projector-free Path-A monomial-hop family
generated by `scripts/bloch_path_a_stepwise.py` with `pattern_count = 10`,
`cycle_count = 24`, `shift_count = 10`, sampled period `12`, and joint algebra
`A = R<T(zeta^0),...,T(zeta^11)>`. Then the finite census has:

1. `2400` total candidates, all closing below the cap `48`;
2. `1320` candidates with `dim Z(A) = 4`;
3. `360` dim-`26` / center-`4` candidates with central idempotent ranks
   `[0,2,8,10]`, hence a lower-rank central refinement and no-locking failure;
4. `960` dim-`34` / center-`4` candidates with central idempotent ranks
   `[0,4,6,10]`;
5. for all `960` dim-`34` candidates, the rule-generated local-center `J`
   count is `0`, so no candidate supplies a strict rule-generated `J`;
6. after the structural filters, zero timeout candidates remain.

Consequently the full projector-free monomial-hop Bloch Path-A family is closed
as a strict bridge route. The dim-`26` branch fails before any `J` question by
rank-2 locking. The dim-`34` branch produces the coarse center but never
produces the microscopic complex structure in the rule-generated local center.
This is stronger than the original six-candidate witness and sharper than the
on-site Route-1 obstruction: Route 1 had compatible `J`s outside the rule
algebra; the monomial-hop census never produces a rule-local `J` on the coarse
branch.

Proof. This is an exact finite calculation, not a sampled numerical claim.
The Laurent symbol is sampled at the twelfth roots only to form the joint
Bloch algebra. Algebra closure is exact over SymPy expressions. The center is
computed from the exact algebra multiplication table or from exact
commutators with a reduced generator set. Central idempotent rank profiles are
certified by exact center-algebra equations plus structural certificates for
the dim-`26` and dim-`34` monomial-hop geometries. The dim-`26` structural
certificate identifies a rank-`2`/rank-`8` central split. The dim-`34`
certificate identifies the coarse rank-`(6,4)` split and, in generated-`J`
diagnostics, records the solved-empty local-center `J` result.

The earlier projected-centralizer witness remains a useful diagnostic subset:
the first six dim-`34` candidates also have compatible centralizer dimension
`4`, generated and compatible `J` counts both equal to `0`, and projected
centralizers `rank6: R^3`, `rank4: R`. That split-real shape motivates
Conditional Lemma 4b.1, but Proposition 4a itself now rests on the full finite
classification:

```bash
uv run python scripts/bloch_path_a_stepwise.py \
  --family monomial-hop \
  --pattern-count 10 --cycle-count 24 --shift-count 10 \
  --target-cache-file data/scans/bloch_path_a_monomial_center_v5.jsonl \
  --target-center-dim 4 --target-generated-dim 26 \
  --max-algebra-dim 48 --idempotents --j-solve \
  --generated-j-only --coarse-only-diagnostics \
  --jobs 8 --chunk-size 8 --timeout-seconds 60 --check

uv run python scripts/bloch_path_a_stepwise.py \
  --family monomial-hop \
  --pattern-count 10 --cycle-count 24 --shift-count 10 \
  --target-cache-file data/scans/bloch_path_a_monomial_center_dim34_timeout_targets_v5.jsonl \
  --target-center-dim 4 --target-generated-dim 34 \
  --max-algebra-dim 48 --idempotents --j-solve \
  --generated-j-only --coarse-only-diagnostics \
  --jobs 8 --chunk-size 8 --timeout-seconds 60 --check
```

The committed artifacts are
`data/scans/bloch_path_a_monomial_center_dim26_rank_filter_v7.jsonl`,
`data/scans/bloch_path_a_monomial_center_dim34_timeout_filter_v6.jsonl`, and
`data/scans/bloch_path_a_monomial_center_dim4_bridge_fast_v5.jsonl`.

Conjectural Proposition 4b (coprime monomial-hop incompatibility). Let
`T(z) = sum_s M_s z^s` be a projector-free 1D Bloch rule on `R^10` whose
coefficients `M_s` are partial real-orthogonal monomial hops. Suppose the
effective coarse blocks, if produced by the joint sampled algebra, carry
constant per-block winding numbers `w_alpha` on the rank-6 block and `w_eta`
on the rank-4 block, with `gcd(w_alpha, w_eta) = 1`, and suppose the
coefficient algebra does not already generate `P_alpha` or `P_eta`. If the
joint sampled algebra `A = R<T(zeta^0),...,T(zeta^(L-1))>` for
`L = lcm(w_alpha, w_eta)` produces the coarse rank-`(6,4)` center, then its
compatible centralizer should be commutative on each coarse spectral block and
should admit no real orthogonal complex structure.

Equivalently, the natural projector-free monomial-hop class can force the
coarse center but not the microscopic `J`. The obstruction is stronger than
the Route-1 centralizer gap: in Route 1 the compatible `J` set is finite and
non-empty but disjoint from the rule algebra; here the compatible `J` set is
empty already in the joint centralizer.

Proof status. The tempting proof is to say that, on each momentum slice, the
monomial hops only contribute root-of-unity phases
`zeta^(j w_alpha)` and `zeta^(j w_eta)` on the two effective blocks. Under that
extra block-scalar hypothesis, the restricted joint algebra is a small
commutative real algebra. Such an algebra cannot contain an orthogonal
operator `J_block` with `J_block^2 = -P_block` unless its image already contains
a `C` factor acting through a fixed complex structure on the whole block.

That argument is not yet a proof of the broader coprime Path-A class. The
implemented finite monomial-hop family is closed by Proposition 4a, but its
coefficients are `U_local E_s`, where `U_local` is the Route-1 noncommuting
on-site update and `E_s` is a mode-edge matrix. In a structural theorem beyond
the finite census, mode cycles may cross the eventual coarse blocks, and
`P_alpha/P_eta` are outputs of the joint sampled algebra rather than inputs
available before the proof. Therefore one cannot assume, without an additional
invariant, that the projected algebra `P A P` is generated only by block-scalar
root-of-unity phases.

The missing invariant for promoting 4b from conjecture to theorem is:
whenever a projector-free coprime monomial-hop rule produces the coarse
rank-`(6,4)` center without seeding `P_alpha/P_eta`, each projected compatible
centralizer is a split real algebra, or equivalently contains no `C` factor
acting orthogonally on the full coarse block. Proving this invariant would
close the coprime monomial-hop class. Finding a coprime monomial-hop rule whose
projected compatible centralizer contains such a `C` factor would refute 4b
and become the next bridge candidate.

Conditional Lemma 4b.1 (split-real projected centralizer obstruction). Let
`P` be one of the produced coarse central idempotents, and let `B_P` be the
image of the compatible centralizer on `P R^10`. If `B_P` is a split real
commutative semisimple algebra, `B_P ~= R^m`, then `B_P` contains no real
orthogonal complex structure on `P R^10`.

Proof. In the algebra `R^m`, multiplication is coordinatewise. If
`J in B_P` satisfied `J^2 = -P`, then each character coordinate `lambda_i(J)`
would satisfy `lambda_i(J)^2 = -1` in `R`. No real number has square `-1`.
Thus no such `J` exists in `B_P`; orthogonality is an additional condition and
cannot restore a missing square root of `-P`.

The negative `J` result is a structural fact about the monomial-hop class,
not a reason to abandon spatial constructions broadly. The seeded transfer
`T(z) = P_alpha z^4 + P_eta z^3` is the effective rule the projector-free
monomial-hop class cannot reach. An unseeded Route-2 search outside the
present obstruction classification would include finite-order on-site
primitives, noncommuting orientation twists, and translation-protected
real-orthogonal hopping generators, with an algebraic seed guardrail: a
candidate is seeded if its coefficient algebra generates `P_alpha` or
`P_eta`, even when no raw coefficient is visibly a diagonal projector.

The second physical family is Defect-β, documented in
[Defect Beta Report](literature/defect_beta_report.md). It computes
round-trip monodromy from two distinct orientation-reversing wall-cycle
transition functions. All ten defect charge patterns produce the same coarse
center and canonical monodromy `J`, but that monodromy is exactly the matching
Floquet-α operator. The checker evaluates β on the actual noncommuting
transition pair `(T_entry, T_exit)`, which has generated algebra dimension `8`
and compatible centralizer dimension `13`, and still does not force `J`.
Defect-β is therefore archived as a tested notation-only sidecar: its bounded
form does not provide an independent bridge mechanism beyond Floquet-α. A
genuine non-translation-invariant wall/defect QCA with different alpha patterns
on the two sides and rule-generated transition data is a possible future
direction outside the present obstruction classification.

## Write-Up Readiness

The project meets the conditions for the obstruction write-up:

1. Propositions 1, 2, 3, 4a, and 5 have independently readable proofs with
   hypotheses no broader than the implemented or proved class.
2. Every empirical witness has one documented `uv run ... --check` command at
   the publication git revision (see
   [Reproduction Index](results/reproduction.md)).
3. Claims are tagged by status: structurally closed for Propositions 1-3,
   empirically witnessed for 4a, conjectural for 4b, and theorem-standard
   decision for 5.
4. The publication plan is recorded in
   [Publication Plan](PUBLICATION_PLAN.md) with section names, theorem
   placement, and the open mechanisms named in Section 8:
   translation-symmetry-breaking defects, parameterized rule families,
   higher-dimensional carriers, and the open Proposition 4b structural
   extension.

## Load-Bearing Conditions

The bridge becomes nontrivial only if QCA data supplies all of the following:

1. A real ten-dimensional local carrier or ten Majoranas.
2. A local finite-depth or micromotion-derived `J` with `J^2 = -I`.
3. A structural `J`-invariant `6+4` real split.
4. Projectors `P_3,P_2` but no smaller rank-one addressability.
5. A geometric gate algebra contained in `C P_3 ⊕ C P_2`.
6. Only then, the standard `Lambda^even(C^5)` spinor reconstruction.

Until those conditions are met, the verdict remains `notation_only`.
