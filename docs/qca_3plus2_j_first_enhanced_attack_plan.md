# Enhanced J-First QCA Attack Plan For The `3+2 -> D5 / Spin(10)` Bridge

**Document type:** enhanced main attack plan
**Project:** real QCA derivation of a one-generation `Spin(10)` carrier
**Status:** theorem plan, falsifier plan, computational implementation plan
**Tooling:** `uv`, Python, SymPy, NumPy/SciPy, NetworkX, pytest, Hypothesis, optional SMT/numerical search
**Date:** 2026-05-13

---

## 0. Executive Verdict

The bridge must be rebuilt around the missing object:

```text
a microscopic QCA-derived real complex structure J with J^2 = -I.
```

The correct target is not:

```text
find five anticommuting complex matrices.
```

The correct target is:

```text
real finite-depth QCA data
  -> forced local J on R^10
  -> structural J-invariant 6+4 real split
  -> C^5 = C^3 ⊕ C^2
  -> geometric gate algebra in the SM commutant
  -> Λ^even(C^5) = Spin(10) chiral 16.
```

The enhanced plan has one central rule:

```text
Do not allow the QCA to resolve individual color axes or individual weak axes.
```

A structural `3+2` split is not enough. The microscopic update must preserve only the whole `3` block and the whole `2` block. If it contains rank-one projectors inside either block, the bridge fails.

---

# 1. Non-Negotiable Bridge Conditions

The bridge is accepted only if all of the following are produced before invoking `Spin(10)` representation theory.

## 1.1 Real Carrier

There must be a real local carrier

```text
K_x ≅ R^10
```

or equivalently ten local Majorana generators

```text
γ_1, ..., γ_10,
{γ_a, γ_b} = 2 δ_ab.
```

The carrier must be real before complex notation is introduced.

---

## 1.2 QCA-Generated Complex Structure

The QCA data must generate

```text
J: K_x -> K_x
J^2 = -I
J^T J = I.
```

Accepted origins:

```text
J = finite-depth local gate word
J = quarter-period micromotion
J = forced monodromy around a microscopic defect
J = canonical real module structure forced by the QCA rules
```

Rejected origins:

```text
J chosen by hand
J copied from ambient scalar i
J selected because it gives SU(5)
J introduced only after color and weak labels are named
```

---

## 1.3 Structural `3+2` Split

There must be real projectors

```text
P_3, P_2 ∈ End_R(K_x)
```

satisfying

```text
P_3 + P_2 = I
P_3 P_2 = 0
rank_R(P_3) = 6
rank_R(P_2) = 4
[J, P_3] = [J, P_2] = 0.
```

Then

```text
W := (K_x, J) ≅ C^5
W = W_3 ⊕ W_2
dim_C W_3 = 3
dim_C W_2 = 2.
```

The split must be structural. It cannot be a post-hoc relabeling of five arbitrary complex modes.

---

## 1.4 No Within-Block Addressability

The QCA may distinguish the whole `3` block from the whole `2` block.

It may not distinguish:

```text
individual axes inside C^3,
individual axes inside C^2.
```

Allowed structural projectors:

```text
P_3
P_2
```

Forbidden projectors:

```text
|color 1><color 1|
|color 2><color 2|
|color 3><color 3|
|weak 1><weak 1|
|weak 2><weak 2|
```

This is the central obstruction test.

---

## 1.5 Geometric Gate Algebra Must Lie In The SM Commutant

On

```text
W = C^3 ⊕ C^2,
```

with

```text
SU(3) acting on C^3,
SU(2) acting on C^2,
Y = -1/3 P_3 + 1/2 P_2,
```

the safe complex-linear one-particle endomorphisms are

```text
End_{SU(3) x SU(2) x U(1)}(W)
= C P_3 ⊕ C P_2.
```

Therefore the safe anti-Hermitian real orthogonal generators are only

```text
R · J P_3 ⊕ R · J P_2.
```

Passing geometric gates are block scalars:

```text
λ_3 I_3 ⊕ λ_2 I_2.
```

Failing geometric gates include:

```text
Hom(C^3, C^2)
Hom(C^2, C^3)
rank-one color projectors
rank-one weak projectors
direction-conditioned internal projectors
BCC-body-diagonal D5 Cartan projectors
```

---

# 2. Enhanced Theorem Target

The theorem should be stated in carrier-first form.

## Theorem Target

Let `K_x` be a real ten-dimensional local QCA carrier. Suppose the microscopic finite-depth QCA data determine:

```text
J ∈ SO(K_x),
J^2 = -I,
```

where `J` is produced by a local gate word, quarter-period micromotion, or forced microscopic monodromy.

Suppose also that the same QCA data determine a structural decomposition

```text
K_x = K_3 ⊕ K_2
```

with

```text
dim_R K_3 = 6,
dim_R K_2 = 4,
J K_3 = K_3,
J K_2 = K_2.
```

Then

```text
W := (K_x, J) ≅ C^5
W = W_3 ⊕ W_2
dim_C W_3 = 3
dim_C W_2 = 2.
```

Assume the geometric/local QCA gate algebra satisfies

```text
A_geom ⊂ End_{SU(3) x SU(2) x U(1)}(W)
       = C P_3 ⊕ C P_2.
```

Then

```text
S^+ := Λ^even(W)
```

is the positive-chirality `Spin(10)` spinor representation, and the operator

```text
Y = -1/3 N_3 + 1/2 N_2
```

reproduces the one-generation Standard Model hypercharge pattern under the usual

```text
Spin(10) -> SU(5) -> SU(3) x SU(2) x U(1)
```

branching.

---

## What The Theorem Does Not Claim

Even if the bridge succeeds, it does not by itself prove:

```text
three generations,
mirror decoupling,
continuum gauge dynamics,
Higgs realism,
Yukawa structure,
phenomenology,
or uniqueness of Spin(10).
```

Those remain separate problems.

---

# 3. Core Algebraic Convention

Use the real basis

```text
K_x = R^2_clock ⊗ R^5_mode.
```

Order the basis as

```text
x_1, x_2, x_3, x_4, x_5,
y_1, y_2, y_3, y_4, y_5.
```

Define

```text
ε =
[ 0  -1
  1   0 ].
```

The canonical candidate complex structure is

```text
J = ε ⊗ I_5
  =
[ 0   -I_5
  I_5  0  ].
```

Then

```text
J^2 = -I_10.
```

The structural split projectors are

```text
π_3 = diag(1,1,1,0,0)
π_2 = diag(0,0,0,1,1)
```

and

```text
P_3 = I_2 ⊗ π_3
P_2 = I_2 ⊗ π_2.
```

They satisfy

```text
P_3 + P_2 = I
P_3 P_2 = 0
[J, P_3] = [J, P_2] = 0.
```

This algebraic model is only an ansatz. It is accepted only if the QCA rules force `J`, `P_3`, and `P_2`.

---

# 4. Central Falsifier

The first fatal obstruction is direction-locking.

Suppose the QCA shift has the form

```text
U_shift =
S_1 ⊗ |1><1|
+ S_2 ⊗ |2><2|
+ S_3 ⊗ |3><3|.
```

Let `E_12` be an `SU(3)` ladder generator. Then

```text
[U_shift, I ⊗ E_12]
=
(S_1 - S_2) ⊗ E_12.
```

Therefore the shift commutes with color only if

```text
S_1 = S_2.
```

Similarly, all color directions require

```text
S_1 = S_2 = S_3.
```

So any QCA that sends internal mode `1` along spatial direction `1`, internal mode `2` along spatial direction `2`, and internal mode `3` along spatial direction `3` breaks the would-be `SU(3)` color symmetry.

The same applies to the weak block:

```text
S_4 ⊗ |4><4| + S_5 ⊗ |5><5|
```

breaks `SU(2)` unless

```text
S_4 = S_5.
```

Therefore:

```text
The QCA may produce a rank-3 block and a rank-2 block,
but it may not use individual basis vectors inside either block as
directional controls.
```

---

# 5. Phase 0 — Toolchain, Repository, And Exact Arithmetic

## Goal

Create a reproducible computational environment for exact matrix algebra, symbolic commutant calculations, QCA layer certificates, and falsifier tests.

Use `uv` for the project environment. `uv` supports project initialization, dependency management, locking, syncing, and running commands inside the project environment; the official docs list `uv init`, `uv add`, `uv sync`, `uv lock`, and `uv run` for this workflow. ([Astral Docs][1])

## Initial Commands

```bash
uv init qca-jfirst --lib
cd qca-jfirst

uv python install 3.12
uv add sympy numpy scipy networkx pytest hypothesis pydantic rich matplotlib jupyter
uv sync
```

Run commands through the project environment with:

```bash
uv run python -m qca_jfirst.scripts.check_candidate
uv run pytest
```

The official `uv run` documentation says it runs commands in the project environment and ensures the environment is up to date before execution. ([Astral Docs][2])

## Repository Layout

```text
qca-jfirst/
  pyproject.toml
  uv.lock

  src/qca_jfirst/
    __init__.py

    algebra/
      realification.py
      matrices.py
      lie.py
      commutants.py
      projectors.py
      exterior.py

    qca/
      carrier.py
      gates.py
      layers.py
      locality.py
      gate_words.py
      certificates.py

    sm/
      embedding.py
      hypercharge.py
      spinor16.py
      classifications.py

    search/
      enumerate_gate_words.py
      solve_complex_structures.py
      normalizer_tests.py
      falsifiers.py

    scripts/
      check_minimal_ansatz.py
      check_commutant.py
      build_spinor_table.py
      scan_gate_words.py

  tests/
    test_real_carrier.py
    test_j_structure.py
    test_projectors.py
    test_commutant.py
    test_gate_classification.py
    test_spinor16.py
    test_qca_layers.py
    test_no_locking.py

  notebooks/
    00_minimal_ansatz.ipynb
    01_commutant_tests.ipynb
    02_gate_word_search.ipynb
    03_spinor_table.ipynb

  docs/
    theorem_plan.md
    falsifier_log.md
    certificates.md
```

## Library Roles

Use libraries as follows:

```text
SymPy:
  exact matrices, rational arithmetic, commutants, nullspaces, symbolic tests

NumPy/SciPy:
  numerical scouting only, never final proof

NetworkX:
  local graph, finite-depth layer support, locality-radius checks

pytest:
  deterministic acceptance/falsifier tests

Hypothesis:
  randomized stress tests for generated gate words and projector classifications

Pydantic:
  structured candidate/certificate schemas

Rich:
  readable terminal reports

Matplotlib:
  optional visualization of graph/layer structure
```

SymPy’s matrix module provides exact matrix operations including nullspace, eigenvalues, and related linear algebra functionality; the nullspace method returns a basis for the nullspace, which is exactly what the commutant calculations need. ([SymPy Documentation][3])

---

# 6. Phase 1 — Real Carrier Construction

## Goal

Define the real local carrier and the primitive real orthogonal gates before introducing complex notation.

## Carrier

Use

```text
K_x = R^10.
```

Preferred explicit model:

```text
K_x = R^2_clock ⊗ R^5_mode.
```

Basis:

```text
x_1, x_2, x_3, x_4, x_5,
y_1, y_2, y_3, y_4, y_5.
```

## Majorana Equivalent

Equivalently use ten Majoranas:

```text
γ_{x1}, ..., γ_{x5}, γ_{y1}, ..., γ_{y5}.
```

One-particle real orthogonal transformations act by

```text
γ_a -> Σ_b O_{ba} γ_b,
O ∈ SO(10).
```

## Primitive Generators

Define real antisymmetric generators

```text
A_{ab} = E_{ab} - E_{ba}.
```

Then

```text
exp(θ A_{ab}) ∈ SO(10).
```

For exact quarter-turns, avoid numerical exponentials. Use explicit symbolic matrices.

For a pair `(x_a, y_a)`:

```text
J_a x_a = y_a
J_a y_a = -x_a.
```

But individual `J_a` gates are dangerous. If the microscopic rule allows each `J_a` independently, then the model has rank-one addressability and fails.

## Acceptance Criteria

Phase 1 passes only if the carrier is genuinely real and ten-dimensional:

```text
dim_R K_x = 10.
```

It fails if:

```text
the construction starts from C^5,
the scalar i is used before J is derived,
the carrier is R^8 or another incompatible dimension,
or the real carrier is only implicit notation.
```

## Computational Deliverables

```text
carrier_certificate.json
real_basis_report.md
test_real_carrier.py
```

Required tests:

```python
def test_carrier_dimension():
    assert K.shape == (10, 10)

def test_metric_is_real_symmetric():
    assert G == G.T
    assert all(entry.is_real for entry in G)

def test_generators_are_skew():
    for A in generators:
        assert A.T + A == zeros(10)
```

---

# 7. Phase 2 — Forced `J` Search

## Goal

Find a QCA-generated operator

```text
J ∈ SO(10)
```

such that

```text
J^2 = -I.
```

This is the first make-or-break phase.

---

## Candidate Minimal Ansatz

Use

```text
J = ε ⊗ I_5,
```

where

```text
ε =
[ 0  -1
  1   0 ].
```

Thus

```text
J =
[ 0   -I_5
  I_5  0  ].
```

Then

```text
J^2 = -I_10.
```

But this matrix is not accepted merely because it works algebraically. It must be generated by the microscopic QCA.

---

## Route 2A: Period-Four Micromotion

Try to realize

```text
J = U(T/4)
```

with

```text
J^2 = U(T/2) = -I
J^4 = U(T) = I
```

on the real one-particle carrier.

Important wording:

```text
U(T/2) = -I on the real carrier K_x.
```

On the full fermionic Fock space, the lift of `-I` on Majoranas is not simply scalar `-I`; it is related to fermion parity up to phase. Treat the carrier and Fock actions separately.

---

## Route 2B: Gate-Word Generation

Define a finite-depth word

```text
J = G_q ... G_2 G_1
```

where each `G_i` is a local real orthogonal gate or a layer of disjoint local real orthogonal gates.

Then check exactly:

```text
J.T J = I
det(J) = 1
J^2 = -I
```

---

## Forcedness Test

Define the admissible complex structures:

```text
C(Data) =
{
  J' ∈ SO(K_x)
  :
  (J')^2 = -I
  and J' is compatible with the microscopic rule data
}.
```

Strong pass:

```text
C(Data) = {J, -J}
```

or one harmless gauge-equivalent orbit.

Fail:

```text
C(Data) ≅ SO(10)/U(5),
```

because then the QCA data did not select a complex structure.

---

## Addressability Falsifier

Even if

```text
J = Π_a J_a
```

as a product over five real pairs, the construction fails if the rule set allows the gates

```text
J_1, J_2, J_3, J_4, J_5
```

to be independently switched, tuned, omitted, or conditioned.

Allowed:

```text
global clock tick: ε ⊗ I_5
block clock phases: ε ⊗ π_3, ε ⊗ π_2
```

Forbidden as geometric primitive gates:

```text
ε ⊗ |1><1|
ε ⊗ |2><2|
ε ⊗ |3><3|
ε ⊗ |4><4|
ε ⊗ |5><5|
```

---

## Computational Method

Implement an exact checker:

```python
def is_complex_structure(J, G):
    return (
        J.T * G * J == G
        and J * J == -sp.eye(J.rows)
        and J.det() == 1
    )
```

Implement a gate-word scanner:

```python
def scan_gate_words(primitives, max_depth):
    for word in words(primitives, max_depth):
        M = product(word)
        if is_complex_structure(M, G):
            yield word, M
```

Implement a forcedness test by computing the linear commutant of the rule data. If the commutant is too large, the rule data probably do not force `J`.

For generators `R_i`, solve

```text
X R_i = R_i X
```

for unknown real `10 x 10` matrices `X`.

Then impose:

```text
X^T = -X
X^2 = -I.
```

The linear part uses exact nullspaces.

---

## Acceptance Criteria

Phase 2 passes if:

```text
J is produced by microscopic QCA data,
J^2 = -I exactly,
J is local/finite-depth,
J is unique up to harmless equivalence,
the rule data do not allow independent rank-one pair rotations.
```

Phase 2 fails if:

```text
J is chosen after the fact,
J is ambient scalar i,
many inequivalent J choices are equally valid,
or the microscopic rule set allows independently addressable J_a.
```

## Computational Deliverables

```text
j_certificate.json
gate_word_certificate.md
forcedness_report.md
test_j_structure.py
test_gate_word_search.py
```

---

# 8. Phase 3 — Structural `3+2` Split

## Goal

Derive a `J`-invariant split

```text
K_x = K_3 ⊕ K_2
```

with

```text
dim_R K_3 = 6
dim_R K_2 = 4.
```

In complex notation:

```text
(K_x, J) = C^3 ⊕ C^2.
```

---

## Candidate Projectors

Use

```text
π_3 = diag(1,1,1,0,0)
π_2 = diag(0,0,0,1,1).
```

Then

```text
P_3 = I_2 ⊗ π_3
P_2 = I_2 ⊗ π_2.
```

Required identities:

```text
P_3 + P_2 = I
P_3 P_2 = 0
P_3^2 = P_3
P_2^2 = P_2
rank_R(P_3) = 6
rank_R(P_2) = 4
[J, P_3] = 0
[J, P_2] = 0.
```

---

## Structural Origin Requirement

The projectors must come from QCA rule data, for example:

```text
a forced orbit decomposition of the local graph,
a forced wall/Floquet sector decomposition,
a symmetry-protected local module decomposition,
a periodized drive architecture with one 3-dimensional block and one 2-dimensional block.
```

Rejected origins:

```text
choose three modes and call them color,
choose two modes and call them weak,
use physical coordinate axes as color axes,
use BCC body diagonals as D5 Cartan axes,
select P_3 only because the Standard Model needs rank 3.
```

---

## No Smaller Central Projectors

The structural data should force only

```text
P_3, P_2
```

and not rank-one projectors inside them.

Strong pass:

```text
Center(rule data) = span_R{I, P_3}
```

equivalently

```text
span_R{P_3, P_2}.
```

Fail:

```text
Center(rule data) contains |1><1|, |2><2|, |3><3|
```

or

```text
Center(rule data) contains |4><4|, |5><5|.
```

---

## Acceptance Criteria

Phase 3 passes if:

```text
P_3 and P_2 are QCA-structural,
P_3 and P_2 commute with J,
the split is unique up to block rotations,
there are no rank-one projectors inside either block.
```

Phase 3 fails if:

```text
the 3+2 split is arbitrary,
the QCA can resolve individual color axes,
the QCA can resolve individual weak axes,
or the split is introduced only after Standard Model labels appear.
```

## Computational Deliverables

```text
split_certificate.json
projector_lattice_report.md
test_projectors.py
test_no_rank_one_addressability.py
```

---

# 9. Phase 4 — SM Commutant And Gate Classification

## Goal

Classify every geometric QCA gate generator as:

```text
safe_sm_commutant
block_mixing_fail
color_breaking_fail
weak_breaking_fail
antilinear_fail
unknown_fail
```

---

## Gauge Algebra Versus Geometric Gate Algebra

Keep two algebras separate.

### Gauge algebra

The gauge algebra acts on the internal carrier:

```text
su(3) ⊕ su(2) ⊕ u(1).
```

Gauge transformations do not need to commute with themselves. They are the symmetry group.

### Geometric QCA gate algebra

The geometric QCA update must not break the gauge symmetry. Therefore its internal action must commute with the gauge algebra.

This is the required condition:

```text
A_geom ⊂ Comm(SU(3) x SU(2) x U(1)).
```

---

## Exact Commutant Test

On

```text
W = C^3 ⊕ C^2,
```

define the gauge generators using exact rational matrices.

For `SU(3)`:

```text
E_12, E_21, E_23, E_32,
H_12 = E_11 - E_22,
H_23 = E_22 - E_33.
```

For `SU(2)`:

```text
E_45, E_54,
H_45 = E_44 - E_55.
```

For hypercharge:

```text
Y = diag(-1/3, -1/3, -1/3, 1/2, 1/2).
```

A complex one-particle matrix `A` is safe iff

```text
[A, G] = 0
```

for every gauge generator `G`.

The expected exact answer is:

```text
A = λ_3 P_3 + λ_2 P_2.
```

---

## Realification

For a complex matrix

```text
C = A + i B,
```

the realification on

```text
z = x + i y
```

is

```text
R(C) =
[ A  -B
  B   A ].
```

The real complex structure is

```text
J = R(i I_5)
  =
[ 0   -I_5
  I_5  0  ].
```

A real matrix `M` is complex-linear iff

```text
[M, J] = 0.
```

If

```text
[M, J] != 0,
```

then `M` is not a safe complex-linear internal gate after `J` has been derived.

---

## Gate Classifier

Given a real `10 x 10` internal generator `M`:

1. Check complex-linearity:

```text
[M, J] = 0.
```

If no:

```text
antilinear_fail
```

2. Convert to complex `5 x 5` form.

3. Decompose into blocks:

```text
M =
[ M_33  M_32
  M_23  M_22 ].
```

4. Classify:

```text
M_32 != 0 or M_23 != 0:
  block_mixing_fail

M_33 not scalar multiple of I_3:
  color_breaking_fail

M_22 not scalar multiple of I_2:
  weak_breaking_fail

otherwise:
  safe_sm_commutant
```

---

## Special Warning: Block Diagonal Is Not Enough

The following matrix is block diagonal:

```text
diag(1,0,0,0,0).
```

But it breaks `SU(3)` because it does not commute with `E_12`.

Therefore:

```text
block diagonal ≠ safe.
```

Safe means:

```text
block scalar.
```

---

## Acceptance Criteria

Phase 4 passes if:

```text
every geometric QCA gate generator lies in C P_3 ⊕ C P_2,
all block mixers are rejected,
all rank-one color projectors are rejected,
all rank-one weak projectors are rejected,
closure of the safe algebra is proven.
```

Phase 4 fails if:

```text
one allowed geometric gate distinguishes color axes,
one allowed geometric gate distinguishes weak axes,
one allowed geometric gate mixes C^3 and C^2,
or the classifier cannot determine the gate class.
```

## Computational Deliverables

```text
commutant_basis.json
gate_classification_report.md
unsafe_gate_witnesses.md
test_commutant.py
test_gate_classification.py
```

---

# 10. Phase 5 — Finite-Depth Microscopic QCA Construction

## Goal

Construct an actual local update

```text
U = U_q ... U_2 U_1
```

where each `U_j` is a local gate or a layer of disjoint local gates.

No final Hamiltonian logarithm is allowed as a substitute for microscopic micromotion.

---

## Layer Certificate

Each layer must specify:

```text
layer name
support sets
gate matrix
locality radius
whether supports are disjoint
whether gate is real orthogonal/unitary
whether internal action passes the SM commutant test
```

Example schema:

```python
class GateCertificate(BaseModel):
    name: str
    support: tuple[int, ...]
    matrix_shape: tuple[int, int]
    is_real: bool
    is_orthogonal: bool
    locality_radius: int
    internal_classification: str
```

---

## Candidate Period-Four Update

Minimal carrier-level drive:

```text
U_1 = J
U_2 = J
U_3 = J
U_4 = J
```

Then

```text
U(T/4) = J
U(T/2) = J^2 = -I
U(T) = J^4 = I.
```

But this is only acceptable if the microscopic QCA architecture forces the global clock tick

```text
J = ε ⊗ I_5
```

as one indivisible rule-layer.

It fails if the architecture merely permits arbitrary independent rotations

```text
ε ⊗ |a><a|.
```

---

## More Realistic Layer Architecture

A candidate finite-depth local update should have layers of the following type:

```text
U_clock:
  global real quarter-turn on the clock factor
  internal action ε ⊗ I_5

U_block_phase:
  optional block-scalar phases
  exp(θ_3 J P_3 + θ_2 J P_2)

U_geometry:
  spacetime/locality update tensor block-scalar internal action

U_wall:
  wall/Floquet geometry update, allowed only if it acts as a whole-block
  operation on W_2, not as separate weak-axis projectors
```

Allowed internal factors:

```text
I
P_3
P_2
J
J P_3
J P_2
```

Forbidden internal factors:

```text
|a><a| inside C^3
|α><α| inside C^2
Hom(C^3, C^2)
Hom(C^2, C^3)
```

---

## Spacetime Coupling Rule

If the spacetime update has symbolic shifts

```text
S_r,
```

and internal coefficients

```text
A_r,
```

so that

```text
U = Σ_r S_r ⊗ A_r,
```

then every `A_r` must be classified.

Passing condition:

```text
A_r ∈ C P_3 ⊕ C P_2
```

for all `r`.

Failing condition:

```text
A_r = |color i><color i|
```

or any non-scalar matrix inside `C^3` or `C^2`.

---

## Acceptance Criteria

Phase 5 passes if:

```text
the update is finite-depth,
each layer is local,
each layer is exactly real orthogonal/unitary,
J appears as actual micromotion,
spacetime coupling does not use within-block projectors,
all internal geometric actions pass the SM commutant classifier.
```

Phase 5 fails if:

```text
U is only defined by exp(-i H_eff T),
the quarter-period is nonlocal,
spacetime propagation requires color-axis projectors,
wall/Floquet propagation requires weak-axis projectors,
or locality is asserted but not certified layer by layer.
```

## Computational Deliverables

```text
qca_update_certificate.json
layer_report.md
locality_report.md
test_qca_layers.py
test_no_locking.py
```

---

# 11. Phase 6 — Spinor Reconstruction

## Goal

Only after `J` and the `3+2` split pass do we construct

```text
Λ^even(C^5).
```

This phase is algebraically standard. It should not be used to justify earlier choices.

---

## Complex Fermion Modes

From the real basis and `J`, define complex modes

```text
z_a = x_a + i y_a,
a = 1,...,5.
```

Then

```text
W = span_C{z_1,...,z_5}.
```

The split is

```text
W_3 = span_C{z_1,z_2,z_3}
W_2 = span_C{z_4,z_5}.
```

---

## Even Fock Space

Construct

```text
S^+ = Λ^even(W)
    = Λ^0(W) ⊕ Λ^2(W) ⊕ Λ^4(W).
```

Dimension:

```text
dim Λ^0(C^5) = 1
dim Λ^2(C^5) = 10
dim Λ^4(C^5) = 5
total = 16.
```

---

## Hypercharge Operator

For a Fock basis state, let

```text
N_3 = number of occupied modes among {1,2,3}
N_2 = number of occupied modes among {4,5}.
```

Define

```text
Y = -1/3 N_3 + 1/2 N_2.
```

---

## Expected Table

| Exterior sector     | Dimension | SM representation | Hypercharge | Mnemonic |
| ------------------- | --------: | ----------------: | ----------: | -------- |
| `Λ^0`               |       `1` |           `(1,1)` |         `0` | `ν^c`    |
| `Λ^2 C^3`           |       `3` |      `(\bar 3,1)` |      `-2/3` | `u^c`    |
| `C^3 ⊗ C^2`         |       `6` |           `(3,2)` |       `1/6` | `Q`      |
| `Λ^2 C^2`           |       `1` |           `(1,1)` |         `1` | `e^c`    |
| `Λ^2 C^3 ⊗ Λ^2 C^2` |       `3` |      `(\bar 3,1)` |       `1/3` | `d^c`    |
| `Λ^3 C^3 ⊗ C^2`     |       `2` |           `(1,2)` |      `-1/2` | `L`      |

Total dimension:

```text
1 + 3 + 6 + 1 + 3 + 2 = 16.
```

---

## Acceptance Criteria

Phase 6 passes if:

```text
the even Fock basis has dimension 16,
the hypercharge table is exact,
the branching uses the previously derived J and P_3/P_2,
no new choice of C^5 appears in this phase.
```

Phase 6 fails if:

```text
C^5 is introduced for the first time here,
J is chosen for the first time here,
the 3+2 split is selected here,
or the hypercharge table requires manual relabeling.
```

## Computational Deliverables

```text
spinor16_table.csv
spinor16_report.md
test_spinor16.py
test_hypercharge.py
```

---

# 12. Phase 7 — Forcedness And Normalizer Analysis

## Goal

Prove the construction is not merely one convenient coordinate choice among many.

---

## Complex Structure Normalizer Test

Given microscopic rule data `Data`, compute or approximate:

```text
Aut(Data)
```

the real orthogonal transformations preserving the microscopic rules.

Then study:

```text
{J' : J'^2 = -I and J' compatible with Data}.
```

Strong pass:

```text
J' = ±J
```

up to harmless global equivalence.

Weak pass:

```text
J' belongs to one orbit whose stabilizer preserves P_3/P_2.
```

Fail:

```text
there is a continuous SO(10)/U(5) family of equally valid J choices.
```

---

## Split Normalizer Test

Study admissible projectors

```text
P
```

satisfying:

```text
P^2 = P
[P, J] = 0
rank_C(P) = 3.
```

Strong pass:

```text
P = P_3
```

up to block equivalence.

Fail:

```text
any rank-3 complex subspace can be chosen equally well.
```

---

## Addressability Algebra Test

Let `A_addr` be the algebra generated by all switchable, tunable, conditionable microscopic internal controls.

Passing condition:

```text
A_addr ⊂ span_C{P_3, P_2}
```

after complexification.

Failing condition:

```text
A_addr contains rank-one projectors inside P_3 or P_2.
```

This distinction is crucial:

```text
A mandatory global layer may factor algebraically into local pair rotations,
but the QCA fails only if those pair rotations are independently addressable.
```

---

## Acceptance Criteria

Phase 7 passes if:

```text
J is unique or canonically forced,
P_3/P_2 are unique structural central projectors,
the microscopic controls do not generate smaller projectors,
the normalizer does not erase the 3+2 split.
```

Phase 7 fails if:

```text
J has arbitrary alternatives,
the 3+2 split has arbitrary alternatives,
rank-one internal projectors are addressable,
or the rule data secretly generate full U(5).
```

## Computational Deliverables

```text
normalizer_report.md
addressability_algebra.json
forcedness_certificate.json
test_normalizer.py
```

---

# 13. Phase 8 — Route Branches If The Minimal Ansatz Fails

The preferred path is still:

```text
real carrier + period-four micromotion.
```

If that fails, branch in the following order.

---

## Branch A — Stronger Real-QCA First Route

Search over real finite-depth gate words on `R^10`.

Allowed primitive gates:

```text
global clock rotations,
whole-block rotations,
structural graph-local real orthogonal gates,
mandatory translation-invariant layers.
```

Forbidden primitive gates:

```text
independently tunable rank-one pair rotations,
direction-conditioned color projectors,
direction-conditioned weak projectors.
```

Goal:

```text
find a finite-depth word forcing J and P_3/P_2.
```

---

## Branch B — Floquet-Kähler Micromotion

Search period-four drives:

```text
U(T) = U_4 U_3 U_2 U_1
```

with

```text
U(T/4) = J
U(T/2) = -I on K_x.
```

The period-four structure must be required by the microscopic rule, not added as decoration.

---

## Branch C — Defect/Monodromy Complex Structure

Build a punctured or wall-cycle QCA with local transition functions.

Compute monodromy:

```text
M_defect = product of transition functions around cycle.
```

Try to prove:

```text
M_defect^2 = -I.
```

Pass only if the monodromy is forced.

Fail if the defect charge or winding is freely chosen.

---

## Branch D — Five-Axis Parent QCA

Build a parent graph with five structural axes:

```text
V = V_3 ⊕ V_2.
```

Then find five forced conjugate partners:

```text
axis_a'
```

so that

```text
J axis_a = axis_a'
J axis_a' = -axis_a.
```

Pass only if both the five axes and their conjugate partners are structural.

---

## Branch E — `D5` / `SU(5)` Root-Lattice QCA

Use only as a later algebraic route.

Risk:

```text
D5 root geometry is real and does not automatically supply J.
```

This branch passes only if the root geometry itself forces the complex structure and the `3+2` split.

---

## Branch F — Passive `Spin(10)` Fiber Fallback

If no route derives `J`, narrow the claim:

```text
QCA geometry supplies spacetime locality/topology/family index.
Spin(10) remains an independent internal fiber.
```

This is honest but does not prove the geometric `3+2 -> D5` bridge.

---

# 14. Phase 9 — Family Number Is Separate

## Goal

Avoid hiding three generations inside the construction.

The following do not count as deriving three generations:

```text
choose winding n = 3
stack three copies
choose three defects
repeat a unit pump three times
choose a degree-three map
```

A family claim passes only if:

```text
the microscopic rules forbid n = 1 and force n = 3,
```

or if:

```text
a stable index canonically fixed by the geometry equals 3.
```

Until then, the bridge claims only:

```text
one-generation internal carrier.
```

## Computational Deliverables

```text
family_index_status.md
no_hidden_family_insertion.md
```

---

# 15. Exact Implementation Milestones

## Milestone M0 — Algebra Kernel

Implement exact matrix utilities.

Files:

```text
src/qca_jfirst/algebra/matrices.py
src/qca_jfirst/algebra/commutants.py
src/qca_jfirst/algebra/realification.py
```

Core functions:

```python
def epsilon() -> sp.Matrix:
    return sp.Matrix([[0, -1], [1, 0]])

def J_clock(n: int) -> sp.Matrix:
    return sp.kronecker_product(epsilon(), sp.eye(n))

def projector_3_2() -> tuple[sp.Matrix, sp.Matrix]:
    pi3 = sp.diag(1, 1, 1, 0, 0)
    pi2 = sp.diag(0, 0, 0, 1, 1)
    I2 = sp.eye(2)
    return sp.kronecker_product(I2, pi3), sp.kronecker_product(I2, pi2)

def commutator(A: sp.Matrix, B: sp.Matrix) -> sp.Matrix:
    return A * B - B * A
```

Tests:

```text
J^2 = -I
P_3 + P_2 = I
P_i^2 = P_i
[J, P_i] = 0
```

---

## Milestone M1 — SM Commutant Engine

Implement exact `SU(3) x SU(2) x U(1)` commutant.

Files:

```text
src/qca_jfirst/sm/embedding.py
src/qca_jfirst/algebra/commutants.py
```

Core function:

```python
def commutant_basis(gens: list[sp.Matrix]) -> list[sp.Matrix]:
    n = gens[0].rows
    xs = sp.symbols(f"x0:{n*n}")
    X = sp.Matrix(n, n, xs)

    equations = []
    for G in gens:
        equations.extend(list(X * G - G * X))

    A, _ = sp.linear_eq_to_matrix(equations, xs)
    basis_vecs = A.nullspace()
    return [sp.Matrix(n, n, v) for v in basis_vecs]
```

Expected result:

```text
basis = {P_3, P_2}
```

over the complex one-particle space.

---

## Milestone M2 — Gate Classifier

Files:

```text
src/qca_jfirst/sm/classifications.py
```

Classifications:

```python
class GateClass(str, Enum):
    SAFE_SM_COMMUTANT = "safe_sm_commutant"
    BLOCK_MIXING_FAIL = "block_mixing_fail"
    COLOR_BREAKING_FAIL = "color_breaking_fail"
    WEAK_BREAKING_FAIL = "weak_breaking_fail"
    ANTILINEAR_FAIL = "antilinear_fail"
    UNKNOWN_FAIL = "unknown_fail"
```

Required examples:

```text
P_3 -> structural projector, safe as block projector
J P_3 -> safe generator
J P_2 -> safe generator
|1><1| inside C^3 -> color_breaking_fail
|4><4| inside C^2 -> weak_breaking_fail
E_14 -> block_mixing_fail
```

---

## Milestone M3 — Gate-Word Search

Files:

```text
src/qca_jfirst/search/enumerate_gate_words.py
src/qca_jfirst/search/solve_complex_structures.py
```

Search for words satisfying:

```text
J_word^2 = -I
J_word.T J_word = I
det(J_word) = 1
```

Then immediately run:

```text
forcedness test
addressability test
commutant test
```

A found `J` is not enough.

---

## Milestone M4 — QCA Layer Certificate

Files:

```text
src/qca_jfirst/qca/gates.py
src/qca_jfirst/qca/layers.py
src/qca_jfirst/qca/locality.py
src/qca_jfirst/qca/certificates.py
```

Each candidate update must output:

```text
finite_depth: true/false
layer_count
max_locality_radius
all_layers_orthogonal
quarter_period_operator
half_period_operator
J_certificate
gate_classification_summary
unsafe_gate_witnesses
```

---

## Milestone M5 — Spinor Table

Files:

```text
src/qca_jfirst/sm/spinor16.py
src/qca_jfirst/sm/hypercharge.py
```

Generate exact table using Python `Fraction` or SymPy rationals.

Expected bitstring basis:

```text
00000

11000, 10100, 01100
10010, 10001, 01010, 01001, 00110, 00101
00011

11110, 11101
11011, 10111, 01111
```

Grouped by:

```text
N_3 = occupancy among first three modes
N_2 = occupancy among last two modes
Y = -N_3/3 + N_2/2.
```

---

# 16. Master Test Suite

The project should have one master command:

```bash
uv run pytest
```

The test suite should enforce the bridge conditions.

## Required Tests

```text
test_real_carrier_dimension
test_real_metric
test_j_squared
test_j_orthogonal
test_j_generated_by_gate_word
test_j_not_ambient_scalar

test_projector_identities
test_projector_ranks
test_projectors_commute_with_j
test_no_rank_one_projectors_in_rule_data

test_sm_commutant_basis
test_color_axis_projector_fails
test_weak_axis_projector_fails
test_block_mixer_fails
test_block_scalar_passes

test_qca_layers_are_finite_depth
test_qca_layers_are_local
test_quarter_period_is_j
test_half_period_is_minus_identity_on_carrier
test_spacetime_update_has_no_color_locking

test_spinor_even_dimension_is_16
test_hypercharge_table_exact
test_spinor_reconstruction_uses_prior_j_and_split

test_no_hidden_family_insertion
```

---

# 17. Decision Tree

```text
Can the microscopic QCA generate J?
|
+-- no
|   -> D5 bridge fails for this construction.
|
+-- yes
    |
    Is J forced, not arbitrary?
    |
    +-- no
    |   -> complex structure is notation-only.
    |
    +-- yes
        |
        Is there a structural J-invariant 3+2 split?
        |
        +-- no
        |   -> SU(5) branching is notation-only.
        |
        +-- yes
            |
            Does the QCA avoid rank-one addressability inside C^3 and C^2?
            |
            +-- no
            |   -> color/weak locking obstruction.
            |
            +-- yes
                |
                Are all geometric gates in the SM commutant?
                |
                +-- no
                |   -> gauge compatibility fails.
                |
                +-- yes
                    |
                    Does Λ^even(C^5) reconstruct the 16?
                    |
                    +-- no
                    |   -> algebraic Spin(10) bridge fails.
                    |
                    +-- yes
                        -> one-generation J-first QCA bridge survives.
```

Then separately:

```text
Does the QCA force family index 3?
|
+-- no
|   -> one-generation bridge only.
|
+-- yes
    -> candidate three-family mechanism.
```

---

# 18. First Concrete Work Package

The first work package should be narrow.

## Work Package 1: Minimal Real Period-Four Ansatz

Use

```text
K_x = R^2_clock ⊗ (R^3 ⊕ R^2).
```

Define

```text
J = ε ⊗ I_5
P_3 = I_2 ⊗ π_3
P_2 = I_2 ⊗ π_2.
```

Verify exactly:

```text
J^2 = -I
J^T J = I
P_3 + P_2 = I
P_i^2 = P_i
[J, P_i] = 0.
```

Then test the dangerous question:

```text
Does the microscopic rule set generate only P_3 and P_2,
or does it generate rank-one projectors inside them?
```

If it generates rank-one projectors, kill the ansatz immediately.

---

## Work Package 2: Commutant Classifier

Build the exact SM commutant engine.

Confirm:

```text
Comm(SU(3) x SU(2) x U(1) on C^3 ⊕ C^2)
= C P_3 ⊕ C P_2.
```

Then classify candidate QCA gates.

This becomes the permanent gate-safety oracle.

---

## Work Package 3: Finite-Depth QCA Candidate

Construct a period-four layer certificate:

```text
U(T/4) = J
U(T/2) = -I on K_x
U(T) = I on K_x.
```

Then add spacetime/locality layers only if their internal coefficients pass:

```text
A_r ∈ C P_3 ⊕ C P_2.
```

Reject any shift of the form:

```text
S_i ⊗ |i><i|
```

inside the future color or weak blocks.

---

## Work Package 4: Spinor Reconstruction

After the earlier tests pass, generate:

```text
Λ^even(C^5)
```

and the exact hypercharge table.

This phase should be automatic and should not introduce new structure.

---

# 19. Bottom Line

The enhanced attack is:

```text
J first,
then forced 3+2,
then no-locking commutant,
then finite-depth QCA,
then Spin(10) spinor.
```

The strongest concrete ansatz is:

```text
real period-four QCA on K_x = R^2_clock ⊗ (R^3 ⊕ R^2)
```

with

```text
J = ε ⊗ I_5,
P_3 = I_2 ⊗ π_3,
P_2 = I_2 ⊗ π_2.
```

But the ansatz survives only if the QCA rule data force the global clock structure and the `3+2` block structure while forbidding rank-one addressability inside either block.

The bridge fails as soon as the QCA contains gates like:

```text
S_1 ⊗ |1><1| + S_2 ⊗ |2><2| + S_3 ⊗ |3><3|
```

unless all `S_i` are identical.

So the practical research slogan is:

```text
derive J,
derive only P_3 and P_2,
never derive |1><1|, |2><2|, |3><3|, |4><4|, or |5><5|.
```

That is the precise enhanced QCA attack plan.

[1]: https://docs.astral.sh/uv/getting-started/features/?utm_source=chatgpt.com "Features | uv"
[2]: https://docs.astral.sh/uv/concepts/projects/run/?utm_source=chatgpt.com "Running commands | uv"
[3]: https://docs.sympy.org/latest/tutorials/intro-tutorial/matrices.html?utm_source=chatgpt.com "Matrices - SymPy 1.14.0 documentation"
