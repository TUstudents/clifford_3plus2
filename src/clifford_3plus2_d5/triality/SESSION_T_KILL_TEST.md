# Session T — Triality Kill Test Report

**Verdict: K1 FAIL.**  Triality does not preserve the SM-inside-Spin(8)
Cartan subalgebra under the natural index-`{0..7}` embedding.  The
sidecar is closed.

## Setup

The sidecar asks one binary question (per ``PLAN.md``):

> Can explicit Spin(8) triality produce three equivalent SM-generation
> carriers without declaring three generations by hand?

The carrier is the chiral-16 of Spin(0,10), realized on a real-32-dim
chiral subspace by ``patisalam_chiral16_block_matrix`` from
``lepton.clifford_patisalam``.

The pinned Spin(8) embedding uses Cl(0,10) gamma indices ``{0..7}``, with
Cartan basis ``H_k = (1/2) γ_{2k} γ_{2k+1}`` for `k = 0, 1, 2, 3`.

The triality outer automorphism ``τ`` is constructed via its action on
the Spin(8) Cartan as a 4×4 orthogonal matrix of order 3, derived from
the requirement that ``τ`` cyclically permute the three outer Dynkin
nodes ``{α_1, α_3, α_4}`` while fixing the central node ``α_2`` of D_4.
The resulting Cartan matrix is

```text
T = (1/2) * [[ 1,  1,  1,  1],
             [ 1,  1, -1, -1],
             [ 1, -1,  1, -1],
             [-1,  1,  1, -1]]
```

with `T^T T = I`, `T^3 = I`, `det(T) = +1`, and the +1 eigenspace
2-dimensional.

## Pati-Salam SM does not fit inside Spin(8)

Reading ``lepton.clifford_patisalam`` exposes a structural obstruction
named in ``PLAN.md``:  Pati-Salam ``SU(4) x SU(2)_L x SU(2)_R`` has rank
5, but ``Spin(8)`` has rank 4.  Every generator of ``SU(2)_L``
(``su2_l_generators_from_spin04``) is a sum of one bivector with indices
in ``{6, 7}`` (inside Spin(8)) and one with indices in ``{8, 9}`` (outside
Spin(8)).  ``SU(2)_L`` therefore cannot be preserved by any Spin(8)
inside Spin(10) using a contiguous index set.

The kill test focuses instead on the Spin(8)-restricted SM:

```text
g_SM(8) = SU(3)_c ⊕ U(1)_{Y'}
```

where ``Y'`` is the projection of physical hypercharge onto the Spin(8)
Cartan span.  The Cartan of this restricted SM is **3-dimensional**
inside the 4-dimensional Cartan of so(8):

- two ``SU(3)_c`` Cartan generators (from ``su3_c_generators_from_su4``,
  indices `(4, 7)` are the ones in the Cartan span);
- ``Y'`` itself.

## K1 — Cartan necessary condition

Apply ``T`` to each of the three SM Cartan basis vectors in
``(H_0, H_1, H_2, H_3)`` coordinates.  Check whether the image is in the
3-dimensional SM Cartan subspace.

Computed by ``sm_restriction.k1_failure_witnesses``:

```text
SU(3)_c Cartan v_0 = (-1, 1, 0, 0)
  T · v_0          = ( 0, 0, -1, 1)        ✗ outside SM Cartan span

SU(3)_c Cartan v_1 = (-1, 0, 1, 0)
  T · v_1          = ( 0, -1, 0, 1)        ✗ outside SM Cartan span

Y'                 = (1/3, 1/3, 1/3, 1/2)
  T · Y'           = (3/4, -1/12, -1/12, -1/12)   ✗ outside SM Cartan span
```

All three SM Cartan generators map outside.  K1 fails on every basis
element.

### Why K1 fails (interpretation)

The SU(3) Cartan vectors are differences `H_i - H_j` between the first
three Cartan elements (Cl(0,6) sub-Cartan).  ``T`` is the Hadamard-style
mixing matrix that combines all four ``H_k`` symmetrically.  Under ``T``,
a difference like `H_0 - H_1` maps to a difference involving `H_2` and
`H_3`, which is outside the SU(3) Cartan because the SU(3) Cartan only
involves the first three.

The Y' vector is asymmetric in the H_3 coordinate (`1/2` vs `1/3` for
H_0..H_2).  ``T`` does not preserve this asymmetry; the image has a
heavily skewed distribution (`3/4` on H_0, `-1/12` on the rest).

These failures express the fundamental tension:  triality is a Z/3 cycle
in the four-dimensional Spin(8) Cartan, but the SM Cartan inside Spin(8)
is the 3-dim subspace **aligned with the Pati-Salam factorization
``Cl(0,6) ⊗ Cl(0,4)``**.  The Pati-Salam factorization makes ``H_3``
qualitatively different from ``H_0, H_1, H_2``.  Triality does not
respect this qualitative difference because the D_4 Dynkin symmetry sees
all four ``H_k`` on equal footing.

## K2 — Y' spectrum on the chiral-16

Computed by ``sm_restriction.y_prime_complex_spectrum``:

```text
Y'  eigenvalue   complex multiplicity
    +3/4         1
    +5/12        3
    +1/4         1
    +1/12        3
    -1/12        3
    -1/4         1
    -5/12        3
    -3/4         1
                ─────
    total       16
```

The spectrum is symmetric under `Y' → -Y'` (as required of a U(1)
generator), has 8 distinct eigenvalues, and totals to 16 (the complex
chiral-16 dimension).  This confirms ``Y'`` is a non-trivial U(1)
generator on the chiral-16.

The spectrum does **not** match the SM physical hypercharge spectrum
``{1/6:6, -2/3:3, 1/3:3, -1/2:2, 1:1, 0:1}``.  This is expected: ``Y'``
is the Spin(8)-restricted projection of ``Y``, not ``Y`` itself.  The
chiral-16 is a one-generation SM carrier under ``Y``, but under ``Y'``
it decomposes into a different multiset.

K2 is informational under a K1 fail.  If K1 had passed, the three
triality copies would have had the same Y'-decomposition (by symmetry),
and one could then ask whether the multiset matches expectations.
Under K1 fail, the three copies have inequivalent SM-content positions,
so K2 across them does not produce three equivalent generations.

## Hypercharge residual

``physical_hypercharge_generator()`` is not contained in the Spin(8)
Cartan: the residual

```text
||Y - Y'||_Frobenius^2 = 2
```

confirms that physical hypercharge has out-of-Spin(8) content (the
``γ_8 γ_9`` direction).  This is a numerical sanity check on the
Spin(8)-vs-Spin(10) rank obstruction.

## Verdict

**K1 FAIL.**  Triality maps every one of the three SM-inside-Spin(8)
Cartan generators outside the SM Cartan subspace.  Therefore:

- The Spin(8) triality outer automorphism does not preserve
  ``g_SM(8) = SU(3)_c ⊕ U(1)_{Y'}`` as a subalgebra.
- The three triality-rotated chiral-16 carriers ``C_0, C_1, C_2`` carry
  inequivalent SM-content positions.
- The three copies cannot represent three equivalent generations of the
  same Standard Model.

The program dies cleanly at K1.

## What this rules out

This negative result rules out the following specific program under the
specific Spin(8) embedding tested:

> Three generations as three triality copies of one chiral-16 under
> ``Spin(8) ⊂ Spin(10)`` with indices ``{0..7}``, the natural Cartan
> ``H_k = (1/2) γ_{2k} γ_{2k+1}``, and the Pati-Salam-derived SM
> subalgebra from ``lepton``.

It does **not** rule out:

- Triality-based generation schemes that do not use the
  Pati-Salam-aligned Spin(8) embedding.
- Triality-based schemes that use a larger ambient group with explicit
  Z/3 outer automorphism (Spin(10) has only Z/2; F_4, E_6, E_7, E_8 do
  not have Z/3 outer automorphisms either, by inspection of their Dynkin
  diagrams).
- Discrete-flavor-symmetry schemes in which the Z/3 acts differently
  (e.g., on a flavor index introduced by hand rather than emerging from
  group structure).

## What this confirms about the original CP-from-quantization hope

The user's earlier hope was that approximate embedding of continuous
symmetries (with quantized invariants forcing tiny deviations) would
produce CP violation while preserving CPT.  This kill test addresses a
**different** hypothesis — that Z/3 triality of Spin(8) provides the
three-generation structure exactly — and confirms that the exact
formulation fails on the natural Spin(8) embedding.

The original CP-from-quantization hope survives this kill test.  That
program would treat triality as an approximate / broken symmetry,
introducing the breaking parameters as the source of generation
differences and CP phase.  Whether the broken-triality version is viable
is a separate question that this audit does not answer.

## Cross-references

- ``PLAN.md`` — the load-bearing question, the predicted failure modes
  (F3 is what we hit), and the kill-test definitions.
- ``parameter_ledger.md`` — exact choices made for this audit.
- ``STATUS.md`` — updated to reflect the closed sidecar.
- ``../lepton/`` — provided all Pati-Salam, SM, and chiral-16
  infrastructure via ``reuse.py``.

## Tests

21 passing tests:

```text
tests/test_spin8_triality.py        9 passed
tests/test_sm_restriction.py       12 passed
                                  ─────────
                                   21 passed
```

Run with:

```bash
uv run pytest src/clifford_3plus2_d5/triality/tests -q
```

## Effort

Implementation total: scaffolding + reading + plan + 3 modules + 21
tests + this report.  Roughly one focused session.  The negative result
fits the budget; a positive result would have justified the longer
mass/mixing investigation outlined in the original sidecar sketch.

The reviewer's structural recommendation — make T4 the first milestone,
not T6 — was vindicated.  The kill test cost ~10% of the originally
sketched sidecar effort and produced a clean, publishable negative
result.
