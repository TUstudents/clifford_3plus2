# Session BT — Broken-Triality Kill-Test Report

**Verdict: BT-2 FAIL.**  The broken-triality program produces an
essentially flat mass spectrum at leading order; no SM-like hierarchy is
available without invoking additional structure beyond pure-triality
projection.  Sidecar closed.

Of the four kill tests in ``PLAN.md``, two ran:

| Kill | Verdict |
|---|---|
| BT-1 (Yukawa overlap structure) | PASS (with rank-deficit caveat) |
| BT-2 (mass hierarchy)           | **FAIL** |
| BT-3 (CP phase)                 | not run (program closed at BT-2) |
| BT-4 (parameter audit)          | not run (program closed at BT-2) |

## Setup

The sidecar uses the same Spin(8) embedding and triality construction
verified in ``../triality/``.  Specifically:

- Spin(8) ⊂ Spin(10) via Cl(0,10) gamma indices ``{0..7}``.
- Cartan ``H_k = (1/2) γ_{2k} γ_{2k+1}`` for `k = 0..3`.
- Triality Cartan matrix ``T`` from ``triality.spin8_triality``.

The default ``v_*`` for the Yukawa construction is ``Y'`` (restricted
hypercharge), matching the physical picture of a Higgs aligned with
``U(1)_Y``.  ``Y'`` in Cartan coords is ``(1/3, 1/3, 1/3, 1/2)``.

## BT-1 result: PASS with rank-deficit caveat

The three triality-rotated, SM-projected vectors are:

```text
u_0 = Pi_SM(Y')         = ( 1/3,    1/3,    1/3,     1/2 )
u_1 = Pi_SM(tau Y')     = ( 9/14,  -4/21,  -4/21,   11/84 )
u_2 = Pi_SM(tau^2 Y')   = (-1/42,   1/7,    1/7,    11/84 )
```

The 3×3 Yukawa overlap matrix ``Y_ij = <u_i, u_j>``:

```text
Y = [[ 7/12,    11/72,    11/72   ],
     [ 11/72,   169/336,  -53/1008 ],
     [ 11/72,  -53/1008,  59/1008  ]]
```

Eigenvalues: ``{5/7, 31/72, 0}``.  Off-diagonal entries all non-zero.
Three distinct eigenvalues.  **BT-1 passes the literal pass condition**:
non-trivial off-diagonal mixing + non-degenerate eigenvalues.

**Documented feature**: one eigenvalue is exactly zero, so ``Y`` has
rank 2.  This is forced by a residual ``H_1 <-> H_2`` swap symmetry in
the SM Cartan (visible in the entries of ``u_0, u_1, u_2``: position 1
equals position 2 in every projected vector).  ``Y'`` itself has this
symmetry by construction (``Y' = (1/3, 1/3, 1/3, 1/2)``), and the
symmetry survives both the triality cycle and the Cartan projection.

The rank deficit carries through to BT-2.

## BT-2 result: FAIL

Non-zero eigenvalues of ``Y``: ``(5/7, 31/72)``.

Ratio: ``(5/7) / (31/72) = 360 / 217 ≈ 1.659``.

Pass threshold: 100.  Fail threshold: 10.  The ratio is well below the
fail threshold.

**Interpretation**: the pure triality-projection construction with the
hypercharge-aligned starting vector produces a Yukawa whose non-zero
eigenvalues differ by less than a factor of 2.  This is essentially flat
on the relevant scale (SM ratios range from ~100 within a sector up to
~10^5 across the entire fermion content).

The rank deficit (zero eigenvalue) further obstructs a
phenomenologically sensible mass spectrum: one generation would be
exactly massless at leading order, requiring an additional mechanism to
generate any first-generation mass at all.

**Verdict: BT-2 FAIL.**  Program closes per ``PLAN.md`` decision tree.

## What this rules out

The broken-triality program, as defined in ``PLAN.md`` with:

- Spin(8) embedding via Cl(0,10) gamma indices ``{0..7}``;
- Cartan basis aligned with the Pati-Salam factorization;
- Yukawa construction from triality projection of ``Y'``;

does not produce SM-like mass hierarchy.  The structure is fixed by the
existing lepton/triality infrastructure and is not adjustable without
additional external choices.

## What this does NOT rule out

This kill test fixed the simplest natural choices.  The negative result
does not address:

1. **Alternative starting vectors ``v_*``**.  ``Y' = (1/3, 1/3, 1/3, 1/2)``
   has an accidental ``H_1 <-> H_2`` symmetry that forces the rank
   deficit.  An asymmetric ``v_*`` would lift the symmetry; whether the
   resulting Yukawa has SM-like hierarchy is not addressed here.

2. **Alternative inner products**.  The construction uses the Euclidean
   inner product on R^4.  Using a non-trivial Killing-form-derived inner
   product (with weights aligned to the SM hypercharge structure) could
   change the overlaps.  Not addressed.

3. **Higher-order corrections**.  Pure triality projection is the leading
   order.  Higher-order corrections (one-loop running of overlaps,
   non-linear contributions, etc.) could enhance the mass spread.  Not
   addressed.

4. **A different ambient algebra**.  The Spin(8) triality is the natural
   Z/3 candidate inside Spin(10).  If the program drops the
   Spin(8)-inside-Spin(10) embedding and uses Z/3 acting on a 3 ×
   chiral-16 carrier abstractly (the "generic discrete flavor symmetry"
   approach), the kill condition does not apply.  But that is a different
   program with no microscopic motivation from the Spin(10) carrier.

5. **The "approximate embedding from quantization" hope**.  The original
   user-stated program (CP from lattice anisotropy, CPT preserved by
   discrete symmetry) is **orthogonal** to broken triality.  This kill
   test does not address it.

## What this does rule in

Reinforces the structural lesson from the exact-triality K1 failure:
the Pati-Salam factorization makes ``H_3`` qualitatively different from
``H_0, H_1, H_2``.  This asymmetry is fundamental to the Pati-Salam
construction and is what makes the SM-inside-Spin(8) Cartan non-generic.

Triality, being the D_4 Dynkin symmetry, treats all four Cartan elements
on equal footing.  When applied to a Pati-Salam-aligned SM Cartan, the
result is **structurally constrained** — and the constraint is not in
the direction of SM-like flavor structure.

## Honest assessment

The exact-triality program died at K1 (subgroup not preserved).  The
broken-triality program dies at BT-2 (mass spectrum flat / rank deficient).

Both results point in the same direction: **Spin(8) triality and the
Pati-Salam factorization are aligned in different ways**.  Triality
treats H_0..H_3 symmetrically; Pati-Salam picks out H_3 (the Cl(0,4)
direction).  When triality is forced to act on the Pati-Salam-aligned SM
content, the action either escapes the SM subgroup entirely (exact case,
K1) or produces flat / rank-deficient flavor structure (broken case,
BT-2).

The most honest reading of these two negative results: **Spin(8)
triality is not the source of three-generation structure for an SM
carrier whose color sector comes from Pati-Salam**.  The mechanism, if
it exists, has to come from elsewhere.

## What's saved

- 21 passing tests in ``tests/``.
- 7 helper functions in ``yukawa_overlaps.py`` and ``mass_hierarchy.py``.
- The triality-restricted projection ``Pi_SM`` and the triality orbit
  helpers are useful regardless; they could be lifted into ``triality/``
  if a similar audit is wanted with a different starting vector.

## What's saved for the future

If anyone returns to this program with one of the "not ruled out"
extensions above, the BT-1 / BT-2 modules give the right scaffolding to
test it:

- Replace ``v_*`` in ``yukawa_overlap_matrix()`` with a different
  starting vector.
- Run ``bt1_audit_payload()`` and ``bt2_audit_payload()`` on the new
  Yukawa.
- Compare verdicts.

This is a half-day extension if it ever becomes interesting.

## Tests

17 passing tests:

```text
tests/test_yukawa_overlaps.py    12 passed
tests/test_mass_hierarchy.py      5 passed
                                ─────────
                                 17 passed
```

Run with:

```bash
uv run pytest src/clifford_3plus2_d5/broken_triality/tests -q
```

## Effort

- Scaffolding: 30 minutes.
- BT-1 (yukawa_overlaps.py + tests): ~2 hours including the exploratory
  calculation.
- BT-2 (mass_hierarchy.py + tests): ~1 hour.
- Reports + status updates: 30 minutes.

Total: ~4 hours of focused work.  Far below the ~3-day budget; the
early kill at BT-2 saved the sidecar from wasted work on BT-3 and BT-4.

This is exactly the "early killing" discipline the PLAN.md was designed
to enable.  The negative result is clean, publishable, and reusable.

## Cross-references

- ``PLAN.md`` — kill-test design with predicted failure modes.
- ``../triality/SESSION_T_KILL_TEST.md`` — exact-triality kill test
  (K1 fail).  The structural lesson is the same.
- ``parameter_ledger.md`` — choices made for this audit.
- ``STATUS.md`` — updated to reflect the closed sidecar.
