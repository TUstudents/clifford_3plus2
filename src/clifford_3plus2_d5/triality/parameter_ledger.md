# Triality Sidecar Parameter Ledger

This document lists every choice — continuous, discrete, embedding-related,
or convention-related — that the triality sidecar relies on.

## Discrete / embedding choices

1. **Spin(8) embedding inside Spin(10)**
   - Choice: gamma indices `{0, 1, 2, 3, 4, 5, 6, 7}` from
     ``patisalam_cl010_gamma_matrices()``.
   - Where: ``spin8_triality.spin8_bivector_indices()`` /
     ``spin8_triality.spin8_generator_on_chiral16``.
   - Status: free (one-of-several).  Alternatives not tried in this
     pass: `{0..5, 8, 9}`, `{0..3, 6..9}`, etc.
   - Real DOF: 1 discrete pick from a finite set of natural index
     subsets.

2. **Cartan basis for Spin(8)**
   - Choice: ``H_k = (1/2) γ_{2k} γ_{2k+1}`` for `k = 0, 1, 2, 3`.
   - Where: ``spin8_triality.spin8_cartan_pairs`` returns
     `((0, 1), (2, 3), (4, 5), (6, 7))`.
   - Status: free (one-of-several).  Any rank-4 commuting subalgebra
     would work; this one is aligned with the Pati-Salam factorization
     ``Cl(0,6) ⊗ Cl(0,4)`` in lepton.
   - Real DOF: 1 discrete pick.  The Pati-Salam alignment makes it
     natural but not unique.

3. **Triality construction**
   - Choice: Cartan-action-derived.  The 4×4 ``T_cartan`` is fixed by
     requiring triality to permute the outer D_4 Dynkin nodes
     ``α_1, α_3, α_4`` cyclically while fixing the central node ``α_2``.
   - Where: ``spin8_triality.triality_cartan_matrix``.
   - Status: forced (up to Z/2 inversion `τ ↔ τ²`).
   - Real DOF: 1 discrete binary (`τ` vs `τ²`).

4. **Triality direction (`τ` vs `τ²`)**
   - Choice: the cycle ``α_1 → α_3 → α_4 → α_1``.
   - Where: docstring of ``triality_cartan_matrix``.
   - Status: free binary.  Swapping to the inverse cycle gives the same
     outer automorphism class; the K1 outcome is identical.
   - Real DOF: 1 discrete binary.

5. **Convention: real-32 carrier vs complex-16 carrier**
   - Choice: real-32 chiral-16 carrier via
     ``patisalam_chiral16_block_matrix``.  All generators are 32×32 real-skew.
   - Where: inherited from ``clifford_patisalam`` (no fresh choice here).
   - Status: forced by the lepton conventions.
   - Real DOF: 0.

## Derived (not free) choices

6. **Restricted hypercharge ``Y'``**
   - Definition: least-squares projection of
     ``physical_hypercharge_generator()`` onto the Spin(8) Cartan span.
   - Where: ``sm_restriction.restricted_hypercharge_cartan_coords``.
   - Result: `Y' = (1/3, 1/3, 1/3, 1/2)` in `(H_0, H_1, H_2, H_3)` basis.
   - Status: derived from choices 1, 2, and the physical hypercharge from
     ``lepton``.
   - Real DOF: 0.

7. **SU(3)_c Cartan inside Spin(8) Cartan**
   - Definition: those generators of ``su3_c_generators_from_su4`` whose
     Cartan coordinates are non-trivial.
   - Where: ``sm_restriction.su3_c_cartan_indices``.
   - Result: indices `(4, 7)`, giving Cartan coords `(-1, 1, 0, 0)` and
     `(-1, 0, 1, 0)`.
   - Status: derived from the existing ``patisalam_sm`` definition of
     ``su3_c``.
   - Real DOF: 0.

## Continuous parameters

None — the kill test introduces no continuous parameters.  Mass /
mixing / breaking parameters are deferred until the kill test passes.

## Convention / basis choices inherited from lepton

These are not choices made by the triality sidecar but are dependencies
that propagate to the K1 outcome:

- **Pati-Salam complex structure ``J``** — chosen as the right-quaternionic
  unit in the ``Cl(0,4)`` commutant (``patisalam_chosen_complex_structure``).
  Affects the Y' charge observable (``-J · Y'``), so the K2 spectrum
  depends on it.
- **B-L direction inside SU(4)** — Cartan sum `su4[0] + su4[9] + su4[14]`
  (``b_minus_l_generator_from_su4``).  Affects which two su3_c generators
  end up in the Cartan span.
- **Hypercharge normalization** — physical `Y = T3_R/2 + (B-L)/3` (the
  Session 19b convention).  Affects the Y' Cartan coords numerically.

These would only become questions for the triality sidecar if K1 had
passed; since K1 failed, they are pinned as the ledger states.

## Total free degrees of freedom

Four discrete choices: embedding (1), Cartan basis (2), triality
construction (3, 4).  Choices 3 and 4 are essentially fixed once choices
1 and 2 are made (only the binary `τ` vs `τ²` remains, and that does not
affect K1 outcome).

So the effective number of free choices for the K1 audit is **two**:
the Spin(8) embedding (#1) and the Cartan basis (#2).  Both are taken to
align naturally with the Pati-Salam factorization.

A negative K1 result on these natural-aligned choices kills the program
on the natural alignment.  A second pass under a non-natural embedding
could in principle yield a different verdict, but would require its own
justification.
