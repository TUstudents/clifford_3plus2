# Session 15 — Clifford Lie Data Extraction

Status: implemented as a minimal dependency layer for Session 16.

## Purpose

This session is not another representation-theory audit. It extracts the exact
Lie data needed for the next physics-facing step: a 1D `Cl(8)`-internal
checkerboard/QCA continuum-limit test with background `SU(3)` links.

## Deliverable

`clifford_lie.py` exports:

- the 28 positive-chirality bivector generators `1/2 gamma_i gamma_j`;
- the exact octonionic `g2` derivation basis;
- the exact `su3` stabilizer basis fixing `e7`;
- a concrete 14-dimensional complement to `g2` inside the chiral `so(8)` span;
- a concrete 6-dimensional complement to `su3` inside `g2`;
- the six imaginary octonion basis vectors transverse to `e7`.

The complements are practical basis complements, not canonical irrep labels.

## Guardrails

The focused tests assert only the data Session 16 depends on:

- chiral bivectors form a 28-dimensional skew basis;
- `g2` has dimension 14, is skew, and annihilates `e0`;
- `su3` has dimension 8, lies in `g2`, and annihilates `e7`;
- both `g2` and `su3` lie in the chiral bivector span;
- the complements have dimensions 14 and 6.

## Next

Session 16 should use this data layer to build the continuum-limit QCA test:

1. massless 1D checkerboard/Weyl walk on the positive chiral `R8_+` block;
2. background `SU(3)` link variables from the exact `su3` basis;
3. first-order effective generator extraction;
4. gauge covariance and gapless-momentum checks.

That is the first dynamics-level test of the `Cl(8)` pivot.
