# lepton — Status

**Status**: positive result through Session 19b; active development possible.

This module derives the Standard Model gauge content (`SU(3) × SU(2)_L ×
U(1)_Y`), a compatible complex structure, and the correct one-generation
hypercharge spectrum on the chiral-16 of Spin(10), with 1+1D massless
Dirac/Weyl continuum dynamics under background gauge links, using the
Pati-Salam factorization of Cl(0,10).

## What is constructed

| Object | From | Choice |
|---|---|---|
| Chiral-16 carrier (real dim 32) | Volume element of Cl(0,10) | — derived |
| Pati-Salam algebra `SU(4) × SU(2)_L × SU(2)_R` | Cl(0,6) ⊗ Cl(0,4) factorization | derived once factorization is chosen |
| Compatible J | Right-quaternionic Cl(0,4) commutant | structurally distinguished; H-unit choice declared |
| Octonion structure | Fano table | declared (1 of 480 tables) |
| Imaginary direction `e_7` | Choice in `Im(O)` | declared |
| SM gauge group `SU(3) × SU(2)_L × U(1)_Y` | PS → SM breaking | conventional |
| Hypercharge spectrum | `Y = T_{3R} + (B-L)/2` with normalization factors | declared (`T_{3R}` factor 1/2 and B-L factor 2/3) |
| 1+1D Dirac/Weyl continuum | Feynman checkerboard tensored with R^32 | derived |
| Background gauge covariance | Standard lattice gauge link | derived |

## Session log

| Session | Result |
|---|---|
| 13 | Exact Cl(0,8) gamma matrices, octonion stabilizer chain G_2 ⊃ SU(3) |
| 14 | Rigid Clifford dynamics: 107 candidates, 7 octonion automorphisms, all SU(3)-flavored |
| 15 | Bivector Lie data export |
| 16 | 1+1D Cl(8) checkerboard continuum + background SU(3) covariance |
| 17 | Cl(0,10) chiral-16 lift; SU(2)_L gap identified |
| 18 | Pati-Salam factorization Cl(0,6) ⊗ Cl(0,4); full PS algebra exposed |
| 19a | SM gauge algebra extraction `SU(3)_c × SU(2)_L × U(1)_Y` |
| 19b | Hypercharge spectrum verified: `{1/6: 6, -2/3: 3, 1/3: 3, -1/2: 2, 1: 1, 0: 1}` |

## Cross-module dependency

This module depends on `obstruction_r10.qca.rule_verdict` for the VerdictProfile
machinery (Lab A / Lab B verdicts). That is the one cross-sidecar coupling.

## Open

Standard open-research problems:

- Dirac mass / Yukawa coupling between chiralities.
- Higgs as condensate.
- Dynamical gauge fields (Yang-Mills, plaquettes).
- 3+1D Lorentz invariance recovery (collaborates with `spacetime_qca`).
- Three generations (Spin(8) triality unexplored).
- SU(3) color rep identification (3 vs 3̄) via Casimir.

## Tests

```bash
uv run pytest src/clifford_3plus2_d5/lepton/tests/ -q
```

Expected: 120+ tests green.
