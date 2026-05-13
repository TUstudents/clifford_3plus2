# Falsifiers

The project should return `notation_only` or `falsified` when any load-bearing
datum is missing or chosen independently of QCA geometry.

1. QCA two-plane is arbitrary.
2. Complex structure `J` is chosen by hand.
3. SU(5) embedding is chosen independently of QCA.
4. Gate algebra allows `V_3 <-> V_2` block mixing.
5. Gate algebra is block-diagonal but not in the `SU(3) x SU(2)` commutant.
6. Gate algebra resolves individual `C^3` basis states.
7. Candidate `J` has no explicit operator matrix.
8. Candidate `J` fails `J^2 = -I`.
9. Candidate `J` fails to preserve `P3` and `P2`.
10. Candidate `J` is not in the allowed QCA gate algebra.
11. Candidate matrices use floating-point entries for exact Clifford or
    projector checks.
12. Project only reproduces Spin(10) branching.

## Implemented Guardrails

The current code enforces these boundaries:

- Missing `data/qca_data.json` returns `notation_only`.
- Incomplete or invalid JSON returns `notation_only`.
- Float matrix entries are rejected.
- Schema contract requires rational strings for exact matrix entries.
- `complex_structure_origin = "by_hand"` is falsifying.
- `complex_structure_origin = "unknown"` cannot pass.
- Off-block one-particle gate generators fail.
- Block-diagonal one-particle gates that resolve color-basis directions fail.

The remaining future guardrail is stronger provenance checking: real QCA data
must justify why the selected two-plane and `J` are structural, not convenient.
