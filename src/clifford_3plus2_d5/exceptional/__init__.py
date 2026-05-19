"""Exceptional algebra sidecar — three-generation kill-test.

Tests whether the Boyle exceptional Jordan algebra J_3(O) (or its
complexification J_3^C(O)) can carry three SM generations of the chiral-16
type under Spin(10) ⊂ E_6.

Decision tree (kill-disciplined):

- Phase 0a: bimultiplication-of-T quick check (confirm triality fail).
- Phase 0b: three Fano lines through e_7 (likely fail SM-shape).
- Phase 1: build J_3(O) (27-dim Jordan algebra).
- Phase 2: decompose 27 under Spin(10) -> expected 16 + 10 + 1 (one
  generation, not three).
- Phase 2b: complexify to J_3^C(O) and re-decompose -> expected
  (16+10+1) + conj.
- Phase 3 (only if Phase 2 or 2b passes): Yukawa + CP across generations.

See ``PLAN.md`` for the full audit design and decision tree.
"""
