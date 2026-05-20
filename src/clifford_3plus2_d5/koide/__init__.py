"""Koide formula audit: does BCC body-diagonal Z₃ predict K = 2/3?

The Koide formula (1981) is the empirical relation

    K = (m_e + m_μ + m_τ) / (√m_e + √m_μ + √m_τ)²  =  2/3

verified to ~10⁻⁴ for charged leptons.  Geometrically: the 3-vector
(√m_e, √m_μ, √m_τ) makes an angle of exactly 45° with (1, 1, 1)/√3.

The striking coincidence: (1, 1, 1)/√3 is the BCC body-diagonal —
exactly the rotation axis audited in topology/SC-2.  This sidecar
tests whether the program's structure naturally places the lepton
mass-vector on the Koide cone.

Three pre-named outcomes:
- KOIDE PREDICTED: structure forces K = 2/3.
- KOIDE CONSISTENT: structure permits K = 2/3 but does not force.
- KOIDE CONFLICT: structure forbids K = 2/3.

Mass-vector identification: σ^a-axes (from cp/H^(1) T_{2g} structure)
↔ generations (e, μ, τ).  See PLAN.md for full design.
"""
