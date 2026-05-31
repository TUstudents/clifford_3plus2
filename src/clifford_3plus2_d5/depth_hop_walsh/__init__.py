"""Raw BCC hop-shell Walsh-support probe (Claim A of the depth mechanism).

Tests whether the genuine Bialynicki-Birula BCC Weyl hop source carries the
primitive ``[111]`` parity-graded cube tower ``A1g(0) + T1u(1) + A2u(3)`` with no
degree-2 even quadrupole ``T2g`` — the angular selection rule the family-depth
conjecture (depths {0,2,6}) needs. See docs/depth_hierarchy_mechanism_review.md.

The object decomposed is the **coefficient-Walsh transform of the eight 2x2 hop
matrices** ``H_v`` over the cube directions ``v in {+-1}^3``:

    Hhat_S = (1/8) sum_v chi_S(v) H_v,   chi_S(v) = prod_{i in S} v_i,

with O_h assignment by Walsh degree |S|: A1g(|S|=0), T1u(1), T2g(2), A2u(3), and
depth = 2|S|. This is the primitive hop-shell alphabet — NOT the Taylor degree of
the Bloch symbol h(k) (the exponential makes a scalar source generate harmless
quadratic Taylor descendants), and NOT the BCH effective Hamiltonian (that is the
low-energy Lorentz grammar; it lives in the W3 diagnostic only).

Gates:
* W1 (hop_walsh_decomposition) — the Walsh coefficients, their O_h/parity
  assignment, and a C3-about-[111] covariance check (coefficient-Walsh vs full
  covariant irrep).
* W2 (hop_walsh_support_audit) — PRIMARY. The [111]-singlet support test, evaluated
  separately per helicity, with granular kill provenance.
* W3 (effective_hamiltonian_diagnostic) — DIAGNOSTIC ONLY. Reports the strong-CP
  effective-Hamiltonian content (A2u(H_eff)=0, H^(1) != 0); never passes or kills.

A combined PASS establishes Claim A only (the angular 0,1,3 source ladder). It does
NOT derive the depths: the bridge ``d_radial = 2 * Walsh-degree`` (Claim B) and the
``sqrt(5) = sqrt(2_BCC + 3_color)`` compatibility (Claim C) remain unproven.
"""
