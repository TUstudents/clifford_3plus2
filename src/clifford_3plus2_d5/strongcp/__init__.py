"""Strong-CP audit: BCC walk contribution to θ_QCD.

Question: does the BCC Bialynicki-Birula walk's effective action
contribute to θ_QCD, and if so at what order in ε relative to the
neutron-EDM bound |θ_QCD| ≤ 10⁻¹⁰?

Three pre-named outcomes:
- STRONG-CP TRIVIAL: θ = 0 to all orders by structural lattice symmetry.
- STRONG-CP SAFE: θ = O(ε^n) for n ≥ 1 → far below 10⁻¹⁰.
- STRONG-CP TENSION: θ > 10⁻¹⁰ → conflicts with bound.

The structural observation: H^(1) (verified by cp/) lives in T_{2g},
a g-irrep of the cubic group O_h.  The θ_QCD operator E·B has
momentum-shape k_x k_y k_z living in A_{2u} (u-irrep).  By the
parity selection rule (g × g = g, u × u = g, g × u = u), no power
of g-irrep operators produces a u-irrep — so if all H^(n) stay in
g-irreps, the BCC walk's contribution to θ_QCD vanishes structurally.
"""
