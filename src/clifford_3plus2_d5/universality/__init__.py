"""Cross-sector universality audit (roadmap gate A2).

The strong universality claim: flavor differences between the sectors
(nu, e, u, d) come entirely from the couplings ``V_f`` — which Standard-Model
quantum numbers connect to the boundary — rather than from different boundary
operators ``H_Q``.  If true, the whole dimensionless flavor pattern is the
spectral data of one ``H_Q`` read through four projections
``Sigma_f(z) = V_f^dagger (z - H_Q)^{-1} V_f``.

This sidecar tests the *necessary conditions* for that claim and can return a
clean KILL.  It does NOT prove universality: the full numerical reproduction of
every ``Sigma_f`` from one ``H_Q`` on lepton's chiral-16 carrier is the flavor
program (roadmap A3) and is explicitly deferred.

Gates:

* U1 — shared transfer invariant: all sectors use the *same* ``epsilon``, and it
  is the decaying root of the residual ``K_3`` graph (a graph-tracking control
  varies ``K_2/K_3/K_4`` and a negative control shows an independent-``epsilon``
  sector is rejected).
* U2 — sector difference is the color quantum number: the quark and lepton
  boundary shells differ by *exactly* the three color ports; their non-color
  cores are the same ``1 + 2`` residual structure.
* U3 — coupling catalog: the per-sector couplings ``V_f`` are SM quantum-number
  projections (color singlet/triplet x weak doublet/singlet x hypercharge), so
  the flavor difference lives in ``V_f``.
"""
