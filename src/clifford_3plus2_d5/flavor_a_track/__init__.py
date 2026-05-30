"""Flavor A-track — one boundary, all sectors (roadmap Track A: A2, A3a, A3b).

The A-track is a single block: it tests whether the Standard-Model flavor pattern
is the spectral data of *one* boundary operator ``H_Q`` read through four
quantum-number projections ``Sigma_f(z) = V_f^dagger (z - H_Q)^{-1} V_f``, and
then audits how much of the CKM/PMNS texture is genuinely derived. It runs three
kill-disciplined phases, each with its own top-level audit:

A2 — universality (``universality_audit``). Necessary conditions for one-boundary
universality:
  * U1 shared transfer invariant: every sector is a power of the single residual
    K3 root ``epsilon = sqrt(2)-1`` (graph-tracking + independent-epsilon controls).
  * U2 sector difference = color label: quark and lepton shells share the ``1+2``
    non-color core; quarks add exactly three color ports.
  * U3 coupling catalog: each field's chiral-16 multiplicity factors as
    ``color x weak``; ``V_f`` is the SM quantum-number projection.
  Verdict ``UNIVERSAL_BOUNDARY_NECESSARY_CONDITIONS_PASS`` (can KILL, cannot
  confirm — the full ``Sigma_f`` reproduction is A3).

A3a — unified transfer boundary (``unified_boundary_audit``). One common sterile
chain ``H_Q``: the lepton core ``K_nu = eps^2 P_u + P_b`` is its Schur complement
(A3-1), the quark transfer amplitudes ``eps^2, eps^4, eps^6`` are powers of the
*same* chain factor (A3-2), and the sector-specific structure (color ``C_F=4/3``,
BCC ``sqrt(2)``/``1/sqrt(2)``, coin ``atan(sqrt(5))``) lives in ``V_f`` (A3-3).
Verdict ``UNIFIED_TRANSFER_BOUNDARY_PASS``.

A3b — texture provenance / "derive or count" (``texture_provenance_audit``). The
honest ledger of the texture factors A3a left as inputs:
  * Derived (B1, machine-checked): ``C_F``, the coin base ``sqrt(5)`` from
    ``Gamma_q^2 = 5I = (2_BCC + 3_color)``, the BCC Clebsches, the charged-lepton
    ``sqrt(3/2)``, and the V10 leptonic phase word — giving the CP phases
    ``atan(sqrt(5))`` and ``5 pi/12`` and the PMNS angle structure.
  * Free input (B2): the depth embedding ``{0,2,6}`` (fit to the CKM hierarchy),
    the charged-lepton depth, the ``r=1`` ergodicity prior, the CP branch.
  * Count (B3): ``N_free = 4 < N_observables = 8`` (surplus 4) — predictive for
    structure, not numerology.
  Verdict ``TEXTURE_STRUCTURE_DERIVED_HIERARCHY_INPUT``; the one remaining input
  ``generation_depth_embedding_derived`` (deriving ``{0,2,6}`` = a generation
  mechanism, ``N=3`` empirical) is recorded, not attempted.

All gates borrow from ``boundary_response`` / ``lepton`` through ``reuse.py``; no
boundary physics is redefined here.
"""
