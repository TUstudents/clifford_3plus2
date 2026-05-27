"""Boundary-response sidecar for the residual BCC flavor core audit.

This package is intentionally narrow.  It audits only the operator path

    H_Q, V_a -> Sigma(z) -> K_nu = epsilon^2 P_u + P_b

and exports no PMNS or CKM texture machinery.
"""

from clifford_3plus2_d5.boundary_response.audit import (
    BoundaryCoreAuditPayload,
    boundary_core_audit_payload,
)
from clifford_3plus2_d5.boundary_response.charged_lepton_leakage import (
    ChargedLeptonLeakageAuditPayload,
    charged_lepton_leakage_audit_payload,
)
from clifford_3plus2_d5.boundary_response.framed_sterile import (
    FramedSterileAuditPayload,
    framed_sterile_audit_payload,
)
from clifford_3plus2_d5.boundary_response.impedance import (
    ImpedanceAuditPayload,
    impedance_audit_payload,
)
from clifford_3plus2_d5.boundary_response.leptonic_phase_word import (
    LeptonicPhaseWordAuditPayload,
    leptonic_phase_word_audit_payload,
)
from clifford_3plus2_d5.boundary_response.product_sterile import (
    ProductSterileAuditPayload,
    product_sterile_audit_payload,
)
from clifford_3plus2_d5.boundary_response.weyl_sterile import (
    WeylSterileAuditPayload,
    weyl_sterile_audit_payload,
)
from clifford_3plus2_d5.boundary_response.explicit_hq import (
    ExplicitHqAuditPayload,
    explicit_hq_audit_payload,
)
from clifford_3plus2_d5.boundary_response.residual_basis import (
    k_nu_operator,
    residual_projectors,
)
from clifford_3plus2_d5.boundary_response.transfer import (
    epsilon,
    epsilon_fourth,
)

__all__ = [
    "BoundaryCoreAuditPayload",
    "ChargedLeptonLeakageAuditPayload",
    "ExplicitHqAuditPayload",
    "FramedSterileAuditPayload",
    "ImpedanceAuditPayload",
    "LeptonicPhaseWordAuditPayload",
    "ProductSterileAuditPayload",
    "WeylSterileAuditPayload",
    "boundary_core_audit_payload",
    "charged_lepton_leakage_audit_payload",
    "epsilon",
    "epsilon_fourth",
    "explicit_hq_audit_payload",
    "framed_sterile_audit_payload",
    "impedance_audit_payload",
    "k_nu_operator",
    "leptonic_phase_word_audit_payload",
    "product_sterile_audit_payload",
    "residual_projectors",
    "weyl_sterile_audit_payload",
]
