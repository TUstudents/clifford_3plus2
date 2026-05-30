"""Tests for A3-1 — one common H_Q; lepton Sigma recovered."""

from __future__ import annotations

import sympy as sp

from clifford_3plus2_d5.unified_boundary.a31_common_hq import (
    common_chain_transfer_factor,
    common_hq_audit_payload,
    lepton_sigma_matches_k_nu,
    transfer_factor_is_epsilon,
)


def test_common_chain_factor_is_silver_ratio() -> None:
    assert sp.simplify(common_chain_transfer_factor() - (sp.sqrt(2) - 1)) == 0
    assert transfer_factor_is_epsilon()


def test_lepton_sigma_equals_k_nu() -> None:
    assert lepton_sigma_matches_k_nu()


def test_payload_pass() -> None:
    payload = common_hq_audit_payload()
    assert payload.final_verdict == "LEPTON_SIGMA_FROM_COMMON_HQ"
    assert payload.transfer_factor_is_epsilon
    assert payload.lepton_sigma_matches_k_nu
