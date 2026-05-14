from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.qca.spatial_1d import (
    alpha_eta_projectors,
    root_of_unity,
    spatial_1d_alpha_certificate,
    spatial_alpha_prototype,
    spatial_orientation_orbits,
    transfer_matrix_at_root,
)


ROOT = Path(__file__).resolve().parents[1]


def test_spatial_transfer_rule_builds_exact_root_of_unity_matrix() -> None:
    rule = spatial_alpha_prototype()
    transfer = transfer_matrix_at_root(rule, 1)
    zeta = root_of_unity(12, 1)

    assert transfer.shape == (10, 10)
    assert transfer[0, 0] == sp.simplify(zeta**4)
    assert transfer[3, 3] == sp.simplify(zeta**3)
    assert transfer[5, 5] == sp.simplify(zeta**4)
    assert transfer[8, 8] == sp.simplify(zeta**3)


def test_spatial_root_samples_are_unitary() -> None:
    rule = spatial_alpha_prototype()

    for sample in range(rule.period):
        transfer = transfer_matrix_at_root(rule, sample)
        assert sp.simplify(transfer.conjugate().T * transfer - identity(10)) == sp.zeros(10)


def test_spatial_alpha_eta_rank_profile() -> None:
    rule = spatial_alpha_prototype()
    alpha_projector, eta_projector = alpha_eta_projectors(rule)

    assert alpha_projector.rank() == 6
    assert eta_projector.rank() == 4
    assert alpha_projector + eta_projector == identity(10)
    assert alpha_projector * eta_projector == sp.zeros(10)


def test_spatial_orientation_orbits_couple_block_signs() -> None:
    rule = spatial_alpha_prototype()
    orbits = spatial_orientation_orbits(rule)
    allowed = tuple(orbit for orbit in orbits if orbit.transport_allowed)

    assert len(orbits) == 4
    assert {(orbit.alpha_sign, orbit.eta_sign) for orbit in allowed} == {
        (1, 1),
        (-1, -1),
    }


def test_spatial_1d_alpha_certificate_reports_sign_coupling() -> None:
    certificate = spatial_1d_alpha_certificate()

    assert certificate.period == 12
    assert certificate.alpha_winding == 4
    assert certificate.eta_winding == 3
    assert certificate.winding_gcd == 1
    assert certificate.winding_lcm == 12
    assert certificate.locality_radius == 4
    assert certificate.transfer_unitary_on_samples
    assert certificate.coarse_6_4_band_split
    assert certificate.orientation_choices_before_transport == 4
    assert certificate.orientation_choices_after_transport == 2
    assert certificate.sign_coupled_to_global_pm
    assert certificate.strict_bridge_candidates == 0
    assert certificate.route_label == "spatial_signs_coupled_to_global_pm"
    assert not certificate.load_bearing_qca_bridge


def test_spatial_1d_alpha_cli_check() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/spatial_1d_alpha_search.py",
            "--json",
            "--check",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["family"] == "spatial_1d_alpha"
    assert payload["candidate_count"] == 1
    assert payload["unitary_candidates"] == 1
    assert payload["coarse_6_4_band_candidates"] == 1
    assert payload["orientation_choices_before_transport"] == 4
    assert payload["orientation_choices_after_transport"] == 2
    assert payload["sign_coupled_candidates"] == 1
    assert payload["strict_bridge_candidates"] == 0
    assert payload["route_label"] == "spatial_signs_coupled_to_global_pm"
    assert payload["load_bearing_qca_bridge"] is False
