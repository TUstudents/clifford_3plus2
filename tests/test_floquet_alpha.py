from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.qca.floquet_alpha import (
    ALPHA_PHASE,
    ETA_PHASE,
    FLOQUET_ALPHA_EXACT_WORKING_FIELD,
    FLOQUET_ALPHA_SCALED_RELATION,
    floquet_alpha_candidates,
    floquet_alpha_canonical_j,
    floquet_alpha_operator,
    floquet_alpha_polarization_certificate,
    floquet_alpha_rule_to_verdict,
    floquet_alpha_scaled_alpha_operator,
    floquet_alpha_spectral_projectors,
    pair_rotation,
)


ROOT = Path(__file__).resolve().parents[1]


def test_pair_rotations_have_exact_orders() -> None:
    alpha_rotation = pair_rotation(0, ALPHA_PHASE)
    eta_rotation = pair_rotation(0, ETA_PHASE)

    assert alpha_rotation.T * alpha_rotation == identity(10)
    assert sp.simplify(alpha_rotation**3) == identity(10)
    assert sp.simplify(eta_rotation**4) == identity(10)
    assert alpha_rotation != eta_rotation


def test_floquet_alpha_enumerates_single_parameter_patterns() -> None:
    candidates = floquet_alpha_candidates()

    assert len(candidates) == 10
    assert candidates[0].alpha_modes == (0, 1, 2)
    assert candidates[0].eta_modes == (3, 4)
    assert all(len(candidate.alpha_modes) == 3 for candidate in candidates)
    assert all(len(candidate.eta_modes) == 2 for candidate in candidates)


def test_floquet_alpha_operator_is_one_mandatory_layer() -> None:
    candidate = floquet_alpha_candidates()[0]
    operator = floquet_alpha_operator(candidate)

    assert operator.T * operator == identity(10)
    assert operator.shape == (10, 10)
    assert operator[0, 5] == -sp.sqrt(3) / 2
    assert operator[3, 8] == -1


def test_floquet_alpha_generates_coarse_center_without_rank_one() -> None:
    result = floquet_alpha_rule_to_verdict(floquet_alpha_candidates()[0])

    assert [item.rank for item in result.central_idempotents] == [0, 4, 6, 10]
    assert result.complementary_rank_6_4_pairs == 1
    assert result.lower_rank_central_idempotents == ()
    assert not result.forced_j_found
    assert not result.pass_rule_to_bridge
    assert result.verdict == "candidate_only_j_not_forced"


def test_floquet_alpha_spectral_projectors_produce_canonical_j() -> None:
    candidate = floquet_alpha_candidates()[0]
    alpha_projector, eta_projector = floquet_alpha_spectral_projectors(candidate)
    scaled_alpha = floquet_alpha_scaled_alpha_operator(candidate)
    canonical_j = floquet_alpha_canonical_j(candidate)

    assert alpha_projector.rank() == 6
    assert eta_projector.rank() == 4
    assert alpha_projector * alpha_projector == alpha_projector
    assert eta_projector * eta_projector == eta_projector
    assert alpha_projector + eta_projector == identity(10)
    assert sp.simplify(scaled_alpha * scaled_alpha + 3 * alpha_projector) == sp.zeros(10)
    assert sp.simplify(scaled_alpha.T * scaled_alpha - 3 * alpha_projector) == sp.zeros(10)
    assert canonical_j * canonical_j == -identity(10)
    assert canonical_j.T * canonical_j == identity(10)


def test_floquet_alpha_plus_reports_polarization_j_and_strict_obstruction() -> None:
    certificate = floquet_alpha_polarization_certificate(floquet_alpha_candidates()[0])

    assert certificate.exact_working_field == FLOQUET_ALPHA_EXACT_WORKING_FIELD
    assert certificate.alpha_projector_rank == 6
    assert certificate.eta_projector_rank == 4
    assert certificate.scaled_alpha_relation == FLOQUET_ALPHA_SCALED_RELATION
    assert certificate.scaled_alpha_square_relation
    assert certificate.scaled_alpha_orthogonality_relation
    assert certificate.scaled_alpha_commutes_with_projectors
    assert certificate.eta_j_square_relation
    assert certificate.eta_j_orthogonality_relation
    assert certificate.scaled_polarization_certified
    assert certificate.normalized_j_requires_sqrt3
    assert certificate.alpha_plus_polarization_passed
    assert certificate.canonical_j_generated_by_floquet
    assert certificate.canonical_j_squared_minus_identity
    assert certificate.canonical_j_orthogonal
    assert not certificate.strict_compatible_j_forced
    assert not certificate.pass_strict_rule_to_bridge
    assert certificate.verdict == "polarization_j_produced_not_strictly_unique"
    assert not certificate.load_bearing_qca_bridge


def test_floquet_alpha_cli_searches_all_patterns() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/floquet_alpha_search.py",
            "--json",
            "--check",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["candidate_count"] == 10
    assert payload["rank_6_4_pair_candidates"] == 10
    assert payload["rank_one_falsified_candidates"] == 0
    assert payload["bridge_candidates"] == 0
    assert payload["verdict_counts"] == {"candidate_only_j_not_forced": 10}
    assert payload["load_bearing_qca_bridge"] is False


def test_floquet_alpha_plus_cli_searches_all_patterns() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/floquet_alpha_plus_search.py",
            "--json",
            "--check",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["candidate_count"] == 10
    assert payload["polarization_j_candidates"] == 10
    assert payload["scaled_polarization_certified_candidates"] == 10
    assert payload["strict_compatible_j_forced_candidates"] == 0
    assert payload["strict_bridge_candidates"] == 0
    assert payload["verdict_counts"] == {
        "polarization_j_produced_not_strictly_unique": 10
    }
    assert payload["load_bearing_qca_bridge"] is False
