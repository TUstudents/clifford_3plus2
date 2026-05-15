from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.qca.bloch_rule import (
    BlochRuleLayerInput,
    bloch_path_a_search_summary,
    bloch_symbol_at_root,
)
from clifford_3plus2_d5.qca.spatial_1d import SpatialHoppingTerm, root_of_unity


ROOT = Path(__file__).resolve().parents[1]


def test_bloch_symbol_evaluates_laurent_terms_at_roots() -> None:
    layer = BlochRuleLayerInput(
        name="identity_shift",
        period=12,
        dimension=10,
        terms=(SpatialHoppingTerm(shift=4, matrix=identity(10)),),
    )

    assert bloch_symbol_at_root(layer, 1) == root_of_unity(12, 1) ** 4 * identity(10)


def test_bloch_path_a_summary_reports_seeded_shape_only() -> None:
    summary = bloch_path_a_search_summary()

    assert summary.candidate_count == 6
    assert summary.seed_guardrail_rejections == 4
    assert summary.unseeded_candidate_count == 2
    assert summary.stable_6_4_band_candidates == 0
    assert summary.topological_pm_candidates == 4
    assert summary.rule_generated_j_section_candidates == 0
    assert summary.strict_bridge_candidates == 0
    assert summary.route_label == "bloch_path_a_seeded_shape_only"
    assert not summary.load_bearing_qca_bridge

    by_name = {candidate.rule_name: candidate for candidate in summary.candidates}
    combined = by_name["path_a_combined_route1_route2"]
    assert not combined.seed_guardrail_passed
    assert combined.algebraic_seed_witnesses == (
        "coefficient_algebra:P_alpha",
        "coefficient_algebra:P_eta",
    )
    assert combined.global_transport_pm_candidate
    assert combined.rule_generated_transported_j_section_count == 0
    assert combined.route_label == "bloch_path_a_seeded_guardrail_rejected"

    identity_shift = by_name["path_a_unseeded_uniform_identity_shift"]
    assert identity_shift.seed_guardrail_passed
    assert not identity_shift.coarse_6_4_band_split_at_all_samples
    assert identity_shift.route_label == "bloch_path_a_no_stable_6_4_band_split"


def test_bloch_path_a_cli_check() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/bloch_path_a_search.py",
            "--json",
            "--check",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["family"] == "bloch_path_a"
    assert payload["candidate_count"] == 6
    assert payload["seed_guardrail_rejections"] == 4
    assert payload["unseeded_candidate_count"] == 2
    assert payload["topological_pm_candidates"] == 4
    assert payload["strict_bridge_candidates"] == 0
    assert payload["route_label"] == "bloch_path_a_seeded_shape_only"
    assert payload["load_bearing_qca_bridge"] is False
