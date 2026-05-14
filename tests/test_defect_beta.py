from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import sympy as sp

from clifford_3plus2_d5.algebra.matrices import identity
from clifford_3plus2_d5.qca.defect_beta import (
    DEFECT_BETA_COMPATIBLE_CENTRALIZER_DIMENSION,
    DEFECT_BETA_COMPATIBLE_J_MODULI_DIMENSION,
    DEFECT_BETA_EXACT_WORKING_FIELD,
    DEFECT_BETA_I_SECTOR_CENTRALIZER_DIMENSION,
    DEFECT_BETA_OMEGA_SECTOR_CENTRALIZER_DIMENSION,
    DEFECT_BETA_SCALED_RELATION,
    defect_beta_clutching_reflection,
    defect_beta_candidates,
    defect_beta_canonical_j,
    defect_beta_certificate,
    defect_beta_monodromy_core,
    defect_beta_monodromy_operator,
    defect_beta_scaled_omega_operator,
    defect_beta_spectral_projectors,
    defect_beta_transition_functions,
)


ROOT = Path(__file__).resolve().parents[1]


def test_defect_beta_enumerates_defect_charge_patterns() -> None:
    candidates = defect_beta_candidates()

    assert len(candidates) == 10
    assert candidates[0].omega_modes == (0, 1, 2)
    assert candidates[0].i_modes == (3, 4)
    assert all(len(candidate.omega_modes) == 3 for candidate in candidates)
    assert all(len(candidate.i_modes) == 2 for candidate in candidates)


def test_defect_beta_monodromy_is_computed_from_transitions() -> None:
    candidate = defect_beta_candidates()[0]
    transitions = defect_beta_transition_functions(candidate)
    clutching = defect_beta_clutching_reflection()
    monodromy_core = defect_beta_monodromy_core(candidate)
    monodromy = identity(10)
    for transition in transitions:
        monodromy = transition.matrix * monodromy

    assert len(transitions) == 2
    assert transitions[0].matrix != transitions[1].matrix
    assert [transition.matrix.det() for transition in transitions] == [-1, -1]
    assert clutching.det() == -1
    assert clutching * clutching == identity(10)
    assert transitions[0].matrix == clutching
    assert transitions[1].matrix == monodromy_core * clutching
    assert monodromy == monodromy_core
    assert sp.simplify(monodromy) == defect_beta_monodromy_operator(candidate)
    assert monodromy.T * monodromy == identity(10)


def test_defect_beta_spectral_projectors_and_j() -> None:
    candidate = defect_beta_candidates()[0]
    omega_projector, i_projector = defect_beta_spectral_projectors(candidate)
    scaled_omega = defect_beta_scaled_omega_operator(candidate)
    canonical_j = defect_beta_canonical_j(candidate)

    assert omega_projector.rank() == 6
    assert i_projector.rank() == 4
    assert omega_projector * omega_projector == omega_projector
    assert i_projector * i_projector == i_projector
    assert omega_projector + i_projector == identity(10)
    assert sp.simplify(scaled_omega * scaled_omega + 3 * omega_projector) == sp.zeros(10)
    assert sp.simplify(scaled_omega.T * scaled_omega - 3 * omega_projector) == sp.zeros(10)
    assert canonical_j * canonical_j == -identity(10)
    assert canonical_j.T * canonical_j == identity(10)


def test_defect_beta_reports_monodromy_j_and_strict_obstruction() -> None:
    certificate = defect_beta_certificate(defect_beta_candidates()[0])

    assert certificate.exact_working_field == DEFECT_BETA_EXACT_WORKING_FIELD
    assert certificate.transition_count == 2
    assert certificate.monodromy_computed_from_transitions
    assert certificate.entry_exit_transitions_distinct
    assert certificate.transition_determinants == (-1, -1)
    assert certificate.clutching_reflection_determinant == -1
    assert certificate.clutching_identity_passed
    assert certificate.omega_projector_rank == 6
    assert certificate.i_projector_rank == 4
    assert certificate.scaled_omega_relation == DEFECT_BETA_SCALED_RELATION
    assert certificate.scaled_omega_square_relation
    assert certificate.scaled_omega_orthogonality_relation
    assert certificate.scaled_omega_commutes_with_projectors
    assert certificate.i_j_square_relation
    assert certificate.i_j_orthogonality_relation
    assert certificate.scaled_monodromy_certified
    assert certificate.normalized_j_requires_sqrt3
    assert certificate.generated_j_moduli_dimension == 0
    assert (
        certificate.omega_sector_centralizer_dimension
        == DEFECT_BETA_OMEGA_SECTOR_CENTRALIZER_DIMENSION
    )
    assert (
        certificate.i_sector_centralizer_dimension
        == DEFECT_BETA_I_SECTOR_CENTRALIZER_DIMENSION
    )
    assert (
        certificate.compatible_centralizer_dimension
        == DEFECT_BETA_COMPATIBLE_CENTRALIZER_DIMENSION
    )
    assert (
        certificate.compatible_j_moduli_dimension
        == DEFECT_BETA_COMPATIBLE_J_MODULI_DIMENSION
    )
    assert certificate.beta_monodromy_passed
    assert certificate.canonical_j_generated_by_monodromy
    assert certificate.canonical_j_squared_minus_identity
    assert certificate.canonical_j_orthogonal
    assert not certificate.strict_compatible_j_forced
    assert not certificate.pass_strict_rule_to_bridge
    assert certificate.verdict == "monodromy_j_produced_not_strictly_unique"
    assert not certificate.load_bearing_qca_bridge


def test_defect_beta_cli_single_pattern() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/defect_beta_search.py",
            "--json",
            "--check",
            "--pattern-index",
            "0",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["candidate_count"] == 1
    assert payload["monodromy_candidates"] == 1
    assert payload["scaled_monodromy_certified_candidates"] == 1
    assert payload["generated_j_moduli_dimension"] == 0
    assert payload["compatible_centralizer_dimension"] == 26
    assert payload["compatible_j_moduli_dimension"] == 9
    assert payload["results"][0]["generated_j_moduli_dimension"] == 0
    assert payload["results"][0]["compatible_centralizer_dimension"] == 26
    assert payload["results"][0]["compatible_j_moduli_dimension"] == 9
    assert payload["results"][0]["entry_exit_transitions_distinct"] is True
    assert payload["results"][0]["transition_determinants"] == [-1, -1]
    assert payload["results"][0]["clutching_identity_passed"] is True
    assert payload["strict_compatible_j_forced_candidates"] == 0
    assert payload["strict_bridge_candidates"] == 0
    assert payload["verdict_counts"] == {
        "monodromy_j_produced_not_strictly_unique": 1
    }
    assert payload["load_bearing_qca_bridge"] is False
