"""Tests for the CUSP-PLAN target certificates."""

import sympy as sp

from clifford_3plus2_d5.cusp.targets import (
    BCCEdgeQClock,
    CenterChargeRecirculationAutomaton,
    CenterClosureSource,
    CenterHolonomyPath,
    CoefficientPathMeasure,
    CuspBoundaryDilationAudit,
    CuspBoundaryMaterialSource,
    CuspBoundaryMaterialOriginAudit,
    CuspCenterTopologyAudit,
    CuspCoefficientMeasureAudit,
    CuspRightChargeOriginAudit,
    CuspShearMatchingPrincipleAudit,
    NumericalSemigroup,
    QLocalLockingOriginAudit,
    RecirculationGraph,
    additive_center_control_powers,
    bilinear_center_linking_powers,
    complete_graph_distance_center_powers,
    cusp_boundary_dilation_audit,
    cusp_boundary_material_origin_audit,
    cusp_boundary_material_source,
    cusp_center_topology_audit,
    cusp_coefficient_measure_audit,
    cusp_module_path_amplitudes,
    cusp_module_path_count,
    cusp_right_charge_origin_audit,
    cusp_shear_matching_principle_audit,
    cusp_targets_payload,
    cyclic_difference_center_powers,
    exact_omega,
    finite_recirculation_graph_from_automata,
    finite_recirculation_graph_from_sources,
    graph_low_valuation_summary,
    minimal_center_recirculation_walk_graph,
    oriented_double_up_right_charges,
    path_flag_distance_center_powers,
    path_measure_from_center_powers,
    subtract_conductor_floor,
    standard_model_center_automata,
    target_a_neighbor_walk_graphs,
    target_a_payload,
    target_b_payload,
    target_c_payload,
    target_d_payload,
)


def test_bcc_edge_q_clock_splits_same_normal_and_leakage_branches() -> None:
    q_clock = BCCEdgeQClock()

    assert q_clock.same_normal_count == 2
    assert q_clock.mixed_normal_count == 2
    assert q_clock.same_normal_delta_q_values == (0, 0)
    assert sorted(q_clock.mixed_normal_delta_q_values) == [-2, 2]
    assert q_clock.q_clock_pass


def test_cusp_boundary_material_source_derives_weak_source_from_bcc_parity() -> None:
    material = cusp_boundary_material_source()

    assert isinstance(material, CuspBoundaryMaterialSource)
    assert material.q_clock.q_clock_pass
    assert material.same_normal_norm == sp.eye(2) / 2
    assert material.mixed_normal_norm == sp.eye(2) / 2
    assert material.total_norm_is_identity
    assert material.norm_split_pass
    assert material.weak_source.name == "weak_SU2_doublet"
    assert material.weak_source.center_order == material.q_clock.same_normal_count == 2
    assert material.weak_source.channel_count == material.q_clock.same_normal_count
    assert material.weak_source.primitive_closure_length == 2
    assert material.weak_source_from_bcc_pass
    assert material.color_source.name == "color_SU3_triplet"
    assert material.color_source.center_order == 3
    assert material.color_source.channel_count == 3
    assert material.color_source.primitive_closure_length == 3
    assert material.color_source_pass
    assert material.material_source_pass


def test_cusp_boundary_dilation_audit_completes_local_bb_isometry() -> None:
    audit = cusp_boundary_dilation_audit()

    assert isinstance(audit, CuspBoundaryDilationAudit)
    assert audit.local_isometry.shape == (8, 2)
    assert audit.local_isometry_norm == sp.eye(2)
    assert audit.local_isometry_pass
    assert audit.local_input_dimension == 2
    assert audit.local_output_dimension == 8
    assert audit.unitary_completion_nullity == 6
    assert audit.unitary_dilation_exists_pass
    assert audit.finite_gap_feedback_scalar
    assert audit.hard_gap_feedback_zero
    assert audit.retarded_feedback_limit_zero
    assert audit.retarded_visible_powers_match_survival
    assert audit.recurrent_wedge_return == sp.Matrix(
        [[-(1 + sp.I) / 4, 0], [0, -(1 - sp.I) / 4]]
    )
    assert audit.recurrent_wedge_return_nonzero
    assert audit.recurrent_visible_powers_differ
    assert audit.outgoing_leakage_pass
    assert audit.boundary_dilation_pass
    assert "single_clock_locking_field" in audit.remaining_declared_inputs[0]


def test_q_local_locking_origin_selects_unique_q_squared_penalty() -> None:
    audit = cusp_boundary_material_origin_audit()
    locking = audit.q_locking
    q = sp.symbols("q")
    g = locking.stiffness

    assert isinstance(audit, CuspBoundaryMaterialOriginAudit)
    assert isinstance(locking, QLocalLockingOriginAudit)
    assert locking.selected_polynomial == g * q**2
    assert locking.selected_penalty_values == (0, 4 * g, 4 * g)
    assert locking.selected_gap == 4 * g
    assert locking.visible_penalty_zero
    assert locking.mixed_penalties_degenerate
    assert locking.mixed_penalty_positive
    assert locking.uniqueness_solution[sp.symbols("a")] == g
    assert locking.uniqueness_solution[sp.symbols("b")] == 0
    assert locking.uniqueness_solution[sp.symbols("c")] == 0
    assert locking.linear_control_values == (0, -2, 2)
    assert locking.constant_control_values == (1, 1, 1)
    assert locking.linear_control_rejected
    assert locking.constant_control_rejected
    assert locking.q_locking_origin_pass


def test_causal_outgoing_origin_selects_retarded_closure_over_recurrence() -> None:
    audit = cusp_boundary_material_origin_audit()

    assert audit.outgoing.causal_rule == "no incoming mixed-normal clock-error data"
    assert audit.outgoing.retarded_visible_powers_match_survival
    assert audit.outgoing.recurrent_wedge_return == sp.Matrix(
        [[-(1 + sp.I) / 4, 0], [0, -(1 - sp.I) / 4]]
    )
    assert audit.outgoing.recurrent_wedge_return_nonzero
    assert audit.outgoing.recurrent_visible_powers_differ
    assert audit.outgoing.outgoing_origin_pass
    assert audit.material_origin_pass
    assert "q_reflection_positive_stiffness" in audit.remaining_physical_axioms[0]


def test_center_closure_source_forbids_one_step_and_closes_at_center_order() -> None:
    weak = CenterClosureSource("weak", center_order=2, channel_count=2)
    color = CenterClosureSource("color", center_order=3, channel_count=3)

    assert weak.center_charge(1) == 1
    assert color.center_charge(1) == 1
    assert weak.one_step_forbidden
    assert color.one_step_forbidden
    assert weak.primitive_closure_length == 2
    assert color.primitive_closure_length == 3


def test_center_charge_automata_force_visible_return_orders() -> None:
    weak, color = standard_model_center_automata()

    assert isinstance(weak, CenterChargeRecirculationAutomaton)
    assert weak.source.name == "weak_SU2_doublet"
    assert color.source.name == "color_SU3_triplet"
    assert weak.charge_table(6) == (0, 1, 0, 1, 0, 1, 0)
    assert color.charge_table(6) == (0, 1, 2, 0, 1, 2, 0)
    assert weak.return_ticks_through(6) == (2, 4, 6)
    assert color.return_ticks_through(6) == (3, 6)
    assert weak.one_step_forbidden
    assert color.one_step_forbidden
    assert weak.primitive_return_length == 2
    assert color.primitive_return_length == 3


def test_finite_recirculation_walk_graph_recovers_primitive_two_three_closures() -> None:
    graph = minimal_center_recirculation_walk_graph()
    automaton_graph = finite_recirculation_graph_from_automata(
        standard_model_center_automata()
    )
    source_graph = finite_recirculation_graph_from_sources(
        (
            CenterClosureSource("weak_SU2_doublet", center_order=2, channel_count=2),
            CenterClosureSource("color_SU3_triplet", center_order=3, channel_count=3),
        )
    )

    assert graph.edges == automaton_graph.edges
    assert graph.one_step_forbidden
    assert graph.closed_walk_counts_through(6) == (1, 0, 2, 3, 4, 12, 17)
    assert graph.closed_walk_lengths_through(6) == (0, 2, 3, 4, 5, 6)
    assert graph.primitive_closed_walk_lengths_through(6) == (2, 3)
    assert graph.closed_walk_count(2) == 2
    assert graph.closed_walk_count(3) == 3
    assert source_graph.closed_walk_counts_through(6) == graph.closed_walk_counts_through(6)
    assert source_graph.primitive_closed_walk_lengths_through(6) == (2, 3)


def test_target_a_neighbor_graph_controls_miss_the_cusp_valuations() -> None:
    controls = target_a_neighbor_walk_graphs()

    assert graph_low_valuation_summary(controls["one_step_visible_loop"]) == (0, 1, 2)
    assert graph_low_valuation_summary(controls["weak_only_two_cycle"]) == (0, 2, 4)
    assert graph_low_valuation_summary(controls["color_only_three_cycle"]) == (0, 3, 6)
    assert graph_low_valuation_summary(controls["wrong_color_length_four"]) == (0, 2, 4)
    assert all(
        graph_low_valuation_summary(control) != (0, 2, 3)
        for control in controls.values()
    )


def test_target_a_minimal_graph_has_cusp_valuations() -> None:
    payload = target_a_payload()

    assert payload.graph_closures == (2, 3)
    assert payload.bcc_same_normal_count == 2
    assert payload.bcc_mixed_normal_count == 2
    assert payload.bcc_same_normal_delta_q_values == (0, 0)
    assert sorted(payload.bcc_mixed_normal_delta_q_values) == [-2, 2]
    assert payload.bcc_q_clock_pass
    assert payload.closed_walk_counts_through_six == (1, 0, 2, 3, 4, 12, 17)
    assert payload.closed_walk_lengths_through_six == (0, 2, 3, 4, 5, 6)
    assert payload.graph_primitive_closures == (2, 3)
    assert payload.graph_one_step_forbidden
    assert payload.graph_model_pass
    assert payload.source_derived_graph_pass
    assert tuple(source.name for source in payload.closure_sources) == (
        "weak_SU2_doublet",
        "color_SU3_triplet",
    )
    assert payload.closure_source_orders == (2, 3)
    assert payload.closure_source_one_step_forbidden == (True, True)
    assert payload.sm_center_source_pass
    assert tuple(source.name for source in payload.material_closure_sources) == (
        "weak_SU2_doublet",
        "color_SU3_triplet",
    )
    assert payload.material_same_normal_norm == sp.eye(2) / 2
    assert payload.material_mixed_normal_norm == sp.eye(2) / 2
    assert payload.material_total_norm_is_identity
    assert payload.material_norm_split_pass
    assert payload.material_weak_source_from_bcc_pass
    assert payload.material_color_source_pass
    assert payload.material_source_pass
    assert payload.boundary_local_isometry_norm == sp.eye(2)
    assert payload.boundary_local_isometry_pass
    assert payload.boundary_local_input_dimension == 2
    assert payload.boundary_local_output_dimension == 8
    assert payload.boundary_unitary_completion_nullity == 6
    assert payload.boundary_unitary_dilation_exists_pass
    assert payload.boundary_finite_gap_feedback_scalar
    assert payload.boundary_hard_gap_feedback_zero
    assert payload.boundary_retarded_feedback_limit_zero
    assert payload.boundary_retarded_visible_powers_match_survival
    assert payload.boundary_recurrent_wedge_return_nonzero
    assert payload.boundary_recurrent_visible_powers_differ
    assert payload.boundary_outgoing_leakage_pass
    assert payload.boundary_dilation_pass
    assert "mixed_normal_clock_error_ports" in payload.boundary_remaining_declared_inputs[1]
    q = sp.symbols("q")
    g = sp.symbols("g", positive=True)
    assert payload.origin_q_locking_polynomial == g * q**2
    assert payload.origin_q_locking_values == (0, 4 * g, 4 * g)
    assert payload.origin_q_locking_gap == 4 * g
    assert payload.origin_q_locking_unique_solution[sp.symbols("a")] == g
    assert payload.origin_q_locking_unique_solution[sp.symbols("b")] == 0
    assert payload.origin_q_locking_unique_solution[sp.symbols("c")] == 0
    assert payload.origin_q_locking_pass
    assert payload.origin_linear_control_rejected
    assert payload.origin_constant_control_rejected
    assert payload.origin_causal_rule == "no incoming mixed-normal clock-error data"
    assert payload.origin_outgoing_pass
    assert payload.origin_material_pass
    assert "retarded_asymptotics" in payload.origin_remaining_physical_axioms[1]
    assert payload.automaton_return_orders == (2, 3)
    assert payload.automaton_one_step_forbidden == (True, True)
    assert payload.automaton_charge_tables_through_six == (
        (0, 1, 0, 1, 0, 1, 0),
        (0, 1, 2, 0, 1, 2, 0),
    )
    assert payload.automaton_return_ticks_through_six == ((2, 4, 6), (3, 6))
    assert len(payload.automaton_graph_edges) == 13
    assert payload.automaton_graph_pass
    assert payload.one_step_forbidden
    assert payload.first_three_valuations == (0, 2, 3)
    assert payload.gaps_through_six == (1,)
    assert payload.target_algebra == "C[t^2,t^3]"
    assert payload.graph_controls == {
        "one_step_visible_loop": (0, 1, 2),
        "weak_only_two_cycle": (0, 2, 4),
        "color_only_three_cycle": (0, 3, 6),
        "wrong_color_length_four": (0, 2, 4),
    }
    assert payload.graph_controls_rejected
    assert payload.target_a_pass


def test_target_a_controls_are_not_the_cusp_low_valuations() -> None:
    payload = target_a_payload()

    assert payload.controls["C[t]"] == (0, 1, 2)
    assert payload.controls["C[t^2]"] == (0, 2, 4)
    assert payload.controls["C[t^3]"] == (0, 3, 6)
    assert all(control != payload.first_three_valuations for control in payload.controls.values())


def test_one_step_return_collapses_to_vanilla_t_algebra() -> None:
    graph = RecirculationGraph((2, 3), one_step_return_allowed=True)

    assert graph.semigroup.first_values(3) == (0, 1, 2)
    assert not graph.one_step_forbidden


def test_numerical_semigroup_decompositions_are_exact() -> None:
    semigroup = NumericalSemigroup((2, 3))

    assert semigroup.values_through(6) == (0, 2, 3, 4, 5, 6)
    assert semigroup.decompositions(1) == ()
    assert semigroup.decompositions(6) == ((0, 2), (3, 0))


def test_target_b_cabibbo_shear_rejects_reflection_coefficient() -> None:
    payload = target_b_payload()

    assert payload.matching_principle == (
        "normalized amplitude matching: (1+t)sqrt(N_w)=sqrt(N_c)"
    )
    assert payload.weak_channels == 2
    assert payload.color_channels == 3
    assert payload.weak_amplitude == sp.sqrt(2)
    assert payload.color_amplitude == sp.sqrt(3)
    assert sp.simplify(payload.lambda_rec - (sp.sqrt(sp.Rational(3, 2)) - 1)) == 0
    assert payload.matching_residual == 0
    assert payload.matching_function_value == 0
    assert payload.matching_function_derivative_at_lambda == 0
    assert payload.matching_function_second_derivative == 4
    assert payload.matching_function_stable_minimum_pass
    assert payload.boundary_matching_condition.startswith(
        "one-sided retarded recirculation"
    )
    assert payload.one_sided_matching_residual_at_lambda == 0
    assert payload.ordinary_reflection_two_sided_residual == 0
    assert payload.ordinary_reflection_solves_two_sided_control
    assert payload.ordinary_reflection_excluded_by_one_sided_boundary
    assert payload.positive_oriented_uniformizer_pass
    assert not payload.uses_ckm_data
    assert payload.shear_matching_theorem_pass
    assert payload.matching_solution_unique
    assert payload.shear_formula_pass
    assert payload.control_shears == {
        "count_ratio_minus_one": sp.Rational(1, 2),
        "inverse_amplitude_shear": -1 + sp.sqrt(sp.Rational(2, 3)),
        "ordinary_reflection": (sp.sqrt(3) - sp.sqrt(2)) / (sp.sqrt(2) + sp.sqrt(3)),
    }
    assert set(payload.control_matching_residuals) == set(payload.control_shears)
    assert all(
        sp.simplify(residual) != 0
        for residual in payload.control_matching_residuals.values()
    )
    assert payload.reflection_control_rejected
    assert payload.wrong_shear_controls_rejected
    assert sp.simplify(payload.reflection_matching_residual) != 0
    assert sp.N(payload.lambda_rec, 10) == sp.N(sp.sqrt(sp.Rational(3, 2)) - 1, 10)


def test_cusp_shear_matching_principle_is_one_sided_not_reflection() -> None:
    audit = cusp_shear_matching_principle_audit()

    assert isinstance(audit, CuspShearMatchingPrincipleAudit)
    assert audit.matcher.weak_channels == 2
    assert audit.matcher.color_channels == 3
    assert sp.simplify(audit.lambda_rec - (sp.sqrt(sp.Rational(3, 2)) - 1)) == 0
    assert audit.one_sided_matching_residual_at_lambda == 0
    assert audit.ordinary_reflection_two_sided_residual == 0
    assert sp.simplify(audit.ordinary_reflection_one_sided_residual) != 0
    assert audit.ordinary_reflection_solves_two_sided_control
    assert audit.ordinary_reflection_excluded_by_one_sided_boundary
    assert audit.positive_oriented_uniformizer_pass
    assert not audit.uses_ckm_data
    assert audit.shear_matching_theorem_pass


def test_target_c_candidate_reproduces_fn_power_skeleton() -> None:
    payload = target_c_payload()

    assert payload.left_charges_light_to_heavy == (3, 2, 0)
    assert payload.cusp_semigroup_values_through_six == (0, 2, 3, 4, 5, 6)
    assert payload.cusp_conductor == 2
    assert payload.cusp_frobenius_number == 1
    assert payload.conductor_from_semigroup_pass
    assert payload.up_yukawa_hypercharge_sum == 0
    assert payload.down_yukawa_hypercharge_sum == 0
    assert payload.swapped_up_hypercharge_sum != 0
    assert payload.swapped_down_hypercharge_sum != 0
    assert payload.hypercharge_doors_forced
    assert payload.boundary_module_rule.name == "conductor_floor_plus_oriented_double_lift"
    assert payload.boundary_module_rule.conductor == 2
    assert payload.boundary_module_rule.frobenius_number == 1
    assert payload.boundary_module_rule.down_right_charges == (1, 0, 0)
    assert payload.boundary_module_rule.up_right_charges == (5, 2, 0)
    assert payload.boundary_module_rule.down_diagonal_exponents == (4, 2, 0)
    assert payload.boundary_module_rule.up_diagonal_exponents == (8, 4, 0)
    assert payload.boundary_module_rule_pass
    assert payload.wrong_conductor_down_diagonal_controls == {
        1: (5, 3, 0),
        2: (4, 2, 0),
        3: (3, 2, 0),
    }
    assert payload.orientation_lift_factor == 2
    assert payload.orientation_lift_up_diagonal_controls == {
        1: (4, 2, 0),
        2: (8, 4, 0),
        3: (12, 6, 0),
    }
    assert subtract_conductor_floor((3, 2, 0), 2) == (1, 0, 0)
    assert oriented_double_up_right_charges((3, 2, 0), (1, 0, 0)) == (5, 2, 0)
    assert payload.up_right_candidate == (5, 2, 0)
    assert payload.down_right_candidate == (1, 0, 0)
    assert payload.down_right_from_conductor_rule == (1, 0, 0)
    assert payload.up_right_from_oriented_double_rule == (5, 2, 0)
    assert isinstance(payload.right_charge_origin_audit, CuspRightChargeOriginAudit)
    assert payload.right_charge_origin_audit.down_right_charges == (1, 0, 0)
    assert payload.right_charge_origin_audit.up_right_charges == (5, 2, 0)
    assert payload.right_charge_origin_pass
    assert not payload.right_charge_origin_uses_mass_fits
    assert payload.right_charge_origin_down_rule == (
        "right residue of cusp conductor ideal: r_d=max(q-c,0)"
    )
    assert payload.right_charge_origin_up_rule == (
        "H_tilde oriented weak-double cover: diag_u=N_w diag_d"
    )
    assert payload.right_charge_origin_lift_factor_from_weak_order == 2
    assert payload.right_charge_origin_trivial_lift_control == (4, 2, 0)
    assert payload.right_charge_origin_color_lift_control == (12, 6, 0)
    assert payload.conductor_orientation_rule_pass
    assert payload.up_diagonal_exponents == (8, 4, 0)
    assert payload.down_diagonal_exponents == (4, 2, 0)
    assert payload.up_solver_solutions == ((5, 2, 0),)
    assert payload.down_solver_solutions == ((1, 0, 0),)
    assert payload.right_charges_forced_by_diagonal_targets
    assert payload.ckm_power_exponents == {"V_us": 1, "V_cb": 2, "V_ub": 3}
    assert payload.target_c_candidate_pass
    assert "semigroup derives conductor c=2" in payload.derivation_status


def test_cusp_right_charge_origin_uses_conductor_and_weak_order_not_mass_fits() -> None:
    audit = cusp_right_charge_origin_audit()

    assert isinstance(audit, CuspRightChargeOriginAudit)
    assert audit.left_charges == (3, 2, 0)
    assert audit.conductor == 2
    assert audit.weak_lift_factor == 2
    assert audit.color_lift_control_factor == 3
    assert audit.down_right_charges == (1, 0, 0)
    assert audit.up_right_charges == (5, 2, 0)
    assert audit.down_diagonal_exponents == (4, 2, 0)
    assert audit.up_diagonal_exponents == (8, 4, 0)
    assert audit.trivial_lift_control == (4, 2, 0)
    assert audit.color_lift_control == (12, 6, 0)
    assert audit.trivial_lift_control != audit.up_diagonal_exponents
    assert audit.color_lift_control != audit.up_diagonal_exponents
    assert audit.all_right_charges_nonnegative
    assert not audit.uses_mass_fits
    assert audit.right_charge_origin_pass


def test_target_d_center_holonomy_witness_has_nonzero_cp_invariant() -> None:
    payload = target_d_payload()

    assert sp.simplify(payload.center_phase**3 - 1) == 0
    assert sp.simplify(payload.center_phase - 1) != 0
    assert payload.up_holonomy_rule.name == "up_flag_winding"
    assert payload.down_holonomy_rule.name == "down_bilinear_linking"
    assert payload.holonomy_rule_pair_pass
    assert payload.up_holonomy_rule.center_powers == (
        (0, 1, 2),
        (1, 0, 1),
        (2, 1, 0),
    )
    assert payload.down_holonomy_rule.center_powers == (
        (0, 0, 0),
        (0, 1, 2),
        (0, 2, 1),
    )
    assert payload.up_path_measure.center_power_matrix() == (
        (0, 1, 2),
        (1, 0, 1),
        (2, 1, 0),
    )
    assert payload.down_path_measure.center_power_matrix() == (
        (0, 0, 0),
        (0, 1, 2),
        (0, 2, 1),
    )
    assert payload.up_coefficient_matrix[0][1] == payload.center_phase
    assert sp.simplify(payload.up_coefficient_matrix[0][2] - payload.center_phase**2) == 0
    assert payload.target_d_witness_pass
    assert sp.simplify(payload.cp_invariant) != 0
    assert payload.cp_invariant_numeric != 0
    assert payload.positive_up_path_measure.center_power_matrix() == (
        (0, 1, 2),
        (1, 0, 1),
        (2, 1, 0),
    )
    assert payload.positive_down_path_measure.center_power_matrix() == (
        (0, 0, 0),
        (0, 1, 2),
        (0, 2, 1),
    )
    assert payload.positive_up_amplitudes != ((1, 1, 1), (1, 1, 1), (1, 1, 1))
    assert payload.positive_down_amplitudes != ((1, 1, 1), (1, 1, 1), (1, 1, 1))
    assert all(
        entry.is_positive
        for row in payload.positive_up_amplitudes + payload.positive_down_amplitudes
        for entry in row
    )
    assert sp.simplify(payload.positive_amplitude_cp_invariant) != 0
    assert payload.positive_amplitude_cp_invariant_numeric != 0
    assert sp.simplify(payload.positive_real_control_invariant) == 0
    assert payload.positive_amplitude_witness_pass
    assert isinstance(payload.coefficient_measure_audit, CuspCoefficientMeasureAudit)
    assert payload.coefficient_measure_audit.amplitude_rule == (
        "A_ij=max(1,# decompositions of q_i+r_j in <2,3>)"
    )
    assert payload.derived_up_amplitudes == (
        (sp.Integer(2), sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
    )
    assert payload.derived_down_amplitudes == (
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
    )
    assert sp.simplify(payload.derived_measure_cp_invariant) != 0
    assert payload.derived_measure_cp_invariant_numeric != 0
    assert sp.simplify(payload.derived_measure_real_control_invariant) == 0
    assert sp.simplify(
        payload.derived_measure_cp_invariant - payload.derived_measure_rephased_invariant
    ) == 0
    assert payload.derived_measure_pass
    assert not payload.derived_measure_topology_selected
    assert isinstance(payload.center_topology_audit, CuspCenterTopologyAudit)
    assert payload.center_topology_finite_selection_pass
    assert not payload.center_topology_boundary_selected
    assert payload.center_topology_up_control_powers == {
        "cyclic_group_difference": cyclic_difference_center_powers(),
        "complete_graph_distance": complete_graph_distance_center_powers(),
        "all_trivial": ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
    }
    assert payload.center_topology_down_control_powers == {
        "all_trivial": ((0, 0, 0), (0, 0, 0), (0, 0, 0)),
        "additive_row_plus_col": additive_center_control_powers(),
        "cyclic_difference": cyclic_difference_center_powers(),
        "orientation_conjugate": bilinear_center_linking_powers(unit=2),
    }
    assert sp.simplify(payload.real_coefficient_control_invariant) == 0
    assert payload.real_control_rejected
    assert set(payload.holonomy_rule_control_invariants) == {
        "up_rule_down_trivial",
        "up_trivial_down_rule",
        "separable_row_col",
        "row_only_down_rule",
        "up_rule_col_only",
    }
    assert all(
        sp.simplify(invariant) == 0
        for invariant in payload.holonomy_rule_control_invariants.values()
    )
    assert payload.holonomy_rule_controls_rejected
    assert sp.simplify(payload.cp_invariant - payload.rephased_cp_invariant) == 0
    assert payload.rephasing_invariant_pass
    assert sp.simplify(payload.cp_invariant - payload.field_rephased_cp_invariant) == 0
    assert payload.full_field_rephasing_invariant_pass
    assert "derived cusp-module amplitudes" in payload.derivation_status


def test_cusp_center_topology_selects_flag_distance_and_bilinear_pairing() -> None:
    audit = cusp_center_topology_audit()

    assert isinstance(audit, CuspCenterTopologyAudit)
    assert path_flag_distance_center_powers() == (
        (0, 1, 2),
        (1, 0, 1),
        (2, 1, 0),
    )
    assert cyclic_difference_center_powers() == (
        (0, 1, 2),
        (2, 0, 1),
        (1, 2, 0),
    )
    assert complete_graph_distance_center_powers() == (
        (0, 1, 1),
        (1, 0, 1),
        (1, 1, 0),
    )
    assert bilinear_center_linking_powers(unit=1) == (
        (0, 0, 0),
        (0, 1, 2),
        (0, 2, 1),
    )
    assert bilinear_center_linking_powers(unit=2) == (
        (0, 0, 0),
        (0, 2, 1),
        (0, 1, 2),
    )
    assert additive_center_control_powers() == (
        (0, 1, 2),
        (1, 2, 0),
        (2, 0, 1),
    )
    assert audit.up_center_powers == path_flag_distance_center_powers()
    assert audit.down_center_powers == bilinear_center_linking_powers(unit=1)
    assert audit.up_rule.center_powers == audit.up_center_powers
    assert audit.down_rule.center_powers == audit.down_center_powers
    assert audit.up_controls_rejected
    assert audit.down_controls_rejected_except_conjugate
    assert audit.down_direct_row_col_zero
    assert audit.down_unit_pairing_normalized
    assert audit.finite_topology_selection_pass
    assert not audit.boundary_microphysics_selected


def test_cusp_coefficient_measure_uses_module_path_counts() -> None:
    audit = cusp_coefficient_measure_audit()
    semigroup = NumericalSemigroup((2, 3))

    assert isinstance(audit, CuspCoefficientMeasureAudit)
    assert cusp_module_path_count(0, semigroup) == 1
    assert cusp_module_path_count(1, semigroup) == 1
    assert cusp_module_path_count(2, semigroup) == 1
    assert cusp_module_path_count(3, semigroup) == 1
    assert cusp_module_path_count(8, semigroup) == 2
    assert cusp_module_path_amplitudes(((8, 1, 0), (3, 2, 5)), semigroup) == (
        (sp.Integer(2), sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
    )
    assert audit.up_amplitudes == (
        (sp.Integer(2), sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
    )
    assert audit.down_amplitudes == (
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
        (sp.Integer(1), sp.Integer(1), sp.Integer(1)),
    )
    assert audit.all_amplitudes_positive
    assert not audit.uses_fitted_amplitudes
    assert sp.simplify(audit.cp_invariant) != 0
    assert sp.simplify(audit.real_control_invariant) == 0
    assert sp.simplify(audit.cp_invariant - audit.rephased_cp_invariant) == 0
    assert not audit.holonomy_topology_selected_by_boundary
    assert audit.coefficient_measure_pass


def test_center_holonomy_path_measure_sums_paths() -> None:
    measure = CoefficientPathMeasure(
        paths=(
            CenterHolonomyPath(row=0, col=0, amplitude=1, center_power=0),
            CenterHolonomyPath(row=0, col=0, amplitude=1, center_power=1),
        )
    )

    matrix = measure.coefficient_matrix()
    assert sp.simplify(matrix[0, 0] - (1 + exact_omega())) == 0
    assert matrix[1, 1] == 0


def test_path_measure_accepts_entrywise_positive_amplitudes() -> None:
    measure = path_measure_from_center_powers(
        ((0, 1, 2), (1, 0, 1), (2, 1, 0)),
        amplitude=((1, 2, 3), (4, 5, 6), (7, 8, 9)),
    )

    matrix = measure.coefficient_matrix()
    assert matrix[0, 0] == 1
    assert sp.simplify(matrix[0, 1] - 2 * exact_omega()) == 0
    assert sp.simplify(matrix[0, 2] - 3 * exact_omega() ** 2) == 0
    assert measure.center_power_matrix() == ((0, 1, 2), (1, 0, 1), (2, 1, 0))


def test_combined_payload_records_current_status_honestly() -> None:
    payload = cusp_targets_payload()

    assert payload.final_verdict == "CUSP_TARGETS_A_D_FINITE_PASS_MICRO_GATES_OPEN"
    assert payload.target_a.target_a_pass
    assert payload.target_b.shear_formula_pass
    assert payload.target_c.target_c_candidate_pass
    assert payload.target_d.target_d_witness_pass
    assert "Target C proves the SM hypercharge Yukawa doors" in payload.interpretation
    assert "Target D has a finite center-holonomy topology selection" in (
        payload.interpretation
    )
