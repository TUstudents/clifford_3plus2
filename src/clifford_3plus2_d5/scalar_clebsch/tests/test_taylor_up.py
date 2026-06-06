"""Tests for the up-sector nilpotent Taylor scalar response."""

import sympy as sp

from clifford_3plus2_d5.scalar_clebsch.taylor_up import (
    bernstein_cumulative_alternative,
    empirical_rational_up_control,
    is_positive_scalar_grade_profile,
    nilpotent_flag,
    nilpotent_order_is_three,
    old_up_clebsch_vector,
    one_step_amplitude_from_charm,
    taylor_kernel_matrix,
    taylor_repair_amplitude,
    taylor_shell_profile,
    taylor_up_audit_payload,
    up_clebsch_vector,
)


def test_nilpotent_flag_has_length_three() -> None:
    flag = nilpotent_flag()
    assert flag**3 == sp.zeros(3, 3)
    assert flag**2 != sp.zeros(3, 3)
    assert nilpotent_order_is_three()


def test_normalized_taylor_profile_gives_up_vector() -> None:
    assert taylor_repair_amplitude() == 1 / sp.sqrt(2)
    assert taylor_shell_profile() == up_clebsch_vector()
    assert up_clebsch_vector() == (sp.Rational(1, 4), 1 / sp.sqrt(2), sp.Integer(1))


def test_taylor_kernel_contains_second_order_light_coefficient() -> None:
    flag = nilpotent_flag()
    expected = sp.eye(3) + flag / sp.sqrt(2) + flag**2 / 4
    assert taylor_kernel_matrix() == expected
    assert taylor_kernel_matrix()[0, 2] == sp.Rational(1, 4)


def test_nearby_rational_control_is_not_the_preferred_theorem() -> None:
    assert empirical_rational_up_control() == (sp.Rational(1, 4), sp.Rational(3, 4), sp.Integer(1))
    assert bernstein_cumulative_alternative() == empirical_rational_up_control()
    assert empirical_rational_up_control() != up_clebsch_vector()
    assert sp.N(up_clebsch_vector()[1] / empirical_rational_up_control()[1] - 1) < 0


def test_old_sqrt2_vector_is_not_positive_scalar_profile() -> None:
    assert not is_positive_scalar_grade_profile(old_up_clebsch_vector())


def test_one_step_amplitude_is_charm_coefficient() -> None:
    assert one_step_amplitude_from_charm(up_clebsch_vector()) == 1 / sp.sqrt(2)


def test_taylor_up_payload_passes() -> None:
    payload = taylor_up_audit_payload()
    assert payload.final_verdict == "NILPOTENT_TAYLOR_UP_CLEBSCH_PASS"
    assert payload.nilpotent_order_three
    assert payload.taylor_matches_target
    assert payload.empirical_rational_control_close
    assert payload.old_sqrt2_vector_rejected
