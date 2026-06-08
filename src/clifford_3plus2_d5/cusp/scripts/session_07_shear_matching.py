"""Run the Target-B one-sided shear matching audit."""

from __future__ import annotations

from clifford_3plus2_d5.cusp.targets import cusp_shear_matching_principle_audit


def main() -> None:
    """Print the CUSP Target-B matching payload."""

    audit = cusp_shear_matching_principle_audit()
    print("boundary matching condition =", audit.boundary_matching_condition)
    print("weak channels =", audit.matcher.weak_channels)
    print("color channels =", audit.matcher.color_channels)
    print("weak amplitude =", audit.matcher.weak_amplitude)
    print("color amplitude =", audit.matcher.color_amplitude)
    print("lambda_rec =", audit.lambda_rec)
    print("lambda_rec numeric =", audit.lambda_rec.evalf(15))
    print(
        "one-sided residual at lambda =",
        audit.one_sided_matching_residual_at_lambda,
    )
    print("ordinary reflection =", audit.ordinary_reflection)
    print(
        "ordinary reflection two-sided residual =",
        audit.ordinary_reflection_two_sided_residual,
    )
    print(
        "ordinary reflection one-sided residual =",
        audit.ordinary_reflection_one_sided_residual,
    )
    print(
        "reflection solves two-sided control =",
        audit.ordinary_reflection_solves_two_sided_control,
    )
    print(
        "reflection excluded by one-sided boundary =",
        audit.ordinary_reflection_excluded_by_one_sided_boundary,
    )
    print(
        "positive oriented uniformizer pass =",
        audit.positive_oriented_uniformizer_pass,
    )
    print("uses CKM data =", audit.uses_ckm_data)
    print("shear matching theorem pass =", audit.shear_matching_theorem_pass)


if __name__ == "__main__":
    main()
