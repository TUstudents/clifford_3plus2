"""Run MicroCUSP Session F: one-sided lambda-rec audit."""

from __future__ import annotations

from clifford_3plus2_d5.cusp.targets import microscopic_lambda_rec_audit


def main() -> None:
    """Print the microscopic lambda-rec audit."""

    audit = microscopic_lambda_rec_audit()
    print("weak return moment M2 =", audit.weak_return_moment)
    print("color return moment M3 =", audit.color_return_moment)
    print("lambda_rec =", audit.lambda_rec)
    print("one-sided residual =", audit.one_sided_residual)
    print("one-sided stable minimum =", audit.one_sided_stable_minimum)
    print("ordinary reflection control =", audit.ordinary_reflection_control)
    print("control residuals =", audit.control_residuals)
    print("controls rejected =", audit.controls_rejected)
    print("session F pass =", audit.session_f_pass)
    print("remaining microscopic gates =", audit.microscopic_gate_remaining)


if __name__ == "__main__":
    main()
