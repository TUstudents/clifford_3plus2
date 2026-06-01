"""V2 role-separation audit for the depth-scar theory.

The spectrum ``{0,2,6}`` is built in once ``D_scar = 2 Delta(P3)`` is accepted.
The real V2 question is what this operator is allowed to mean.  This audit
records the conservative interpretation: the path scar is a quark
mixing-depth/transfer operator, not by itself a mass model or a CP source.
"""

from __future__ import annotations

from dataclasses import dataclass

from clifford_3plus2_d5.depth_scar.prediction_ledger import (
    ckm_lambda_exponents,
    endpoint_parity_blocks_even_odd,
    leading_kernel_is_democratic_rank_one,
    port_transfer_relations_hold,
    projectors_resolve_identity,
    pure_path_has_no_intrinsic_cp_holonomy,
    restored_triangle_has_one_loop,
    transfer_kernel_matches_v1,
    two_sided_mass_depth_semigroup,
    two_sided_lambda_power_semigroup,
)


@dataclass(frozen=True)
class DepthScarPredictionLedgerPayload:
    """V2 payload: exact predictions and no-overclaim role separation."""

    final_verdict: str
    projectors_resolve_identity: bool
    transfer_kernel_matches_v1: bool
    port_transfer_relations_hold: bool
    endpoint_parity_blocks_even_odd: bool
    leading_kernel_is_democratic_rank_one: bool
    pure_path_has_no_intrinsic_cp_holonomy: bool
    restored_triangle_has_one_loop: bool
    ckm_lambda_exponents: dict[tuple[int, int], int]
    two_sided_mass_depth_semigroup: tuple[int, ...]
    two_sided_lambda_power_semigroup: tuple[int, ...]
    p3_is_mass_model_by_default: bool
    p3_is_cp_source_by_default: bool
    p3_is_universal_lepton_scar_by_default: bool
    interpretation: str


def depth_scar_prediction_ledger_payload() -> DepthScarPredictionLedgerPayload:
    """Return the V2 prediction-ledger verdict."""

    projectors_pass = projectors_resolve_identity()
    transfer_pass = transfer_kernel_matches_v1()
    port_relations_pass = port_transfer_relations_hold()
    parity_pass = endpoint_parity_blocks_even_odd()
    leading_pass = leading_kernel_is_democratic_rank_one()
    no_path_cp = pure_path_has_no_intrinsic_cp_holonomy()
    restored_loop = restored_triangle_has_one_loop()
    ckm_exponents = ckm_lambda_exponents()
    mass_semigroup = two_sided_mass_depth_semigroup()
    lambda_semigroup = two_sided_lambda_power_semigroup()

    checks_pass = (
        projectors_pass
        and transfer_pass
        and port_relations_pass
        and parity_pass
        and leading_pass
        and no_path_cp
        and restored_loop
        and ckm_exponents == {(1, 2): 1, (2, 3): 2, (1, 3): 3}
        and mass_semigroup == (0, 2, 4, 6, 8, 12)
        and lambda_semigroup == (0, 1, 2, 3, 4, 6)
    )

    if checks_pass:
        final_verdict = "DEPTH_SCAR_PREDICTION_LEDGER_PASS"
        interpretation = (
            "The P3 scar has a fixed projector ledger and transfer kernel: "
            "T = P0 + epsilon^2 P2 + epsilon^6 P6. It predicts P3 normal-mode "
            "families, exact port transfer relations, a rank-one democratic "
            "leading kernel, and CKM exponents lambda:lambda^2:lambda^3. The "
            "pure path has no intrinsic CP holonomy, so CP must come from a "
            "loop/holonomy/non-normal deformation. Mass exponents are only a "
            "conditional semigroup until a left/right Yukawa factorization is "
            "specified."
        )
    elif not projectors_pass:
        final_verdict = "DEPTH_SCAR_PROJECTOR_LEDGER_KILL"
        interpretation = "The P3 normal-mode projectors failed orthogonality or completeness."
    elif not transfer_pass:
        final_verdict = "DEPTH_SCAR_TRANSFER_LEDGER_KILL"
        interpretation = "The projector transfer kernel does not match the V1 transfer operator."
    elif not parity_pass:
        final_verdict = "DEPTH_SCAR_ENDPOINT_PARITY_KILL"
        interpretation = "Endpoint parity does not block even/odd mixing as predicted."
    elif not no_path_cp:
        final_verdict = "DEPTH_SCAR_CP_NO_GO_KILL"
        interpretation = "The pure path unexpectedly has a graph cycle for CP holonomy."
    else:
        final_verdict = "DEPTH_SCAR_PREDICTION_LEDGER_CONTROL_KILL"
        interpretation = "One of the role-separation controls failed."

    return DepthScarPredictionLedgerPayload(
        final_verdict=final_verdict,
        projectors_resolve_identity=projectors_pass,
        transfer_kernel_matches_v1=transfer_pass,
        port_transfer_relations_hold=port_relations_pass,
        endpoint_parity_blocks_even_odd=parity_pass,
        leading_kernel_is_democratic_rank_one=leading_pass,
        pure_path_has_no_intrinsic_cp_holonomy=no_path_cp,
        restored_triangle_has_one_loop=restored_loop,
        ckm_lambda_exponents=ckm_exponents,
        two_sided_mass_depth_semigroup=mass_semigroup,
        two_sided_lambda_power_semigroup=lambda_semigroup,
        p3_is_mass_model_by_default=False,
        p3_is_cp_source_by_default=False,
        p3_is_universal_lepton_scar_by_default=False,
        interpretation=interpretation,
    )

