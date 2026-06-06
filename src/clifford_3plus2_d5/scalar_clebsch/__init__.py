"""Scalar Clebsch sidecar for quark mass coefficient structure.

This sidecar is deliberately narrower than ``flavor_a_track``. It asks whether
quark mass Clebsches can be interpreted as scalar boundary-response data:

* up sector: a length-3 nilpotent Taylor response ``exp(xN)`` with
  ``x = 1/sqrt(2)`` gives ``(1/4, 1/sqrt(2), 1)``;
* down sector: the natural S3 projector baseline gives
  ``(1, 1/sqrt(3), sqrt(2/3))``, while the data-improved odd-shell candidate
  gives ``(1, 1/sqrt(3), sqrt(5/6))``.

It does not fit masses, run RGEs, or alter the CKM charged-current Clebsches.
"""

from clifford_3plus2_d5.scalar_clebsch.audit import (
    ScalarClebschAuditPayload,
    scalar_clebsch_audit_payload,
)
from clifford_3plus2_d5.scalar_clebsch.down_subset_counts import (
    DownSubsetAuditPayload,
    color_only_middle_control,
    compressed_partition_cannot_derive_bcc_count,
    down_baseline_clebsch_vector,
    down_baseline_counts,
    down_baseline_mass_ratio_predictions,
    down_candidate_clebsch_vector,
    down_candidate_counts,
    down_candidate_mass_ratio_predictions,
    down_subset_audit_payload,
    odd_shell_plus_one_is_open,
    primitive_channel_subsets,
    s3_projector_bottom_control,
)
from clifford_3plus2_d5.scalar_clebsch.s3_projector_audit import (
    S3ProjectorAuditPayload,
    central_projector_ranks,
    central_projectors,
    central_s3_does_not_force_rank_two,
    rank_five_is_not_unique,
    rank_five_projector_ranks,
    rank_five_projectors,
    s3_projector_audit_payload,
    standard_copy_projector,
    standard_copy_projector_rank,
)
from clifford_3plus2_d5.scalar_clebsch.taylor_up import (
    TaylorUpAuditPayload,
    bernstein_cumulative_alternative,
    empirical_rational_up_control,
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

__all__ = [
    "DownSubsetAuditPayload",
    "S3ProjectorAuditPayload",
    "ScalarClebschAuditPayload",
    "TaylorUpAuditPayload",
    "bernstein_cumulative_alternative",
    "central_projector_ranks",
    "central_projectors",
    "central_s3_does_not_force_rank_two",
    "color_only_middle_control",
    "compressed_partition_cannot_derive_bcc_count",
    "down_baseline_clebsch_vector",
    "down_baseline_counts",
    "down_baseline_mass_ratio_predictions",
    "down_candidate_clebsch_vector",
    "down_candidate_counts",
    "down_candidate_mass_ratio_predictions",
    "down_subset_audit_payload",
    "empirical_rational_up_control",
    "nilpotent_flag",
    "nilpotent_order_is_three",
    "odd_shell_plus_one_is_open",
    "old_up_clebsch_vector",
    "one_step_amplitude_from_charm",
    "primitive_channel_subsets",
    "rank_five_is_not_unique",
    "rank_five_projector_ranks",
    "rank_five_projectors",
    "scalar_clebsch_audit_payload",
    "s3_projector_audit_payload",
    "s3_projector_bottom_control",
    "standard_copy_projector",
    "standard_copy_projector_rank",
    "taylor_kernel_matrix",
    "taylor_repair_amplitude",
    "taylor_shell_profile",
    "taylor_up_audit_payload",
    "up_clebsch_vector",
]
