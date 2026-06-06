"""Tests for the down-sector dark-line framing."""

from clifford_3plus2_d5.radial_response.down_dark_line import down_dark_line_payload


def test_down_dark_line_payload_passes_with_open_selection_rule() -> None:
    payload = down_dark_line_payload()
    assert payload.final_verdict == "DOWN_DARK_LINE_AVAILABLE_NOT_DERIVED"
    assert payload.rank_five_available
    assert not payload.rank_five_forced
    assert not payload.rank_two_standard_forced
    assert payload.s3_projectors.final_verdict == "S3_PROJECTOR_COUNT_AVAILABILITY_PASS"
