"""Placeholder tests for the cusp sidecar infrastructure."""

from clifford_3plus2_d5.cusp import SIDECAR_NAME


def test_cusp_sidecar_imports() -> None:
    """The sidecar package imports before theory modules are added."""

    assert SIDECAR_NAME == "cusp"
