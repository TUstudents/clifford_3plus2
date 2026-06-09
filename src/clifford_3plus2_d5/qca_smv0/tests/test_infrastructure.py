"""Infrastructure tests for QCA_SMv0."""

from pathlib import Path

from clifford_3plus2_d5 import qca_smv0


def test_qca_smv0_sidecar_metadata() -> None:
    assert qca_smv0.SIDECAR_NAME == "qca_smv0"
    assert qca_smv0.SIDECAR_TITLE == "QCA_SMv0"
    assert qca_smv0.SIDECAR_STATUS == "Stage 20 physical-right sourced tick implemented"


def test_qca_smv0_scaffold_files_exist() -> None:
    sidecar_root = Path(qca_smv0.__file__).parent

    assert (sidecar_root / "README.md").is_file()
    assert (sidecar_root / "PLAN.md").is_file()
    assert (sidecar_root / "STATUS.md").is_file()
    assert (sidecar_root / "scripts").is_dir()
    assert (sidecar_root / "tests").is_dir()
