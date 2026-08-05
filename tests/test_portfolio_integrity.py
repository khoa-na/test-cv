from pathlib import Path

from tools.validate_portfolio import REPOSITORY_ROOT, validate_repository


def test_portfolio_integrity() -> None:
    result = validate_repository()
    assert result["status"] == "ok", result["errors"]
    assert result["checks"]["headline_claims"] == 17


def test_license_matches_model_distribution_terms() -> None:
    license_text = (REPOSITORY_ROOT / "LICENSE").read_text(encoding="utf-8")
    notices = (REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "GNU AFFERO GENERAL PUBLIC LICENSE" in license_text
    assert "models/pothole_yolo26n_seg.onnx" in notices
    assert "AGPL-3.0" in notices


def test_clone_ready_repository_commands_exist() -> None:
    assert (Path(REPOSITORY_ROOT) / "demo/model_smoke.py").is_file()
    assert (Path(REPOSITORY_ROOT) / "tools/validate_portfolio.py").is_file()
    assert (Path(REPOSITORY_ROOT) / ".github/workflows/ci.yml").is_file()
