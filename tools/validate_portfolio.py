#!/usr/bin/env python3
"""Validate tracked portfolio claims, links, receipts, and model identity."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any
from urllib.parse import unquote


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "3ab52bdc4b41cc59b4b845b090bcddc2d927870ce272fdcff2dbb473b3a598c5"
PORTFOLIO_DOCUMENTS = (
    "README.md",
    "BENCHMARK.md",
    "MODEL_CARD.md",
    "THIRD_PARTY_NOTICES.md",
    "artifacts/README.md",
)
MARKDOWN_LINK = re.compile(r"!?\[[^]]*]\(([^)]+)\)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(actual: float, expected: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def portfolio_markdown_files(root: Path) -> list[Path]:
    documents = [root / path for path in PORTFOLIO_DOCUMENTS]
    documents.extend(sorted((root / "docs").rglob("*.md")))
    return list(dict.fromkeys(documents))


def local_link_errors(root: Path) -> list[str]:
    errors = []
    for document in portfolio_markdown_files(root):
        relative_document = document.relative_to(root)
        if not document.is_file():
            errors.append(f"missing portfolio document: {relative_document}")
            continue
        text = document.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            path_text = unquote(target.split("#", 1)[0])
            linked_path = (document.parent / path_text).resolve()
            if not linked_path.exists():
                errors.append(f"broken link in {relative_document}: {target}")
    return errors


def value_at(document: Any, path: list[Any]) -> Any:
    value = document
    for key in path:
        value = value[key]
    return value


def validate_file_receipt(root: Path, receipt: dict[str, Any], label: str) -> list[str]:
    errors = []
    path = root / receipt["path"]
    if not path.is_file():
        return [f"{label} file is missing: {receipt['path']}"]
    if sha256(path) != receipt["sha256"]:
        errors.append(f"{label} SHA-256 mismatch: {receipt['path']}")
    if path.stat().st_size != receipt["bytes"]:
        errors.append(f"{label} byte-size mismatch: {receipt['path']}")
    return errors


def validate_cpu_environment(environment: dict[str, Any], label: str) -> list[str]:
    errors = []
    if environment.get("onnxruntime_gpu") is not None:
        errors.append(f"{label} was not produced in a CPU-only ONNX Runtime environment")
    torch_version = environment.get("torch")
    if torch_version is not None and not str(torch_version).endswith("+cpu"):
        errors.append(f"{label} does not identify a CPU-only PyTorch build")
    return errors


def validate_claims(root: Path, manifest: dict[str, Any]) -> list[str]:
    errors = []
    receipt_cache: dict[str, dict[str, Any]] = {}
    for claim in manifest["claims"]:
        claim_id = claim["id"]
        receipt_path = claim["receipt"]
        try:
            receipt = receipt_cache.setdefault(
                receipt_path, load_json(root / receipt_path)
            )
            actual = value_at(receipt, claim["path"])
            if claim.get("mode", "value") == "length":
                actual = len(actual)
            expected = claim["expected"]
            if isinstance(expected, bool):
                passed = actual is expected
            elif isinstance(expected, (int, float)):
                passed = close(float(actual), float(expected), claim.get("tolerance", 1e-9))
            else:
                passed = actual == expected
            if not passed:
                errors.append(f"claim {claim_id} mismatch: expected {expected!r}, got {actual!r}")
        except (KeyError, IndexError, TypeError, FileNotFoundError, json.JSONDecodeError) as error:
            errors.append(f"claim {claim_id} cannot be resolved: {error}")
            continue

        for document_check in claim.get("documents", []):
            document_path = root / document_check["path"]
            if not document_path.is_file():
                errors.append(f"claim {claim_id} document is missing: {document_check['path']}")
            elif document_check["contains"] not in document_path.read_text(encoding="utf-8"):
                errors.append(
                    f"claim {claim_id} is not represented as {document_check['contains']!r} "
                    f"in {document_check['path']}"
                )
    return errors


def validate_repository(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    errors = []
    required = (
        "LICENSE",
        "CITATION.cff",
        "pyproject.toml",
        ".github/workflows/ci.yml",
        "models/pothole_yolo26n_seg.onnx",
        "artifacts/portfolio-claims.json",
        "artifacts/portfolio-detection/a1.json",
        "artifacts/portfolio-detection/cross-domain.json",
        "artifacts/portfolio-stereo/benchmark.json",
        "artifacts/portfolio-stereo/rows.csv",
        "training/combined_recipe.yaml",
        "third_party_licenses/FAN_STEREO_MIT.txt",
    )
    for relative_path in required:
        if not (root / relative_path).is_file():
            errors.append(f"missing required file: {relative_path}")

    errors.extend(local_link_errors(root))
    if errors:
        return {"status": "failed", "errors": errors}

    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    if "[project]" in pyproject or "[build-system]" in pyproject:
        errors.append("pyproject must describe tooling only; this repository is not a wheel")

    model_path = root / "models/pothole_yolo26n_seg.onnx"
    model_digest = sha256(model_path)
    if model_digest != MODEL_SHA256:
        errors.append(f"model SHA-256 mismatch: {model_digest}")

    detection = load_json(root / "artifacts/portfolio-detection/a1.json")
    evaluated = detection["model"]["evaluated"]
    deployed = detection["model"]["deployed"]
    if evaluated["sha256"] != model_digest or deployed["sha256"] != model_digest:
        errors.append("detection receipt does not identify the tracked ONNX model")
    if not detection["model"].get("same_artifact"):
        errors.append("detection evaluation and deployment artifacts differ")
    errors.extend(validate_file_receipt(root, detection["receipt"]["source"], "detection source"))
    errors.extend(validate_cpu_environment(detection["receipt"]["environment"], "detection receipt"))

    cross = load_json(root / "artifacts/portfolio-detection/cross-domain.json")
    if cross["receipt"]["model"]["sha256"] != model_digest:
        errors.append("cross-domain receipt does not identify the tracked ONNX model")
    errors.extend(validate_file_receipt(root, cross["receipt"]["source"], "cross-domain source"))
    errors.extend(
        validate_file_receipt(
            root, cross["receipt"]["in_domain_receipt"], "cross-domain in-domain receipt"
        )
    )
    errors.extend(validate_cpu_environment(cross["receipt"]["environment"], "cross-domain receipt"))

    stereo = load_json(root / "artifacts/portfolio-stereo/benchmark.json")
    if stereo["receipt"]["model"]["sha256"] != model_digest:
        errors.append("stereo receipt does not identify the tracked ONNX model")
    source_receipts = stereo["receipt"].get("source_files", [])
    for source_receipt in source_receipts:
        errors.extend(validate_file_receipt(root, source_receipt, "stereo source"))
    if len(source_receipts) != 4:
        errors.append("stereo receipt must identify four source files")
    errors.extend(validate_cpu_environment(stereo["receipt"]["environment"], "stereo receipt"))

    with (root / "artifacts/portfolio-stereo/rows.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))
    held_out = [row for row in rows if row["model"] in {"model2", "model3"}]
    fused = [row for row in held_out if row["fusion_success"] == "True"]
    depth_errors = [float(row["relative_error"]) for row in fused]
    area_errors = [float(row["area_relative_error"]) for row in fused]
    latencies = [float(row["total_ms"]) for row in rows]

    metric_checks = {
        "receipt pair count": stereo["pairs"] == len(rows) == 27,
        "held-out pair count": stereo["held_out"]["pairs"] == len(held_out) == 19,
        "depth median": close(
            stereo["held_out"]["median_relative_error"], statistics.median(depth_errors)
        ),
        "area median": close(
            stereo["held_out"]["area_median_relative_error"], statistics.median(area_errors)
        ),
        "latency median": close(
            stereo["performance"]["median_latency_ms"], statistics.median(latencies)
        ),
        "15 FPS coverage": close(
            stereo["performance"]["pairs_at_least_15_fps"],
            sum(1000 / latency >= 15 for latency in latencies) / len(latencies),
        ),
    }
    errors.extend(name for name, passed in metric_checks.items() if not passed)

    claims = load_json(root / "artifacts/portfolio-claims.json")
    errors.extend(validate_claims(root, claims))

    return {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "checks": {
            "portfolio_documents": len(portfolio_markdown_files(root)),
            "model_sha256": model_digest,
            "stereo_rows": len(rows),
            "held_out_rows": len(held_out),
            "receipt_metrics_recomputed": len(metric_checks),
            "receipt_source_files": len(source_receipts) + 3,
            "headline_claims": len(claims["claims"]),
        },
    }


def main() -> None:
    result = validate_repository()
    print(json.dumps(result, indent=2))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
