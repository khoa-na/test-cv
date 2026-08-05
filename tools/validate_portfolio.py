#!/usr/bin/env python3
"""Validate the tracked portfolio claims, links, receipt, and model identity."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from urllib.parse import unquote


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "3ab52bdc4b41cc59b4b845b090bcddc2d927870ce272fdcff2dbb473b3a598c5"
PORTFOLIO_DOCUMENTS = (
    "README.md",
    "BENCHMARK.md",
    "REPORT.md",
    "MODEL_CARD.md",
    "THIRD_PARTY_NOTICES.md",
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


def local_link_errors(root: Path) -> list[str]:
    errors = []
    for relative_document in PORTFOLIO_DOCUMENTS:
        document = root / relative_document
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


def validate_repository(root: Path = REPOSITORY_ROOT) -> dict:
    errors = []
    required = (
        "LICENSE",
        "CITATION.cff",
        "pyproject.toml",
        ".github/workflows/ci.yml",
        "models/pothole_yolo26n_seg.onnx",
        "artifacts/portfolio-stereo/benchmark.json",
        "artifacts/portfolio-stereo/rows.csv",
    )
    for relative_path in required:
        if not (root / relative_path).is_file():
            errors.append(f"missing required file: {relative_path}")

    errors.extend(local_link_errors(root))
    if errors:
        return {"status": "failed", "errors": errors}

    model_path = root / "models/pothole_yolo26n_seg.onnx"
    model_digest = sha256(model_path)
    if model_digest != MODEL_SHA256:
        errors.append(f"model SHA-256 mismatch: {model_digest}")

    detection = json.loads(
        (root / "artifacts/verify-final/a1/benchmark.json").read_text(encoding="utf-8")
    )
    detection_digest = detection["model"]["onnx_deployed"]["sha256"]
    if detection_digest != model_digest:
        errors.append("detection receipt does not identify the tracked ONNX model")

    stereo = json.loads(
        (root / "artifacts/portfolio-stereo/benchmark.json").read_text(encoding="utf-8")
    )
    stereo_digest = stereo["receipt"]["model"]["sha256"]
    if stereo_digest != model_digest:
        errors.append("stereo receipt does not identify the tracked ONNX model")
    source_receipts = stereo["receipt"].get("source_files", [])
    for source_receipt in source_receipts:
        source_path = root / source_receipt["path"]
        if not source_path.is_file():
            errors.append(f"receipt source file is missing: {source_receipt['path']}")
        elif sha256(source_path) != source_receipt["sha256"]:
            errors.append(f"receipt source hash mismatch: {source_receipt['path']}")
    if len(source_receipts) != 4:
        errors.append("stereo receipt must identify four source files")

    with (root / "artifacts/portfolio-stereo/rows.csv").open(
        newline="", encoding="utf-8"
    ) as file:
        rows = list(csv.DictReader(file))
    held_out = [row for row in rows if row["model"] in {"model2", "model3"}]
    fused = [row for row in held_out if row["fusion_success"] == "True"]
    depth_errors = [float(row["relative_error"]) for row in fused]
    area_errors = [float(row["area_relative_error"]) for row in fused]
    latencies = [float(row["total_ms"]) for row in rows]

    checks = {
        "receipt pair count": stereo["pairs"] == len(rows) == 27,
        "held-out pair count": stereo["held_out"]["pairs"] == len(held_out) == 19,
        "depth median": close(
            stereo["held_out"]["median_relative_error"], statistics.median(depth_errors)
        ),
        "area median": close(
            stereo["held_out"]["area_median_relative_error"],
            statistics.median(area_errors),
        ),
        "latency median": close(
            stereo["performance"]["median_latency_ms"], statistics.median(latencies)
        ),
        "15 FPS coverage": close(
            stereo["performance"]["pairs_at_least_15_fps"],
            sum(1000 / latency >= 15 for latency in latencies) / len(latencies),
        ),
    }
    errors.extend(name for name, passed in checks.items() if not passed)

    readme = (root / "README.md").read_text(encoding="utf-8").casefold()
    for stale_claim in ("python 3.10 or newer", "end-to-end throughput | 44.3"):
        if stale_claim in readme:
            errors.append(f"stale README claim: {stale_claim}")

    return {
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "checks": {
            "portfolio_documents": len(PORTFOLIO_DOCUMENTS),
            "model_sha256": model_digest,
            "stereo_rows": len(rows),
            "held_out_rows": len(held_out),
            "receipt_metrics_recomputed": len(checks),
            "receipt_source_files": len(source_receipts),
        },
    }


def main() -> None:
    result = validate_repository()
    print(json.dumps(result, indent=2))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
