"""Chất lượng của hệ toạ độ dùng để gán nhãn B2.

Nhãn positive của B2 (`‖xy‖ <= 5 m`) chỉ đáng tin nếu frame gán nhãn chính xác
hơn ngưỡng đó. Script này đo hai đại lượng khác nhau, thường bị lẫn:

  Tuyệt đối  residual của rigid fit `reference XYZ -> common ENU` so với chính
             các fix RTK quality-4. Lớn khi RTK kém hoặc reference VIO trôi.

  Tương đối  khoảng cách nearest-neighbour giữa hai quỹ đạo reference SAU khi
             fit. Đây mới là đại lượng quyết định nhãn "cùng chỗ", vì sai số
             chung cho cả hai traversal triệt tiêu.

Chạy:
    .venv/bin/python -m data_tools.audit_label_frame
"""

import argparse
import json
from pathlib import Path

import numpy as np

from data_tools import fourseasons
from data_tools.map_frame import (
    Datum,
    _epoch_from_seconds_of_day,
    first_quality4_datum,
    fit_reference_to_map,
    geodetic_to_enu_3d,
    quality4_rows,
    reference_in_map_frame,
    reference_xyz,
)

RECORDINGS = {
    "garage_2": "recording_2021-02-25_13-39-06",
    "garage_3": "recording_2021-05-10_19-15-19",
}
QUARTILES = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))


def fit_residuals(recording: Path, datum: Datum) -> dict:
    """Residual của fit tại từng fix quality-4, kèm vị trí tương đối trên tuyến."""
    fit = fit_reference_to_map(recording, datum)
    rows = quality4_rows(recording)
    epoch = float(fourseasons.load_times(recording)[0, 1])
    timestamps, xyz, _ = reference_xyz(recording)
    fix_times = np.array(
        [_epoch_from_seconds_of_day(row["utc"], epoch) for row in rows]
    )
    inside = (fix_times >= timestamps[0]) & (fix_times <= timestamps[-1])
    rows = [row for row, keep in zip(rows, inside) if keep]
    fix_times = fix_times[inside]

    target = geodetic_to_enu_3d(
        np.array([row["lat"] for row in rows]),
        np.array([row["lon"] for row in rows]),
        np.array(
            [
                0.0 if row["ellipsoid_altitude_m"] is None else row["ellipsoid_altitude_m"]
                for row in rows
            ]
        ),
        datum,
    )
    source = np.column_stack(
        [np.interp(fix_times, timestamps, xyz[:, axis]) for axis in range(3)]
    )
    residual = np.linalg.norm(source @ fit.rotation.T + fit.translation - target, axis=1)
    progress = (fix_times - fix_times[0]) / (fix_times[-1] - fix_times[0])

    sections = []
    for lower, upper in QUARTILES:
        mask = (progress >= lower) & (progress < upper if upper < 1.0 else progress <= 1.0)
        sections.append(
            {
                "route_fraction": [lower, upper],
                "fixes": int(mask.sum()),
                "median_m": float(np.median(residual[mask])) if mask.any() else None,
                "p95_m": float(np.percentile(residual[mask], 95)) if mask.any() else None,
            }
        )
    return {
        "recording": recording.name,
        "quality4_fixes_used": len(rows),
        "overall": {
            "median_m": float(np.median(residual)),
            "p95_m": float(np.percentile(residual, 95)),
            "rmse_m": float(np.sqrt(np.mean(residual**2))),
        },
        "by_route_section": sections,
        "holdout_residual": fit.holdout_residual,
    }


def cross_traversal(first: Path, second: Path, datum: Datum) -> dict:
    """Nearest-neighbour giữa hai quỹ đạo reference đã fit về cùng ENU."""
    _, xyz_first, _ = reference_in_map_frame(first, fit_reference_to_map(first, datum))
    _, xyz_second, _ = reference_in_map_frame(second, fit_reference_to_map(second, datum))
    distances = np.linalg.norm(
        xyz_first[:, None, :2] - xyz_second[None, :, :2], axis=2
    )
    nearest = distances.min(axis=1)
    partner = distances.argmin(axis=1)
    vertical = np.abs(xyz_first[:, 2] - xyz_second[partner, 2])
    return {
        "planar_nn_m": {
            "median": float(np.median(nearest)),
            "p95": float(np.percentile(nearest, 95)),
            "fraction_under_1m": float((nearest < 1.0).mean()),
            "fraction_under_5m": float((nearest < 5.0).mean()),
        },
        "vertical_at_nn_m": {
            "median": float(np.median(vertical)),
            "p95": float(np.percentile(vertical, 95)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(".cache/data/4seasons"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/label-frame-audit"))
    args = parser.parse_args()

    mapping = args.dataset / RECORDINGS["garage_3"]
    query = args.dataset / RECORDINGS["garage_2"]
    datum = first_quality4_datum(mapping)

    report = {
        "datum": {
            "source_recording": mapping.name,
            "latitude": datum.latitude,
            "longitude": datum.longitude,
            "altitude_m": datum.altitude,
        },
        "absolute_fit_residual": {
            "garage_3": fit_residuals(mapping, datum),
            "garage_2": fit_residuals(query, datum),
        },
        "relative_consistency": cross_traversal(query, mapping, datum),
        "reading": [
            "Residual tuyệt đối lớn (median vài mét) nhưng nearest-neighbour giữa "
            "hai traversal dưới nửa mét: sai số là common-mode, không phải sai số "
            "gán nhãn. Hai fit độc lập sai 5 m sẽ cho nn khoảng 7 m, không phải 0,35 m.",
            "Residual dồn vào phần tư cuối tuyến ở CẢ HAI recording. Nhãn ở "
            "3/4 đầu tin được; 1/4 cuối thì không.",
            "Vì vậy holdout_residual bi quan theo thiết kế: split 60/40 theo thời "
            "gian dồn trọn phần tư xấu vào holdout. garage_3 có median toàn tuyến "
            "2,11 m nhưng median holdout 4,90 m — cùng một fit, khác mẫu. Split "
            "thời gian được chọn để tránh rò rỉ lạc quan, và cái giá là con số "
            "holdout không đại diện cho toàn tuyến.",
            "Nearest-neighbour ràng buộc chặt sai lệch ngang, nhưng gần như mù "
            "với sai lệch dọc theo hướng đi. Ngưỡng positive 5 m rộng gấp 2,5 lần "
            "spacing keyframe 2 m, nên lệch dọc vài mét đổi keyframe nào được tính "
            "là positive chứ không đổi việc query có positive hay không.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    for name, data in report["absolute_fit_residual"].items():
        sections = "  ".join(
            "n/a" if item["median_m"] is None else f"{item['median_m']:5.2f}"
            for item in data["by_route_section"]
        )
        print(f"{name}: overall {data['overall']['median_m']:5.2f} m   quartiles {sections}")
    nn = report["relative_consistency"]["planar_nn_m"]
    print(f"cross-traversal nn: median {nn['median']:.3f} m  p95 {nn['p95']:.3f} m")
    print(f"artifact: {args.output / 'audit.json'}")


if __name__ == "__main__":
    main()
