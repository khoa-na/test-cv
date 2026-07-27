"""Quét mọi cửa sổ hiệu chỉnh bias gyro khả dĩ trong một recording.

`IMUYawIntegrator` hiệu chỉnh bias trên 2 s ĐẦU recording và bỏ IMU yaw nếu
p95 gyro norm trong cửa sổ đó vượt `stationary_gyro_p95_rad_s`. Script này hỏi
câu mà cổng không hỏi: trong cả recording có cửa sổ nào đứng yên hơn không, và
bias thật đáng lẽ là bao nhiêu.

Chỉ đọc imu.txt, không chạy VO. Không có tham số nào ở đây đi vào production.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from data_tools.gps_sources import load_reference_trajectory
from data_tools.imu_yaw import IMUYawConfig, load_camera_imu_rotation

SEQUENCES = {
    "office_loop_1": "recording_2020-03-24_17-36-22",
    "neighborhood_4": "recording_2020-12-22_11-54-24",
    "garage_2": "recording_2021-02-25_13-39-06",
    "garage_3": "recording_2021-05-10_19-15-19",
}


def scan(recording: Path, window_s: float, step_s: float, gate: float) -> dict:
    imu = np.loadtxt(recording / "imu.txt")
    timestamps = imu[:, 0] / 1e9
    gyro_camera = imu[:, 1:4] @ load_camera_imu_rotation(recording).T
    norm = np.linalg.norm(gyro_camera, axis=1)
    reference_start = float(load_reference_trajectory(recording)[0][0])

    candidates = []
    for start in np.arange(timestamps[0], timestamps[-1] - window_s, step_s):
        mask = (timestamps >= start) & (timestamps <= start + window_s)
        if mask.sum() < 10:
            continue
        candidates.append(
            {
                "offset_from_reference_start_s": float(start - reference_start),
                "gyro_norm_p95_rad_s": float(np.percentile(norm[mask], 95)),
                "bias_camera_y_rad_s": float(np.median(gyro_camera[mask, 1])),
            }
        )
    if not candidates:
        raise ValueError(f"Không dựng được cửa sổ nào cho {recording.name}")

    quietest = min(candidates, key=lambda c: c["gyro_norm_p95_rad_s"])
    # Cửa sổ production thật sự dùng: cái bắt đầu tại reference start.
    used = min(
        candidates, key=lambda c: abs(c["offset_from_reference_start_s"])
    )
    passing = [c for c in candidates if c["gyro_norm_p95_rad_s"] <= gate]
    return {
        "recording": recording.name,
        "windows_scanned": len(candidates),
        "windows_passing_gate": len(passing),
        "gate_rad_s": gate,
        "production_window": used,
        "quietest_window": quietest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(".cache/data/4seasons"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/gyro-calibration-audit")
    )
    parser.add_argument("--step-seconds", type=float, default=1.0)
    args = parser.parse_args()

    defaults = IMUYawConfig()
    results = {}
    for name, folder in SEQUENCES.items():
        recording = args.dataset / folder
        if not recording.is_dir():
            continue
        results[name] = scan(
            recording,
            defaults.bias_calibration_seconds,
            args.step_seconds,
            defaults.stationary_gyro_p95_rad_s,
        )
        entry = results[name]
        production = entry["production_window"]
        quietest = entry["quietest_window"]
        print(
            f"{name:16s} pass {entry['windows_passing_gate']:3d}/"
            f"{entry['windows_scanned']:3d}  "
            f"production p95={production['gyro_norm_p95_rad_s']:.4f} "
            f"bias={production['bias_camera_y_rad_s']:+.6f}  "
            f"quietest p95={quietest['gyro_norm_p95_rad_s']:.4f} "
            f"bias={quietest['bias_camera_y_rad_s']:+.6f} "
            f"@t0+{quietest['offset_from_reference_start_s']:.0f}s"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(
        json.dumps(
            {
                "question": (
                    "Bias gyro hiệu chỉnh trên 2 s đầu recording. Có cửa sổ nào "
                    "yên hơn ở chỗ khác không, và bias thật là bao nhiêu?"
                ),
                "caveat": (
                    "production_window neo theo lưới bước step_seconds tính từ "
                    "reference start, còn IMUYawIntegrator neo đúng timestamp "
                    "stereo frame đầu, nên p95 lệch vài phần nghìn (garage_3: "
                    "0.0451 ở đây vs 0.0439 trong vo-drift-final). Không đổi kết "
                    "luận vì cả hai đều nằm cùng phía ngưỡng 0.02."
                ),
                "window_seconds": defaults.bias_calibration_seconds,
                "step_seconds": args.step_seconds,
                "sequences": results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nartifact: {args.output / 'audit.json'}")


if __name__ == "__main__":
    main()
