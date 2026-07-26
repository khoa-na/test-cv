"""Gate #1 của Bước 3: kiểm cặp traversal garage trước khi code landmark DB.

Trả lời hai câu hỏi mà STEP3 để mở:

1. Chênh lệch độ cao giữa hai recording là sai số đăng ký hay hai traversal
   thật sự chạy khác tầng? Script dựng **common map frame** của mục 2
   (``data_tools.map_frame``: reference → ENU 3D neo bằng RTK quality-4 của
   chính recording đó, datum chung), rồi xuất cặp ảnh nearest-neighbour cho cả
   hai nhóm ``|dz|`` nhỏ và ``|dz|`` lớn để người đọc kết luận bằng mắt.
2. Xe vào/ra hầm lúc nào? Candidate lấy từ exposure của ``times.txt`` (cột 3)
   và fix quality NMEA thật, chốt vào segment JSON để B5 dùng chung cho A/B/C.

Toàn bộ script là evaluation-side. Không phần nào đi vào production pipeline.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

from data_tools import fourseasons, map_frame
from data_tools.gps_sources import _epoch_from_seconds_of_day

MAPPING_RECORDING = "recording_2021-05-10_19-15-19"  # garage_3, dựng DB
QUERY_RECORDING = "recording_2021-02-25_13-39-06"  # garage_2, query + datum
SAME_LEVEL_DZ_M = 1.0
DIFFERENT_LEVEL_DZ_M = 2.0


def exposure_series(recording: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = np.loadtxt(recording / "times.txt", usecols=(1, 2))
    return raw[:, 0], raw[:, 1]


def indoor_candidate(
    timestamps: np.ndarray, exposure: np.ndarray, factor: float = 3.0
) -> dict:
    """Đề xuất cửa sổ trong hầm: exposure vọt lên so với mức ngoài trời.

    Ngưỡng là ``factor`` lần exposure trung vị của 10% frame phơi ngắn nhất
    (ngoài trời). Chỉ là candidate — timestamp cuối phải xác nhận bằng ảnh.
    """
    outdoor = float(np.median(np.sort(exposure)[: max(1, len(exposure) // 10)]))
    threshold = outdoor * factor
    indoor = exposure > threshold
    runs = []
    for value, group in itertools.groupby(enumerate(indoor), key=lambda item: item[1]):
        if not value:
            continue
        indices = [index for index, _ in group]
        runs.append((indices[0], indices[-1]))
    runs.sort(key=lambda run: run[1] - run[0], reverse=True)
    longest = runs[0] if runs else None
    return {
        "outdoor_exposure_ms": outdoor,
        "threshold_ms": threshold,
        "run_count": len(runs),
        "enter_timestamp": None if longest is None else float(timestamps[longest[0]]),
        "exit_timestamp": None if longest is None else float(timestamps[longest[1]]),
        "duration_s": (
            None
            if longest is None
            else float(timestamps[longest[1]] - timestamps[longest[0]])
        ),
    }


def gps_outage_candidate(recording: Path) -> dict:
    frame_epoch = float(fourseasons.load_times(recording)[0, 1])
    rows = fourseasons.load_nmea_gga(recording)
    qualities = [row["fix_quality"] for row in rows]
    runs = []
    index = 0
    for value, group in itertools.groupby(qualities):
        length = len(list(group))
        if value == 0:
            runs.append((index, index + length - 1, length))
        index += length
    runs.sort(key=lambda run: run[2], reverse=True)
    counts: dict[int, int] = {}
    for quality in qualities:
        counts[quality] = counts.get(quality, 0) + 1
    longest = runs[0] if runs else None
    return {
        "message_count": len(rows),
        "quality_counts": {str(key): counts[key] for key in sorted(counts)},
        "longest_zero_run_messages": None if longest is None else longest[2],
        "longest_zero_run_utc": (
            None
            if longest is None
            else [rows[longest[0]]["utc"], rows[longest[1]]["utc"]]
        ),
        "longest_zero_run_timestamp": (
            None
            if longest is None
            else [
                _epoch_from_seconds_of_day(rows[index]["utc"], frame_epoch)
                for index in (longest[0], longest[1])
            ]
        ),
    }


def write_pair_montage(
    output: Path, query_frame: Path, mapping_frame: Path, caption: str
) -> None:
    left = cv2.imread(str(query_frame), cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(str(mapping_frame), cv2.IMREAD_GRAYSCALE)
    if left is None or right is None:
        raise ValueError(f"Không đọc được {query_frame} hoặc {mapping_frame}")
    canvas = cv2.cvtColor(np.vstack((left, right)), cv2.COLOR_GRAY2BGR)
    cv2.putText(
        canvas, "query garage_2", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1
    )
    cv2.putText(
        canvas,
        "db garage_3",
        (8, left.shape[0] + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        1,
    )
    cv2.putText(
        canvas,
        caption,
        (8, canvas.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 200, 0),
        1,
    )
    cv2.imwrite(str(output), canvas)


def frame_for_timestamp(
    frames: list[Path], frame_times: np.ndarray, target: float
) -> Path:
    return frames[int(np.argmin(np.abs(frame_times - target)))]


def audit(dataset: Path, output: Path, pairs_per_group: int) -> dict:
    mapping_recording = dataset / MAPPING_RECORDING
    query_recording = dataset / QUERY_RECORDING
    output.mkdir(parents=True, exist_ok=True)

    datum = map_frame.first_quality4_datum(query_recording)
    query_fit = map_frame.fit_reference_to_map(query_recording, datum)
    mapping_fit = map_frame.fit_reference_to_map(mapping_recording, datum)
    query_times, query_xyz, query_heading = map_frame.reference_in_map_frame(
        query_recording, query_fit
    )
    mapping_times, mapping_xyz, mapping_heading = map_frame.reference_in_map_frame(
        mapping_recording, mapping_fit
    )

    planar = np.linalg.norm(
        query_xyz[:, None, :2] - mapping_xyz[None, :, :2], axis=2
    )
    nearest_planar = planar.argmin(1)
    planar_distance = planar.min(1)
    spatial = np.linalg.norm(query_xyz[:, None, :] - mapping_xyz[None, :, :], axis=2)
    spatial_distance = spatial.min(1)
    height_delta = query_xyz[:, 2] - mapping_xyz[nearest_planar, 2]

    close = planar_distance < 1.0
    histogram, edges = np.histogram(height_delta[close], bins=16)

    mapping_frames = sorted(
        (mapping_recording / "undistorted_images" / "cam0").glob("*.png")
    )
    query_frames = sorted((query_recording / "undistorted_images" / "cam0").glob("*.png"))
    mapping_frame_times, mapping_exposure = exposure_series(mapping_recording)
    query_frame_times, query_exposure = exposure_series(query_recording)

    groups = {
        "same_level": np.flatnonzero(close & (np.abs(height_delta) < SAME_LEVEL_DZ_M)),
        "different_level": np.flatnonzero(
            close & (np.abs(height_delta) > DIFFERENT_LEVEL_DZ_M)
        ),
    }
    exported = []
    for group, indices in groups.items():
        if not len(indices):
            continue
        for step in np.linspace(0, len(indices) - 1, pairs_per_group).astype(int):
            query_index = int(indices[step])
            mapping_index = int(nearest_planar[query_index])
            heading_delta = float(
                np.rad2deg(
                    np.abs(
                        (
                            query_heading[query_index]
                            - mapping_heading[mapping_index]
                            + np.pi
                        )
                        % (2 * np.pi)
                        - np.pi
                    )
                )
            )
            query_frame = frame_for_timestamp(
                query_frames, query_frame_times, query_times[query_index]
            )
            mapping_frame = frame_for_timestamp(
                mapping_frames, mapping_frame_times, mapping_times[mapping_index]
            )
            name = f"{group}_{query_index:05d}.png"
            write_pair_montage(
                output / name,
                query_frame,
                mapping_frame,
                f"dxy={planar_distance[query_index]:.2f}m "
                f"dz={height_delta[query_index]:+.2f}m "
                f"dheading={heading_delta:.0f}deg",
            )
            exported.append(
                {
                    "group": group,
                    "image": name,
                    "query_pose_index": query_index,
                    "query_timestamp": float(query_times[query_index]),
                    "query_frame": query_frame.name,
                    "mapping_pose_index": mapping_index,
                    "mapping_timestamp": float(mapping_times[mapping_index]),
                    "mapping_frame": mapping_frame.name,
                    "planar_distance_m": float(planar_distance[query_index]),
                    "height_delta_m": float(height_delta[query_index]),
                    "heading_delta_deg": heading_delta,
                }
            )

    report = {
        "question": (
            "Chênh cao giữa hai traversal là lệch đăng ký hay khác tầng? "
            "Kết luận đọc từ same_level_*.png và different_level_*.png; "
            "script không tự kết luận."
        ),
        "frame": query_fit.metadata()["direction"],
        "datum": query_fit.metadata()["datum"],
        "fit": {
            "query": query_fit.metadata(),
            "mapping": mapping_fit.metadata(),
        },
        "cross_traversal_nearest_neighbour": {
            "planar_median_m": float(np.median(planar_distance)),
            "planar_p95_m": float(np.percentile(planar_distance, 95)),
            "spatial_median_m": float(np.median(spatial_distance)),
            "spatial_p95_m": float(np.percentile(spatial_distance, 95)),
            "fraction_planar_within_2m": float((planar_distance < 2.0).mean()),
            "fraction_spatial_within_2m": float((spatial_distance < 2.0).mean()),
        },
        "height_delta_at_planar_close_pairs": {
            "planar_threshold_m": 1.0,
            "sample_count": int(close.sum()),
            "median_m": float(np.median(height_delta[close])),
            "fraction_same_level": float(
                (np.abs(height_delta[close]) < SAME_LEVEL_DZ_M).mean()
            ),
            "fraction_different_level": float(
                (np.abs(height_delta[close]) > DIFFERENT_LEVEL_DZ_M).mean()
            ),
            "histogram": [
                {"edge_m": float(edges[index]), "count": int(histogram[index])}
                for index in range(len(histogram))
                if histogram[index]
            ],
        },
        "indoor_candidate": {
            "query": indoor_candidate(query_frame_times, query_exposure),
            "mapping": indoor_candidate(mapping_frame_times, mapping_exposure),
        },
        "gps_outage": {
            "query": gps_outage_candidate(query_recording),
            "mapping": gps_outage_candidate(mapping_recording),
        },
        "pairs": exported,
        "reference_usage": "Evaluation only: nhãn và candidate segment",
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2))

    covered = report["indoor_candidate"]["query"]
    outage = report["gps_outage"]["query"]["longest_zero_run_timestamp"]
    segments = {
        "recording": QUERY_RECORDING,
        "role": "B5 query traversal",
        "covered_segment": {
            "enter_timestamp": covered["enter_timestamp"],
            "exit_timestamp": covered["exit_timestamp"],
            "duration_s": covered["duration_s"],
            "evidence": "exposure jump + xác nhận bằng ảnh biên ±3 s",
        },
        "real_gps_outage": {
            "enter_timestamp": None if outage is None else outage[0],
            "exit_timestamp": None if outage is None else outage[1],
            "note": (
                "Receiver giữ fix một lúc sau khi xe đã vào mái che và mất fix "
                "thêm một lúc sau khi ra; outage thật là tập con lệch trễ của "
                "covered_segment, không trùng biên"
            ),
        },
        "usage": (
            "Chốt trước khi chạy B5; ba cấu hình A/B/C dùng chung, không trim "
            "theo error"
        ),
    }
    (output / "segments.json").write_text(json.dumps(segments, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path(".cache/data/4seasons"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/garage-pair-audit")
    )
    parser.add_argument("--pairs-per-group", type=int, default=4)
    args = parser.parse_args()
    report = audit(args.dataset, args.output, args.pairs_per_group)
    print(json.dumps({k: v for k, v in report.items() if k != "pairs"}, indent=2))


if __name__ == "__main__":
    main()
