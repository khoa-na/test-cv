"""B2 — landmark re-identification giữa hai lần đi qua cùng bãi đỗ.

DB dựng từ ``garage_3``, query bằng ``garage_2``. Chiều này chốt ở STEP3 mục 0
theo chất lượng NMEA, trước khi thấy bất kỳ số recall nào.

Ranh giới dữ liệu:
  - Vị trí DB entry và prior khi query đều là pose NHÂN QUẢ do hệ tự ước lượng
    (VO + NMEA thật + EKF). Reference pose không bao giờ đi vào đường suy luận.
  - Reference pose chỉ dùng để gán nhãn positive và để đo sai số. Toàn bộ nhãn
    đi qua một phép fit duy nhất của STEP3 mục 2 (rigid 3D Kabsch, datum chung).

Định nghĩa positive (chốt trước, không đổi sau khi thấy recall):
    ‖xy_q - xy_d‖ <= 5.0 m  AND  |z_q - z_d| <= 2.0 m  AND  |dheading| <= 45 deg
Gate |dz| là bắt buộc: gate #1 cho thấy 35% pose trùng nhau trên mặt phẳng lại
nằm khác tầng.
"""

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_tools.gps_sources import load_nmea_replay, wrap_angle
from data_tools.imu_yaw import load_imu_yaw_integrator
from data_tools.map_frame import (
    Datum,
    first_quality4_datum,
    fit_reference_to_map,
    reference_in_map_frame,
)
from data_tools.stereo_odometry import load_calibration, stereo_frames
from pipelines.landmark_db import (
    LandmarkConfig,
    LandmarkMatcher,
    build_database,
)
from pipelines.localization_ekf import LocalizationFusion
from pipelines.stereo_vo import StereoVO, StereoVOConfig

RECORDINGS = {
    "garage_2": "recording_2021-02-25_13-39-06",
    "garage_3": "recording_2021-05-10_19-15-19",
}
POSITIVE_XY_M = 5.0
POSITIVE_DZ_M = 2.0
POSITIVE_HEADING_RAD = np.deg2rad(45.0)
SENSITIVITY_XY_M = (3.0, 10.0)


def run_causal_pipeline(
    recording: Path,
    calibration_dir: Path,
    datum: Datum,
    *,
    max_frames: int | None = None,
) -> dict:
    """VO + NMEA thật + EKF, trả pose nhân quả từng frame. Không đọc reference."""
    frames = stereo_frames(recording)
    if max_frames is not None:
        frames = frames[:max_frames]
    calibration = load_calibration(calibration_dir)
    vo = StereoVO(calibration, StereoVOConfig())
    imu_yaw = load_imu_yaw_integrator(recording, frames[0].timestamp)
    # raw_enu + datum chung: map dựng ở traversal này phải dùng được ở traversal
    # kia. reference_rigid bị cấm trong đường production của Bước 3.
    replay = load_nmea_replay(
        recording, alignment_mode="raw_enu", datum=datum.as_tuple()
    )
    replay.seek(float(frames[0].timestamp))
    fusion = LocalizationFusion()

    poses = np.zeros((len(frames), 3), dtype=np.float64)
    covariances = np.zeros((len(frames), 2, 2), dtype=np.float64)
    previous_timestamp = float(frames[0].timestamp)
    dropouts = 0

    for index, frame in enumerate(frames):
        left = cv2.imread(str(frame.left_path), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(frame.right_path), cv2.IMREAD_GRAYSCALE)
        measurement = vo.process(left, right, frame.timestamp)
        if measurement is not None:
            heading = imu_yaw.delta_heading(
                measurement.timestamp - measurement.dt, measurement.timestamp
            )
            if heading is not None:
                measurement = replace(
                    measurement, dtheta=heading, source="stereo_vo_imu_yaw"
                )
        for gps_measurement in replay.pop_until(frame.timestamp):
            fusion.process_gps(gps_measurement)
        if measurement is None:
            dropouts += 1
            dt = (
                1.0 / 30.0
                if index == 0
                else float(frame.timestamp - previous_timestamp)
            )
            fusion.predict_only(
                frame.timestamp,
                dt,
                translation_process_std=0.05,
                rotation_process_std=np.deg2rad(1.0),
            )
        else:
            fusion.process_odometry(measurement)
        previous_timestamp = float(frame.timestamp)
        poses[index] = fusion.global_pose
        covariances[index] = fusion.global_position_covariance()

    return {
        "frames": frames,
        "poses": poses,
        "covariances": covariances,
        "dropouts": dropouts,
        "calibration": calibration,
    }


def reference_labels(recording: Path, datum: Datum) -> dict:
    """XYZ + heading trong common ENU. Chỉ dùng để gán nhãn và chấm điểm."""
    fit = fit_reference_to_map(recording, datum)
    timestamps, xyz, heading = reference_in_map_frame(recording, fit)
    return {
        "timestamps": timestamps,
        "xyz": xyz,
        "heading": heading,
        "fit": fit,
    }


def sample_labels(labels: dict, timestamps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nội suy XYZ; heading nội suy trên vector đơn vị để không nhảy ở +-pi."""
    reference_timestamps = labels["timestamps"]
    xyz = np.column_stack(
        [
            np.interp(timestamps, reference_timestamps, labels["xyz"][:, axis])
            for axis in range(3)
        ]
    )
    cosine = np.interp(timestamps, reference_timestamps, np.cos(labels["heading"]))
    sine = np.interp(timestamps, reference_timestamps, np.sin(labels["heading"]))
    return xyz, np.arctan2(sine, cosine)


def positive_matrix(
    query_xyz: np.ndarray,
    query_heading: np.ndarray,
    db_xyz: np.ndarray,
    db_heading: np.ndarray,
    *,
    xy_threshold: float = POSITIVE_XY_M,
) -> np.ndarray:
    planar = np.linalg.norm(
        query_xyz[:, None, :2] - db_xyz[None, :, :2], axis=2
    )
    vertical = np.abs(query_xyz[:, None, 2] - db_xyz[None, :, 2])
    turned = np.abs(wrap_angle(query_heading[:, None] - db_heading[None, :]))
    return (
        (planar <= xy_threshold)
        & (vertical <= POSITIVE_DZ_M)
        & (turned <= POSITIVE_HEADING_RAD)
    )


def keyframe_indices_causal(poses: np.ndarray, config: LandmarkConfig) -> list[int]:
    """Cùng luật với select_keyframes, viết dạng tiến để dùng trong vòng query."""
    selected = [0]
    anchor = poses[0]
    for index in range(1, len(poses)):
        moved = float(np.linalg.norm(poses[index][:2] - anchor[:2]))
        turned = abs(float(wrap_angle(poses[index][2] - anchor[2])))
        if moved >= config.keyframe_distance_m or turned >= config.keyframe_rotation_rad:
            selected.append(index)
            anchor = poses[index]
    return selected


def landmark_nis(
    predicted: np.ndarray,
    predicted_covariance: np.ndarray,
    observed: np.ndarray,
    observation_covariance: np.ndarray,
) -> float:
    """NIS của update vị trí, tính mà không đụng vào EKF."""
    residual = np.asarray(observed, dtype=np.float64) - np.asarray(
        predicted, dtype=np.float64
    )
    innovation = np.asarray(predicted_covariance) + np.asarray(observation_covariance)
    return float(residual @ np.linalg.solve(innovation, residual))


def evaluate_regime(
    name: str,
    matcher: LandmarkMatcher,
    query: dict,
    query_keyframes: list[int],
    query_xyz: np.ndarray,
    query_heading: np.ndarray,
    positives: np.ndarray,
    db_entry_ids: list[int],
    db_xyz: np.ndarray,
    *,
    use_prior: bool,
) -> dict:
    entry_position = {entry_id: index for index, entry_id in enumerate(db_entry_ids)}
    eligible = positives.any(axis=1)
    recall_at_1 = 0
    recall_at_5 = 0
    verified_correct = 0
    accepted = 0
    accepted_correct = 0
    association_errors: list[float] = []
    frame_offsets: list[float] = []
    nis_values: list[float] = []
    nis_rejected = 0
    reacquisitions = 0
    durations: list[float] = []

    for query_index, frame_index in enumerate(query_keyframes):
        frame = query["frames"][frame_index]
        left = cv2.imread(str(frame.left_path), cv2.IMREAD_GRAYSCALE)
        right = cv2.imread(str(frame.right_path), cv2.IMREAD_GRAYSCALE)
        pose = query["poses"][frame_index]
        covariance = query["covariances"][frame_index]

        started = time.perf_counter()
        result = matcher.query(
            left,
            right,
            prior_position=pose[:2] if use_prior else None,
            prior_covariance=covariance if use_prior else None,
        )
        durations.append((time.perf_counter() - started) * 1000.0)
        reacquisitions += bool(result.used_full_database and use_prior)

        row = positives[query_index]
        if eligible[query_index]:
            ranked = [entry_position[entry_id] for entry_id in result.ranked_entry_ids]
            if ranked and row[ranked[0]]:
                recall_at_1 += 1
            if any(row[index] for index in ranked[:5]):
                recall_at_5 += 1

        if result.match is not None:
            accepted += 1
            matched = entry_position[result.match.entry_id]
            correct = bool(row[matched])
            accepted_correct += correct
            if eligible[query_index]:
                verified_correct += correct
            # Hai đại lượng khác hẳn nhau, gộp lại là sai. Association error đo
            # chất lượng ghép cặp, cả hai vế cùng nằm trong frame nhãn. Frame
            # offset đo lệch giữa pose hệ tự dựng DB và frame nhãn — đó là sai số
            # global của EKF lúc mapping, không phải lỗi của landmark.
            association_errors.append(
                float(np.linalg.norm(db_xyz[matched, :2] - query_xyz[query_index, :2]))
            )
            frame_offsets.append(
                float(np.linalg.norm(result.match.position - db_xyz[matched, :2]))
            )
            nis = landmark_nis(
                pose[:2], covariance, result.match.position, result.match.covariance
            )
            nis_values.append(nis)
            nis_rejected += nis > 5.991

    eligible_count = int(eligible.sum())
    return {
        "regime": name,
        "uses_spatial_prior": use_prior,
        "queries": len(query_keyframes),
        "eligible_queries": eligible_count,
        "coverage": eligible_count / max(len(query_keyframes), 1),
        "retrieval_recall_at_1": recall_at_1 / max(eligible_count, 1),
        "retrieval_recall_at_5": recall_at_5 / max(eligible_count, 1),
        "verified_recall_at_1": verified_correct / max(eligible_count, 1),
        "verified_accepts": accepted,
        "verified_precision": accepted_correct / max(accepted, 1),
        "incorrect_accept_rate": 1.0 - accepted_correct / max(accepted, 1),
        "association_error_m": {
            "definition": "‖label(db entry) - label(query)‖, cả hai trong frame nhãn",
            "median": float(np.median(association_errors)) if association_errors else None,
            "p95": float(np.percentile(association_errors, 95)) if association_errors else None,
        },
        "db_frame_offset_m": {
            "definition": (
                "‖pose hệ của db entry - label(db entry)‖; sai số global của EKF "
                "lúc dựng map, không phải sai số association"
            ),
            "median": float(np.median(frame_offsets)) if frame_offsets else None,
            "p95": float(np.percentile(frame_offsets, 95)) if frame_offsets else None,
        },
        "landmark_nis": {
            "gate": 5.991,
            "median": float(np.median(nis_values)) if nis_values else None,
            "p95": float(np.percentile(nis_values, 95)) if nis_values else None,
            "reject_rate": nis_rejected / max(len(nis_values), 1),
            "histogram": np.histogram(
                np.clip(nis_values, 0.0, 30.0), bins=15, range=(0.0, 30.0)
            )[0].tolist()
            if nis_values
            else [],
        },
        "full_database_reacquisitions": reacquisitions,
        "query_latency_ms": {
            "median": float(np.median(durations)) if durations else None,
            "p95": float(np.percentile(durations, 95)) if durations else None,
        },
    }


def run_pair(
    dataset: Path,
    db_name: str,
    query_name: str,
    config: LandmarkConfig,
    max_frames: int | None,
) -> dict:
    db_recording = dataset / RECORDINGS[db_name]
    query_recording = dataset / RECORDINGS[query_name]
    calibration_dir = dataset / "calibration"
    # Datum chung lấy từ recording dựng DB; mọi thứ sau đó nằm cùng một ENU.
    datum = first_quality4_datum(db_recording)

    db_run = run_causal_pipeline(
        db_recording, calibration_dir, datum, max_frames=max_frames
    )
    database = build_database(
        db_run["frames"],
        db_run["poses"],
        db_run["covariances"],
        db_run["calibration"],
        config=config,
        metadata={
            "mapping_recording": db_recording.name,
            "datum": [datum.latitude, datum.longitude, datum.altitude],
            "pose_source": "causal fusion (stereo VO + real NMEA + EKF)",
        },
    )
    if not len(database):
        raise RuntimeError("DB rỗng, không thể chấm B2")

    query_run = run_causal_pipeline(
        query_recording, calibration_dir, datum, max_frames=max_frames
    )
    query_keyframes = keyframe_indices_causal(query_run["poses"], config)

    db_labels = reference_labels(db_recording, datum)
    query_labels = reference_labels(query_recording, datum)
    db_timestamps = np.array(
        [entry.t_first for entry in database.entries], dtype=np.float64
    )
    db_xyz, db_heading = sample_labels(db_labels, db_timestamps)
    query_timestamps = np.array(
        [query_run["frames"][index].timestamp for index in query_keyframes],
        dtype=np.float64,
    )
    query_xyz, query_heading = sample_labels(query_labels, query_timestamps)

    positives = positive_matrix(query_xyz, query_heading, db_xyz, db_heading)
    db_entry_ids = [entry.id for entry in database.entries]

    regimes = []
    for regime_name, use_prior in (("full_pipeline", True), ("prior_free", False)):
        matcher = LandmarkMatcher(database, db_run["calibration"], config)
        regimes.append(
            evaluate_regime(
                regime_name,
                matcher,
                query_run,
                query_keyframes,
                query_xyz,
                query_heading,
                positives,
                db_entry_ids,
                db_xyz,
                use_prior=use_prior,
            )
        )

    sensitivity = {}
    for threshold in SENSITIVITY_XY_M:
        alternate = positive_matrix(
            query_xyz, query_heading, db_xyz, db_heading, xy_threshold=threshold
        )
        sensitivity[f"coverage_at_{threshold:g}m"] = float(
            alternate.any(axis=1).mean()
        )

    return {
        "database_recording": db_recording.name,
        "query_recording": query_recording.name,
        "database": {
            "keyframes": len(database),
            "spacing_median_m": database.spacing_m(),
            "vo_dropouts": db_run["dropouts"],
            "metadata": database.metadata,
        },
        "query": {
            "frames": len(query_run["frames"]),
            "keyframes": len(query_keyframes),
            "vo_dropouts": query_run["dropouts"],
        },
        "label_frame": {
            "database_fit": db_labels["fit"].metadata(),
            "query_fit": query_labels["fit"].metadata(),
        },
        "positive_definition": {
            "xy_m": POSITIVE_XY_M,
            "dz_m": POSITIVE_DZ_M,
            "heading_deg": 45.0,
            "uses_ground_truth": True,
        },
        "coverage_sensitivity": sensitivity,
        "regimes": regimes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(".cache/data/4seasons"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/landmark-reid"))
    parser.add_argument("--max-frames", type=int, help="Smoke test; bỏ trống khi chạy thật.")
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Chạy thêm chiều ngược (DB=garage_2) làm sanity check frame.",
    )
    args = parser.parse_args()

    config = LandmarkConfig()
    pairs = {"forward": ("garage_3", "garage_2")}
    if args.reverse:
        pairs["reverse_sanity_check"] = ("garage_2", "garage_3")

    results = {}
    for label, (db_name, query_name) in pairs.items():
        results[label] = run_pair(
            args.dataset, db_name, query_name, config, args.max_frames
        )
        headline = results[label]["regimes"][0]
        print(
            f"{label} ({db_name} -> {query_name}): "
            f"coverage {headline['coverage']:.3f}  "
            f"R@1 {headline['retrieval_recall_at_1']:.3f}  "
            f"R@5 {headline['retrieval_recall_at_5']:.3f}  "
            f"verified R@1 {headline['verified_recall_at_1']:.3f}  "
            f"precision {headline['verified_precision']:.3f}"
        )

    report = {
        "kpi": {"retrieval_recall_target": 0.85},
        "direction_decision": (
            "DB=garage_3, query=garage_2 chốt ở STEP3 mục 0 theo chất lượng NMEA, "
            "trước mọi số recall."
        ),
        "headline_regime": "full_pipeline",
        "pairs": results,
        "disclosure": [
            "Dev probe ở STEP3 mục 0 đã chạy trên toàn tuyến garage — cùng loại "
            "contamination với dev segment 1800 frame của Bước 2.",
            "Vị trí DB và prior đều là pose nhân quả của hệ; reference pose chỉ "
            "gán nhãn và chấm điểm.",
            "Tầng landmark chưa nằm trong vòng đo FPS của B7.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "benchmark.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nartifact: {args.output / 'benchmark.json'}")


if __name__ == "__main__":
    main()
