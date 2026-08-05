"""Video demo Phần B — localization + GPS integrity trên garage_2.

Script này chạy vòng fusion riêng thay vì gọi ``benchmark_gps_fusion.run_fusion``
vì cần state của integrity monitor ở từng frame để vẽ. Mọi con số KPI trong báo
cáo vẫn đến từ benchmark, không đến từ đây.

Không có footage thực địa tự quay trong time-box, nên video render từ 4Seasons
và banner ghi rõ nguồn — không trình bày như footage tự quay.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_tools.gps_sources import load_nmea_replay, load_reference_trajectory
from data_tools.stereo_odometry import run_stereo_odometry, stereo_frames
from pipelines.localization_ekf import LocalizationFusion, UTurnDetector
from pipelines.stereo_vo import StereoVOConfig

BANNER = (
    "Source: 4Seasons dataset (Wenzel et al. 2020) - recording_2021-02-25_13-39-06."
    " Rendered, NOT self-recorded HCMC footage."
)
STATE_COLORS = {
    "GOOD": (96, 216, 96),
    "DEGRADED": (64, 208, 240),
    "LOST": (72, 72, 232),
    "RECOVERING": (48, 160, 248),
}
MAP_SIZE = 400
BANNER_HEIGHT = 34
STATUS_HEIGHT = 108
UTURN_HOLD_S = 2.5


class MapCanvas:
    """Chiếu XY sang pixel, giữ tỉ lệ, gốc cố định theo bound của reference."""

    def __init__(self, reference: np.ndarray, size: int = MAP_SIZE) -> None:
        self.size = size
        lower = reference.min(axis=0)
        upper = reference.max(axis=0)
        span = float(np.max(upper - lower))
        self.scale = (size - 60) / max(span, 1e-6)
        self.center = (lower + upper) / 2.0

    def project(self, points: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(np.asarray(points, dtype=np.float64))
        offset = (points - self.center) * self.scale
        pixels = np.empty_like(offset)
        pixels[:, 0] = offset[:, 0] + self.size / 2.0
        # ảnh có trục y hướng xuống, lật để bắc nằm trên
        pixels[:, 1] = self.size / 2.0 - offset[:, 1]
        return pixels.astype(np.int32)


def draw_polyline(
    canvas: np.ndarray,
    pixels: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    if len(pixels) >= 2:
        cv2.polylines(canvas, [pixels.reshape(-1, 1, 2)], False, color, thickness)


def put(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.45,
    color: tuple[int, int, int] = (235, 235, 235),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def build_frames(recording: Path, calibration: Path, max_frames: int | None):
    timestamps, reference, _ = load_reference_trajectory(recording)
    vo_frames = run_stereo_odometry(
        recording,
        calibration,
        config=StereoVOConfig(),
        max_frames=max_frames,
        timestamp_min=float(timestamps[0]),
        timestamp_max=float(timestamps[-1]),
        yaw_source="imu",
    )
    images = {frame.frame_id: frame for frame in stereo_frames(recording)}
    return timestamps, reference, vo_frames, images


def render(
    recording: Path,
    calibration: Path,
    output: Path,
    max_frames: int | None,
    fps: float,
) -> dict:
    timestamps, reference, vo_frames, images = build_frames(
        recording, calibration, max_frames
    )
    replay = load_nmea_replay(recording, alignment_mode="reference_rigid")
    replay.seek(float(timestamps[0]))

    fusion = LocalizationFusion()
    detector = UTurnDetector()
    canvas_map = MapCanvas(reference)
    reference_pixels = canvas_map.project(reference)

    width = 800 + MAP_SIZE
    height = BANNER_HEIGHT + 400 + STATUS_HEIGHT
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Không mở được VideoWriter: {output}")

    track: list[tuple[int, int, str]] = []
    last_gps = None
    last_uturn = None
    uturn_count = 0
    previous_timestamp = float(vo_frames[0].timestamp)

    for index, frame in enumerate(vo_frames):
        timestamp = float(frame.timestamp)
        for measurement in replay.pop_until(timestamp):
            fusion.process_gps(measurement)
            last_gps = measurement
        if frame.measurement is None:
            dt = 1.0 / 30.0 if index == 0 else timestamp - previous_timestamp
            fusion.predict_only(
                timestamp,
                dt,
                translation_process_std=0.05,
                rotation_process_std=np.deg2rad(1.0),
            )
        else:
            fusion.process_odometry(frame.measurement)
        previous_timestamp = timestamp

        pose = fusion.global_pose
        state = fusion.integrity.state.value
        color = STATE_COLORS.get(state, (200, 200, 200))
        pixel = canvas_map.project(pose[:2])[0]
        track.append((int(pixel[0]), int(pixel[1]), state))

        event = detector.update(timestamp, float(pose[2]))
        if event is not None:
            uturn_count += 1
            last_uturn = (timestamp, event["heading_change_degrees"])

        left = cv2.imread(str(images[frame.frame_id].left_path), cv2.IMREAD_GRAYSCALE)
        if left is None:
            continue
        camera = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        if camera.shape[:2] != (400, 800):
            camera = cv2.resize(camera, (800, 400))

        map_panel = np.full((MAP_SIZE, MAP_SIZE, 3), 24, dtype=np.uint8)
        draw_polyline(map_panel, reference_pixels, (86, 86, 86), 1)
        for start, end in zip(track, track[1:]):
            cv2.line(
                map_panel,
                start[:2],
                end[:2],
                STATE_COLORS.get(end[2], (200, 200, 200)),
                2,
            )
        heading = float(pose[2])
        tip = (
            int(pixel[0] + 16 * np.cos(heading)),
            int(pixel[1] - 16 * np.sin(heading)),
        )
        cv2.arrowedLine(map_panel, tuple(pixel), tip, (255, 255, 255), 2, tipLength=0.4)
        cv2.circle(map_panel, tuple(pixel), 4, (255, 255, 255), -1)
        put(map_panel, "map frame (top-down)", (8, 18), scale=0.4, color=(170, 170, 170))
        put(map_panel, "grey = reference", (8, MAP_SIZE - 10), scale=0.38, color=(140, 140, 140))

        canvas = np.full((height, width, 3), 18, dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (width, BANNER_HEIGHT), (52, 40, 32), -1)
        put(canvas, BANNER, (10, 22), scale=0.42, color=(210, 210, 210))
        canvas[BANNER_HEIGHT : BANNER_HEIGHT + 400, 0:800] = camera
        canvas[BANNER_HEIGHT : BANNER_HEIGHT + 400, 800:] = map_panel

        base = BANNER_HEIGHT + 400
        elapsed = timestamp - float(vo_frames[0].timestamp)
        put(canvas, f"t = {elapsed:6.1f} s", (12, base + 24))
        put(canvas, "GPS integrity:", (150, base + 24))
        put(canvas, state, (270, base + 24), scale=0.55, color=color, thickness=2)
        if last_gps is not None:
            # NMEA thật có bản tin thiếu HDOP/sats, nhất là quanh lúc mất fix.
            hdop = (
                f"{last_gps.hdop:.1f}" if last_gps.hdop is not None else "n/a"
            )
            sats = (
                last_gps.satellites if last_gps.satellites is not None else "n/a"
            )
            put(
                canvas,
                f"fix q={last_gps.fix_quality}  sats={sats}  hdop={hdop}",
                (400, base + 24),
            )
        else:
            put(canvas, "no GPS fix yet", (400, base + 24), color=(120, 120, 200))

        odom = "VO dropout - predict only" if frame.measurement is None else "stereo VO + IMU yaw"
        put(canvas, f"odometry: {odom}", (12, base + 50))
        put(
            canvas,
            f"pose (map): x={pose[0]:8.2f}  y={pose[1]:8.2f}"
            f"  yaw={np.rad2deg(pose[2]):7.1f} deg",
            (12, base + 76),
        )
        put(
            canvas,
            f"local (odom): x={fusion.local_pose[0]:8.2f}"
            f"  y={fusion.local_pose[1]:8.2f}",
            (430, base + 76),
        )
        put(canvas, f"U-turns: {uturn_count}", (700, base + 50))
        if last_uturn is not None and timestamp - last_uturn[0] < UTURN_HOLD_S:
            cv2.rectangle(
                canvas,
                (820, base + 34),
                (width - 12, base + 66),
                (40, 140, 240),
                -1,
            )
            put(
                canvas,
                f"U-TURN {last_uturn[1]:+.0f} deg",
                (836, base + 56),
                scale=0.6,
                color=(20, 20, 20),
                thickness=2,
            )

        writer.write(canvas)

    writer.release()
    return {
        "frames": len(vo_frames),
        "duration_s": len(vo_frames) / fps,
        "uturns": uturn_count,
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path(".cache/data/4seasons"))
    parser.add_argument(
        "--recording", default="recording_2021-02-25_13-39-06"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/demo/part_b.mp4")
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    summary = render(
        args.dataset / args.recording,
        args.dataset / "calibration",
        args.output,
        args.max_frames,
        args.fps,
    )
    print(
        f"{summary['frames']} frames -> {summary['duration_s']:.1f}s, "
        f"{summary['uturns']} U-turns, {summary['output']}"
    )


if __name__ == "__main__":
    main()
