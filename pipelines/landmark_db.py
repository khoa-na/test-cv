"""Landmark database + association cho Bước 3 (B2, B5).

Kiến trúc bốn tầng, thứ tự cố định:

1. spatial prior nhân quả từ pose fusion — không bao giờ từ reference,
2. global descriptor top-K,
3. sequence consistency trên cửa sổ keyframe,
4. geometric verification bằng PnP trên điểm 3D stereo của DB keyframe.

Mọi phép đổi camera → base dùng lại ``StereoVO.camera_motion_to_se2`` để cả VO
lẫn landmark chung một convention; ``TS_cam_imu`` lệch hoán vị trục 0.903° và
việc trộn hai convention sẽ tạo bias xoay hệ thống giữa map và measurement.

Entry DB là planar SE(2) (``position[2] = 0``): pose fusion không có z/pitch/roll
thật, nên dựng ``T_map_camera`` đầy đủ từ nó là bịa. Điểm 3D vì vậy được giữ
nguyên trong ``cam0 optical frame`` của chính keyframe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from data_tools.gps_sources import wrap_angle
from pipelines.stereo_vo import StereoVO, StereoVOConfig

DESCRIPTOR_SIZE = (64, 32)  # (width, height) sau khi resize
DESCRIPTOR_BLOCK = 8
POINT_FRAME = "cam0_optical"
POSE_FRAME = "base_link_se2"


@dataclass(frozen=True)
class LandmarkConfig:
    keyframe_distance_m: float = 2.0
    keyframe_rotation_rad: float = np.deg2rad(15.0)
    max_keypoints: int = 500
    top_k: int = 20
    sequence_window: int = 5
    prior_sigma_scale: float = 3.0
    prior_min_radius_m: float = 10.0
    prior_max_radius_m: float = 50.0
    reacquisition_after_misses: int = 5
    match_ratio: float = 0.8
    min_inliers: int = 15
    pnp_reprojection_error_px: float = 2.0
    pnp_iterations: int = 100
    pnp_confidence: float = 0.999
    max_reprojection_rms_px: float = 2.0
    covariance_eigenvalue_floor_m: float = 0.25
    sigma_model_m: float = 1.5
    lever_arm_m: float = 0.185
    min_sigma_geometry_m: float = 0.25
    rng_seed: int = 7


def global_descriptor(
    image: np.ndarray,
    size: tuple[int, int] = DESCRIPTOR_SIZE,
    block: int = DESCRIPTOR_BLOCK,
) -> np.ndarray:
    """Thumbnail patch-normalized, L2 normalize.

    Patch-normalize khử gain/offset toàn cục của ảnh, thứ thay đổi mạnh giữa hai
    traversal khác giờ. Deterministic, không training, không dependency mới.
    """
    if image is None or image.size == 0:
        raise ValueError("Ảnh rỗng")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    width, height = size
    if width % block or height % block:
        raise ValueError("Kích thước descriptor phải chia hết cho block")
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    blocks = small.reshape(height // block, block, width // block, block)
    mean = blocks.mean(axis=(1, 3), keepdims=True)
    std = blocks.std(axis=(1, 3), keepdims=True)
    normalized = ((blocks - mean) / (std + 1e-6)).reshape(height, width)
    vector = normalized.ravel()
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def select_keyframes(
    poses: np.ndarray,
    *,
    distance_m: float,
    rotation_rad: float,
) -> list[int]:
    """Chọn keyframe theo quãng đường/góc, **không** theo thời gian.

    Query và DB phải dùng chung luật này: tầng sequence so đường chéo nên
    spacing hai bên phải tương đương, còn luật theo tần suất sẽ cho spacing
    khác nhau khi tốc độ khác nhau (ramp chậm vs đường thẳng).
    """
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 3:
        raise ValueError("poses phải có shape [N,3] gồm x, y, theta")
    selected = [0]
    anchor = poses[0]
    for index in range(1, len(poses)):
        pose = poses[index]
        moved = float(np.linalg.norm(pose[:2] - anchor[:2]))
        turned = abs(float(wrap_angle(pose[2] - anchor[2])))
        if moved >= distance_m or turned >= rotation_rad:
            selected.append(index)
            anchor = pose
    return selected


@dataclass
class LandmarkEntry:
    id: int
    landmark_class: str
    position: np.ndarray  # [x, y, 0] map frame
    heading: float
    descriptor: np.ndarray
    keypoints: np.ndarray
    points_3d: np.ndarray
    orb_desc: np.ndarray
    t_first: float
    t_last: float
    n_obs: int
    covariance: np.ndarray


@dataclass
class LandmarkDatabase:
    entries: list[LandmarkEntry] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def descriptors(self) -> np.ndarray:
        if not self.entries:
            return np.zeros((0, DESCRIPTOR_SIZE[0] * DESCRIPTOR_SIZE[1]), np.float32)
        return np.stack([entry.descriptor for entry in self.entries])

    @property
    def positions(self) -> np.ndarray:
        if not self.entries:
            return np.zeros((0, 3), np.float64)
        return np.stack([entry.position for entry in self.entries])

    def spacing_m(self) -> float:
        if len(self.entries) < 2:
            return 0.0
        positions = self.positions[:, :2]
        return float(np.median(np.linalg.norm(np.diff(positions, axis=0), axis=1)))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        counts = np.array([len(entry.keypoints) for entry in self.entries], np.int64)
        payload = {
            "ids": np.array([entry.id for entry in self.entries], np.int64),
            "classes": np.array(
                [entry.landmark_class for entry in self.entries], dtype="<U32"
            ),
            "positions": self.positions,
            "headings": np.array([entry.heading for entry in self.entries], np.float64),
            "descriptors": self.descriptors,
            "t_first": np.array([entry.t_first for entry in self.entries], np.float64),
            "t_last": np.array([entry.t_last for entry in self.entries], np.float64),
            "n_obs": np.array([entry.n_obs for entry in self.entries], np.int64),
            "covariances": (
                np.stack([entry.covariance for entry in self.entries])
                if self.entries
                else np.zeros((0, 2, 2))
            ),
            "counts": counts,
            "keypoints": (
                np.concatenate([entry.keypoints for entry in self.entries])
                if self.entries
                else np.zeros((0, 2), np.float32)
            ),
            "points_3d": (
                np.concatenate([entry.points_3d for entry in self.entries])
                if self.entries
                else np.zeros((0, 3), np.float32)
            ),
            "orb_desc": (
                np.concatenate([entry.orb_desc for entry in self.entries])
                if self.entries
                else np.zeros((0, 32), np.uint8)
            ),
            "metadata": np.array(json.dumps(self.metadata)),
        }
        np.savez(path, **payload)

    @staticmethod
    def load(path: str | Path) -> LandmarkDatabase:
        data = np.load(Path(path), allow_pickle=False)
        offsets = np.concatenate(([0], np.cumsum(data["counts"])))
        entries = [
            LandmarkEntry(
                id=int(data["ids"][index]),
                landmark_class=str(data["classes"][index]),
                position=data["positions"][index],
                heading=float(data["headings"][index]),
                descriptor=data["descriptors"][index],
                keypoints=data["keypoints"][offsets[index] : offsets[index + 1]],
                points_3d=data["points_3d"][offsets[index] : offsets[index + 1]],
                orb_desc=data["orb_desc"][offsets[index] : offsets[index + 1]],
                t_first=float(data["t_first"][index]),
                t_last=float(data["t_last"][index]),
                n_obs=int(data["n_obs"][index]),
                covariance=data["covariances"][index],
            )
            for index in range(len(data["ids"]))
        ]
        return LandmarkDatabase(entries, json.loads(str(data["metadata"])))


def _floor_covariance(covariance: np.ndarray, floor_m: float) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=np.float64)
    symmetric = (covariance + covariance.T) / 2
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, floor_m**2)
    return vectors @ np.diag(values) @ vectors.T


def build_database(
    frames: list,
    poses: np.ndarray,
    pose_covariances: np.ndarray,
    calibration: dict,
    *,
    config: LandmarkConfig | None = None,
    vo_config: StereoVOConfig | None = None,
    metadata: dict | None = None,
    read_image=None,
) -> LandmarkDatabase:
    """Dựng DB từ mapping traversal.

    ``frames`` là list ``StereoOdometryFrame``-like có ``timestamp``,
    ``left_path``, ``right_path``. ``poses[i]`` là SE(2) map-frame **nhân quả**
    của frame ``i`` (từ pose log của fusion, không phải reference).
    """
    config = config or LandmarkConfig()
    poses = np.asarray(poses, dtype=np.float64)
    pose_covariances = np.asarray(pose_covariances, dtype=np.float64)
    if len(frames) != len(poses) or len(frames) != len(pose_covariances):
        raise ValueError("frames, poses và pose_covariances phải cùng độ dài")
    reader = read_image or (
        lambda path: cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    )
    stereo = StereoVO(calibration, vo_config)
    entries = []
    for entry_id, index in enumerate(
        select_keyframes(
            poses,
            distance_m=config.keyframe_distance_m,
            rotation_rad=config.keyframe_rotation_rad,
        )
    ):
        frame = frames[index]
        left = reader(frame.left_path)
        right = reader(frame.right_path)
        if left is None or right is None:
            raise ValueError(f"Không đọc được stereo frame {frame.timestamp}")
        pixels, descriptors, points = stereo.keyframe_features(left, right)
        if len(pixels) < config.min_inliers:
            continue
        if len(pixels) > config.max_keypoints:
            keep = np.argsort(points[:, 2])[: config.max_keypoints]
            pixels, descriptors, points = pixels[keep], descriptors[keep], points[keep]
        entries.append(
            LandmarkEntry(
                id=entry_id,
                landmark_class="keyframe_orb",
                position=np.array([poses[index][0], poses[index][1], 0.0]),
                heading=float(poses[index][2]),
                descriptor=global_descriptor(left).astype(np.float32),
                keypoints=pixels,
                points_3d=points,
                orb_desc=descriptors,
                t_first=float(frame.timestamp),
                t_last=float(frame.timestamp),
                n_obs=1,
                covariance=_floor_covariance(
                    pose_covariances[index], config.covariance_eigenvalue_floor_m
                ),
            )
        )
    database = LandmarkDatabase(entries, dict(metadata or {}))
    database.metadata.update(
        {
            "point_frame": POINT_FRAME,
            "pose_frame": POSE_FRAME,
            "keyframe_count": len(entries),
            "keyframe_spacing_median_m": database.spacing_m(),
            "keyframe_rule": "distance>=2m hoặc rotation>=15deg (query dùng chung)",
            "descriptor": "thumbnail 64x32 patch-normalized L2",
            "n_obs_semantics": "MVP chưa gộp track: t_first=t_last, n_obs=1",
        }
    )
    return database


@dataclass(frozen=True)
class LandmarkMatch:
    entry_id: int
    position: np.ndarray
    covariance: np.ndarray
    inliers: int
    reprojection_rms_px: float


@dataclass(frozen=True)
class QueryResult:
    """Ranked list luôn được trả về, kể cả khi verification trượt.

    B2 lấy Recall@K từ ranked list (sau tầng 3, trước PnP); nếu chỉ trả match đã
    verify thì mọi query trượt PnP sẽ biến mất khỏi metric retrieval.
    """

    ranked_entry_ids: list[int]
    match: LandmarkMatch | None
    used_full_database: bool
    candidate_count: int


class LandmarkMatcher:
    """Query nhân quả: prior → descriptor → sequence → geometric verification."""

    def __init__(
        self,
        database: LandmarkDatabase,
        calibration: dict,
        config: LandmarkConfig | None = None,
        vo_config: StereoVOConfig | None = None,
    ) -> None:
        self.database = database
        self.config = config or LandmarkConfig()
        self.stereo = StereoVO(calibration, vo_config)
        self.camera_matrix = self.stereo.camera_matrix
        self._similarity_history: list[np.ndarray] = []
        self._consecutive_misses = 0
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def prior_radius(self, covariance: np.ndarray | None) -> float:
        if covariance is None:
            return float("inf")
        trace = float(np.trace(np.asarray(covariance, dtype=np.float64)))
        radius = self.config.prior_sigma_scale * np.sqrt(max(trace, 0.0))
        return float(
            np.clip(radius, self.config.prior_min_radius_m, self.config.prior_max_radius_m)
        )

    def _candidate_mask(
        self, prior_position: np.ndarray | None, covariance: np.ndarray | None
    ) -> tuple[np.ndarray, bool]:
        count = len(self.database)
        if prior_position is None:
            return np.ones(count, dtype=bool), True
        forced_full = self._consecutive_misses >= self.config.reacquisition_after_misses
        if forced_full:
            return np.ones(count, dtype=bool), True
        radius = self.prior_radius(covariance)
        distances = np.linalg.norm(
            self.database.positions[:, :2] - np.asarray(prior_position)[:2], axis=1
        )
        mask = distances <= radius
        if not mask.any():
            return np.ones(count, dtype=bool), True
        return mask, False

    def _sequence_scores(self, similarity: np.ndarray) -> np.ndarray:
        """Điểm dọc đường chéo: query ``i-k`` ↔ DB ``j-k``."""
        history = self._similarity_history[-(self.config.sequence_window - 1) :]
        scores = similarity.copy()
        weights = np.ones_like(scores)
        for step, previous in enumerate(reversed(history), start=1):
            shifted = np.full_like(scores, np.nan)
            if step < len(previous):
                shifted[step:] = previous[:-step]
            valid = ~np.isnan(shifted)
            scores[valid] += shifted[valid]
            weights[valid] += 1.0
        return scores / weights

    def _verify(
        self, entry: LandmarkEntry, pixels: np.ndarray, descriptors: np.ndarray
    ) -> tuple[np.ndarray, int, float, float] | None:
        if len(descriptors) < 2 or len(entry.orb_desc) < 2:
            return None
        pairs = self._matcher.knnMatch(entry.orb_desc, descriptors, k=2)
        matched = [
            (pair[0].queryIdx, pair[0].trainIdx)
            for pair in pairs
            if len(pair) == 2
            and pair[0].distance < self.config.match_ratio * pair[1].distance
        ]
        if len(matched) < self.config.min_inliers:
            return None
        object_points = np.array(
            [entry.points_3d[db_index] for db_index, _ in matched], dtype=np.float64
        )
        image_points = np.array(
            [pixels[query_index] for _, query_index in matched], dtype=np.float64
        )
        cv2.setRNGSeed(self.config.rng_seed)
        success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            self.camera_matrix,
            None,
            reprojectionError=self.config.pnp_reprojection_error_px,
            iterationsCount=self.config.pnp_iterations,
            confidence=self.config.pnp_confidence,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success or inliers is None or len(inliers) < self.config.min_inliers:
            return None
        index = inliers.ravel()
        rotation_vector, translation = cv2.solvePnPRefineLM(
            object_points[index],
            image_points[index],
            self.camera_matrix,
            None,
            rotation_vector,
            translation,
        )
        projected, _ = cv2.projectPoints(
            object_points[index],
            rotation_vector,
            translation,
            self.camera_matrix,
            None,
        )
        residual = projected.reshape(-1, 2) - image_points[index]
        rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        if rms > self.config.max_reprojection_rms_px:
            return None
        rotation, _ = cv2.Rodrigues(rotation_vector)
        motion_rotation, motion_translation = StereoVO.invert_pnp_transform(
            rotation, translation
        )
        delta = StereoVO.camera_motion_to_se2(motion_rotation, motion_translation)
        pose = StereoVO.integrate_relative(
            np.array([entry.position[0], entry.position[1], entry.heading]),
            *delta,
        )
        depth = float(np.median(object_points[index][:, 2]))
        return pose, len(index), rms, depth

    def measurement_covariance(
        self, entry: LandmarkEntry, inliers: int, rms: float, median_depth: float
    ) -> tuple[np.ndarray, dict]:
        sigma_geometry = max(
            self.config.min_sigma_geometry_m,
            median_depth / self.stereo.fx * rms / np.sqrt(max(inliers, 1)),
        )
        extra = (
            self.config.sigma_model_m**2
            + sigma_geometry**2
            + self.config.lever_arm_m**2
        )
        covariance = np.asarray(entry.covariance, dtype=np.float64) + np.eye(2) * extra
        return covariance, {
            "covariance_db_trace": float(np.trace(entry.covariance)),
            "sigma_model_m": self.config.sigma_model_m,
            "sigma_geometry_m": float(sigma_geometry),
            "lever_arm_m": self.config.lever_arm_m,
        }

    def query(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        *,
        prior_position: np.ndarray | None = None,
        prior_covariance: np.ndarray | None = None,
    ) -> QueryResult:
        if not len(self.database):
            return QueryResult([], None, True, 0)
        descriptor = global_descriptor(img_left)
        similarity = self.database.descriptors @ descriptor
        # Tính điểm sequence TRƯỚC khi đẩy similarity hiện tại vào history, nếu
        # không cửa sổ sẽ tự so query với chính nó lệch một bước.
        mask, used_full = self._candidate_mask(prior_position, prior_covariance)
        scores = self._sequence_scores(similarity)
        self._similarity_history.append(similarity)
        if len(self._similarity_history) > self.config.sequence_window:
            self._similarity_history.pop(0)

        scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[: self.config.top_k]
        ranked = [int(self.database.entries[index].id) for index in order]

        pixels, descriptors, _ = self.stereo.keyframe_features(img_left, img_right)
        for index in order:
            if not np.isfinite(scores[index]):
                break
            entry = self.database.entries[index]
            verified = self._verify(entry, pixels, descriptors)
            if verified is None:
                continue
            pose, inliers, rms, depth = verified
            covariance, _ = self.measurement_covariance(entry, inliers, rms, depth)
            self._consecutive_misses = 0
            return QueryResult(
                ranked_entry_ids=ranked,
                match=LandmarkMatch(
                    entry_id=entry.id,
                    position=pose[:2],
                    covariance=covariance,
                    inliers=inliers,
                    reprojection_rms_px=rms,
                ),
                used_full_database=used_full,
                candidate_count=int(mask.sum()),
            )
        self._consecutive_misses += 1
        return QueryResult(ranked, None, used_full, int(mask.sum()))
