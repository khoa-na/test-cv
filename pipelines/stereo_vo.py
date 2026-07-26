"""Stereo visual odometry nhẹ bằng ORB + 3D-2D PnP.

Ảnh đầu vào phải là stereo pinhole đã rectified. Reference pose không được
đọc trong module này; output chỉ là relative SE(2) trong vehicle frame trước.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from data_tools.gps_sources import OdometryMeasurement, wrap_angle


@dataclass(frozen=True)
class StereoVOConfig:
    nfeatures: int = 1200
    grid_columns: int = 4
    grid_rows: int = 2
    stereo_ratio: float = 0.75
    temporal_ratio: float = 0.8
    epipolar_tolerance_px: float = 1.0
    min_depth_m: float = 1.0
    max_depth_m: float = 60.0
    min_pnp_points: int = 15
    pnp_reprojection_error_px: float = 2.0
    pnp_iterations: int = 100
    pnp_confidence: float = 0.999
    temporal_min_radius_px: float = 40.0
    temporal_max_radius_px: float = 240.0
    temporal_radius_margin_px: float = 20.0
    forward_backward_interval: int = 1
    use_symmetric_yaw: bool = True
    forward_backward_translation_m: float = 0.25
    forward_backward_rotation_rad: float = np.deg2rad(3.0)
    inconsistent_std_inflation: float = 3.0
    translation_std_base_m: float = 0.08
    rotation_std_base_rad: float = np.deg2rad(1.0)
    reprojection_reference_px: float = 1.0
    min_quality: float = 0.05
    min_translation_std_m: float = 0.002
    max_translation_std_m: float = 0.5
    min_rotation_std_rad: float = np.deg2rad(0.05)
    max_rotation_std_rad: float = np.deg2rad(10.0)
    rng_seed: int = 7


@dataclass(frozen=True)
class MotionEstimate:
    rotation: np.ndarray
    translation: np.ndarray
    inlier_indices: np.ndarray
    inlier_ratio: float
    reprojection_rms_px: float


class StereoVO:
    """Causal frame-to-frame stereo VO."""

    def __init__(
        self,
        calibration: dict,
        config: StereoVOConfig | None = None,
    ) -> None:
        self.config = config or StereoVOConfig()
        camera = calibration["cam0"]
        self.fx = float(camera["fx"])
        self.fy = float(camera["fy"])
        self.cx = float(camera["cx"])
        self.cy = float(camera["cy"])
        self.baseline_m = float(calibration["baseline_m"])
        if min(self.fx, self.fy, self.baseline_m) <= 0:
            raise ValueError("Calibration stereo phải có focal và baseline > 0")
        self.camera_matrix = np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.min_disparity_px = self.fx * self.baseline_m / self.config.max_depth_m
        self.max_disparity_px = self.fx * self.baseline_m / self.config.min_depth_m
        per_cell = int(
            np.ceil(
                self.config.nfeatures
                / (self.config.grid_columns * self.config.grid_rows)
            )
        )
        self._orb = cv2.ORB_create(
            nfeatures=max(per_cell * 2, 100),
            fastThreshold=12,
        )
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self._previous_keypoints: list[cv2.KeyPoint] | None = None
        self._previous_descriptors: np.ndarray | None = None
        self._previous_points_3d: dict[int, np.ndarray] = {}
        self._previous_timestamp: float | None = None
        self._temporal_radius_px = self.config.temporal_max_radius_px
        self._frame_index = 0
        self._consistency_inflation = 1.0

    @staticmethod
    def depth_from_disparity(
        disparity_px: float | np.ndarray,
        fx: float,
        baseline_m: float,
    ) -> float | np.ndarray:
        disparity = np.asarray(disparity_px, dtype=np.float64)
        if np.any(disparity <= 0):
            raise ValueError("Disparity phải > 0")
        depth = fx * baseline_m / disparity
        return float(depth) if depth.ndim == 0 else depth

    @staticmethod
    def invert_pnp_transform(
        rotation_pnp: np.ndarray,
        translation_pnp: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        rotation_motion = np.asarray(rotation_pnp, dtype=np.float64).T
        translation_motion = (
            -rotation_motion
            @ np.asarray(translation_pnp, dtype=np.float64).reshape(3)
        )
        return rotation_motion, translation_motion

    @staticmethod
    def camera_motion_to_se2(
        rotation_motion: np.ndarray,
        translation_motion: np.ndarray,
    ) -> tuple[float, float, float]:
        rotation = np.asarray(rotation_motion, dtype=np.float64)
        translation = np.asarray(translation_motion, dtype=np.float64).reshape(3)
        camera_yaw = float(np.arctan2(rotation[0, 2], rotation[2, 2]))
        return (
            float(translation[2]),
            float(-translation[0]),
            float(wrap_angle(-camera_yaw)),
        )

    @staticmethod
    def integrate_pose(
        pose: np.ndarray,
        measurement: OdometryMeasurement,
    ) -> np.ndarray:
        """Tích phân full relative SE(2), dùng cho benchmark VO độc lập."""
        return StereoVO.integrate_relative(
            pose,
            measurement.dx,
            measurement.dy,
            measurement.dtheta,
        )

    @staticmethod
    def integrate_relative(
        pose: np.ndarray,
        dx: float,
        dy: float,
        dtheta: float,
    ) -> np.ndarray:
        x, y, theta = np.asarray(pose, dtype=np.float64)
        cosine, sine = np.cos(theta), np.sin(theta)
        return np.array(
            [
                x + cosine * dx - sine * dy,
                y + sine * dx + cosine * dy,
                float(wrap_angle(theta + dtheta)),
            ],
            dtype=np.float64,
        )

    def reset(self) -> None:
        self._previous_keypoints = None
        self._previous_descriptors = None
        self._previous_points_3d = {}
        self._previous_timestamp = None
        self._temporal_radius_px = self.config.temporal_max_radius_px
        self._frame_index = 0
        self._consistency_inflation = 1.0

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise ValueError("Ảnh stereo rỗng")
        if image.ndim == 2:
            return image
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        raise ValueError("Ảnh phải là grayscale hoặc BGR")

    def _detect_grid(
        self, image: np.ndarray
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        height, width = image.shape
        cell_limit = int(
            np.ceil(
                self.config.nfeatures
                / (self.config.grid_columns * self.config.grid_rows)
            )
        )
        output: list[cv2.KeyPoint] = []
        for row in range(self.config.grid_rows):
            y0 = row * height // self.config.grid_rows
            y1 = (row + 1) * height // self.config.grid_rows
            for column in range(self.config.grid_columns):
                x0 = column * width // self.config.grid_columns
                x1 = (column + 1) * width // self.config.grid_columns
                keypoints = self._orb.detect(image[y0:y1, x0:x1], None)
                keypoints = sorted(
                    keypoints,
                    key=lambda item: (-item.response, item.pt[1], item.pt[0]),
                )[:cell_limit]
                for keypoint in keypoints:
                    keypoint.pt = (
                        keypoint.pt[0] + x0,
                        keypoint.pt[1] + y0,
                    )
                output.extend(keypoints)
        output, descriptors = self._orb.compute(image, output)
        return list(output or []), descriptors

    def _mutual_ratio_matches(
        self,
        descriptors_a: np.ndarray | None,
        descriptors_b: np.ndarray | None,
        ratio: float,
    ) -> list[cv2.DMatch]:
        if (
            descriptors_a is None
            or descriptors_b is None
            or len(descriptors_a) < 2
            or len(descriptors_b) < 2
        ):
            return []

        def accepted(
            query: np.ndarray, train: np.ndarray
        ) -> dict[tuple[int, int], cv2.DMatch]:
            result = {}
            for pair in self._matcher.knnMatch(query, train, k=2):
                if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
                    match = pair[0]
                    result[(match.queryIdx, match.trainIdx)] = match
            return result

        forward = accepted(descriptors_a, descriptors_b)
        backward = accepted(descriptors_b, descriptors_a)
        mutual = [
            match
            for (query_index, train_index), match in forward.items()
            if (train_index, query_index) in backward
        ]
        return sorted(mutual, key=lambda item: (item.distance, item.queryIdx))

    def _stereo_points(
        self,
        left_keypoints: list[cv2.KeyPoint],
        left_descriptors: np.ndarray | None,
        right_keypoints: list[cv2.KeyPoint],
        right_descriptors: np.ndarray | None,
    ) -> dict[int, np.ndarray]:
        points = {}
        matches = self._mutual_ratio_matches(
            left_descriptors,
            right_descriptors,
            self.config.stereo_ratio,
        )
        for match in matches:
            left = left_keypoints[match.queryIdx].pt
            right = right_keypoints[match.trainIdx].pt
            disparity = left[0] - right[0]
            if (
                abs(left[1] - right[1]) > self.config.epipolar_tolerance_px
                or disparity < self.min_disparity_px
                or disparity > self.max_disparity_px
            ):
                continue
            depth = self.depth_from_disparity(
                disparity,
                self.fx,
                self.baseline_m,
            )
            points[match.queryIdx] = np.array(
                [
                    (left[0] - self.cx) * depth / self.fx,
                    (left[1] - self.cy) * depth / self.fy,
                    depth,
                ],
                dtype=np.float64,
            )
        return points

    def _temporal_correspondences(
        self,
        current_keypoints: list[cv2.KeyPoint],
        current_descriptors: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
        matches = self._mutual_ratio_matches(
            self._previous_descriptors,
            current_descriptors,
            self.config.temporal_ratio,
        )
        object_points = []
        image_points = []
        indices = []
        for match in matches:
            if match.queryIdx not in self._previous_points_3d:
                continue
            previous_pixel = np.array(
                self._previous_keypoints[match.queryIdx].pt
            )
            current_pixel = np.array(current_keypoints[match.trainIdx].pt)
            if (
                np.linalg.norm(current_pixel - previous_pixel)
                > self._temporal_radius_px
            ):
                continue
            object_points.append(self._previous_points_3d[match.queryIdx])
            image_points.append(current_pixel)
            indices.append((match.queryIdx, match.trainIdx))
        return (
            np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
            np.asarray(image_points, dtype=np.float64).reshape(-1, 2),
            indices,
        )

    def _estimate_motion(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
    ) -> MotionEstimate | None:
        if len(object_points) < self.config.min_pnp_points:
            return None
        cv2.setRNGSeed(self.config.rng_seed)
        success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            self.camera_matrix,
            None,
            iterationsCount=self.config.pnp_iterations,
            reprojectionError=self.config.pnp_reprojection_error_px,
            confidence=self.config.pnp_confidence,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not success or inliers is None:
            return None
        inlier_indices = inliers.reshape(-1)
        if len(inlier_indices) < self.config.min_pnp_points:
            return None
        inlier_object = object_points[inlier_indices]
        inlier_image = image_points[inlier_indices]
        rotation_vector, translation = cv2.solvePnPRefineLM(
            inlier_object,
            inlier_image,
            self.camera_matrix,
            None,
            rotation_vector,
            translation,
        )
        rotation_pnp, _ = cv2.Rodrigues(rotation_vector)
        projected, _ = cv2.projectPoints(
            inlier_object,
            rotation_vector,
            translation,
            self.camera_matrix,
            None,
        )
        residual = projected.reshape(-1, 2) - inlier_image
        rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        rotation_motion, translation_motion = self.invert_pnp_transform(
            rotation_pnp,
            translation,
        )
        return MotionEstimate(
            rotation=rotation_motion,
            translation=translation_motion,
            inlier_indices=inlier_indices,
            inlier_ratio=len(inlier_indices) / len(object_points),
            reprojection_rms_px=rms,
        )

    def _grid_coverage(
        self,
        points: np.ndarray,
        image_shape: tuple[int, int],
    ) -> int:
        if len(points) == 0:
            return 0
        height, width = image_shape
        columns = np.clip(
            (points[:, 0] * self.config.grid_columns / width).astype(int),
            0,
            self.config.grid_columns - 1,
        )
        rows = np.clip(
            (points[:, 1] * self.config.grid_rows / height).astype(int),
            0,
            self.config.grid_rows - 1,
        )
        return len(set(zip(rows.tolist(), columns.tolist())))

    def _standard_deviations(
        self,
        estimate: MotionEstimate,
        grid_coverage: int,
    ) -> tuple[float, float]:
        cell_count = self.config.grid_columns * self.config.grid_rows
        quality = max(
            self.config.min_quality,
            estimate.inlier_ratio
            * max(grid_coverage / cell_count, self.config.min_quality),
        )
        factor = (
            1.0
            + estimate.reprojection_rms_px
            / self.config.reprojection_reference_px
        ) / np.sqrt(len(estimate.inlier_indices) * quality)
        factor *= self._consistency_inflation
        translation_std = float(
            np.clip(
                self.config.translation_std_base_m * factor,
                self.config.min_translation_std_m,
                self.config.max_translation_std_m,
            )
        )
        rotation_std = float(
            np.clip(
                self.config.rotation_std_base_rad * factor,
                self.config.min_rotation_std_rad,
                self.config.max_rotation_std_rad,
            )
        )
        return translation_std, rotation_std

    def _check_forward_backward(
        self,
        forward: MotionEstimate,
        current_points_3d: dict[int, np.ndarray],
        indices: list[tuple[int, int]],
    ) -> MotionEstimate | None:
        reverse_object = []
        reverse_image = []
        for previous_index, current_index in indices:
            if current_index not in current_points_3d:
                continue
            reverse_object.append(current_points_3d[current_index])
            reverse_image.append(self._previous_keypoints[previous_index].pt)
        reverse = self._estimate_motion(
            np.asarray(reverse_object, dtype=np.float64).reshape(-1, 3),
            np.asarray(reverse_image, dtype=np.float64).reshape(-1, 2),
        )
        if reverse is None:
            self._consistency_inflation = self.config.inconsistent_std_inflation
            return None
        composed_rotation = reverse.rotation @ forward.rotation
        composed_translation = (
            reverse.rotation @ forward.translation + reverse.translation
        )
        rotation_vector, _ = cv2.Rodrigues(composed_rotation)
        inconsistent = (
            np.linalg.norm(composed_translation)
            > self.config.forward_backward_translation_m
            or np.linalg.norm(rotation_vector)
            > self.config.forward_backward_rotation_rad
        )
        self._consistency_inflation = (
            self.config.inconsistent_std_inflation if inconsistent else 1.0
        )
        return reverse

    def _store_current(
        self,
        keypoints: list[cv2.KeyPoint],
        descriptors: np.ndarray | None,
        points_3d: dict[int, np.ndarray],
        timestamp: float,
    ) -> None:
        self._previous_keypoints = keypoints
        self._previous_descriptors = descriptors
        self._previous_points_3d = points_3d
        self._previous_timestamp = timestamp

    def process(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        timestamp: float,
    ) -> OdometryMeasurement | None:
        left = self._gray(img_left)
        right = self._gray(img_right)
        if left.shape != right.shape:
            raise ValueError("Hai ảnh stereo phải cùng kích thước")
        timestamp = float(timestamp)
        if self._previous_timestamp is not None and timestamp <= self._previous_timestamp:
            raise ValueError("Timestamp VO phải tăng nghiêm ngặt")

        left_keypoints, left_descriptors = self._detect_grid(left)
        right_keypoints, right_descriptors = self._detect_grid(right)
        current_points_3d = self._stereo_points(
            left_keypoints,
            left_descriptors,
            right_keypoints,
            right_descriptors,
        )
        self._frame_index += 1
        if self._previous_timestamp is None:
            self._store_current(
                left_keypoints,
                left_descriptors,
                current_points_3d,
                timestamp,
            )
            return None

        dt = timestamp - self._previous_timestamp
        object_points, image_points, indices = self._temporal_correspondences(
            left_keypoints,
            left_descriptors,
        )
        estimate = self._estimate_motion(object_points, image_points)
        if estimate is None:
            self._temporal_radius_px = self.config.temporal_max_radius_px
            self._store_current(
                left_keypoints,
                left_descriptors,
                current_points_3d,
                timestamp,
            )
            return None

        inlier_pixels = image_points[estimate.inlier_indices]
        displacement = [
            np.linalg.norm(
                np.asarray(left_keypoints[current_index].pt)
                - np.asarray(self._previous_keypoints[previous_index].pt)
            )
            for previous_index, current_index in (
                indices[index] for index in estimate.inlier_indices
            )
        ]
        median_displacement = float(np.median(displacement))
        self._temporal_radius_px = float(
            np.clip(
                2.0 * median_displacement + self.config.temporal_radius_margin_px,
                self.config.temporal_min_radius_px,
                self.config.temporal_max_radius_px,
            )
        )
        reverse = None
        if (
            self.config.forward_backward_interval > 0
            and self._frame_index % self.config.forward_backward_interval == 0
        ):
            reverse = self._check_forward_backward(
                estimate,
                current_points_3d,
                indices,
            )
        coverage = self._grid_coverage(inlier_pixels, left.shape)
        translation_std, rotation_std = self._standard_deviations(
            estimate,
            coverage,
        )
        dx, dy, dtheta = self.camera_motion_to_se2(
            estimate.rotation,
            estimate.translation,
        )
        if self.config.use_symmetric_yaw and reverse is not None:
            reverse_as_forward_rotation = reverse.rotation.T
            reverse_as_forward_translation = (
                -reverse_as_forward_rotation @ reverse.translation
            )
            _, _, reverse_dtheta = self.camera_motion_to_se2(
                reverse_as_forward_rotation,
                reverse_as_forward_translation,
            )
            dtheta = float(
                wrap_angle(
                    dtheta
                    + 0.5 * float(wrap_angle(reverse_dtheta - dtheta))
                )
            )
        measurement = OdometryMeasurement(
            timestamp=timestamp,
            dt=dt,
            dx=dx,
            dy=dy,
            dtheta=dtheta,
            translation_std=translation_std,
            rotation_std=rotation_std,
            source="stereo_vo",
        )
        self._store_current(
            left_keypoints,
            left_descriptors,
            current_points_3d,
            timestamp,
        )
        return measurement
