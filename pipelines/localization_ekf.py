"""GPS integrity, local EKF và global map->odom correction.

API odometry chỉ phụ thuộc relative SE(2), vì vậy Bước 2 có thể thay
ODOM_proxy bằng stereo VO mà không đổi fusion core.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

from data_tools.gps_sources import GPSMeasurement, OdometryMeasurement, wrap_angle


class GPSState(str, Enum):
    GOOD = "GOOD"
    DEGRADED = "DEGRADED"
    LOST = "LOST"
    RECOVERING = "RECOVERING"


@dataclass(frozen=True)
class IntegrityConfig:
    max_hdop: float = 5.0
    min_satellites: int = 4
    timeout_seconds: float = 1.5
    unusable_timeout_seconds: float = 1.5
    fix_zero_debounce: int = 3
    reject_limit: int = 3
    recovery_quality_count: int = 5
    recovery_nis_count: int = 3


@dataclass(frozen=True)
class StateTransition:
    timestamp: float
    previous: GPSState
    current: GPSState
    reason: str


class GPSIntegrityMonitor:
    def __init__(
        self,
        config: IntegrityConfig | None = None,
        initial_state: GPSState = GPSState.LOST,
    ) -> None:
        self.config = config or IntegrityConfig()
        self.state = initial_state
        self.last_message_time: float | None = None
        self.last_usable_fix_time: float | None = None
        self.state_entered_time: float | None = None
        self.fix_zero_streak = 0
        self.nis_reject_streak = 0
        self.quality_accept_streak = 0
        self.recovery_required = initial_state != GPSState.GOOD
        self.recent_recovery_nis: deque[bool] = deque(
            maxlen=self.config.recovery_nis_count
        )
        self.transitions: list[StateTransition] = []

    @property
    def uses_recovery_gate(self) -> bool:
        return self.recovery_required or self.state in {
            GPSState.LOST,
            GPSState.RECOVERING,
        }

    def quality_good(self, measurement: GPSMeasurement) -> bool:
        return (
            measurement.fix_quality >= 1
            and measurement.has_position
            and measurement.hdop is not None
            and measurement.hdop <= self.config.max_hdop
            and measurement.satellites >= self.config.min_satellites
        )

    def _transition(
        self, state: GPSState, timestamp: float, reason: str
    ) -> StateTransition | None:
        if state == self.state:
            return None
        transition = StateTransition(timestamp, self.state, state, reason)
        self.state = state
        self.state_entered_time = timestamp
        self.transitions.append(transition)
        self.nis_reject_streak = 0
        self.quality_accept_streak = 0
        self.recent_recovery_nis.clear()
        return transition

    def complete_recovery(self) -> None:
        if self.state != GPSState.GOOD:
            raise RuntimeError("Chỉ complete recovery khi integrity đã GOOD")
        self.recovery_required = False
        self.nis_reject_streak = 0
        self.quality_accept_streak = 0
        self.recent_recovery_nis.clear()

    def tick(self, timestamp: float) -> StateTransition | None:
        if (
            self.last_message_time is not None
            and self.state != GPSState.LOST
            and timestamp - self.last_message_time > self.config.timeout_seconds
        ):
            self.recovery_required = True
            return self._transition(GPSState.LOST, timestamp, "receiver_timeout")
        if self.state in {GPSState.DEGRADED, GPSState.RECOVERING}:
            usable_reference = (
                self.last_usable_fix_time
                if self.last_usable_fix_time is not None
                else self.state_entered_time
            )
            if (
                usable_reference is not None
                and timestamp - usable_reference
                > self.config.unusable_timeout_seconds
            ):
                self.recovery_required = True
                return self._transition(
                    GPSState.LOST, timestamp, "no_usable_fix_timeout"
                )
        return None

    def observe(
        self,
        measurement: GPSMeasurement,
        nis_accepted: bool,
        *,
        used_recovery_gate: bool | None = None,
    ) -> StateTransition | None:
        self.last_message_time = measurement.timestamp
        if used_recovery_gate is None:
            used_recovery_gate = self.uses_recovery_gate

        if measurement.fix_quality == 0 or not measurement.has_position:
            self.fix_zero_streak += 1
            self.quality_accept_streak = 0
            self.recent_recovery_nis.clear()
            self.recovery_required = True
            transition = None
            if self.state in {GPSState.GOOD, GPSState.RECOVERING}:
                transition = self._transition(
                    GPSState.DEGRADED,
                    measurement.timestamp,
                    "fix_quality_zero_degraded",
                )
            if (
                self.fix_zero_streak >= self.config.fix_zero_debounce
                and self.state != GPSState.LOST
            ):
                transition = self._transition(
                    GPSState.LOST,
                    measurement.timestamp,
                    "fix_quality_zero_confirmed",
                )
            return transition

        self.fix_zero_streak = 0
        quality_good = self.quality_good(measurement)
        if quality_good and nis_accepted:
            self.last_usable_fix_time = measurement.timestamp

        if self.state == GPSState.LOST:
            self.recovery_required = True
            if quality_good and nis_accepted:
                transition = self._transition(
                    GPSState.RECOVERING, measurement.timestamp, "valid_fix_returned"
                )
                self.quality_accept_streak = 1
                self.recent_recovery_nis.append(nis_accepted)
                return transition
            return None

        if self.state == GPSState.GOOD:
            if not quality_good:
                return self._transition(
                    GPSState.DEGRADED, measurement.timestamp, "quality_gate_failed"
                )
            self.nis_reject_streak = (
                0 if nis_accepted else self.nis_reject_streak + 1
            )
            if self.nis_reject_streak >= self.config.reject_limit:
                self.recovery_required = True
                return self._transition(
                    GPSState.DEGRADED, measurement.timestamp, "nis_reject_streak"
                )
            return None

        if self.state == GPSState.DEGRADED:
            if not quality_good:
                self.quality_accept_streak = 0
                return None
            if self.recovery_required:
                if nis_accepted and used_recovery_gate:
                    transition = self._transition(
                        GPSState.RECOVERING,
                        measurement.timestamp,
                        "recovery_fix_accepted",
                    )
                    self.quality_accept_streak = 1
                    self.recent_recovery_nis.append(True)
                    return transition
                return None
            if quality_good and nis_accepted:
                self.quality_accept_streak += 1
                self.nis_reject_streak = 0
                if self.quality_accept_streak >= self.config.recovery_quality_count:
                    return self._transition(
                        GPSState.GOOD, measurement.timestamp, "quality_hysteresis_passed"
                    )
            else:
                self.quality_accept_streak = 0
                self.nis_reject_streak += 1
                if self.nis_reject_streak >= self.config.reject_limit:
                    self.recovery_required = True
                    return self._transition(
                        GPSState.RECOVERING,
                        measurement.timestamp,
                        "normal_nis_deadlock_escape",
                    )
            return None

        if not quality_good:
            return self._transition(
                GPSState.DEGRADED,
                measurement.timestamp,
                "recovery_quality_gate_failed",
            )
        self.quality_accept_streak += 1
        self.recent_recovery_nis.append(nis_accepted)
        self.nis_reject_streak = 0 if nis_accepted else self.nis_reject_streak + 1
        if self.nis_reject_streak >= self.config.reject_limit:
            return self._transition(
                GPSState.DEGRADED, measurement.timestamp, "recovery_nis_reject_streak"
            )
        if (
            self.quality_accept_streak >= self.config.recovery_quality_count
            and len(self.recent_recovery_nis) == self.config.recovery_nis_count
            and all(self.recent_recovery_nis)
        ):
            return self._transition(
                GPSState.GOOD, measurement.timestamp, "recovery_hysteresis_passed"
            )
        return None


class LocalOdometryEKF:
    """EKF state [x, y, theta, v, omega] trong frame odom."""

    def __init__(self) -> None:
        self.state = np.zeros(5, dtype=np.float64)
        self.covariance = np.diag([1e-3, 1e-3, 1e-3, 1.0, 1.0])

    @property
    def pose(self) -> np.ndarray:
        return self.state[:3].copy()

    def predict(
        self,
        dt: float,
        translation_std: float = 0.01,
        rotation_std: float = 0.002,
    ) -> None:
        if dt <= 0:
            raise ValueError("dt phải > 0")
        x, y, theta, velocity, omega = self.state
        cosine, sine = np.cos(theta), np.sin(theta)
        self.state[0] = x + velocity * dt * cosine
        self.state[1] = y + velocity * dt * sine
        self.state[2] = float(wrap_angle(theta + omega * dt))
        jacobian = np.eye(5)
        jacobian[0, 2] = -velocity * dt * sine
        jacobian[0, 3] = dt * cosine
        jacobian[1, 2] = velocity * dt * cosine
        jacobian[1, 3] = dt * sine
        jacobian[2, 4] = dt
        velocity_std = translation_std / dt
        omega_std = rotation_std / dt
        process_noise = np.diag(
            [
                translation_std**2,
                translation_std**2,
                rotation_std**2,
                velocity_std**2 * dt,
                omega_std**2 * dt,
            ]
        )
        self.covariance = jacobian @ self.covariance @ jacobian.T + process_noise
        self.covariance = (self.covariance + self.covariance.T) / 2

    def update_odometry(self, measurement: OdometryMeasurement) -> None:
        lateral_ratio = abs(measurement.dy) / max(abs(measurement.dx), 0.01)
        inflation = 1.0 + min(lateral_ratio, 10.0)
        measurement_vector = np.array(
            [measurement.v_meas, measurement.omega_meas], dtype=np.float64
        )
        observation = np.zeros((2, 5), dtype=np.float64)
        observation[0, 3] = 1.0
        observation[1, 4] = 1.0
        measurement_covariance = np.diag(
            [
                (measurement.translation_std / measurement.dt) ** 2 * inflation,
                (measurement.rotation_std / measurement.dt) ** 2 * inflation,
            ]
        )
        innovation = measurement_vector - observation @ self.state
        innovation_covariance = (
            observation @ self.covariance @ observation.T + measurement_covariance
        )
        gain = (
            self.covariance
            @ observation.T
            @ np.linalg.inv(innovation_covariance)
        )
        self.state += gain @ innovation
        identity = np.eye(5)
        residual = identity - gain @ observation
        self.covariance = (
            residual @ self.covariance @ residual.T
            + gain @ measurement_covariance @ gain.T
        )
        self.covariance = (self.covariance + self.covariance.T) / 2

    def step(self, measurement: OdometryMeasurement) -> None:
        self.update_odometry(measurement)
        self.predict(
            measurement.dt,
            measurement.translation_std,
            measurement.rotation_std,
        )


@dataclass(frozen=True)
class FusionConfig:
    normal_nis_threshold: float = 5.991
    recovery_nis_threshold: float = 9.210
    gps_sigma_floor: float = 0.2
    recovery_sigma_floor: float = 2.0
    max_translation_correction_mps: float = 3.0
    recovery_tau_seconds: float = 1.0
    max_correction_step_m: float = 0.4
    recovery_consensus_count: int = 3
    recovery_consensus_radius_m: float = 4.0
    recovery_consensus_timeout_seconds: float = 5.0
    recovery_max_duration_seconds: float = 10.0
    min_fast_correction_quality: int = 2
    recovery_end_gap_m: float = 1.0
    max_rotation_correction_rps: float = np.deg2rad(30.0)
    heading_baseline_m: float = 2.0
    heading_update_baseline_m: float = 10.0
    heading_update_alpha: float = 0.2
    map_process_variance_per_second: float = 0.05


class LocalizationFusion:
    def __init__(
        self,
        config: FusionConfig | None = None,
        integrity_config: IntegrityConfig | None = None,
    ) -> None:
        self.config = config or FusionConfig()
        self.local = LocalOdometryEKF()
        self.integrity = GPSIntegrityMonitor(integrity_config)
        self.map_to_odom = np.zeros(3, dtype=np.float64)
        self.target_map_to_odom = np.zeros(3, dtype=np.float64)
        self.map_covariance = np.eye(2, dtype=np.float64) * 25.0
        self.global_initialized = False
        self.heading_initialized = False
        self.heading_anchor: tuple[np.ndarray, np.ndarray] | None = None
        self.recovery_candidates: deque[np.ndarray] = deque(maxlen=10)
        self.recovery_episode_started: float | None = None
        self.recovery_consensus_achieved = False
        self.recovery_consensus_timeout_logged = False
        self.recovery_unconverged_logged = False
        self.fast_correction_active = False
        self.recovery_log: list[dict] = []
        self.global_confidence = "low"
        self.last_recovery_outcome: str | None = None
        self.latched_global_pose: np.ndarray | None = None
        self.latched_covariance: np.ndarray | None = None
        self.latched_timestamp: float | None = None
        self.gps_log: list[dict] = []
        self.pose_log: list[dict] = []

    @staticmethod
    def _rotate(vector: np.ndarray, angle: float) -> np.ndarray:
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.array(
            [
                cosine * vector[0] - sine * vector[1],
                sine * vector[0] + cosine * vector[1],
            ]
        )

    @property
    def local_pose(self) -> np.ndarray:
        return self.local.pose

    @property
    def global_pose(self) -> np.ndarray:
        return self._pose_with_transform(self.map_to_odom)

    def _pose_with_transform(self, transform: np.ndarray) -> np.ndarray:
        local = self.local.pose
        position = self._rotate(local[:2], transform[2])
        return np.array(
            [
                position[0] + transform[0],
                position[1] + transform[1],
                float(wrap_angle(local[2] + transform[2])),
            ]
        )

    def _apply_transition(self, transition: StateTransition | None) -> None:
        if transition is None:
            return
        if transition.previous == GPSState.GOOD and transition.current != GPSState.GOOD:
            self.latched_global_pose = self.global_pose
            self.latched_covariance = self.global_position_covariance()
            self.latched_timestamp = transition.timestamp
            self._start_recovery_episode(transition.timestamp)
        if transition.current == GPSState.LOST:
            self._start_recovery_episode(transition.timestamp, reset=True)
        if (
            transition.current == GPSState.GOOD
            and not self.integrity.recovery_required
        ):
            self.latched_global_pose = None
            self.latched_covariance = None
            self.latched_timestamp = None

    def _start_recovery_episode(
        self, timestamp: float, *, reset: bool = False
    ) -> None:
        if self.recovery_episode_started is not None and not reset:
            return
        self.recovery_candidates.clear()
        self.recovery_episode_started = timestamp
        self.recovery_consensus_achieved = False
        self.recovery_consensus_timeout_logged = False
        self.recovery_unconverged_logged = False
        self.fast_correction_active = False
        self.global_confidence = "low"
        self.last_recovery_outcome = None
        self.recovery_log.append(
            {
                "timestamp": timestamp,
                "event": "episode_started",
                "reset": reset,
            }
        )

    def _record_recovery_candidate(
        self, measurement: GPSMeasurement
    ) -> None:
        if (
            not measurement.has_position
            or measurement.fix_quality
            < self.config.min_fast_correction_quality
        ):
            return
        candidate = measurement.position - self._rotate(
            self.local_pose[:2], self.target_map_to_odom[2]
        )
        self.recovery_candidates.append(candidate)
        count = self.config.recovery_consensus_count
        if len(self.recovery_candidates) < count:
            return
        recent = np.asarray(list(self.recovery_candidates)[-count:])
        median = np.median(recent, axis=0)
        radius = float(np.max(np.linalg.norm(recent - median, axis=1)))
        if radius > self.config.recovery_consensus_radius_m:
            return
        first_consensus = not self.recovery_consensus_achieved
        self.target_map_to_odom[:2] = median
        self.recovery_consensus_achieved = True
        self.fast_correction_active = True
        if first_consensus:
            self.recovery_log.append(
                {
                    "timestamp": measurement.timestamp,
                    "event": "consensus_achieved",
                    "candidate_count": count,
                    "radius_m": radius,
                    "target": median.tolist(),
                    "fix_quality": measurement.fix_quality,
                    "hdop": measurement.hdop,
                    "satellites": measurement.satellites,
                }
            )

    def _check_recovery_timeout(self, timestamp: float) -> None:
        if (
            self.recovery_episode_started is not None
            and not self.recovery_consensus_achieved
            and not self.recovery_consensus_timeout_logged
            and timestamp - self.recovery_episode_started
            > self.config.recovery_consensus_timeout_seconds
        ):
            self.recovery_consensus_timeout_logged = True
            self.recovery_log.append(
                {
                    "timestamp": timestamp,
                    "event": "consensus_timeout",
                    "candidate_count": len(self.recovery_candidates),
                }
            )

    def _finish_recovery(
        self, timestamp: float, gap: float, outcome: str
    ) -> None:
        latched_pose = (
            None
            if self.latched_global_pose is None
            else self.latched_global_pose.tolist()
        )
        latched_covariance = (
            None
            if self.latched_covariance is None
            else self.latched_covariance.tolist()
        )
        self.integrity.complete_recovery()
        self.fast_correction_active = False
        self.global_confidence = (
            "high" if outcome == "consensus_settled" else "low"
        )
        self.last_recovery_outcome = outcome
        self.recovery_log.append(
            {
                "timestamp": timestamp,
                "event": "episode_completed",
                "outcome": outcome,
                "terminal": True,
                "gap_m": gap,
                "consensus_achieved": self.recovery_consensus_achieved,
                "latched_pose": latched_pose,
                "latched_covariance": latched_covariance,
                "latched_timestamp": self.latched_timestamp,
            }
        )
        self.recovery_candidates.clear()
        self.recovery_episode_started = None
        self.recovery_consensus_achieved = False
        self.recovery_consensus_timeout_logged = False
        self.recovery_unconverged_logged = False
        self.latched_global_pose = None
        self.latched_covariance = None
        self.latched_timestamp = None

    def _maybe_complete_recovery(self, timestamp: float) -> None:
        if not self.integrity.recovery_required:
            return
        gap = float(
            np.linalg.norm(
                self.target_map_to_odom[:2] - self.map_to_odom[:2]
            )
        )
        if (
            self.integrity.state == GPSState.GOOD
            and gap <= self.config.recovery_end_gap_m
        ):
            if self.recovery_consensus_achieved:
                self._finish_recovery(
                    timestamp, gap, "consensus_settled"
                )
                return
            if self.recovery_consensus_timeout_logged:
                self._finish_recovery(
                    timestamp,
                    gap,
                    "timeout_settled_low_confidence",
                )
                return
        if (
            self.recovery_episode_started is not None
            and not self.recovery_unconverged_logged
            and timestamp - self.recovery_episode_started
            > self.config.recovery_max_duration_seconds
        ):
            self.recovery_unconverged_logged = True
            self.global_confidence = "low"
            self.last_recovery_outcome = "unconverged"
            self.recovery_log.append(
                {
                    "timestamp": timestamp,
                    "event": "recovery_unconverged",
                    "outcome": "unconverged",
                    "terminal": False,
                    "gap_m": gap,
                    "consensus_achieved": self.recovery_consensus_achieved,
                }
            )

    def global_position_covariance(self) -> np.ndarray:
        angle = self.map_to_odom[2]
        cosine, sine = np.cos(angle), np.sin(angle)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        return (
            self.map_covariance
            + rotation @ self.local.covariance[:2, :2] @ rotation.T
        )

    def _advance_global_correction(self, dt: float) -> None:
        angle_delta = float(
            wrap_angle(self.target_map_to_odom[2] - self.map_to_odom[2])
        )
        angle_limit = self.config.max_rotation_correction_rps * dt
        angle_step = float(np.clip(angle_delta, -angle_limit, angle_limit))
        if angle_step:
            global_position = self.global_pose[:2]
            self.map_to_odom[2] = float(
                wrap_angle(self.map_to_odom[2] + angle_step)
            )
            self.map_to_odom[:2] = global_position - self._rotate(
                self.local_pose[:2], self.map_to_odom[2]
            )
        translation_delta = self.target_map_to_odom[:2] - self.map_to_odom[:2]
        distance = float(np.linalg.norm(translation_delta))
        rate = self.config.max_translation_correction_mps
        if self.fast_correction_active:
            rate = max(rate, distance / self.config.recovery_tau_seconds)
            maximum = min(
                rate * dt,
                self.config.max_correction_step_m,
            )
        else:
            maximum = rate * dt
        if distance > maximum > 0:
            translation_delta *= maximum / distance
        self.map_to_odom[:2] += translation_delta

    def process_odometry(self, measurement: OdometryMeasurement) -> np.ndarray:
        self.local.step(measurement)
        self.map_covariance += (
            np.eye(2)
            * self.config.map_process_variance_per_second
            * measurement.dt
        )
        self._advance_global_correction(measurement.dt)
        self._apply_transition(self.integrity.tick(measurement.timestamp))
        if self.integrity.recovery_required:
            self._start_recovery_episode(measurement.timestamp)
        self._check_recovery_timeout(measurement.timestamp)
        self._maybe_complete_recovery(measurement.timestamp)
        pose = self.global_pose
        self.pose_log.append(
            {
                "timestamp": measurement.timestamp,
                "local": self.local_pose.tolist(),
                "global": pose.tolist(),
                "state": self.integrity.state.value,
            }
        )
        return pose

    def _gps_covariance(
        self, measurement: GPSMeasurement, recovering: bool
    ) -> np.ndarray:
        base_sigma = {
            1: 1.5,
            2: 0.5,
            4: 0.05,
            5: 0.5,
        }.get(measurement.fix_quality, 2.0)
        hdop = measurement.hdop if measurement.hdop is not None else 10.0
        sigma = max(self.config.gps_sigma_floor, hdop * base_sigma)
        if recovering:
            sigma = max(self.config.recovery_sigma_floor, sigma * np.sqrt(10.0))
        return np.eye(2) * sigma**2

    def _maybe_initialize_heading(
        self,
        position: np.ndarray,
        local_position: np.ndarray,
        *,
        allow_update: bool = False,
    ) -> None:
        if self.heading_initialized and not allow_update:
            return
        if self.heading_anchor is None:
            self.heading_anchor = (position.copy(), local_position.copy())
            return
        gps_anchor, local_anchor = self.heading_anchor
        gps_delta = position - gps_anchor
        local_delta = local_position - local_anchor
        required_baseline = (
            self.config.heading_update_baseline_m
            if self.heading_initialized
            else self.config.heading_baseline_m
        )
        if (
            np.linalg.norm(gps_delta) < required_baseline
            or np.linalg.norm(local_delta) < required_baseline
        ):
            return
        candidate = float(
            wrap_angle(
                np.arctan2(gps_delta[1], gps_delta[0])
                - np.arctan2(local_delta[1], local_delta[0])
            )
        )
        if self.heading_initialized:
            correction = float(
                wrap_angle(candidate - self.target_map_to_odom[2])
            )
            self.target_map_to_odom[2] = float(
                wrap_angle(
                    self.target_map_to_odom[2]
                    + self.config.heading_update_alpha * correction
                )
            )
        else:
            self.target_map_to_odom[2] = candidate
            self.heading_initialized = True
        self.target_map_to_odom[:2] = position - self._rotate(
            local_position, self.target_map_to_odom[2]
        )
        self.heading_anchor = (position.copy(), local_position.copy())

    def _position_innovation(
        self, position: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        innovation = position - self.global_pose[:2]
        innovation_covariance = self.global_position_covariance() + covariance
        nis = float(
            innovation.T @ np.linalg.solve(innovation_covariance, innovation)
        )
        return innovation, innovation_covariance, nis

    def update_position(
        self,
        position: np.ndarray,
        covariance: np.ndarray,
        *,
        gate: bool = True,
        nis_threshold: float | None = None,
    ) -> tuple[bool, float]:
        """Hook chung cho GPS/landmark; chỉ sửa target map->odom."""
        position = np.asarray(position, dtype=np.float64)
        covariance = np.asarray(covariance, dtype=np.float64)
        if position.shape != (2,) or covariance.shape != (2, 2):
            raise ValueError("position [2] và covariance [2,2] là bắt buộc")
        threshold = (
            self.config.normal_nis_threshold
            if nis_threshold is None
            else nis_threshold
        )
        innovation, innovation_covariance, nis = self._position_innovation(
            position, covariance
        )
        accepted = not gate or nis <= threshold
        if accepted:
            predicted_covariance = self.global_position_covariance()
            gain = predicted_covariance @ np.linalg.inv(innovation_covariance)
            target_innovation = (
                position - self._pose_with_transform(self.target_map_to_odom)[:2]
            )
            self.target_map_to_odom[:2] += gain @ target_innovation
            self.map_covariance = (
                np.eye(2) - gain
            ) @ predicted_covariance
            self.map_covariance = (
                self.map_covariance + self.map_covariance.T
            ) / 2
        return accepted, nis

    def process_gps(self, measurement: GPSMeasurement) -> dict:
        state_before = self.integrity.state
        used_recovery_gate = self.integrity.uses_recovery_gate
        if used_recovery_gate:
            self._start_recovery_episode(measurement.timestamp)
        quality_good = self.integrity.quality_good(measurement)
        accepted = False
        nis = float("inf")
        covariance = self._gps_covariance(measurement, used_recovery_gate)

        if measurement.has_position and quality_good:
            local_position = self.local_pose[:2]
            position = measurement.position
            if not self.global_initialized:
                self.map_to_odom[:2] = position - self._rotate(
                    local_position, self.map_to_odom[2]
                )
                self.target_map_to_odom[:] = self.map_to_odom
                self.map_covariance = covariance.copy()
                self.global_initialized = True
                self.heading_anchor = (position.copy(), local_position.copy())
                accepted, nis = True, 0.0
            else:
                self._maybe_initialize_heading(position, local_position)
                threshold = (
                    self.config.recovery_nis_threshold
                    if used_recovery_gate
                    else self.config.normal_nis_threshold
                )
                accepted, nis = self.update_position(
                    position, covariance, gate=True, nis_threshold=threshold
                )
                if accepted:
                    self._maybe_initialize_heading(
                        position, local_position, allow_update=True
                    )

        transition = self.integrity.observe(
            measurement,
            accepted,
            used_recovery_gate=used_recovery_gate,
        )
        self._apply_transition(transition)
        if self.integrity.recovery_required:
            self._start_recovery_episode(measurement.timestamp)
            if accepted and quality_good:
                self._record_recovery_candidate(measurement)
        self._check_recovery_timeout(measurement.timestamp)
        self._maybe_complete_recovery(measurement.timestamp)
        if (
            accepted
            and quality_good
            and measurement.fix_quality
            >= self.config.min_fast_correction_quality
            and not self.integrity.recovery_required
        ):
            self.global_confidence = "high"
        record = {
            "timestamp": measurement.timestamp,
            "state_before": state_before.value,
            "state_after": self.integrity.state.value,
            "quality_good": quality_good,
            "accepted": accepted,
            "used_recovery_gate": used_recovery_gate,
            "nis": nis,
            "fix_quality": measurement.fix_quality,
            "hdop": measurement.hdop,
            "satellites": measurement.satellites,
            "mode": measurement.mode,
            "recovery_required": self.integrity.recovery_required,
            "recovery_consensus": self.recovery_consensus_achieved,
            "fast_correction": self.fast_correction_active,
            "global_confidence": self.global_confidence,
            "recovery_outcome": self.last_recovery_outcome,
        }
        self.gps_log.append(record)
        return record


class UTurnDetector:
    def __init__(
        self,
        threshold_degrees: float = 150.0,
        window_seconds: float = 8.0,
        cooldown_seconds: float = 8.0,
    ) -> None:
        self.threshold = np.deg2rad(threshold_degrees)
        self.window_seconds = window_seconds
        self.cooldown_seconds = cooldown_seconds
        self.history: deque[tuple[float, float]] = deque()
        self.last_raw_heading: float | None = None
        self.unwrapped_heading = 0.0
        self.last_detection_time = -np.inf

    def update(self, timestamp: float, heading: float) -> dict | None:
        heading = float(wrap_angle(heading))
        if self.last_raw_heading is None:
            self.unwrapped_heading = heading
        else:
            self.unwrapped_heading += float(
                wrap_angle(heading - self.last_raw_heading)
            )
        self.last_raw_heading = heading
        self.history.append((timestamp, self.unwrapped_heading))
        while (
            len(self.history) > 1
            and timestamp - self.history[0][0] > self.window_seconds
        ):
            self.history.popleft()
        delta = self.unwrapped_heading - self.history[0][1]
        if (
            abs(delta) >= self.threshold
            and timestamp - self.last_detection_time >= self.cooldown_seconds
        ):
            self.last_detection_time = timestamp
            return {
                "timestamp": timestamp,
                "heading_change_degrees": float(np.rad2deg(delta)),
                "window_start": self.history[0][0],
            }
        return None
