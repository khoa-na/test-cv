"""ROS 2 node đóng gói localization pipeline (mục 3.3 của đề).

Môi trường test không có ROS 2 cài đặt, nên file này tách làm hai lớp:

``FusionBridge``   thuần Python, không import rclpy. Nhận số liệu đã bóc khỏi
                   message, chạy ``LocalizationFusion``, trả về dict đủ để lấp
                   vào ``nav_msgs/Odometry``. Lớp này CÓ test offline.

``LocalizationNode`` lớp rclpy mỏng, chỉ bóc/đóng message và gọi bridge.
                   KHÔNG được kiểm chứng runtime vì không có ROS 2 ở đây.

Chạy được không cần ROS 2:

    .venv/bin/python -m ros2.localization_node --self-check

Topic khi chạy dưới ROS 2:
    sub  /gps/fix              sensor_msgs/NavSatFix
    sub  /odom/visual          nav_msgs/Odometry        (stereo VO)
    pub  /localization/odometry  nav_msgs/Odometry      (frame map)
    pub  /localization/pose      geometry_msgs/PoseStamped
    pub  /localization/status    std_msgs/String        (GPS integrity state)
    tf   map -> odom, odom -> base_link
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from data_tools.gps_sources import GPSMeasurement, OdometryMeasurement
from data_tools.map_frame import Datum, geodetic_to_enu_3d
from pipelines.localization_ekf import LocalizationFusion

# NavSatFix.status.status: -1 NO_FIX, 0 FIX, 1 SBAS, 2 GBAS.
# Hệ tích hợp dùng thang fix quality của NMEA GGA nên phải ánh xạ.
NAVSAT_TO_GGA = {-1: 0, 0: 1, 1: 2, 2: 4}


class FusionBridge:
    """Lõi không phụ thuộc ROS. Datum chốt ở fix hợp lệ đầu tiên."""

    def __init__(self, *, datum: Datum | None = None) -> None:
        self.fusion = LocalizationFusion()
        self.datum = datum
        self.datum_source = "provided" if datum is not None else "pending"
        self.last_timestamp: float | None = None

    def on_gps(
        self,
        timestamp: float,
        latitude: float,
        longitude: float,
        altitude: float,
        status: int,
        *,
        satellites: int = 0,
        hdop: float | None = None,
    ) -> None:
        quality = NAVSAT_TO_GGA.get(int(status), 0)
        position = None
        finite = all(math.isfinite(value) for value in (latitude, longitude, altitude))
        if quality > 0 and finite:
            if self.datum is None:
                self.datum = Datum(latitude, longitude, altitude)
                self.datum_source = "first_valid_fix"
            east_north_up = geodetic_to_enu_3d(
                np.array([latitude]),
                np.array([longitude]),
                np.array([altitude]),
                self.datum,
            )[0]
            position = (float(east_north_up[0]), float(east_north_up[1]))

        self.fusion.process_gps(
            GPSMeasurement(
                timestamp=float(timestamp),
                x=position[0] if position else None,
                y=position[1] if position else None,
                fix_quality=quality,
                satellites=int(satellites),
                hdop=hdop,
                source="navsatfix",
            )
        )

    def on_odometry(
        self,
        timestamp: float,
        dx: float,
        dy: float,
        dtheta: float,
        dt: float,
        *,
        translation_std: float = 0.05,
        rotation_std: float = math.radians(1.0),
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt phải dương")
        self.fusion.process_odometry(
            OdometryMeasurement(
                timestamp=float(timestamp),
                dt=float(dt),
                dx=float(dx),
                dy=float(dy),
                dtheta=float(dtheta),
                translation_std=translation_std,
                rotation_std=rotation_std,
                source="ros_visual_odometry",
            )
        )
        self.last_timestamp = float(timestamp)

    def on_odometry_dropout(self, timestamp: float, dt: float) -> None:
        """VO mất frame: chỉ predict, không bịa measurement."""
        self.fusion.predict_only(
            float(timestamp),
            float(dt),
            translation_process_std=0.05,
            rotation_process_std=math.radians(1.0),
        )
        self.last_timestamp = float(timestamp)

    def snapshot(self) -> dict:
        """Đủ để lấp nav_msgs/Odometry và cả hai TF."""
        pose = self.fusion.global_pose
        local = self.fusion.local_pose
        map_to_odom = self.fusion.map_to_odom
        covariance = self.fusion.global_position_covariance()
        return {
            "timestamp": self.last_timestamp,
            "map": {
                "x": float(pose[0]),
                "y": float(pose[1]),
                "yaw": float(pose[2]),
            },
            "odom": {
                "x": float(local[0]),
                "y": float(local[1]),
                "yaw": float(local[2]),
            },
            "map_to_odom": {
                "x": float(map_to_odom[0]),
                "y": float(map_to_odom[1]),
                "yaw": float(map_to_odom[2]),
            },
            "position_covariance": [
                [float(covariance[0, 0]), float(covariance[0, 1])],
                [float(covariance[1, 0]), float(covariance[1, 1])],
            ],
            "gps_state": self.fusion.integrity.state.value,
            "datum": None
            if self.datum is None
            else {
                "latitude": self.datum.latitude,
                "longitude": self.datum.longitude,
                "altitude": self.datum.altitude,
                "source": self.datum_source,
            },
        }

    def odometry_covariance_row_major(self) -> list[float]:
        """36 phần tử của nav_msgs/Odometry.pose.covariance."""
        covariance = self.fusion.global_position_covariance()
        flat = [0.0] * 36
        flat[0] = float(covariance[0, 0])
        flat[1] = float(covariance[0, 1])
        flat[6] = float(covariance[1, 0])
        flat[7] = float(covariance[1, 1])
        flat[14] = 1e6  # z không quan sát trong mô hình SE(2)
        flat[21] = 1e6  # roll
        flat[28] = 1e6  # pitch
        flat[35] = float(self.fusion.local.covariance[2, 2])
        return flat


def yaw_to_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


class LocalizationNode:  # pragma: no cover - cần ROS 2 runtime
    """Lớp rclpy mỏng. Không chạy được trong môi trường test hiện tại."""

    def __init__(self) -> None:
        import rclpy
        from geometry_msgs.msg import PoseStamped, TransformStamped
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from sensor_msgs.msg import NavSatFix
        from std_msgs.msg import String
        from tf2_ros import TransformBroadcaster

        self._Odometry = Odometry
        self._PoseStamped = PoseStamped
        self._TransformStamped = TransformStamped
        self._String = String

        self.node = Node("localization_fusion")
        self.bridge = FusionBridge()
        self.broadcaster = TransformBroadcaster(self.node)
        self.previous_odom_stamp: float | None = None

        self.node.declare_parameter("map_frame", "map")
        self.node.declare_parameter("odom_frame", "odom")
        self.node.declare_parameter("base_frame", "base_link")
        self.map_frame = self.node.get_parameter("map_frame").value
        self.odom_frame = self.node.get_parameter("odom_frame").value
        self.base_frame = self.node.get_parameter("base_frame").value

        self.odometry_publisher = self.node.create_publisher(
            Odometry, "/localization/odometry", 10
        )
        self.pose_publisher = self.node.create_publisher(
            PoseStamped, "/localization/pose", 10
        )
        self.status_publisher = self.node.create_publisher(
            String, "/localization/status", 10
        )
        self.node.create_subscription(NavSatFix, "/gps/fix", self.on_fix, 10)
        self.node.create_subscription(
            Odometry, "/odom/visual", self.on_visual_odometry, 10
        )
        self._rclpy = rclpy

    @staticmethod
    def _stamp_seconds(header) -> float:
        return header.stamp.sec + header.stamp.nanosec * 1e-9

    def on_fix(self, message) -> None:
        self.bridge.on_gps(
            self._stamp_seconds(message.header),
            message.latitude,
            message.longitude,
            message.altitude,
            message.status.status,
            hdop=None,
        )

    def on_visual_odometry(self, message) -> None:
        stamp = self._stamp_seconds(message.header)
        dt = 1.0 / 30.0 if self.previous_odom_stamp is None else stamp - self.previous_odom_stamp
        self.previous_odom_stamp = stamp
        if dt <= 0.0:
            return
        twist = message.twist.twist
        self.bridge.on_odometry(
            stamp,
            twist.linear.x * dt,
            twist.linear.y * dt,
            twist.angular.z * dt,
            dt,
        )
        self.publish(message.header.stamp)

    def publish(self, stamp) -> None:
        snapshot = self.bridge.snapshot()
        quaternion = yaw_to_quaternion(snapshot["map"]["yaw"])

        odometry = self._Odometry()
        odometry.header.stamp = stamp
        odometry.header.frame_id = self.map_frame
        odometry.child_frame_id = self.base_frame
        odometry.pose.pose.position.x = snapshot["map"]["x"]
        odometry.pose.pose.position.y = snapshot["map"]["y"]
        (
            odometry.pose.pose.orientation.x,
            odometry.pose.pose.orientation.y,
            odometry.pose.pose.orientation.z,
            odometry.pose.pose.orientation.w,
        ) = quaternion
        odometry.pose.covariance = self.bridge.odometry_covariance_row_major()
        self.odometry_publisher.publish(odometry)

        pose = self._PoseStamped()
        pose.header = odometry.header
        pose.pose = odometry.pose.pose
        self.pose_publisher.publish(pose)

        status = self._String()
        status.data = snapshot["gps_state"]
        self.status_publisher.publish(status)

        # map->odom mang toàn bộ hiệu chỉnh global; odom->base_link liên tục.
        for parent, child, values in (
            (self.map_frame, self.odom_frame, snapshot["map_to_odom"]),
            (self.odom_frame, self.base_frame, snapshot["odom"]),
        ):
            transform = self._TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = parent
            transform.child_frame_id = child
            transform.transform.translation.x = values["x"]
            transform.transform.translation.y = values["y"]
            (
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ) = yaw_to_quaternion(values["yaw"])
            self.broadcaster.sendTransform(transform)

    def spin(self) -> None:
        self._rclpy.spin(self.node)


def self_check() -> dict:
    """Chạy bridge bằng dữ liệu tổng hợp, không cần ROS 2."""
    bridge = FusionBridge()
    latitude, longitude, altitude = 49.0, 11.0, 400.0
    stamp = 1000.0
    bridge.on_gps(stamp, latitude, longitude, altitude, 2, satellites=12, hdop=0.8)
    for step in range(60):
        stamp += 1.0 / 30.0
        bridge.on_odometry(stamp, 0.2, 0.0, 0.0, 1.0 / 30.0)
    snapshot = bridge.snapshot()
    assert snapshot["datum"]["source"] == "first_valid_fix"
    assert snapshot["odom"]["x"] > 11.0, snapshot
    assert len(bridge.odometry_covariance_row_major()) == 36

    # Mất fix: state phải rời GOOD, pose local vẫn tiến.
    before = bridge.snapshot()["odom"]["x"]
    for _ in range(30):
        stamp += 1.0 / 30.0
        bridge.on_gps(stamp, float("nan"), float("nan"), float("nan"), -1)
        bridge.on_odometry(stamp, 0.2, 0.0, 0.0, 1.0 / 30.0)
    after = bridge.snapshot()
    assert after["odom"]["x"] > before
    return after


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Chạy bridge không cần ROS 2 và in snapshot.",
    )
    args = parser.parse_args()
    if args.self_check:
        import json

        print(json.dumps(self_check(), indent=2, ensure_ascii=False))
        return
    node = LocalizationNode()
    try:
        node.spin()
    finally:
        node.node.destroy_node()


if __name__ == "__main__":
    main()
