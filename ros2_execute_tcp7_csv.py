#!/usr/bin/env python3
"""Execute robot-frame TCP7 targets from offline_eval_zarr_episode.py in ROS2.

This script expects MoveIt to be running. It converts each TCP pose to a joint
solution through /compute_ik, then sends one FollowJointTrajectory goal to the
Gazebo joint_trajectory_controller.
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
import time

import numpy as np


def _read_tcp7_csv(path: pathlib.Path, *, max_points: int | None) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    times = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tcp7 = [float(row[f"target_tcp7_{i}"]) for i in range(7)]
            rows.append(tcp7)
            times.append(float(row["rel_time_s"]))
            if max_points is not None and len(rows) >= int(max_points):
                break
    if not rows:
        raise ValueError(f"no target rows in {path}")
    return np.asarray(rows, dtype=np.float64), np.asarray(times, dtype=np.float64)


def _rotvec_to_quat_xyzw(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = rotvec / theta
    half = theta * 0.5
    return np.concatenate([axis * math.sin(half), [math.cos(half)]])


class MoveItIkTrajectoryExecutor:
    def __init__(
        self,
        *,
        group_name: str,
        ik_link_name: str,
        frame_id: str,
        joint_names: list[str],
        ik_service: str,
        action_name: str,
        timeout_s: float,
        ik_timeout_s: float,
    ):
        import rclpy
        from builtin_interfaces.msg import Duration
        from control_msgs.action import FollowJointTrajectory
        from geometry_msgs.msg import PoseStamped
        from moveit_msgs.srv import GetPositionIK
        from rclpy.action import ActionClient
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectoryPoint

        self.rclpy = rclpy
        self.Duration = Duration
        self.FollowJointTrajectory = FollowJointTrajectory
        self.GetPositionIK = GetPositionIK
        self.JointState = JointState
        self.JointTrajectoryPoint = JointTrajectoryPoint
        self.PoseStamped = PoseStamped
        self.group_name = group_name
        self.ik_link_name = ik_link_name
        self.frame_id = frame_id
        self.joint_names = joint_names
        self.timeout_s = float(timeout_s)
        self.ik_timeout_s = float(ik_timeout_s)
        self.current_joint_state = None

        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init(args=None)
        self.node = rclpy.create_node("offline_tcp7_moveit_executor")
        self.ik_client = self.node.create_client(GetPositionIK, ik_service)
        self.action_client = ActionClient(
            self.node,
            FollowJointTrajectory,
            action_name,
        )
        self.node.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)

    def __enter__(self):
        if not self.ik_client.wait_for_service(timeout_sec=self.timeout_s):
            raise RuntimeError("MoveIt IK service is not available")
        if not self.action_client.wait_for_server(timeout_sec=self.timeout_s):
            raise RuntimeError("FollowJointTrajectory action server is not available")
        self.wait_for_joint_state(timeout_s=3.0, required=False)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.node.destroy_node()
        if self._owns_rclpy:
            self.rclpy.shutdown()

    def _joint_state_cb(self, msg):
        self.current_joint_state = msg

    def wait_for_joint_state(self, *, timeout_s: float, required: bool = True):
        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            if self.current_joint_state is not None:
                return self.current_joint_state
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        if required:
            raise RuntimeError("no /joint_states received")
        return None

    def _duration(self, seconds: float):
        seconds = max(0.0, float(seconds))
        sec = int(math.floor(seconds))
        nanosec = int(round((seconds - sec) * 1_000_000_000))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        return self.Duration(sec=sec, nanosec=nanosec)

    def compute_ik(self, tcp6: np.ndarray, seed_positions: np.ndarray | None) -> np.ndarray:
        req = self.GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.ik_link_name
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout = self._duration(self.ik_timeout_s)

        pose = self.PoseStamped()
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = float(tcp6[0])
        pose.pose.position.y = float(tcp6[1])
        pose.pose.position.z = float(tcp6[2])
        quat = _rotvec_to_quat_xyzw(tcp6[3:6])
        pose.pose.orientation.x = float(quat[0])
        pose.pose.orientation.y = float(quat[1])
        pose.pose.orientation.z = float(quat[2])
        pose.pose.orientation.w = float(quat[3])
        req.ik_request.pose_stamped = pose

        if seed_positions is not None:
            seed = self.JointState()
            seed.name = list(self.joint_names)
            seed.position = [float(x) for x in seed_positions]
            req.ik_request.robot_state.joint_state = seed
        elif self.current_joint_state is not None:
            req.ik_request.robot_state.joint_state = self.current_joint_state

        future = self.ik_client.call_async(req)
        self.rclpy.spin_until_future_complete(self.node, future, timeout_sec=self.timeout_s)
        if not future.done():
            raise TimeoutError("IK request timed out")
        resp = future.result()
        if int(resp.error_code.val) != 1:
            raise RuntimeError(f"IK failed with MoveIt error code {resp.error_code.val}")

        js = resp.solution.joint_state
        by_name = {name: pos for name, pos in zip(js.name, js.position)}
        missing = [name for name in self.joint_names if name not in by_name]
        if missing:
            raise RuntimeError(f"IK solution missing joints: {missing}")
        return np.asarray([by_name[name] for name in self.joint_names], dtype=np.float64)

    def send_trajectory(
        self,
        joint_points: np.ndarray,
        rel_times: np.ndarray,
        *,
        first_point_delay_s: float,
        speed_scale: float,
    ) -> None:
        rel_times = np.asarray(rel_times, dtype=np.float64)
        rel_times = (rel_times - rel_times[0]) * float(speed_scale) + float(first_point_delay_s)
        rel_times = np.maximum.accumulate(rel_times)
        for i in range(1, len(rel_times)):
            if rel_times[i] <= rel_times[i - 1]:
                rel_times[i] = rel_times[i - 1] + 0.02

        goal = self.FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.joint_names)
        for q, t in zip(joint_points, rel_times):
            point = self.JointTrajectoryPoint()
            point.positions = [float(v) for v in q]
            point.velocities = [0.0] * len(q)
            point.time_from_start = self._duration(float(t))
            goal.trajectory.points.append(point)

        future = self.action_client.send_goal_async(goal)
        self.rclpy.spin_until_future_complete(self.node, future, timeout_sec=self.timeout_s)
        if not future.done():
            raise TimeoutError("sending trajectory goal timed out")
        handle = future.result()
        if not handle.accepted:
            raise RuntimeError("trajectory goal rejected")

        result_future = handle.get_result_async()
        wait_s = float(rel_times[-1]) + self.timeout_s
        self.rclpy.spin_until_future_complete(self.node, result_future, timeout_sec=wait_s)
        if not result_future.done():
            raise TimeoutError("trajectory execution timed out")
        result = result_future.result().result
        if int(result.error_code) != 0:
            msg = result.error_string or f"error_code={result.error_code}"
            raise RuntimeError(f"trajectory execution failed: {msg}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replay offline TCP7 targets in Indy Gazebo.")
    p.add_argument("csv", type=pathlib.Path, help="sim_tcp7_targets_epXXX.csv")
    p.add_argument("--group_name", default="indy_manipulator")
    p.add_argument("--ik_link_name", default="tcp")
    p.add_argument("--frame_id", default="link0")
    p.add_argument("--joint_names", default="joint0,joint1,joint2,joint3,joint4,joint5,joint6")
    p.add_argument("--ik_service", default="/compute_ik")
    p.add_argument("--action_name", default="/joint_trajectory_controller/follow_joint_trajectory")
    p.add_argument("--timeout_s", type=float, default=20.0)
    p.add_argument("--ik_timeout_s", type=float, default=0.2)
    p.add_argument("--first_point_delay_s", type=float, default=1.0)
    p.add_argument("--speed_scale", type=float, default=1.0, help=">1 slows execution.")
    p.add_argument("--max_points", type=int, default=None)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--joint_csv", type=pathlib.Path, default=None)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    tcp7, rel_times = _read_tcp7_csv(args.csv, max_points=args.max_points)
    joint_names = [x.strip() for x in args.joint_names.split(",") if x.strip()]
    print(f"loaded {len(tcp7)} TCP targets from {args.csv}")
    print(f"time span: {rel_times[0]:.3f}s -> {rel_times[-1]:.3f}s")

    with MoveItIkTrajectoryExecutor(
        group_name=args.group_name,
        ik_link_name=args.ik_link_name,
        frame_id=args.frame_id,
        joint_names=joint_names,
        ik_service=args.ik_service,
        action_name=args.action_name,
        timeout_s=args.timeout_s,
        ik_timeout_s=args.ik_timeout_s,
    ) as executor:
        joint_points = []
        seed = None
        for i, row in enumerate(tcp7):
            try:
                seed = executor.compute_ik(row[:6], seed)
            except Exception as exc:
                raise RuntimeError(
                    f"IK failed at row {i}, tcp6={np.round(row[:6], 5).tolist()}: {exc}"
                ) from exc
            joint_points.append(seed.copy())
            if i % 25 == 0:
                print(f"IK {i + 1}/{len(tcp7)}")
        joint_points_np = np.asarray(joint_points, dtype=np.float64)

        if args.joint_csv is not None:
            args.joint_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(args.joint_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["rel_time_s"] + joint_names)
                for t, q in zip(rel_times, joint_points_np):
                    writer.writerow([float(t)] + [float(v) for v in q])
            print(f"wrote joint csv: {args.joint_csv}")

        if args.dry_run:
            print("dry-run: IK solved, trajectory not sent.")
            return

        print("sending trajectory...")
        executor.send_trajectory(
            joint_points_np,
            rel_times,
            first_point_delay_s=args.first_point_delay_s,
            speed_scale=args.speed_scale,
        )
        print("done.")


if __name__ == "__main__":
    main()
