from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from umi.common.pose_util import mat_to_pose, pose_to_mat
from umi.real_world.indy_interpolation_controller import IndyInterpolationController


def make_controller() -> IndyInterpolationController:
    """Construct only the pure coordinate-conversion part (no robot process)."""
    ctrl = IndyInterpolationController.__new__(IndyInterpolationController)
    ctrl.task_frame_xyz_signs = np.ones(3, dtype=np.float64)
    ctrl.tool_rot_offset_deg = np.zeros(3, dtype=np.float64)
    ctrl._has_tool_rot_offset = False
    ctrl._tool_rot_offset = Rotation.identity()
    ctrl.task_rot_is_euler = True
    ctrl.task_rot_euler_in_degrees = True
    ctrl._task_euler_scipy_seq = "xyz"
    ctrl.vel_ratio = 0.1
    ctrl.acc_ratio = 0.5
    ctrl.flange_to_tcp_pose = np.array([0, 0, 0.235, 0, 0, 0], dtype=np.float64)
    ctrl._tx_flange_tcp = pose_to_mat(ctrl.flange_to_tcp_pose)
    ctrl._tx_tcp_flange = np.linalg.inv(ctrl._tx_flange_tcp)
    return ctrl


class IndyTcpTransformTest(unittest.TestCase):
    def test_feedback_and_command_are_inverse(self):
        ctrl = make_controller()
        flange_xyz_m = np.array([0.4, -0.1, 0.5])
        flange_uvw_deg = np.array([20.0, -30.0, 40.0])
        flange_matrix = np.eye(4)
        flange_matrix[:3, :3] = Rotation.from_euler(
            "xyz", flange_uvw_deg, degrees=True
        ).as_matrix()
        flange_matrix[:3, 3] = flange_xyz_m
        expected_tcp_matrix = flange_matrix @ ctrl._tx_flange_tcp

        dcp_pose = np.concatenate([flange_xyz_m * 1000.0, flange_uvw_deg])
        observed_tcp_pose = ctrl._extract_pose_m_rad({"p": dcp_pose})
        np.testing.assert_allclose(
            pose_to_mat(observed_tcp_pose), expected_tcp_matrix, atol=1e-10
        )

        sent = {}

        def capture_command(**kwargs):
            sent.update(kwargs)

        ctrl._send_task_pose(
            capture_command,
            "movetelel_abs",
            mat_to_pose(expected_tcp_matrix),
            prefer_mm_deg=True,
        )
        sent_flange = np.asarray(sent["tpos"], dtype=np.float64)
        recovered_flange = np.eye(4)
        recovered_flange[:3, :3] = Rotation.from_euler(
            "xyz", sent_flange[3:], degrees=True
        ).as_matrix()
        recovered_flange[:3, 3] = sent_flange[:3] / 1000.0
        np.testing.assert_allclose(recovered_flange, flange_matrix, atol=1e-10)

    def test_relative_api_is_never_selected(self):
        ctrl = make_controller()
        abs_only = type("AbsOnly", (), {"movetelel_abs": lambda self: None})()
        rel_only = type("RelOnly", (), {"movetelel_rel": lambda self: None})()
        self.assertEqual(ctrl._find_task_cmd(abs_only)[1], "movetelel_abs")
        self.assertIsNone(ctrl._find_task_cmd(rel_only)[0])


if __name__ == "__main__":
    unittest.main()
