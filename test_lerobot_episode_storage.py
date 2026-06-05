import sys
import types
import unittest

import numpy as np


cv2_stub = types.SimpleNamespace(
    COLOR_RGB2BGR=1,
    VideoWriter_fourcc=lambda *args: 0,
)
sys.modules.setdefault('cv2', cv2_stub)

from episode_storage import build_lerobot_frame


class TestLeRobotFrameBuilder(unittest.TestCase):
    def test_builds_uarm_compatible_state_and_action_vectors(self):
        obs = {
            'base_image': np.zeros((2, 3, 3), dtype=np.uint8),
            'wrist_image': np.ones((2, 3, 3), dtype=np.uint8),
            'arm_joints': np.arange(7, dtype=np.float64),
            'arm_pos': np.array([0.1, 0.2, 0.3], dtype=np.float64),
            'arm_quat': np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            'gripper_pos': np.array([0.5], dtype=np.float64),
        }
        action = {
            'arm_joints': np.arange(10, 17, dtype=np.float64),
            'gripper_pos': np.array([0.75], dtype=np.float64),
        }

        frame = build_lerobot_frame(obs, action, task='gamepad_task')

        self.assertEqual(frame['task'], 'gamepad_task')
        self.assertEqual(frame['observation.state'].shape, (15,))
        self.assertEqual(frame['action'].shape, (8,))
        np.testing.assert_allclose(frame['observation.state'][:7], obs['arm_joints'])
        np.testing.assert_allclose(frame['observation.state'][7:10], obs['arm_pos'])
        np.testing.assert_allclose(frame['observation.state'][10:14], obs['arm_quat'])
        np.testing.assert_allclose(frame['observation.state'][14], obs['gripper_pos'][0])
        np.testing.assert_allclose(frame['action'][:7], action['arm_joints'])
        np.testing.assert_allclose(frame['action'][7], action['gripper_pos'][0])
        np.testing.assert_array_equal(frame['observation.images.base'], obs['base_image'])
        np.testing.assert_array_equal(frame['observation.images.wrist'], obs['wrist_image'])

    def test_falls_back_to_observed_joints_for_cartesian_action(self):
        obs = {
            'base_image': np.zeros((2, 3, 3), dtype=np.uint8),
            'wrist_image': np.ones((2, 3, 3), dtype=np.uint8),
            'arm_joints': np.arange(7, dtype=np.float64),
            'arm_pos': np.array([0.1, 0.2, 0.3], dtype=np.float64),
            'arm_quat': np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            'gripper_pos': np.array([0.5], dtype=np.float64),
        }
        action = {
            'arm_pos': np.array([0.2, 0.3, 0.4], dtype=np.float64),
            'arm_quat': np.array([0.0, 0.0, 0.1, 0.99], dtype=np.float64),
            'gripper_pos': np.array([0.25], dtype=np.float64),
        }

        frame = build_lerobot_frame(obs, action, task='gamepad_task')

        np.testing.assert_allclose(frame['action'][:7], obs['arm_joints'])
        np.testing.assert_allclose(frame['action'][7], action['gripper_pos'][0])


if __name__ == '__main__':
    unittest.main()
