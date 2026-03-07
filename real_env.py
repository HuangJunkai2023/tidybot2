# Author: Jimmy Wu
# Date: October 2024

import numpy as np
from cameras import KinovaCamera, LogitechCamera, UVCCamera, DummyCamera
from constants import BASE_RPC_HOST, BASE_RPC_PORT, ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY
from constants import BASE_CAMERA_SERIAL, WRIST_CAMERA_DEVICE, USE_KINOVA_WRIST_CAMERA
from constants import ENABLE_BASE, ENABLE_ARM
from arm_server import ArmManager
from base_server import BaseManager

class RealEnv:
    def __init__(self):
        self.base = None
        self.arm = None

        if ENABLE_BASE:
            base_manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY)
            try:
                base_manager.connect()
            except ConnectionRefusedError as e:
                raise Exception('Could not connect to base RPC server, is base_server.py running?') from e
            self.base = base_manager.Base(max_vel=(0.5, 0.5, 1.57), max_accel=(0.5, 0.5, 1.57))

        if ENABLE_ARM:
            arm_manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
            try:
                arm_manager.connect()
            except ConnectionRefusedError as e:
                raise Exception('Could not connect to arm RPC server, is arm_server.py running?') from e
            self.arm = arm_manager.Arm()

        if not ENABLE_BASE and not ENABLE_ARM:
            raise Exception('At least one subsystem must be enabled')

        # Cameras
        self.base_camera = self._create_base_camera(BASE_CAMERA_SERIAL)
        self.wrist_camera = KinovaCamera() if USE_KINOVA_WRIST_CAMERA else UVCCamera(WRIST_CAMERA_DEVICE, frame_width=640, frame_height=480)

    def _create_base_camera(self, camera_hint):
        hint = str(camera_hint).strip()
        if hint == 'TODO':
            return DummyCamera(frame_width=640, frame_height=360)

        # If hint looks like a device index/path, use generic UVC camera.
        if hint.isdigit() or hint.startswith('/dev/'):
            return UVCCamera(hint, frame_width=640, frame_height=360)

        # Otherwise treat it as Logitech C930e serial suffix.
        return LogitechCamera(hint)

    def get_obs(self):
        obs = {}
        if self.base is not None:
            obs.update(self.base.get_state())
        else:
            obs['base_pose'] = np.zeros(3)

        if self.arm is not None:
            obs.update(self.arm.get_state())
        else:
            obs['arm_pos'] = np.zeros(3)
            obs['arm_quat'] = np.array([0.0, 0.0, 0.0, 1.0])
            obs['gripper_pos'] = np.array([1.0])

        obs['base_image'] = self.base_camera.get_image()
        obs['wrist_image'] = self.wrist_camera.get_image()
        return obs

    def reset(self):
        if self.base is not None:
            print('Resetting base...')
            self.base.reset()

        if self.arm is not None:
            print('Resetting arm...')
            self.arm.reset()

        print('Robot has been reset')

    def step(self, action):
        # Note: We intentionally do not return obs here to prevent the policy from using outdated data
        if self.base is not None:
            self.base.execute_action(action)  # Non-blocking
        if self.arm is not None:
            self.arm.execute_action(action)   # Non-blocking

    def close(self):
        if self.base is not None:
            self.base.close()
        if self.arm is not None:
            self.arm.close()
        self.base_camera.close()
        self.wrist_camera.close()

if __name__ == '__main__':
    import time
    import numpy as np
    from constants import POLICY_CONTROL_PERIOD
    env = RealEnv()
    try:
        while True:
            env.reset()
            for _ in range(100):
                action = {
                    'base_pose': 0.1 * np.random.rand(3) - 0.05,
                    'arm_pos': 0.1 * np.random.rand(3) + np.array([0.55, 0.0, 0.4]),
                    'arm_quat': np.random.rand(4),
                    'gripper_pos': np.random.rand(1),
                }
                env.step(action)
                obs = env.get_obs()
                print([(k, v.shape) if v.ndim == 3 else (k, v) for (k, v) in obs.items()])
                time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    finally:
        env.close()
