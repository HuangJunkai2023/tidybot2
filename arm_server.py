# Author: Jimmy Wu
# Date: October 2024
#
# This RPC server allows other processes to communicate with the Kinova arm
# low-level controller, which runs in its own, dedicated real-time process.
#
# Note: Operations that are not time-sensitive should be run in a separate,
# non-real-time process to avoid interfering with the low-level control and
# causing latency spikes.

import queue
import platform
import sys
import time
from pathlib import Path
from multiprocessing.managers import BaseManager as MPBaseManager
import numpy as np
from scipy.spatial.transform import Rotation as R
from constants import ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY
from constants import ARM_BACKEND
from constants import ER3PRO_IP, ER3PRO_LOCAL_IP, ER3PRO_MOVE_VELOCITY, ER3PRO_MOVE_ZONE
from constants import ER3PRO_GRIPPER_THRESHOLD, ER3PRO_GRIPPER_BOARD, ER3PRO_GRIPPER_DI1_PORT, ER3PRO_GRIPPER_DI2_PORT


def import_xcore_sdk():
    if platform.system() == 'Windows':
        module_name = 'Release.windows.xCoreSDK_python'
    elif platform.system() == 'Linux':
        module_name = 'Release.linux.xCoreSDK_python'
    else:
        raise RuntimeError(f'Unsupported OS: {platform.system()}')

    sdk_root = Path(__file__).resolve().parents[1] / 'xcoresdk_python-v0.6.0'
    example_dir = sdk_root / 'example'
    release_dir = sdk_root / 'Release'
    for path in [
        str(example_dir),
        str(sdk_root),
        str(release_dir),
        str(release_dir / 'linux'),
        str(release_dir / 'windows'),
        str(release_dir / 'linux' / 'xCoreSDK_python'),
        str(release_dir / 'windows' / 'xCoreSDK_python'),
    ]:
        if path not in sys.path:
            sys.path.insert(0, path)

    import importlib
    return importlib.import_module(module_name)


class ER3ProArm:
    def __init__(self):
        self.xcore = import_xcore_sdk()
        self.ec = {}
        self.robot = self._create_robot()
        self.toolset = self.xcore.Toolset()
        self.gripper = None
        self.gripper_pos = np.array([1.0])
        self._setup_robot()
        self._setup_gripper()

    def _create_robot(self):
        if hasattr(self.xcore, 'xMateErProRobot'):
            robot = self.xcore.xMateErProRobot()
            robot.connectToRobot(ER3PRO_IP)
            return robot
        if ER3PRO_LOCAL_IP is None:
            return self.xcore.xMateRobot(ER3PRO_IP)
        return self.xcore.xMateRobot(ER3PRO_IP, ER3PRO_LOCAL_IP)

    def _setup_robot(self):
        self.robot.setOperateMode(self.xcore.OperateMode.automatic, self.ec)
        self.robot.setPowerState(True, self.ec)
        self.robot.setMotionControlMode(self.xcore.MotionControlMode.NrtCommandMode, self.ec)
        self.robot.clearServoAlarm(self.ec)

    def _setup_gripper(self):
        try:
            from jodell.er3_io_ctrl import ER3ProGripperIOController
            self.gripper = ER3ProGripperIOController(
                robot=self.robot,
                board=ER3PRO_GRIPPER_BOARD,
                di1_do_port=ER3PRO_GRIPPER_DI1_PORT,
                di2_do_port=ER3PRO_GRIPPER_DI2_PORT,
            )
            self.gripper.enable_or_preset1()
        except Exception:
            self.gripper = None

    def reset(self):
        self.robot.moveReset(self.ec)
        self.robot.clearServoAlarm(self.ec)
        if self.gripper is not None:
            self.gripper.enable_or_preset1()
        self.gripper_pos[:] = 1.0

    def execute_action(self, action):
        arm_pos = np.asarray(action['arm_pos'], dtype=np.float64)
        arm_quat = np.asarray(action['arm_quat'], dtype=np.float64)
        rpy = R.from_quat(arm_quat).as_euler('xyz')
        cart_pos = self.xcore.CartesianPosition([
            float(arm_pos[0]), float(arm_pos[1]), float(arm_pos[2]),
            float(rpy[0]), float(rpy[1]), float(rpy[2]),
        ])

        self.robot.moveReset(self.ec)
        cmd = None
        try:
            joint_pos = self.robot.model().calcIk(cart_pos, self.toolset, self.ec)
            if joint_pos is not None and len(joint_pos) >= 6:
                cmd = self.xcore.MoveAbsJCommand(self.xcore.JointPosition(list(joint_pos)[:6]), ER3PRO_MOVE_VELOCITY, ER3PRO_MOVE_ZONE)
        except Exception:
            cmd = None

        if cmd is None:
            cmd = self.xcore.MoveJCommand(cart_pos, ER3PRO_MOVE_VELOCITY, ER3PRO_MOVE_ZONE)

        cmd_id = self.xcore.PyString()
        self.robot.moveAppend([cmd], cmd_id, self.ec)
        self.robot.moveStart(self.ec)

        gripper_value = float(np.asarray(action['gripper_pos']).item())
        self.gripper_pos[:] = gripper_value
        if self.gripper is not None:
            if gripper_value >= ER3PRO_GRIPPER_THRESHOLD:
                self.gripper.enable_or_preset1()
            else:
                self.gripper.preset2()

    def get_state(self):
        posture = np.asarray(self.robot.posture(self.xcore.CoordinateType.endInRef, self.ec), dtype=np.float64)
        arm_pos = posture[:3]
        arm_quat = R.from_euler('xyz', posture[3:6]).as_quat()
        if arm_quat[3] < 0.0:
            np.negative(arm_quat, out=arm_quat)
        return {
            'arm_pos': arm_pos,
            'arm_quat': arm_quat,
            'gripper_pos': self.gripper_pos.copy(),
        }

    def close(self):
        try:
            self.robot.moveReset(self.ec)
        except Exception:
            pass
        try:
            self.robot.disconnectFromRobot(self.ec)
        except Exception:
            pass


class KinovaArm:
    def __init__(self):
        from arm_controller import JointCompliantController
        from ik_solver import IKSolver
        from kinova import TorqueControlledArm

        self.arm = TorqueControlledArm()
        self.arm.set_joint_limits(speed_limits=(7 * (30,)), acceleration_limits=(7 * (80,)))
        self.command_queue = queue.Queue(1)
        self.controller = None
        self.ik_solver = IKSolver(ee_offset=0.12)
        self.controller_cls = JointCompliantController

    def reset(self):
        if self.arm.cyclic_running:
            time.sleep(0.75)
            self.arm.stop_cyclic()
        self.arm.clear_faults()
        self.arm.open_gripper()
        self.arm.retract()
        self.controller = self.controller_cls(self.command_queue)
        self.arm.init_cyclic(self.controller.control_callback)
        while not self.arm.cyclic_running:
            time.sleep(0.01)

    def execute_action(self, action):
        qpos = self.ik_solver.solve(action['arm_pos'], action['arm_quat'], self.arm.q)
        self.command_queue.put((qpos, action['gripper_pos'].item()))

    def get_state(self):
        arm_pos, arm_quat = self.arm.get_tool_pose()
        if arm_quat[3] < 0.0:
            np.negative(arm_quat, out=arm_quat)
        return {
            'arm_pos': arm_pos,
            'arm_quat': arm_quat,
            'gripper_pos': np.array([self.arm.gripper_pos]),
        }

    def close(self):
        if self.arm.cyclic_running:
            time.sleep(0.75)
            self.arm.stop_cyclic()
        self.arm.disconnect()

class Arm:
    def __init__(self):
        if ARM_BACKEND == 'er3pro':
            self.impl = ER3ProArm()
        elif ARM_BACKEND == 'kinova':
            self.impl = KinovaArm()
        else:
            raise ValueError(f'Unsupported ARM_BACKEND: {ARM_BACKEND}')

    def reset(self):
        self.impl.reset()

    def execute_action(self, action):
        self.impl.execute_action(action)

    def get_state(self):
        return self.impl.get_state()

    def close(self):
        self.impl.close()

class ArmManager(MPBaseManager):
    pass

ArmManager.register('Arm', Arm)

if __name__ == '__main__':
    manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    server = manager.get_server()
    print(f'Arm manager server started at {ARM_RPC_HOST}:{ARM_RPC_PORT}')
    server.serve_forever()
    # import numpy as np
    # from constants import POLICY_CONTROL_PERIOD
    # manager = ArmManager(address=(ARM_RPC_HOST, ARM_RPC_PORT), authkey=RPC_AUTHKEY)
    # manager.connect()
    # arm = manager.Arm()
    # try:
    #     arm.reset()
    #     for i in range(50):
    #         arm.execute_action({
    #             'arm_pos': np.array([0.135, 0.002, 0.211]),
    #             'arm_quat': np.array([0.706, 0.707, 0.029, 0.029]),
    #             'gripper_pos': np.zeros(1),
    #         })
    #         print(arm.get_state())
    #         time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    # finally:
    #     arm.close()
