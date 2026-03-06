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
import subprocess
import sys
import time
from pathlib import Path
from multiprocessing.managers import BaseManager as MPBaseManager
import numpy as np
from scipy.spatial.transform import Rotation as R
from constants import ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY
from constants import ARM_BACKEND
from constants import ER3PRO_IP, ER3PRO_LOCAL_IP, ER3PRO_MOVE_VELOCITY, ER3PRO_MOVE_ZONE
from constants import ER3PRO_ENABLE_GRIPPER
from constants import ER3PRO_GRIPPER_THRESHOLD, ER3PRO_GRIPPER_BOARD, ER3PRO_GRIPPER_DI1_PORT, ER3PRO_GRIPPER_DI2_PORT
from constants import ER3PRO_CPP_BRIDGE_BIN
from constants import ER3PRO_MAX_POS_SPEED, ER3PRO_MAX_ROT_SPEED
from constants import ER3PRO_MAX_POS_ACCEL, ER3PRO_MAX_ROT_ACCEL, ER3PRO_CMD_TIMEOUT


class ER3ProCppBridgeArm:
    def __init__(self):
        self.gripper_pos = np.array([1.0])

        bridge_path = Path(__file__).resolve().parent / ER3PRO_CPP_BRIDGE_BIN
        if not bridge_path.exists():
            raise RuntimeError(f'ER3Pro C++ bridge not found: {bridge_path}')

        cmd = [
            str(bridge_path),
            '--robot-ip', ER3PRO_IP,
            '--speed', str(ER3PRO_MOVE_VELOCITY),
            '--zone', str(ER3PRO_MOVE_ZONE),
            '--max-pos-speed', str(ER3PRO_MAX_POS_SPEED),
            '--max-rot-speed', str(ER3PRO_MAX_ROT_SPEED),
            '--max-pos-accel', str(ER3PRO_MAX_POS_ACCEL),
            '--max-rot-accel', str(ER3PRO_MAX_ROT_ACCEL),
            '--cmd-timeout', str(ER3PRO_CMD_TIMEOUT),
            '--gripper-threshold', str(ER3PRO_GRIPPER_THRESHOLD),
            '--gripper-board', str(ER3PRO_GRIPPER_BOARD),
            '--gripper-di1-port', str(ER3PRO_GRIPPER_DI1_PORT),
            '--gripper-di2-port', str(ER3PRO_GRIPPER_DI2_PORT),
        ]
        if ER3PRO_LOCAL_IP:
            cmd.extend(['--local-ip', ER3PRO_LOCAL_IP])
        if not ER3PRO_ENABLE_GRIPPER:
            cmd.append('--disable-gripper')

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        ready = self._read_line(timeout=10.0)
        if ready != 'READY':
            err = self.proc.stderr.readline().strip() if self.proc.stderr else ''
            raise RuntimeError(f'Failed to start ER3Pro C++ bridge: {ready} {err}')

    def _read_line(self, timeout=3.0):
        start = time.time()
        while time.time() - start < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError('ER3Pro C++ bridge exited unexpectedly')
            if self.proc.stdout is None:
                break
            line = self.proc.stdout.readline()
            if line:
                return line.strip()
        raise RuntimeError('Timeout waiting for ER3Pro C++ bridge response')

    def _request(self, msg, timeout=3.0):
        if self.proc.stdin is None:
            raise RuntimeError('ER3Pro C++ bridge stdin not available')
        self.proc.stdin.write(msg + '\n')
        self.proc.stdin.flush()
        rep = self._read_line(timeout=timeout)
        if rep.startswith('ERR '):
            raise RuntimeError(f'ER3Pro C++ bridge error: {rep[4:]}')
        return rep

    def reset(self):
        self._request('RESET', timeout=8.0)
        self.gripper_pos[:] = 1.0

    def execute_action(self, action):
        arm_pos = np.asarray(action['arm_pos'], dtype=np.float64)
        arm_quat = np.asarray(action['arm_quat'], dtype=np.float64)
        rpy = R.from_quat(arm_quat).as_euler('xyz')

        gripper_value = float(np.asarray(action['gripper_pos']).item())
        self.gripper_pos[:] = gripper_value
        self._request(
            f'EXEC {arm_pos[0]} {arm_pos[1]} {arm_pos[2]} {rpy[0]} {rpy[1]} {rpy[2]} {gripper_value}',
            timeout=8.0,
        )

    def get_state(self):
        rep = self._request('STATE', timeout=3.0)
        items = rep.split()
        if len(items) != 8 or items[0] != 'STATE':
            raise RuntimeError(f'Unexpected ER3Pro C++ bridge STATE response: {rep}')

        posture = np.array([float(v) for v in items[1:7]], dtype=np.float64)
        arm_pos = posture[:3]
        arm_quat = R.from_euler('xyz', posture[3:]).as_quat()
        self.gripper_pos[:] = float(items[7])
        if arm_quat[3] < 0.0:
            np.negative(arm_quat, out=arm_quat)
        return {
            'arm_pos': arm_pos,
            'arm_quat': arm_quat,
            'gripper_pos': self.gripper_pos.copy(),
        }

    def close(self):
        try:
            self._request('CLOSE', timeout=2.0)
        except Exception:
            pass
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()


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
            self.impl = ER3ProCppBridgeArm()
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
