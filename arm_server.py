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
import threading
import time
import importlib.util
from pathlib import Path
from multiprocessing.managers import BaseManager as MPBaseManager
import numpy as np
from scipy.spatial.transform import Rotation as R
from constants import ARM_RPC_HOST, ARM_RPC_PORT, RPC_AUTHKEY
from constants import ARM_BACKEND
from constants import ER3PRO_IP, ER3PRO_LOCAL_IP, ER3PRO_MOVE_VELOCITY, ER3PRO_MOVE_ZONE
from constants import ER3PRO_ENABLE_GRIPPER
from constants import ER3PRO_GRIPPER_BACKEND
from constants import ER3PRO_GRIPPER_THRESHOLD, ER3PRO_GRIPPER_BOARD, ER3PRO_GRIPPER_DI1_PORT, ER3PRO_GRIPPER_DI2_PORT
from constants import ER3PRO_GRIPPER_RS485_SLAVE_ID, ER3PRO_GRIPPER_RS485_ENABLE_ON_START
from constants import ER3PRO_GRIPPER_RS485_INIT_REG, ER3PRO_GRIPPER_RS485_INIT_VALUE
from constants import ER3PRO_GRIPPER_RS485_TORQUE_REG, ER3PRO_GRIPPER_RS485_POS_REG, ER3PRO_GRIPPER_RS485_SPEED_REG
from constants import ER3PRO_GRIPPER_RS485_POS_NOW_REG
from constants import ER3PRO_GRIPPER_RS485_OPEN_POS, ER3PRO_GRIPPER_RS485_CLOSE_POS
from constants import ER3PRO_GRIPPER_RS485_SPEED, ER3PRO_GRIPPER_RS485_TORQUE
from constants import ER3PRO_GRIPPER_USB_PORT, ER3PRO_GRIPPER_USB_BAUD, ER3PRO_GRIPPER_USB_SLAVE_ID
from constants import ER3PRO_GRIPPER_USB_SPEED, ER3PRO_GRIPPER_USB_TORQUE
from constants import ER3PRO_GRIPPER_USB_OPEN_POS, ER3PRO_GRIPPER_USB_CLOSE_POS
from constants import ER3PRO_GRIPPER_USB_MIN_CMD_INTERVAL
from constants import ER3PRO_CPP_BRIDGE_BIN
from constants import ER3PRO_FOLLOW_SCALE, ER3PRO_RT_FILTER_FREQ
from constants import ER3PRO_MAX_POS_SPEED, ER3PRO_MAX_ROT_SPEED
from constants import ER3PRO_MAX_POS_ACCEL, ER3PRO_MAX_ROT_ACCEL, ER3PRO_CMD_TIMEOUT
from constants import ER3PRO_TCP_OFFSET_Z
from constants import ER3PRO_TELEOP_PRESET_JOINT_DEG


def _check_jodell_runtime_deps():
    missing = []
    for module_name in ('serial', 'modbus_tk'):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    if missing:
        raise ModuleNotFoundError(
            'Missing Jodell runtime deps: ' + ', '.join(missing)
            + '. Install with: pip install pyserial modbus-tk'
        )


def _import_epg_control():
    _check_jodell_runtime_deps()
    try:
        from jodellSdk.jodellSdkDemo import EpgClawControl  # type: ignore
        return EpgClawControl
    except Exception:
        wheel_path = Path(__file__).resolve().parent.parent / 'jodell/python/JodellTool-0.1.5-py3-none-any.whl'
        if wheel_path.exists():
            sys.path.insert(0, str(wheel_path))
            from jodellSdk.jodellSdkDemo import EpgClawControl  # type: ignore
            return EpgClawControl
        raise


class JodellUsbGripper:
    def __init__(self):
        EpgClawControl = _import_epg_control()
        self.claw = EpgClawControl()
        self.connected = False
        self.last_cmd_time = 0.0
        self.last_cmd_pos = None

        ret = self.claw.serialOperation(ER3PRO_GRIPPER_USB_PORT, ER3PRO_GRIPPER_USB_BAUD, True)
        if ret != 1:
            raise RuntimeError(f'Jodell serialOperation connect failed: {ret}')
        self.connected = True

        ret = self.claw.enableClamp(ER3PRO_GRIPPER_USB_SLAVE_ID, True)
        if ret != 1:
            raise RuntimeError(f'Jodell enableClamp failed: {ret}')

    def _norm_to_pos(self, value):
        value = float(np.clip(value, 0.0, 1.0))
        # Keep tidybot convention: 1=open, 0=close.
        return int(round(ER3PRO_GRIPPER_USB_CLOSE_POS + value * (ER3PRO_GRIPPER_USB_OPEN_POS - ER3PRO_GRIPPER_USB_CLOSE_POS)))

    def command(self, value):
        target_pos = self._norm_to_pos(value)
        now = time.monotonic()

        # Avoid saturating the serial bus with tiny updates.
        if self.last_cmd_pos is not None and abs(target_pos - self.last_cmd_pos) < 2:
            return
        if now - self.last_cmd_time < ER3PRO_GRIPPER_USB_MIN_CMD_INTERVAL:
            return

        ret = self.claw.runWithParam(ER3PRO_GRIPPER_USB_SLAVE_ID, target_pos, ER3PRO_GRIPPER_USB_SPEED, ER3PRO_GRIPPER_USB_TORQUE)
        if ret != 1:
            print(f'[jodell_usb] runWithParam failed: {ret}', file=sys.stderr, flush=True)
            return

        self.last_cmd_time = now
        self.last_cmd_pos = target_pos

    def close(self):
        if self.connected:
            ret = self.claw.serialOperation(ER3PRO_GRIPPER_USB_PORT, ER3PRO_GRIPPER_USB_BAUD, False)
            if ret != 1:
                print(f'[jodell_usb] serial close failed: {ret}', file=sys.stderr, flush=True)
            self.connected = False


class ER3ProCppBridgeArm:
    def __init__(self):
        self.gripper_pos = np.array([1.0])
        self.usb_gripper = None

        bridge_path = Path(__file__).resolve().parent / ER3PRO_CPP_BRIDGE_BIN
        if not bridge_path.exists():
            raise RuntimeError(f'ER3Pro C++ bridge not found: {bridge_path}')

        bridge_gripper_backend = ER3PRO_GRIPPER_BACKEND if ER3PRO_GRIPPER_BACKEND in ('di', 'rs485_epg') else 'di'

        cmd = [
            str(bridge_path),
            '--robot-ip', ER3PRO_IP,
            '--speed', str(ER3PRO_MOVE_VELOCITY),
            '--zone', str(ER3PRO_MOVE_ZONE),
            '--follow-scale', str(ER3PRO_FOLLOW_SCALE),
            '--filter-freq', str(ER3PRO_RT_FILTER_FREQ),
            '--max-pos-speed', str(ER3PRO_MAX_POS_SPEED),
            '--max-rot-speed', str(ER3PRO_MAX_ROT_SPEED),
            '--max-pos-accel', str(ER3PRO_MAX_POS_ACCEL),
            '--max-rot-accel', str(ER3PRO_MAX_ROT_ACCEL),
            '--cmd-timeout', str(ER3PRO_CMD_TIMEOUT),
            '--tcp-offset-z', str(ER3PRO_TCP_OFFSET_Z),
            '--preset-joints-deg', ','.join(str(float(v)) for v in ER3PRO_TELEOP_PRESET_JOINT_DEG),
            '--gripper-threshold', str(ER3PRO_GRIPPER_THRESHOLD),
            '--gripper-backend', bridge_gripper_backend,
            '--gripper-board', str(ER3PRO_GRIPPER_BOARD),
            '--gripper-di1-port', str(ER3PRO_GRIPPER_DI1_PORT),
            '--gripper-di2-port', str(ER3PRO_GRIPPER_DI2_PORT),
            '--gripper-rs485-slave-id', str(ER3PRO_GRIPPER_RS485_SLAVE_ID),
            '--gripper-rs485-init-reg', str(ER3PRO_GRIPPER_RS485_INIT_REG),
            '--gripper-rs485-init-value', str(ER3PRO_GRIPPER_RS485_INIT_VALUE),
            '--gripper-rs485-torque-reg', str(ER3PRO_GRIPPER_RS485_TORQUE_REG),
            '--gripper-rs485-pos-reg', str(ER3PRO_GRIPPER_RS485_POS_REG),
            '--gripper-rs485-speed-reg', str(ER3PRO_GRIPPER_RS485_SPEED_REG),
            '--gripper-rs485-pos-now-reg', str(ER3PRO_GRIPPER_RS485_POS_NOW_REG),
            '--gripper-rs485-open-pos', str(ER3PRO_GRIPPER_RS485_OPEN_POS),
            '--gripper-rs485-close-pos', str(ER3PRO_GRIPPER_RS485_CLOSE_POS),
            '--gripper-rs485-speed', str(ER3PRO_GRIPPER_RS485_SPEED),
            '--gripper-rs485-torque', str(ER3PRO_GRIPPER_RS485_TORQUE),
        ]
        if ER3PRO_GRIPPER_RS485_ENABLE_ON_START:
            cmd.append('--gripper-rs485-enable-on-start')
        if ER3PRO_LOCAL_IP:
            cmd.extend(['--local-ip', ER3PRO_LOCAL_IP])
        # If using Python-side Jodell control, disable bridge-side gripper to avoid conflicts.
        if (not ER3PRO_ENABLE_GRIPPER) or ER3PRO_GRIPPER_BACKEND == 'jodell_usb':
            cmd.append('--disable-gripper')

        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Forward bridge stderr so hardware/RS485 warnings are visible to operators.
        if self.proc.stderr is not None:
            self.stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
            self.stderr_thread.start()

        ready = self._read_line(timeout=10.0)
        if ready != 'READY':
            err = self.proc.stderr.readline().strip() if self.proc.stderr else ''
            raise RuntimeError(f'Failed to start ER3Pro C++ bridge: {ready} {err}')

        if ER3PRO_ENABLE_GRIPPER and ER3PRO_GRIPPER_BACKEND == 'jodell_usb':
            self.usb_gripper = JodellUsbGripper()

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

    def _stderr_loop(self):
        if self.proc.stderr is None:
            return
        for line in self.proc.stderr:
            line = line.strip()
            if line:
                print(f'[arm_bridge] {line}', file=sys.stderr, flush=True)

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

    def move_to_teleop_preset(self):
        self._request('PRESET', timeout=8.0)

    def execute_action(self, action):
        arm_pos = np.asarray(action['arm_pos'], dtype=np.float64)
        arm_quat = np.asarray(action['arm_quat'], dtype=np.float64)

        gripper_value = float(np.asarray(action['gripper_pos']).item())
        self.gripper_pos[:] = gripper_value
        self._request(
            f'EXEC {arm_pos[0]} {arm_pos[1]} {arm_pos[2]} {arm_quat[0]} {arm_quat[1]} {arm_quat[2]} {arm_quat[3]} {gripper_value}',
            timeout=8.0,
        )

        if self.usb_gripper is not None:
            self.usb_gripper.command(gripper_value)

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
        if self.usb_gripper is not None:
            self.usb_gripper.close()
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

    def move_to_teleop_preset(self):
        return

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

    def move_to_teleop_preset(self):
        if ARM_BACKEND == 'er3pro':
            self.impl.move_to_teleop_preset()

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
