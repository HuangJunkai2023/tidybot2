# Author: Jimmy Wu
# Date: October 2024
#
# This RPC server allows other processes to communicate with the mobile base.
# It supports two backends:
# - native: direct low-level controller in this process (original behavior)
# - ros1_udp: send velocity commands to a ROS1 bridge (e.g. Docker)

import math
import socket
import threading
import time
from multiprocessing.managers import BaseManager as MPBaseManager

import numpy as np

from constants import BASE_BACKEND
from constants import BASE_CMD_TIMEOUT
from constants import BASE_CMD_UDP_HOST, BASE_CMD_UDP_PORT, BASE_CMD_UDP_PUBLISH_RATE
from constants import BASE_MAX_VEL_THETA, BASE_MAX_VEL_XY
from constants import BASE_POS_KP_ANGULAR, BASE_POS_KP_LINEAR
from constants import BASE_RPC_HOST, BASE_RPC_PORT, RPC_AUTHKEY

TWO_PI = 2.0 * math.pi


def wrap_to_pi(angle):
    return (angle + math.pi) % TWO_PI - math.pi


class NativeBaseBackend:
    def __init__(self, max_vel=(0.5, 0.5, 1.57), max_accel=(0.25, 0.25, 0.79)):
        self.max_vel = max_vel
        self.max_accel = max_accel
        self.vehicle = None

    def reset(self):
        if self.vehicle is not None and self.vehicle.control_loop_running:
            self.vehicle.stop_control()

        from base_controller import Vehicle
        self.vehicle = Vehicle(max_vel=self.max_vel, max_accel=self.max_accel)
        self.vehicle.start_control()
        while not self.vehicle.control_loop_running:
            time.sleep(0.01)

    def execute_action(self, action):
        self.vehicle.set_target_position(action['base_pose'])

    def get_state(self):
        return {'base_pose': self.vehicle.x}

    def close(self):
        if self.vehicle is not None and self.vehicle.control_loop_running:
            self.vehicle.stop_control()


class Ros1UdpBaseBackend:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.remote_addr = (BASE_CMD_UDP_HOST, BASE_CMD_UDP_PORT)

        self.pose = np.zeros(3, dtype=np.float64)
        self.latest_cmd = np.zeros(3, dtype=np.float64)  # vx, vy, wz in local frame
        self.target_pose = np.zeros(3, dtype=np.float64)

        self.last_update_time = time.monotonic()
        self.last_action_time = self.last_update_time

        self.running = False
        self.lock = threading.Lock()
        self.publisher_thread = threading.Thread(target=self._publisher_loop, daemon=True)

    def _format_cmd(self, vx, vy, wz):
        return f'{vx:.6f} {vy:.6f} {wz:.6f}'.encode('utf-8')

    def _send_cmd(self, cmd):
        try:
            self.sock.sendto(self._format_cmd(cmd[0], cmd[1], cmd[2]), self.remote_addr)
        except OSError:
            # Keep backend alive even if ROS bridge is temporarily unavailable.
            pass

    def _integrate_pose(self, cmd, now):
        dt = max(0.0, min(0.2, now - self.last_update_time))
        self.last_update_time = now

        if dt <= 0.0:
            return

        theta = self.pose[2]
        vx_local, vy_local, wz = cmd
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        vx_global = cos_t * vx_local - sin_t * vy_local
        vy_global = sin_t * vx_local + cos_t * vy_local

        self.pose[0] += vx_global * dt
        self.pose[1] += vy_global * dt
        self.pose[2] = wrap_to_pi(self.pose[2] + wz * dt)

    def _publisher_loop(self):
        period = 1.0 / max(BASE_CMD_UDP_PUBLISH_RATE, 1.0)
        while self.running:
            with self.lock:
                now = time.monotonic()
                if now - self.last_action_time > BASE_CMD_TIMEOUT:
                    cmd = np.zeros(3, dtype=np.float64)
                else:
                    cmd = self.latest_cmd.copy()

            self._send_cmd(cmd)
            time.sleep(period)

    def reset(self):
        with self.lock:
            self.pose[:] = 0.0
            self.target_pose[:] = 0.0
            self.latest_cmd[:] = 0.0
            self.last_update_time = time.monotonic()
            self.last_action_time = self.last_update_time

        if not self.running:
            self.running = True
            if not self.publisher_thread.is_alive():
                self.publisher_thread = threading.Thread(target=self._publisher_loop, daemon=True)
                self.publisher_thread.start()

        self._send_cmd(np.zeros(3, dtype=np.float64))

    def execute_action(self, action):
        target_pose = np.asarray(action['base_pose'], dtype=np.float64)
        now = time.monotonic()

        with self.lock:
            self._integrate_pose(self.latest_cmd, now)
            self.target_pose[:] = target_pose

            err_global = self.target_pose - self.pose
            err_global[2] = wrap_to_pi(err_global[2])

            theta = self.pose[2]
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            err_local_x = cos_t * err_global[0] + sin_t * err_global[1]
            err_local_y = -sin_t * err_global[0] + cos_t * err_global[1]

            vx = np.clip(BASE_POS_KP_LINEAR * err_local_x, -BASE_MAX_VEL_XY, BASE_MAX_VEL_XY)
            vy = np.clip(BASE_POS_KP_LINEAR * err_local_y, -BASE_MAX_VEL_XY, BASE_MAX_VEL_XY)
            wz = np.clip(BASE_POS_KP_ANGULAR * err_global[2], -BASE_MAX_VEL_THETA, BASE_MAX_VEL_THETA)

            self.latest_cmd[:] = (vx, vy, wz)
            self.last_action_time = now

    def get_state(self):
        with self.lock:
            return {'base_pose': self.pose.copy()}

    def close(self):
        self.running = False
        self._send_cmd(np.zeros(3, dtype=np.float64))
        try:
            self.sock.close()
        except OSError:
            pass

class Base:
    def __init__(self, max_vel=(0.5, 0.5, 1.57), max_accel=(0.25, 0.25, 0.79)):
        backend = BASE_BACKEND.lower()
        if backend == 'native':
            self.backend = NativeBaseBackend(max_vel=max_vel, max_accel=max_accel)
        elif backend == 'ros1_udp':
            self.backend = Ros1UdpBaseBackend()
        else:
            raise ValueError(f'Unsupported BASE_BACKEND: {BASE_BACKEND}')

    def reset(self):
        self.backend.reset()

    def execute_action(self, action):
        self.backend.execute_action(action)

    def get_state(self):
        return self.backend.get_state()

    def close(self):
        self.backend.close()

class BaseManager(MPBaseManager):
    pass

BaseManager.register('Base', Base)

if __name__ == '__main__':
    manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY)
    server = manager.get_server()
    print(f'Base manager server started at {BASE_RPC_HOST}:{BASE_RPC_PORT}')
    server.serve_forever()
    # import numpy as np
    # from constants import POLICY_CONTROL_PERIOD
    # manager = BaseManager(address=(BASE_RPC_HOST, BASE_RPC_PORT), authkey=RPC_AUTHKEY)
    # manager.connect()
    # base = manager.Base()
    # try:
    #     base.reset()
    #     for i in range(50):
    #         base.execute_action({'base_pose': np.array([(i / 50) * 0.5, 0.0, 0.0])})
    #         print(base.get_state())
    #         time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    # finally:
    #     base.close()
