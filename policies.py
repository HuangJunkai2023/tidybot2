# Author: Jimmy Wu
# Date: October 2024

import logging
import math
import socket
import threading
import time
from queue import Empty, Queue
import cv2 as cv
import numpy as np
import zmq
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from scipy.spatial.transform import Rotation as R
from constants import POLICY_SERVER_HOST, POLICY_SERVER_PORT, POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT
from constants import BASE_DIFF_DRIVE_MODE
from constants import POLICY_CONTROL_PERIOD
from constants import TELEOP_JUMP_GUARD_ENABLE
from constants import TELEOP_MAX_BASE_LINEAR_SPEED, TELEOP_MAX_BASE_ANGULAR_SPEED
from constants import TELEOP_MAX_ARM_LINEAR_SPEED, TELEOP_MAX_ARM_ANGULAR_SPEED
from constants import TELEOP_MAX_GRIPPER_SPEED
from constants import TELEOP_ARM_POSE_REJECT_ENABLE
from constants import TELEOP_ARM_MAX_FRAME_POS_DELTA, TELEOP_ARM_MAX_FRAME_ROT_DELTA

POLICY_SOCKET_TIMEOUT_MS = 1000
POLICY_PROFILE_INTERVAL = 2.0

class Policy:
    def reset(self):
        raise NotImplementedError

    def step(self, obs):
        raise NotImplementedError


class TeleopMessageBuffer:
    def __init__(self):
        self.state_updates = Queue()
        self.latest_teleop = None
        self.lock = threading.Lock()

    def put(self, data):
        if 'state_update' in data:
            self.state_updates.put(data)
            return

        with self.lock:
            self.latest_teleop = data

    def get_state_update(self):
        try:
            return self.state_updates.get_nowait()
        except Empty:
            return None

    def pop_latest_teleop(self):
        with self.lock:
            latest = self.latest_teleop
            self.latest_teleop = None
        return latest

class WebServer:
    def __init__(self, message_buffer):
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.message_buffer = message_buffer
        self.last_echo_time = 0.0

        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.socketio.on('message')
        def handle_message(data):
            # Throttle RTT echos to avoid flooding the browser main thread.
            now = time.time()
            if now - self.last_echo_time >= 0.2:
                emit('echo', data['timestamp'])
                self.last_echo_time = now

            # Stamp with server time so staleness check is clock-independent
            data['server_recv_time'] = now

            # Keep only the newest teleop packet; state updates are queued separately.
            self.message_buffer.put(data)

        # Reduce verbose Flask log output
        logging.getLogger('werkzeug').setLevel(logging.WARNING)

    def run(self, use_ssl=False):
        # Get IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(('8.8.8.8', 1))
            address = s.getsockname()[0]
        except Exception:
            address = '127.0.0.1'
        finally:
            s.close()
        
        protocol = 'https' if use_ssl else 'http'
        print(f'Starting server at {protocol}://{address}:5000')
        
        if use_ssl:
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            cert_file = os.path.join(script_dir, 'cert.pem')
            key_file = os.path.join(script_dir, 'key.pem')
            if os.path.exists(cert_file) and os.path.exists(key_file):
                self.socketio.run(self.app, host='0.0.0.0', ssl_context=(cert_file, key_file))
            else:
                print(f'Warning: SSL certificates not found. Run this command to generate them:')
                print(f'openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365')
                print(f'Error: Cannot start HTTPS server without certificates.')
                return
        else:
            self.socketio.run(self.app, host='0.0.0.0')

DEVICE_CAMERA_OFFSET = np.array([0.0, 0.02, -0.04])  # iPhone 14 Pro

# WebXR viewer space: +x right, +y up, +z back
# Robot teleop space: +x forward, +y left, +z up
WEBXR_TO_ROBOT_BASIS = np.array([
    [0.0, 0.0, -1.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
], dtype=np.float64)
WEBXR_TO_ROBOT_ROT = R.from_matrix(WEBXR_TO_ROBOT_BASIS)


def convert_webxr_pose(pos, quat):
    pos = WEBXR_TO_ROBOT_BASIS @ np.array([pos['x'], pos['y'], pos['z']], dtype=np.float64)
    
    # Use the same basis change for orientation instead of ad-hoc component shuffles.
    webxr_rot = R.from_quat([quat['x'], quat['y'], quat['z'], quat['w']])
    rot = WEBXR_TO_ROBOT_ROT * webxr_rot * WEBXR_TO_ROBOT_ROT.inv()

    # Apply offset so that rotations are around device center instead of device camera.
    pos = pos + rot.apply(DEVICE_CAMERA_OFFSET)
    return pos, rot

TWO_PI = 2 * math.pi


def _clip_vector_step(target, current, max_step):
    delta = target - current
    dist = float(np.linalg.norm(delta))
    if dist <= max_step or dist < 1e-12:
        return target
    return current + delta * (max_step / dist)


def _clip_scalar_step(target, current, max_step):
    delta = float(target - current)
    if abs(delta) <= max_step:
        return float(target)
    return float(current + math.copysign(max_step, delta))


def _clip_angle_step(target, current, max_step):
    delta = (target - current + math.pi) % TWO_PI - math.pi
    if abs(delta) <= max_step:
        return float(target)
    return float(current + math.copysign(max_step, delta))


def _clip_rotation_step(target_rot, current_rot, max_step):
    delta_rotvec = (target_rot * current_rot.inv()).as_rotvec()
    delta_norm = float(np.linalg.norm(delta_rotvec))
    if delta_norm <= max_step or delta_norm < 1e-12:
        return target_rot
    limited_delta = delta_rotvec * (max_step / delta_norm)
    return R.from_rotvec(limited_delta) * current_rot


def _rotation_distance(rot_a, rot_b):
    return float(np.linalg.norm((rot_a * rot_b.inv()).as_rotvec()))

class TeleopController:
    def __init__(self):
        # Teleop device IDs
        self.primary_device_id = None    # Primary device controls either the arm or the base
        self.secondary_device_id = None  # Optional secondary device controls the base
        self.enabled_counts = {}

        # Mobile base pose
        self.base_pose = None

        # Teleop targets
        self.targets_initialized = False
        self.base_target_pose = None
        self.arm_target_pos = None
        self.arm_target_rot = None
        self.gripper_target_pos = None

        # WebXR reference poses
        self.base_xr_ref_pos = None
        self.base_xr_ref_rot_inv = None
        self.arm_xr_ref_pos = None
        self.arm_xr_ref_rot_inv = None

        # Robot reference poses
        self.base_ref_pose = None
        self.arm_ref_pos = None
        self.arm_ref_rot = None
        self.arm_ref_base_pose = None  # For optional secondary control of base
        self.gripper_ref_pos = None

        # Last sent command state (for jump protection).
        self.base_cmd_pose = None
        self.arm_cmd_pos = None
        self.arm_cmd_rot = None
        self.gripper_cmd_pos = None

    def process_message(self, data):
        if not self.targets_initialized:
            return

        # Use device ID to disambiguate between primary and secondary devices
        device_id = data['device_id']

        # Update enabled count for the device that sent this message
        self.enabled_counts[device_id] = self.enabled_counts.get(device_id, 0) + 1 if 'teleop_mode' in data else 0

        # Assign primary and secondary devices
        if self.enabled_counts[device_id] > 2:
            if self.primary_device_id is None and device_id != self.secondary_device_id:
                # Note: We skip the first 2 steps because WebXR pose updates have higher latency than touch events
                self.primary_device_id = device_id
            elif self.secondary_device_id is None and device_id != self.primary_device_id:
                self.secondary_device_id = device_id
        elif self.enabled_counts[device_id] == 0:
            if device_id == self.primary_device_id:
                self.primary_device_id = None  # Primary device no longer enabled
                self.base_xr_ref_pos = None
                self.arm_xr_ref_pos = None
            elif device_id == self.secondary_device_id:
                self.secondary_device_id = None
                self.base_xr_ref_pos = None

        # Teleop is enabled
        if self.primary_device_id is not None and 'teleop_mode' in data:
            pos, rot = convert_webxr_pose(data['position'], data['orientation'])

            # Base movement
            if data['teleop_mode'] == 'base' or device_id == self.secondary_device_id:  # Note: Secondary device can only control base
                # Store reference poses
                if self.base_xr_ref_pos is None:
                    self.base_ref_pose = self.base_pose.copy()
                    self.base_xr_ref_pos = pos[:2]
                    self.base_xr_ref_rot_inv = rot.inv()

                # Position
                if BASE_DIFF_DRIVE_MODE:
                    # Differential-drive base: map gesture translation to forward-only travel.
                    delta = pos[:2] - self.base_xr_ref_pos
                    ref_theta = self.base_ref_pose[2]
                    fwd = np.array([math.cos(ref_theta), math.sin(ref_theta)], dtype=np.float64)
                    forward_delta = float(delta @ fwd)
                    self.base_target_pose[:2] = self.base_ref_pose[:2] + forward_delta * fwd
                else:
                    self.base_target_pose[:2] = self.base_ref_pose[:2] + (pos[:2] - self.base_xr_ref_pos)

                # Orientation
                base_fwd_vec_rotated = (rot * self.base_xr_ref_rot_inv).apply([1.0, 0.0, 0.0])
                base_target_theta = self.base_ref_pose[2] + math.atan2(base_fwd_vec_rotated[1], base_fwd_vec_rotated[0])
                self.base_target_pose[2] += (base_target_theta - self.base_target_pose[2] + math.pi) % TWO_PI - math.pi  # Unwrapped

            # Arm movement
            elif data['teleop_mode'] == 'arm':
                # Store reference poses
                if self.arm_xr_ref_pos is None:
                    self.arm_xr_ref_pos = pos
                    self.arm_xr_ref_rot_inv = rot.inv()
                    self.arm_ref_pos = self.arm_target_pos.copy()
                    self.arm_ref_rot = self.arm_target_rot
                    self.arm_ref_base_pose = self.base_pose.copy()
                    self.gripper_ref_pos = self.gripper_target_pos

                # Rotations around z-axis to go between global frame (base) and local frame (arm)
                z_rot = R.from_rotvec(np.array([0.0, 0.0, 1.0]) * self.base_pose[2])
                z_rot_inv = z_rot.inv()
                ref_z_rot = R.from_rotvec(np.array([0.0, 0.0, 1.0]) * self.arm_ref_base_pose[2])

                # Position
                pos_diff = pos - self.arm_xr_ref_pos  # WebXR
                pos_diff += ref_z_rot.apply(self.arm_ref_pos) - z_rot.apply(self.arm_ref_pos)  # Secondary base control: Compensate for base rotation
                pos_diff[:2] += self.arm_ref_base_pose[:2] - self.base_pose[:2]  # Secondary base control: Compensate for base translation
                candidate_arm_target_pos = self.arm_ref_pos + z_rot_inv.apply(pos_diff)

                # Orientation
                candidate_arm_target_rot = (z_rot_inv * (rot * self.arm_xr_ref_rot_inv) * ref_z_rot) * self.arm_ref_rot

                if TELEOP_ARM_POSE_REJECT_ENABLE:
                    pos_delta = float(np.linalg.norm(candidate_arm_target_pos - self.arm_target_pos))
                    rot_delta = _rotation_distance(candidate_arm_target_rot, self.arm_target_rot)
                    if (
                        pos_delta > TELEOP_ARM_MAX_FRAME_POS_DELTA
                        or rot_delta > TELEOP_ARM_MAX_FRAME_ROT_DELTA
                    ):
                        return

                self.arm_target_pos = candidate_arm_target_pos
                self.arm_target_rot = candidate_arm_target_rot

                # Gripper position
                self.gripper_target_pos = np.clip(self.gripper_ref_pos + data['gripper_delta'], 0.0, 1.0)

        # Teleop is disabled
        elif self.primary_device_id is None:
            # Update target pose in case base is pushed while teleop is disabled
            self.base_target_pose = self.base_pose

    def step(self, obs):
        # Update robot state
        self.base_pose = obs['base_pose']

        # Initialize targets
        if not self.targets_initialized:
            self.base_target_pose = obs['base_pose']
            self.arm_target_pos = obs['arm_pos']
            self.arm_target_rot = R.from_quat(obs['arm_quat'])
            self.gripper_target_pos = obs['gripper_pos']

            self.base_cmd_pose = obs['base_pose'].copy()
            self.arm_cmd_pos = obs['arm_pos'].copy()
            self.arm_cmd_rot = R.from_quat(obs['arm_quat'])
            self.gripper_cmd_pos = obs['gripper_pos'].copy()
            self.targets_initialized = True

        # Return no action if teleop is not enabled
        if self.primary_device_id is None:
            # Keep command state aligned with live robot state while teleop is idle.
            self.base_target_pose = obs['base_pose']
            self.arm_target_pos = obs['arm_pos']
            self.arm_target_rot = R.from_quat(obs['arm_quat'])
            self.gripper_target_pos = obs['gripper_pos']
            self.base_cmd_pose = obs['base_pose'].copy()
            self.arm_cmd_pos = obs['arm_pos'].copy()
            self.arm_cmd_rot = R.from_quat(obs['arm_quat'])
            self.gripper_cmd_pos = obs['gripper_pos'].copy()
            return None

        assert self.base_target_pose is not None
        assert self.arm_target_pos is not None
        assert self.arm_target_rot is not None
        assert self.gripper_target_pos is not None
        assert self.base_cmd_pose is not None
        assert self.arm_cmd_pos is not None
        assert self.arm_cmd_rot is not None
        assert self.gripper_cmd_pos is not None

        # Slew-rate limiter to protect against occasional pose jumps during teleop.
        if TELEOP_JUMP_GUARD_ENABLE:
            dt = float(POLICY_CONTROL_PERIOD)

            max_base_linear_step = TELEOP_MAX_BASE_LINEAR_SPEED * dt
            max_base_angular_step = TELEOP_MAX_BASE_ANGULAR_SPEED * dt
            max_arm_linear_step = TELEOP_MAX_ARM_LINEAR_SPEED * dt
            max_arm_angular_step = TELEOP_MAX_ARM_ANGULAR_SPEED * dt
            max_gripper_step = TELEOP_MAX_GRIPPER_SPEED * dt

            self.base_cmd_pose[:2] = _clip_vector_step(self.base_target_pose[:2], self.base_cmd_pose[:2], max_base_linear_step)
            self.base_cmd_pose[2] = _clip_angle_step(self.base_target_pose[2], self.base_cmd_pose[2], max_base_angular_step)

            self.arm_cmd_pos = _clip_vector_step(self.arm_target_pos, self.arm_cmd_pos, max_arm_linear_step)
            self.arm_cmd_rot = _clip_rotation_step(self.arm_target_rot, self.arm_cmd_rot, max_arm_angular_step)
            self.gripper_cmd_pos[0] = _clip_scalar_step(self.gripper_target_pos[0], self.gripper_cmd_pos[0], max_gripper_step)
        else:
            self.base_cmd_pose = self.base_target_pose.copy()
            self.arm_cmd_pos = self.arm_target_pos.copy()
            self.arm_cmd_rot = self.arm_target_rot
            self.gripper_cmd_pos = self.gripper_target_pos.copy()

        # Get most recent teleop command
        arm_quat = self.arm_cmd_rot.as_quat(canonical=False)
        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness (Note: Not strictly necessary since policy training uses 6D rotation representation)
            np.negative(arm_quat, out=arm_quat)
        action = {
            'base_pose': self.base_cmd_pose.copy(),
            'arm_pos': self.arm_cmd_pos.copy(),
            'arm_quat': arm_quat,
            'gripper_pos': self.gripper_cmd_pos.copy(),
        }

        return action

# Teleop using WebXR phone web app
class TeleopPolicy(Policy):
    def __init__(self, use_ssl=False):
        self.message_buffer = TeleopMessageBuffer()
        self.teleop_controller = None
        self.teleop_state = None  # States: episode_started -> episode_ended -> reset_env
        self.episode_ended = False

        # Web server for serving the WebXR phone web app
        server = WebServer(self.message_buffer)
        threading.Thread(target=lambda: server.run(use_ssl=use_ssl), daemon=True).start()

        # Listener thread to process messages from WebXR client
        threading.Thread(target=self.listener_loop, daemon=True).start()

    def reset(self):
        self.teleop_controller = TeleopController()
        self.episode_ended = False

        # Wait for user to signal that episode has started
        self.teleop_state = None
        while self.teleop_state != 'episode_started':
            time.sleep(0.01)

    def step(self, obs):
        # Signal that user has ended episode
        if not self.episode_ended and self.teleop_state == 'episode_ended':
            self.episode_ended = True
            return 'end_episode'

        # Signal that user is ready for env reset (after ending the episode)
        if self.teleop_state == 'reset_env':
            return 'reset_env'

        return self._step(obs)

    def _step(self, obs):
        return self.teleop_controller.step(obs)

    def listener_loop(self):
        while True:
            # Drain state updates first.
            while True:
                item = self.message_buffer.get_state_update()
                if item is None:
                    break
                self.teleop_state = item['state_update']

            latest_data = self.message_buffer.pop_latest_teleop()

            if latest_data is not None:
                # Use server-side receive time to avoid phone/server clock skew
                age_ms = 1000 * (time.time() - latest_data['server_recv_time'])
                if age_ms < 250:  # 250 ms
                    self._process_message(latest_data)

            time.sleep(0.001)

    def _process_message(self, data):
        self.teleop_controller.process_message(data)

# Execute policy running on remote server
class RemotePolicy(TeleopPolicy):
    def __init__(self, use_ssl=False):
        super().__init__(use_ssl=use_ssl)

        # Use phone as enabling device during policy rollout
        self.enabled = False

        # Connection to policy server
        self.context = zmq.Context()
        self.socket = None
        self.profile_last_time = time.time()
        self.profile_step_count = 0
        self.profile_rtt_total_ms = 0.0
        self.profile_rtt_max_ms = 0.0
        self.profile_encode_total_ms = 0.0
        self.profile_encode_max_ms = 0.0
        self._connect_socket()

    def _connect_socket(self):
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, POLICY_SOCKET_TIMEOUT_MS)
        self.socket.setsockopt(zmq.SNDTIMEO, POLICY_SOCKET_TIMEOUT_MS)
        self.socket.connect(f'tcp://{POLICY_SERVER_HOST}:{POLICY_SERVER_PORT}')
        print(f'Connected to policy server at {POLICY_SERVER_HOST}:{POLICY_SERVER_PORT}')

    def reset(self):
        self.enabled = False

        # Wait for user to signal that episode has started
        super().reset()  # Note: Comment out to run without phone

        # Check connection to policy server and reset policy
        try:
            self.socket.send_pyobj({'reset': True})
            self.socket.recv_pyobj()  # Note: Not secure. Only unpickle data you trust.
        except (zmq.error.Again, zmq.error.ZMQError) as e:
            self._connect_socket()
            raise Exception('Could not communicate with policy server') from e

        # Enable policy execution immediately
        self.enabled = True

    def _step(self, obs):
        # Return teleop command if episode has ended
        if self.episode_ended:
            return self.teleop_controller.step(obs)

        # Return no action if robot is not enabled
        if not self.enabled:
            return None

        # Encode images
        encoded_obs = {}
        encode_start = time.time()
        for k, v in obs.items():
            if v.ndim == 3:
                # Resize image to resolution expected by policy server
                v = cv.resize(v, (POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT))

                # OpenCV expects BGR input while our observation pipeline uses RGB.
                ok, v = cv.imencode('.jpg', cv.cvtColor(v, cv.COLOR_RGB2BGR))
                if not ok:
                    raise RuntimeError(f'Failed to encode image for key: {k}')
                encoded_obs[k] = v
            else:
                encoded_obs[k] = v
        encode_ms = 1000.0 * (time.time() - encode_start)

        # Send obs to policy server
        req = {'obs': encoded_obs}
        try:
            request_start = time.time()
            self.socket.send_pyobj(req)
            rep = self.socket.recv_pyobj()  # Note: Not secure. Only unpickle data you trust.
            rtt_ms = 1000.0 * (time.time() - request_start)
        except (zmq.error.Again, zmq.error.ZMQError):
            print('Warning: Lost communication with policy server, pausing policy execution until next reset.')
            self.enabled = False
            self._connect_socket()
            return None
        action = rep.get('action')
        self.profile_step_count += 1
        self.profile_rtt_total_ms += rtt_ms
        self.profile_rtt_max_ms = max(self.profile_rtt_max_ms, rtt_ms)
        self.profile_encode_total_ms += encode_ms
        self.profile_encode_max_ms = max(self.profile_encode_max_ms, encode_ms)
        self._maybe_print_profile()

        return action

    def _maybe_print_profile(self):
        now = time.time()
        dt = now - self.profile_last_time
        if dt < POLICY_PROFILE_INTERVAL or self.profile_step_count == 0:
            return
        loop_hz = self.profile_step_count / dt
        avg_rtt_ms = self.profile_rtt_total_ms / self.profile_step_count
        avg_encode_ms = self.profile_encode_total_ms / self.profile_step_count
        print(
            f'[remote_policy] req_hz={loop_hz:.1f} '
            f'avg_rtt_ms={avg_rtt_ms:.1f} max_rtt_ms={self.profile_rtt_max_ms:.1f} '
            f'avg_encode_ms={avg_encode_ms:.1f} max_encode_ms={self.profile_encode_max_ms:.1f}'
        )
        self.profile_last_time = now
        self.profile_step_count = 0
        self.profile_rtt_total_ms = 0.0
        self.profile_rtt_max_ms = 0.0
        self.profile_encode_total_ms = 0.0
        self.profile_encode_max_ms = 0.0

    def _process_message(self, data):
        if self.episode_ended:
            # Run teleop controller if episode has ended
            self.teleop_controller.process_message(data)

if __name__ == '__main__':
    # WebServer(Queue()).run(); time.sleep(1000)
    # WebXRListener(); time.sleep(1000)
    from constants import POLICY_CONTROL_PERIOD
    obs = {
        'base_pose': np.zeros(3),
        'arm_pos': np.zeros(3),
        'arm_quat': np.array([0.0, 0.0, 0.0, 1.0]),
        'gripper_pos': np.zeros(1),
        'base_image': np.zeros((720, 1280, 3)),
        'wrist_image': np.zeros((720, 1080, 3)),
    }
    policy = TeleopPolicy()
    # policy = RemotePolicy()
    while True:
        policy.reset()
        for _ in range(100):
            print(policy.step(obs))
            time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
