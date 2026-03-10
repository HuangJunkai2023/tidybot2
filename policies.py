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

# Convert coordinate system from WebXR to robot
# def convert_webxr_pose(pos, quat):
#     # WebXR: +x right, +y up, +z back; Robot: +x forward, +y left, +z up
#     pos = np.array([-pos['y'], pos['x'], -pos['z']], dtype=np.float64)
#     # Quaternion axis remap tuned for ER3Pro teleop:
#     # make phone horizontal rotation (WebXR y-axis) align with robot yaw (z-axis).
#     rot = R.from_quat([-quat['z'], quat['x'], quat['y'], quat['w']])

#     # Apply offset so that rotations are around device center instead of device camera
#     pos = pos + rot.apply(DEVICE_CAMERA_OFFSET)

#     return pos, rot

def convert_webxr_pose(pos, quat):
    pos = np.array([-pos['y'], pos['x'], -pos['z']], dtype=np.float64)

    # yaw 和 roll 极性翻转
    rot = R.from_quat([quat['z'], quat['x'], -quat['y'], quat['w']])

    pos = pos + rot.apply(DEVICE_CAMERA_OFFSET)
    return pos, rot

TWO_PI = 2 * math.pi

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
                self.arm_target_pos = self.arm_ref_pos + z_rot_inv.apply(pos_diff)

                # Orientation
                self.arm_target_rot = (z_rot_inv * (rot * self.arm_xr_ref_rot_inv) * ref_z_rot) * self.arm_ref_rot

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
            self.targets_initialized = True

        # Return no action if teleop is not enabled
        if self.primary_device_id is None:
            return None

        # Get most recent teleop command
        arm_quat = self.arm_target_rot.as_quat()
        if arm_quat[3] < 0.0:  # Enforce quaternion uniqueness (Note: Not strictly necessary since policy training uses 6D rotation representation)
            np.negative(arm_quat, out=arm_quat)
        action = {
            'base_pose': self.base_target_pose.copy(),
            'arm_pos': self.arm_target_pos.copy(),
            'arm_quat': arm_quat,
            'gripper_pos': self.gripper_target_pos.copy(),
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
        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(f'tcp://{POLICY_SERVER_HOST}:{POLICY_SERVER_PORT}')
        print(f'Connected to policy server at {POLICY_SERVER_HOST}:{POLICY_SERVER_PORT}')

    def reset(self):
        # Wait for user to signal that episode has started
        super().reset()  # Note: Comment out to run without phone

        # Check connection to policy server and reset policy
        default_timeout = self.socket.getsockopt(zmq.RCVTIMEO)
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # Temporarily set 1000 ms timeout
        self.socket.send_pyobj({'reset': True})
        try:
            self.socket.recv_pyobj()  # Note: Not secure. Only unpickle data you trust.
        except zmq.error.Again as e:
            raise Exception('Could not communicate with policy server') from e
        self.socket.setsockopt(zmq.RCVTIMEO, default_timeout)  # Put default timeout back

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
        for k, v in obs.items():
            if v.ndim == 3:
                # Resize image to resolution expected by policy server
                v = cv.resize(v, (POLICY_IMAGE_WIDTH, POLICY_IMAGE_HEIGHT))

                # Encode image as JPEG
                _, v = cv.imencode('.jpg', v)  # Note: Interprets RGB as BGR
                encoded_obs[k] = v
            else:
                encoded_obs[k] = v

        # Send obs to policy server
        req = {'obs': encoded_obs}
        self.socket.send_pyobj(req)

        # Get action from policy server
        rep = self.socket.recv_pyobj()  # Note: Not secure. Only unpickle data you trust.
        action = rep['action']

        return action

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
