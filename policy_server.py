# Author: Jimmy Wu
# Date: October 2024
#
# Note: This file is intended to be run within the diffusion_policy repository

import argparse
import math
import queue
import threading
import time
from collections import deque
import cv2 as cv
import dill
import hydra
import numpy as np
import torch
import zmq
try:
    from constants import POLICY_CONTROL_PERIOD
except ImportError:
    POLICY_CONTROL_PERIOD = 0.05  # 20 Hz fallback when copied into diffusion_policy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

LATENCY_BUDGET = 0.2  # 200 ms including policy inference and communication
LATENCY_STEPS = math.ceil(LATENCY_BUDGET / POLICY_CONTROL_PERIOD)  # Up to 3 is okay, 4 is too high
PROFILE_INTERVAL = 2.0

class StubDiffusionPolicy:
    def reset(self):
        pass

    def step(self, obs_sequence):
        obs = obs_sequence[-1]
        act_sequence = [f'{obs + i} (inference {obs})' for i in range(8)]
        time.sleep(0.115)  # 115 ms
        return act_sequence

# Adapted from https://github.com/real-stanford/diffusion_policy/blob/main/eval_real_robot.py
class DiffusionPolicy:
    def __init__(self, ckpt_path):
        # Load checkpoint
        with open(ckpt_path, 'rb') as f:
            payload = torch.load(f, pickle_module=dill)
        cfg = payload['cfg']
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg)
        workspace.load_payload(payload)

        # Load policy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model
        device = torch.device('cuda')
        policy.eval().to(device)

        # Store attributes
        self.policy = policy
        self.device = device
        self.obs_shape_meta = cfg.shape_meta['obs']
        self.rotation_transformer = RotationTransformer(from_rep='rotation_6d', to_rep='quaternion')
        self.warmed_up = False
        self.profile_last_time = time.time()
        self.profile_infer_count = 0
        self.profile_infer_total_ms = 0.0
        self.profile_infer_max_ms = 0.0
        print('[policy_server] checkpoint observation shapes:')
        for key, value in self.obs_shape_meta.items():
            print(f'  {key}: type={value.get("type", "low_dim")} shape={list(value["shape"])}')

    def reset(self):
        self.policy.reset()

    def step(self, obs_sequence):
        obs_dict = self._convert_obs(obs_sequence)
        with torch.no_grad():
            if not self.warmed_up:
                print('Warming up policy...')
                self.policy.predict_action(obs_dict)
                self.warmed_up = True
            start_time = time.time()
            result = self.policy.predict_action(obs_dict)
            elapsed_ms = 1000.0 * (time.time() - start_time)
            self.profile_infer_count += 1
            self.profile_infer_total_ms += elapsed_ms
            self.profile_infer_max_ms = max(self.profile_infer_max_ms, elapsed_ms)
            self._maybe_print_profile()
            action = result['action'][0].detach().to('cpu').numpy()
        act_sequence = self._convert_action(action)
        return act_sequence

    def _maybe_print_profile(self):
        now = time.time()
        dt = now - self.profile_last_time
        if dt < PROFILE_INTERVAL or self.profile_infer_count == 0:
            return
        avg_ms = self.profile_infer_total_ms / self.profile_infer_count
        infer_hz = self.profile_infer_count / dt
        print(
            f'[policy_server] infer_hz={infer_hz:.1f} '
            f'avg_infer_ms={avg_ms:.1f} max_infer_ms={self.profile_infer_max_ms:.1f}'
        )
        self.profile_last_time = now
        self.profile_infer_count = 0
        self.profile_infer_total_ms = 0.0
        self.profile_infer_max_ms = 0.0

    def _convert_obs(self, obs_sequence):
        obs_dict_np = {}
        for key, value in self.obs_shape_meta.items():
            if key not in obs_sequence[-1]:
                raise KeyError(f'Missing observation key required by checkpoint: {key}')

            if value.get('type') == 'rgb':
                target_shape = tuple(value['shape'])
                images = np.stack(
                    [self._prepare_rgb_obs(obs[key], target_shape, key) for obs in obs_sequence],
                    axis=0,
                )
                assert images.dtype == np.uint8
                images = images.astype(np.float32) / 255.0
                images = np.transpose(images, (0, 3, 1, 2))
                if images.shape[1:] != target_shape:
                    raise ValueError(f'{key} shape {images.shape[1:]} != {target_shape}')
                obs_dict_np[key] = images
            else:
                values = np.stack([obs[key] for obs in obs_sequence], axis=0).astype(np.float32)
                expected_shape = tuple(value['shape'])
                if values.shape[1:] != expected_shape:
                    raise ValueError(f'{key} shape {values.shape[1:]} != {expected_shape}')
                obs_dict_np[key] = values
        obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
        return obs_dict

    @staticmethod
    def _prepare_rgb_obs(image, target_chw_shape, key):
        if len(target_chw_shape) != 3 or target_chw_shape[0] != 3:
            raise ValueError(f'{key} expected RGB CHW shape [3, H, W], got {target_chw_shape}')

        image = np.asarray(image)
        target_h, target_w = target_chw_shape[1], target_chw_shape[2]

        if image.ndim == 3 and image.shape[0] == 3 and image.shape[2] != 3:
            image = np.transpose(image, (1, 2, 0))

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f'{key} expected HWC RGB image, got shape {image.shape}')

        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and image.max(initial=0.0) <= 1.0:
                image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)

        if image.shape[:2] != (target_h, target_w):
            image = cv.resize(image, (target_w, target_h), interpolation=cv.INTER_AREA)

        return np.ascontiguousarray(image)

    def _convert_action(self, action):
        act_sequence = []
        for act in action:
            action_dict = {
                'base_pose': act[:3],
                'arm_pos': act[3:6],
                'arm_quat': self.rotation_transformer.forward(act[6:12])[[1, 2, 3, 0]],  # (w, x, y, z) -> (x, y, z, w)
                'gripper_pos': act[12:13],
            }
            act_sequence.append(action_dict)
        return act_sequence

class PolicyWrapper:
    def __init__(self, policy, n_obs_steps=2, n_action_steps=8):
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.obs_queue = queue.Queue()
        self.act_queue = queue.Queue()
        self.profile_last_time = time.time()
        self.profile_infer_trigger_count = 0
        self.profile_idle_count = 0
        self.profile_backlog_count = 0
        self.last_error = None

        # Start inference loop
        threading.Thread(target=self.inference_loop, args=(policy,), daemon=True).start()

    def reset(self):
        self.last_error = None
        self.obs_queue.put('reset')

    def step(self, obs):
        if self.last_error is not None:
            error = self.last_error
            self.last_error = None
            raise RuntimeError(f'Policy inference loop failed: {error}')

        self.obs_queue.put(obs)
        action = None if self.act_queue.empty() else self.act_queue.get()
        if action is None:
            print('Warning: Unexpected idle action queue. Is the latency budget set too low?')
            self.profile_idle_count += 1
        return action

    def inference_loop(self, policy):
        obs_history = deque(maxlen=self.n_obs_steps)
        start_of_episode = True
        while True:
            # Check for new obs
            if not self.obs_queue.empty():
                obs = self.obs_queue.get()

                # Reset policy
                if obs == 'reset':
                    policy.reset()
                    obs_history.clear()
                    start_of_episode = True
                    self.last_error = None
                    while not self.act_queue.empty():
                        self.act_queue.get()
                    continue

                # Append obs to history
                obs_history.append(obs)

            if self.act_queue.qsize() < LATENCY_STEPS and len(obs_history) == self.n_obs_steps:
                self.profile_infer_trigger_count += 1
                obs_sequence = list(obs_history)
                try:
                    act_sequence = policy.step(obs_sequence)
                except Exception as e:
                    self.last_error = e
                    print(f'[policy_queue] inference error: {type(e).__name__}: {e}', flush=True)
                    obs_history.clear()
                    while not self.act_queue.empty():
                        self.act_queue.get()
                    time.sleep(0.1)
                    continue
                if not self.act_queue.empty():
                    print('Warning: Unexpected action queue backlog. Is the latency budget set too high?')
                    self.profile_backlog_count += 1
                if start_of_episode:
                    act_sequence = act_sequence[:self.n_action_steps - LATENCY_STEPS]
                    start_of_episode = False
                else:
                    act_sequence = act_sequence[LATENCY_STEPS:self.n_action_steps]
                for action in act_sequence:
                    self.act_queue.put(action)

            self._maybe_print_profile()
            time.sleep(0.001)

    def _maybe_print_profile(self):
        now = time.time()
        dt = now - self.profile_last_time
        if dt < PROFILE_INTERVAL:
            return
        infer_trigger_hz = self.profile_infer_trigger_count / dt
        print(
            f'[policy_queue] trigger_hz={infer_trigger_hz:.1f} '
            f'queue={self.act_queue.qsize()} idle={self.profile_idle_count} backlog={self.profile_backlog_count}'
        )
        self.profile_last_time = now
        self.profile_infer_trigger_count = 0
        self.profile_idle_count = 0
        self.profile_backlog_count = 0

class PolicyServer:
    def __init__(self, policy):
        self.policy = policy

        # Set up ZMQ server
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        port = 5555
        self.socket.bind(f'tcp://*:{port}')
        print(f'Server started on port {port}')

    def step(self, obs):
        # Decode images
        for k, v in obs.items():
            if k.endswith('image'):
                if isinstance(v, np.ndarray) and v.dtype == np.uint8 and (v.ndim == 1 or (v.ndim == 2 and 1 in v.shape)):
                    bgr = cv.imdecode(v, cv.IMREAD_COLOR)
                    if bgr is None:
                        raise RuntimeError(f'Failed to decode image for key: {k}')
                    obs[k] = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
                else:
                    obs[k] = v

        # Get action
        action = self.policy.step(obs)

        return action

    def run(self):
        while True:
            # Wait for request from client
            req = self.socket.recv_pyobj()  # Note: Not secure. Only unpickle data you trust.
            rep = {}

            # Reset policy
            if 'reset' in req:
                self.policy.reset()
                print('Policy has been reset')

            # Get action
            elif 'obs' in req:
                try:
                    obs = req['obs']
                    action = self.step(obs)
                    rep['action'] = action
                except Exception as e:
                    rep['error'] = f'{type(e).__name__}: {e}'
                    print(f'[policy_server] request error: {rep["error"]}', flush=True)

            # Send reply to client
            self.socket.send_pyobj(rep)

def main(ckpt_path):
    policy = PolicyWrapper(DiffusionPolicy(ckpt_path))
    server = PolicyServer(policy)
    server.run()

if __name__ == '__main__':
    # policy = PolicyWrapper(StubDiffusionPolicy())
    # policy.reset()
    # for step_num in range(1, 9999):
    #     print(f'obs: {step_num}, action: {policy.step(step_num)}')
    #     time.sleep(POLICY_CONTROL_PERIOD)  # Note: Not precise
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', default='data/outputs/2024.10.08/23.42.04_train_diffusion_unet_hybrid_sim-v1/checkpoints/epoch=0500-train_loss=0.001.ckpt')
    args = parser.parse_args()
    main(args.ckpt_path)
