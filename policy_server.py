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
from constants import POLICY_CONTROL_PERIOD
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

LATENCY_BUDGET = 0.1  # 200 ms including policy inference and communication
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
            if value.get('type') == 'rgb':
                images = np.stack([obs[key] for obs in obs_sequence], axis=0)
                assert images.dtype == np.uint8
                images = images.astype(np.float32) / 255.0
                images = np.transpose(images, (0, 3, 1, 2))
                assert images.shape[1:] == tuple(value['shape'])
                obs_dict_np[key] = images
            else:
                obs_dict_np[key] = np.stack([obs[key] for obs in obs_sequence], axis=0).astype(np.float32)
        obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
        return obs_dict

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

        # Start inference loop
        threading.Thread(target=self.inference_loop, args=(policy,), daemon=True).start()

    def reset(self):
        self.obs_queue.put('reset')

    def step(self, obs):
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
                    while not self.act_queue.empty():
                        self.act_queue.get()
                    continue

                # Append obs to history
                obs_history.append(obs)

            if self.act_queue.qsize() < LATENCY_STEPS and len(obs_history) == self.n_obs_steps:
                self.profile_infer_trigger_count += 1
                obs_sequence = list(obs_history)
                act_sequence = policy.step(obs_sequence)
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
                bgr = cv.imdecode(v, cv.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f'Failed to decode image for key: {k}')
                obs[k] = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)

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
                obs = req['obs']
                action = self.step(obs)
                rep['action'] = action

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
