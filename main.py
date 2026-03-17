# Author: Jimmy Wu
# Date: October 2024

import argparse
import time
from itertools import count
import numpy as np
from constants import POLICY_CONTROL_PERIOD
from constants import ENABLE_ARM
from constants import ARM_BACKEND
from constants import ER3PRO_ARM_POSE_OBS_SOURCE
from episode_storage import EpisodeWriter
from policies import TeleopPolicy, RemotePolicy

PROFILE_INTERVAL = 2.0


def move_arm_to_teleop_preset(env):
    if not ENABLE_ARM or ARM_BACKEND != 'er3pro':
        return

    if getattr(env, 'arm', None) is None:
        return

    print('Moving arm to teleop preset pose...')
    env.arm.move_to_teleop_preset()
    print('Arm reached teleop preset pose')

def should_save_episode(writer):
    if len(writer) == 0:
        print('Discarding empty episode')
        return False

    # Prompt user whether to save episode
    while True:
        user_input = input('Save episode (y/n)? ').strip().lower()
        if user_input == 'y':
            return True
        if user_input == 'n':
            print('Discarding episode')
            return False
        print('Invalid response')

def _build_logged_observation(obs, action):
    if not isinstance(action, dict):
        return obs

    logged_obs = dict(obs)
    if ENABLE_ARM and ER3PRO_ARM_POSE_OBS_SOURCE == 'command':
        # Save demonstration command for arm pose so dataset aligns with teleop intent.
        logged_obs['arm_pos'] = np.asarray(action['arm_pos'], dtype=np.float64).copy()
        logged_obs['arm_quat'] = np.asarray(action['arm_quat'], dtype=np.float64).copy()
        logged_obs['gripper_pos'] = np.asarray(action['gripper_pos'], dtype=np.float64).copy()
    return logged_obs

def run_episode(env, policy, writer=None):
    profile = {
        'last_time': time.time(),
        'step_count': 0,
        'get_obs_total_ms': 0.0,
        'get_obs_max_ms': 0.0,
        'policy_total_ms': 0.0,
        'policy_max_ms': 0.0,
        'env_step_total_ms': 0.0,
        'env_step_max_ms': 0.0,
        'loop_total_ms': 0.0,
        'loop_max_ms': 0.0,
        'base_state_total_ms': 0.0,
        'base_state_max_ms': 0.0,
        'arm_state_total_ms': 0.0,
        'arm_state_max_ms': 0.0,
        'base_image_total_ms': 0.0,
        'base_image_max_ms': 0.0,
        'wrist_image_total_ms': 0.0,
        'wrist_image_max_ms': 0.0,
        'base_action_total_ms': 0.0,
        'base_action_max_ms': 0.0,
        'arm_action_total_ms': 0.0,
        'arm_action_max_ms': 0.0,
    }

    def update_profile(get_obs_ms, policy_ms, env_step_ms, loop_ms):
        obs_timing = getattr(env, 'last_obs_timing_ms', {})
        step_timing = getattr(env, 'last_step_timing_ms', {})
        profile['step_count'] += 1
        profile['get_obs_total_ms'] += get_obs_ms
        profile['get_obs_max_ms'] = max(profile['get_obs_max_ms'], get_obs_ms)
        profile['policy_total_ms'] += policy_ms
        profile['policy_max_ms'] = max(profile['policy_max_ms'], policy_ms)
        profile['env_step_total_ms'] += env_step_ms
        profile['env_step_max_ms'] = max(profile['env_step_max_ms'], env_step_ms)
        profile['loop_total_ms'] += loop_ms
        profile['loop_max_ms'] = max(profile['loop_max_ms'], loop_ms)
        for key in ('base_state', 'arm_state', 'base_image', 'wrist_image'):
            value = float(obs_timing.get(key, 0.0))
            profile[f'{key}_total_ms'] += value
            profile[f'{key}_max_ms'] = max(profile[f'{key}_max_ms'], value)
        for key in ('base_action', 'arm_action'):
            value = float(step_timing.get(key, 0.0))
            profile[f'{key}_total_ms'] += value
            profile[f'{key}_max_ms'] = max(profile[f'{key}_max_ms'], value)

    def maybe_print_profile():
        now = time.time()
        dt = now - profile['last_time']
        if dt < PROFILE_INTERVAL or profile['step_count'] == 0:
            return
        step_count = profile['step_count']
        loop_hz = step_count / dt
        print(
            f'[main_loop] hz={loop_hz:.1f} '
            f'avg_get_obs_ms={profile["get_obs_total_ms"] / step_count:.1f} max_get_obs_ms={profile["get_obs_max_ms"]:.1f} '
            f'avg_policy_ms={profile["policy_total_ms"] / step_count:.1f} max_policy_ms={profile["policy_max_ms"]:.1f} '
            f'avg_env_step_ms={profile["env_step_total_ms"] / step_count:.1f} max_env_step_ms={profile["env_step_max_ms"]:.1f} '
            f'avg_loop_ms={profile["loop_total_ms"] / step_count:.1f} max_loop_ms={profile["loop_max_ms"]:.1f} '
            f'avg_base_state_ms={profile["base_state_total_ms"] / step_count:.1f} '
            f'avg_arm_state_ms={profile["arm_state_total_ms"] / step_count:.1f} '
            f'avg_base_image_ms={profile["base_image_total_ms"] / step_count:.1f} '
            f'avg_wrist_image_ms={profile["wrist_image_total_ms"] / step_count:.1f} '
            f'avg_base_action_ms={profile["base_action_total_ms"] / step_count:.1f} '
            f'avg_arm_action_ms={profile["arm_action_total_ms"] / step_count:.1f}'
        )
        profile.update({
            'last_time': now,
            'step_count': 0,
            'get_obs_total_ms': 0.0,
            'get_obs_max_ms': 0.0,
            'policy_total_ms': 0.0,
            'policy_max_ms': 0.0,
            'env_step_total_ms': 0.0,
            'env_step_max_ms': 0.0,
            'loop_total_ms': 0.0,
            'loop_max_ms': 0.0,
            'base_state_total_ms': 0.0,
            'base_state_max_ms': 0.0,
            'arm_state_total_ms': 0.0,
            'arm_state_max_ms': 0.0,
            'base_image_total_ms': 0.0,
            'base_image_max_ms': 0.0,
            'wrist_image_total_ms': 0.0,
            'wrist_image_max_ms': 0.0,
            'base_action_total_ms': 0.0,
            'base_action_max_ms': 0.0,
            'arm_action_total_ms': 0.0,
            'arm_action_max_ms': 0.0,
        })

    # Reset the env
    print('Resetting env...')
    env.reset()
    print('Env has been reset')

    # Wait for user to press "Start episode"
    print('Press "Start episode" in the web app when ready to start new episode')
    policy.reset()

    move_arm_to_teleop_preset(env)

    print('Starting new episode')

    episode_ended = False
    start_time = time.time()
    profile['last_time'] = start_time
    for step_idx in count():
        loop_start_time = time.time()

        # Enforce desired control freq
        step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
        while time.time() < step_end_time:
            time.sleep(0.0001)

        # Get latest observation
        get_obs_start = time.time()
        obs = env.get_obs()
        get_obs_ms = 1000.0 * (time.time() - get_obs_start)

        # Get action
        policy_start = time.time()
        action = policy.step(obs)
        policy_ms = 1000.0 * (time.time() - policy_start)

        # No action if teleop not enabled
        if action is None:
            update_profile(
                get_obs_ms,
                policy_ms,
                0.0,
                1000.0 * (time.time() - loop_start_time),
            )
            maybe_print_profile()
            continue

        # Execute valid action on robot
        if isinstance(action, dict):
            env_step_start = time.time()
            env.step(action)
            env_step_ms = 1000.0 * (time.time() - env_step_start)

            if writer is not None and not episode_ended:
                # Record executed action
                writer.step(_build_logged_observation(obs, action), action)

        # Episode ended
        elif not episode_ended and action == 'end_episode':
            episode_ended = True
            print('Episode ended')

            if writer is not None and should_save_episode(writer):
                # Save to disk in background thread
                writer.flush_async()

            print('Teleop is now active. Press "Reset env" in the web app when ready to proceed.')

        # Ready for env reset
        elif action == 'reset_env':
            break

        update_profile(
            get_obs_ms,
            policy_ms,
            env_step_ms if isinstance(action, dict) else 0.0,
            1000.0 * (time.time() - loop_start_time),
        )
        maybe_print_profile()

    if writer is not None:
        # Wait for writer to finish saving to disk
        writer.wait_for_flush()

def main(args):
    # Create env
    if args.sim:
        from mujoco_env import MujocoEnv
        if args.teleop:
            env = MujocoEnv(show_images=True)
        else:
            env = MujocoEnv()
    else:
        from real_env import RealEnv
        env = RealEnv()

    # Create policy
    if args.teleop:
        policy = TeleopPolicy(use_ssl=args.ssl)
    else:
        policy = RemotePolicy(use_ssl=args.ssl)

    try:
        while True:
            writer = EpisodeWriter(args.output_dir) if args.save else None
            run_episode(env, policy, writer)
    finally:
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--teleop', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--ssl', action='store_true', help='Use HTTPS instead of HTTP (required for WebXR on some devices)')
    parser.add_argument('--output-dir', default='data/demos')
    main(parser.parse_args())
