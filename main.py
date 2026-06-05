# Author: Jimmy Wu
# Date: October 2024

import argparse
import signal
import time
from itertools import count
import numpy as np
from constants import POLICY_CONTROL_PERIOD
from constants import ENABLE_ARM
from constants import ARM_BACKEND
from constants import ER3PRO_ARM_POSE_OBS_SOURCE
from constants import LEROBOT_FPS, LEROBOT_REPO_ID, LEROBOT_ROOT, LEROBOT_TASK
from episode_storage import EpisodeWriter, LeRobotEpisodeWriter
from policies import TeleopPolicy, RemotePolicy, UarmTeleopPolicy, GamepadTeleopPolicy

PROFILE_INTERVAL = 2.0
ENABLE_MAIN_LOOP_PROFILE = False


def _close_quietly(resource, name):
    if resource is None or not hasattr(resource, 'close'):
        return
    try:
        resource.close()
    except Exception as e:
        print(f'Warning: error while closing {name}: {e}', flush=True)


def install_shutdown_handlers(get_resources):
    shutting_down = {'active': False}
    previous_handlers = {}

    def cleanup():
        env, policy = get_resources()
        _close_quietly(policy, 'policy')
        _close_quietly(env, 'env')

    def handle_shutdown(signum, frame):
        if not shutting_down['active']:
            shutting_down['active'] = True
            signal_name = signal.Signals(signum).name
            print(f'\nReceived {signal_name}, shutting down cleanly...', flush=True)
            cleanup()
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + signum)

    for signum in (signal.SIGINT, signal.SIGTERM, getattr(signal, 'SIGTSTP', None)):
        if signum is None:
            continue
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_shutdown)

    def restore():
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    return restore


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
            if hasattr(writer, 'discard'):
                writer.discard()
            return False
        print('Invalid response')

def _build_logged_observation(obs, action):
    if not isinstance(action, dict):
        return obs

    logged_obs = dict(obs)
    if ENABLE_ARM and ER3PRO_ARM_POSE_OBS_SOURCE == 'command':
        # Save demonstration command for arm pose so dataset aligns with teleop intent.
        if 'arm_pos' in action:
            logged_obs['arm_pos'] = np.asarray(action['arm_pos'], dtype=np.float64).copy()
        if 'arm_quat' in action:
            logged_obs['arm_quat'] = np.asarray(action['arm_quat'], dtype=np.float64).copy()
        if 'arm_joints' in action:
            logged_obs['arm_joints'] = np.asarray(action['arm_joints'], dtype=np.float64).copy()
        if 'gripper_pos' in action:
            logged_obs['gripper_pos'] = np.asarray(action['gripper_pos'], dtype=np.float64).copy()
    return logged_obs

def _normalize_action_for_env(obs, action):
    if not isinstance(action, dict):
        return action

    normalized_action = dict(action)
    if 'arm_joints' not in normalized_action:
        for alias in ('arm_joint', 'arm_joint_pos', 'arm_qpos'):
            if alias in normalized_action:
                normalized_action['arm_joints'] = normalized_action[alias]
                break

    # Some teleop sources, such as UArm, only drive the arm. Keep the base
    # command pinned to the current pose so downstream components still receive
    # the full action schema they expect.
    if 'base_pose' not in normalized_action and 'base_pose' in obs:
        normalized_action['base_pose'] = np.asarray(obs['base_pose'], dtype=np.float64).copy()

    return normalized_action

def run_episode(env, policy, writer=None, action_debug=False):
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

    action_debug_state = {'last_time': 0.0}

    def maybe_print_profile():
        if not ENABLE_MAIN_LOOP_PROFILE:
            return
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

    def maybe_print_action_debug(obs, action, env_step_ms):
        if not action_debug or not isinstance(action, dict):
            return
        now = time.monotonic()
        if now - action_debug_state['last_time'] < 0.5:
            return
        action_debug_state['last_time'] = now

        step_timing = getattr(env, 'last_step_timing_ms', {})
        parts = [f'env_step_ms={env_step_ms:.1f}', f'arm_action_ms={float(step_timing.get("arm_action", 0.0)):.1f}']
        if 'arm_pos' in action and 'arm_pos' in obs:
            arm_pos = np.asarray(action['arm_pos'], dtype=np.float64)
            obs_pos = np.asarray(obs['arm_pos'], dtype=np.float64)
            parts.append(f'arm_pos={np.round(arm_pos, 4).tolist()}')
            parts.append(f'arm_pos_delta={np.round(arm_pos - obs_pos, 4).tolist()}')
        if 'gripper_pos' in action:
            parts.append(f'gripper={float(np.asarray(action["gripper_pos"]).reshape(-1)[0]):.3f}')
        print('[gamepad_action_debug] ' + ' '.join(parts), flush=True)

    # Reset the env
    print('Resetting env...')
    env.reset()
    print('Env has been reset')

    # Wait for teleop input source to become active.
    if getattr(policy, 'uses_web_start', False):
        print('Press "Start episode" in the web app when ready to start new episode')
    else:
        print('Initializing teleop input source...')
    policy.reset()

    if not getattr(policy, 'handles_teleop_preset', False):
        move_arm_to_teleop_preset(env)

    print('Starting new episode')

    episode_ended = False
    start_time = time.time()
    profile['last_time'] = start_time
    try:
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
                action = _normalize_action_for_env(obs, action)
                env_step_start = time.time()
                env.step(action)
                env_step_ms = 1000.0 * (time.time() - env_step_start)
                maybe_print_action_debug(obs, action, env_step_ms)

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
    except KeyboardInterrupt:
        if writer is not None and len(writer) > 0 and should_save_episode(writer):
            writer.flush_async()
            writer.wait_for_flush()
        raise

    if writer is not None:
        # Wait for writer to finish saving to disk
        writer.wait_for_flush()

def create_episode_writer(args):
    if not args.save:
        return None
    if getattr(args, 'gamepad', False):
        return LeRobotEpisodeWriter(
            root=getattr(args, 'lerobot_root', LEROBOT_ROOT),
            repo_id=getattr(args, 'lerobot_repo_id', LEROBOT_REPO_ID),
            task=getattr(args, 'lerobot_task', LEROBOT_TASK),
            fps=getattr(args, 'lerobot_fps', LEROBOT_FPS),
        )
    return EpisodeWriter(args.output_dir)


def main(args):
    env = None
    policy = None
    restore_shutdown_handlers = install_shutdown_handlers(lambda: (env, policy))

    try:
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
            if args.uarm:
                policy = UarmTeleopPolicy(env.arm)
            elif args.gamepad:
                policy = GamepadTeleopPolicy(use_ssl=args.ssl, debug=args.gamepad_debug)
            else:
                policy = TeleopPolicy(use_ssl=args.ssl)
        else:
            policy = RemotePolicy(use_ssl=args.ssl)

        persistent_writer = create_episode_writer(args) if args.save and args.gamepad else None
        while True:
            writer = persistent_writer if persistent_writer is not None else create_episode_writer(args)
            run_episode(env, policy, writer, action_debug=args.gamepad_debug)
    finally:
        if 'persistent_writer' in locals():
            _close_quietly(persistent_writer, 'writer')
        _close_quietly(policy, 'policy')
        _close_quietly(env, 'env')
        restore_shutdown_handlers()

def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--teleop', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--ssl', action='store_true', help='Use HTTPS instead of HTTP (required for WebXR on some devices)')
    parser.add_argument('--uarm', action='store_true', help='Use Zhonglin U-Arm as the only teleop input source')
    parser.add_argument('--gamepad', action='store_true', help='Use Logitech F710 gamepad as the only teleop input source')
    parser.add_argument('--gamepad-debug', action='store_true', help='Print Logitech F710 raw axes/buttons and mapped motion deltas')
    parser.add_argument('--output-dir', default='data/demos')
    parser.add_argument('--lerobot-root', default=LEROBOT_ROOT, help='LeRobot dataset root for --teleop --gamepad --save')
    parser.add_argument('--lerobot-repo-id', default=LEROBOT_REPO_ID, help='LeRobot repo id for --teleop --gamepad --save')
    parser.add_argument('--lerobot-task', default=LEROBOT_TASK, help='LeRobot task name for --teleop --gamepad --save')
    parser.add_argument('--lerobot-fps', type=int, default=LEROBOT_FPS, help='LeRobot FPS for --teleop --gamepad --save')
    return parser


def validate_args(parser, args):
    if args.uarm and not args.teleop:
        parser.error('--uarm requires --teleop')
    if args.gamepad and not args.teleop:
        parser.error('--gamepad requires --teleop')
    if args.uarm and args.gamepad:
        parser.error('--uarm and --gamepad are mutually exclusive')


if __name__ == '__main__':
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    main(args)
