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
    # 【退出管理】输入资源对象和用于日志的名称，尝试调用 close() 释放资源。
    # 无返回值；资源为空或没有 close() 时跳过，关闭异常只打印，避免打断后续清理。
    if resource is None or not hasattr(resource, 'close'):
        return
    try:
        resource.close()
    except Exception as e:
        print(f'Warning: error while closing {name}: {e}', flush=True)


def install_shutdown_handlers(get_resources):
    # 【启动/退出管理】输入一个可调用对象，调用它可取得当前的 (env, policy)。
    # 保存并替换进程的信号处理方式，使中断/退出时先清理资源；返回恢复旧处理方式的函数。
    # 内部回调在这里被定义、注册，收到对应信号时才执行；此处不会立即关闭环境。
    shutting_down = {'active': False}
    previous_handlers = {}

    def cleanup():
        # 【退出管理】无显式参数，通过外层 get_resources() 获取最新的环境和策略。
        # 依次关闭策略和环境，无返回值；延迟获取可处理它们尚未创建或已经创建的情况。
        env, policy = get_resources()
        _close_quietly(policy, 'policy')
        _close_quietly(env, 'env')

    def handle_shutdown(signum, frame):
        # 【退出管理】信号回调：signum 是信号编号，frame 是被中断位置的栈帧（此处未使用）。
        # 将外层 active 标志置为 True，避免本回调重复清理；随后抛出异常结束正常执行。
        # Ctrl+C 对应 KeyboardInterrupt，其余注册信号对应 SystemExit，没有正常返回值。
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
        # 【退出管理】无显式输入，使用外层保存的 previous_handlers 恢复原来的信号处理方式。
        # 无返回值；由 main() 的 finally 调用，撤销本次安装对进程信号配置的修改。
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    return restore


def move_arm_to_teleop_preset(env):
    # 【action / 运行前准备】输入环境，仅在启用 ER3Pro 且环境有机械臂时请求移动到预设姿态。
    # 无返回值；可能改变实际机械臂状态，为示教/执行提供起始姿态，其余情况直接跳过。
    if not ENABLE_ARM or ARM_BACKEND != 'er3pro':
        return

    if getattr(env, 'arm', None) is None:
        return

    print('Moving arm to teleop preset pose...')
    env.arm.move_to_teleop_preset()
    print('Arm reached teleop preset pose')

def should_save_episode(writer):
    # 【recording】输入记录器，返回是否保存本轮数据的布尔值；空记录直接返回 False。
    # 非空时等待终端输入；选择不保存时可能调用 writer.discard() 丢弃缓存。
    # 本函数只决定是否保存，实际写盘由调用者发起。
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
    # 【recording】输入本步观测和动作，返回用于记录的观测；不重新读取环境。
    # 动作为字典时浅复制 obs；按配置可用动作中的机械臂/夹爪目标替换记录字段，保存示教意图。
    # 不修改原 obs 字典；未替换的值仍共享引用。非字典动作则直接返回原 obs。
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
    # 【action / 接口适配】输入观测和策略输出，返回环境可识别的动作字段格式。
    # 浅复制动作字典、统一关节字段名；缺少底盘目标时，用当前观测中的底盘位姿补齐。
    # 不直接执行动作，也不修改原字典；非字典输出原样返回。这不是数值归一化。
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
    # 【控制循环：observation -> policy -> action，并按需 recording】
    # 输入环境、动作来源、可选记录器和调试开关；重置后反复获取观测、产生动作、下发并记录。
    # 会改变环境/策略/记录器及本轮统计状态；正常结束无返回值，中断时处理保存后继续抛出异常。
    # end_episode 只结束本轮记录，循环仍可执行动作；收到 reset_env 才跳出循环，交回 main()。
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
        # 【运行诊断】输入本步各阶段耗时（毫秒），并读取外层 env 暴露的细分耗时。
        # 累加外层 profile 的次数、总耗时与最大值，无返回值；用于观察控制循环的时间开销。
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
        # 【运行诊断】无显式参数，检查统计开关和打印间隔，按需输出循环频率与耗时。
        # 打印后清零外层 profile 的窗口统计并更新时间；无返回值，不负责控制循环频率。
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
        # 【action / 运行诊断】输入观测、动作和下发耗时，按需打印目标、目标与观测之差及夹爪命令。
        # 更新外层最近打印时间，最多每 0.5 秒打印一次；无返回值，不修改观测或动作。
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

    # 运行前准备：请求环境重置，再重置策略/输入源；具体实现由传入对象提供。
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
    # start_time 是本轮循环的固定时间基准，不在每步完成后重新赋值。
    start_time = time.time()
    profile['last_time'] = start_time
    try:
        for step_idx in count():
            # step_idx 从 0 开始递增；即使本步没有动作，下一轮编号仍会加一。
            loop_start_time = time.time()

            # POLICY_CONTROL_PERIOD 是目标周期（秒），目标频率为 1 / 周期。
            # step_end_time 虽然名字含 end，实际用于等待本步开始读取观测的计划时刻。
            # 各步对齐到 start_time + 0*T、1*T、2*T……，避免每步额外睡满 T 导致累计延迟。
            step_end_time = start_time + step_idx * POLICY_CONTROL_PERIOD
            # 提前到达就短暂休眠；已经超时就不等待，也不跳过步号，随后可能连续追赶。
            # 执行耗时、系统调度和 time.time() 的时钟变化都会影响频率，不保证严格实时。
            while time.time() < step_end_time:
                time.sleep(0.0001)

            # observation：从环境取得本步观测，赋给 obs；这里还没有产生本步动作。
            get_obs_start = time.time()
            obs = env.get_obs()
            get_obs_ms = 1000.0 * (time.time() - get_obs_start)

            # policy：把 obs 交给动作来源（人工遥操作或远程策略），得到 action。
            # 输出也可能是 None 或控制流程用的字符串，所以下面需要分类处理。
            policy_start = time.time()
            action = policy.step(obs)
            policy_ms = 1000.0 * (time.time() - policy_start)

            # 没有动作（例如尚未使能）时，跳过执行与记录，下一轮仍按计划获取观测。
            if action is None:
                update_profile(
                    get_obs_ms,
                    policy_ms,
                    0.0,
                    1000.0 * (time.time() - loop_start_time),
                )
                maybe_print_profile()
                continue

            # action：字典输出先做字段适配，再由 env.step(action) 下发到环境。
            # 此处不接收新观测；环境变化后的观测在下一轮 env.get_obs() 获取。
            if isinstance(action, dict):
                action = _normalize_action_for_env(obs, action)
                env_step_start = time.time()
                env.step(action)
                env_step_ms = 1000.0 * (time.time() - env_step_start)
                maybe_print_action_debug(obs, action, env_step_ms)

                if writer is not None and not episode_ended:
                    # recording：动作下发后记录本步观测和动作；obs 来自下发前，不是执行后的新观测。
                    writer.step(_build_logged_observation(obs, action), action)

            # 标记记录结束并询问是否保存；这里没有 break，仍允许后续动作执行。
            elif not episode_ended and action == 'end_episode':
                episode_ended = True
                print('Episode ended')

                if writer is not None and should_save_episode(writer):
                    # Save to disk in background thread
                    writer.flush_async()

                print('Teleop is now active. Press "Reset env" in the web app when ready to proceed.')

            # 收到重置信号才离开本函数的循环，由 main() 再次调用 run_episode()。
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
    # 【recording / 创建记录器】输入解析后的启动参数；不保存时返回 None。
    # 保存且使用手柄时创建 LeRobotEpisodeWriter，否则创建 EpisodeWriter；具体初始化由类负责。
    # 返回的对象供主循环逐步记录；此函数不采集观测，也不生成动作。
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
    # 【整体调度】输入启动参数，选择实机/仿真环境、人工/远程策略和记录器，反复运行 episode。
    # 管理这些对象的生命周期，退出时清理并恢复信号处理；不向调用者返回运行结果。
    # 它连接所需的外部服务，不负责启动全部硬件服务、Docker 或 GPU 推理服务。
    env = None
    policy = None
    # lambda 在被调用时读取当前 env、policy，避免只保存初始化时的两个 None。
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

        # policy 表示动作来源：可以是人工输入，也可以是远程策略客户端，不一定是本地模型。
        if args.teleop:
            if args.uarm:
                policy = UarmTeleopPolicy(env.arm)
            elif args.gamepad:
                policy = GamepadTeleopPolicy(use_ssl=args.ssl, debug=args.gamepad_debug)
            else:
                policy = TeleopPolicy(use_ssl=args.ssl)
        else:
            policy = RemotePolicy(use_ssl=args.ssl)

        # 手柄保存模式跨 episode 复用记录器；其他模式每次进入 episode 前创建或返回 None。
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
    # 【启动准备】无显式输入，创建并返回定义好命令行选项的 ArgumentParser。
    # add_argument() 修改的是新建解析器的规则；此处尚未读取本次命令行，也没有启动机器人。
    # 集中定义开关和默认值，让同一入口能选择不同运行方式。
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
    # 【启动准备】输入解析器和已解析的参数，检查遥操作选项的依赖与互斥关系。
    # 合法时不修改参数、默认返回 None；非法时 parser.error() 报错退出，避免进入错误运行组合。
    if args.uarm and not args.teleop:
        parser.error('--uarm requires --teleop')
    if args.gamepad and not args.teleop:
        parser.error('--gamepad requires --teleop')
    if args.uarm and args.gamepad:
        parser.error('--uarm and --gamepad are mutually exclusive')


if __name__ == '__main__':
    # 直接运行此脚本时的入口：定义参数规则 -> 读取命令行 -> 检查组合 -> 进入主调度。
    # 导入此模块时不会执行这个入口块。
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    main(args)
