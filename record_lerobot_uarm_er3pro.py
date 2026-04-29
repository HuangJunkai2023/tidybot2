import argparse
import inspect
import select
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from constants import BASE_CAMERA_DEVICE, BASE_CAMERA_HEIGHT, BASE_CAMERA_WIDTH
from constants import ER3PRO_ENABLE_GRIPPER, ER3PRO_GRIPPER_BACKEND, ER3PRO_GRIPPER_BOARD
from constants import ER3PRO_GRIPPER_DI1_PORT, ER3PRO_GRIPPER_DI2_PORT
from constants import ER3PRO_GRIPPER_RS485_CLOSE_POS, ER3PRO_GRIPPER_RS485_ENABLE_ON_START
from constants import ER3PRO_GRIPPER_RS485_INIT_REG, ER3PRO_GRIPPER_RS485_INIT_VALUE
from constants import ER3PRO_GRIPPER_RS485_OPEN_POS, ER3PRO_GRIPPER_RS485_POS_REG
from constants import ER3PRO_GRIPPER_RS485_SLAVE_ID, ER3PRO_GRIPPER_RS485_SPEED
from constants import ER3PRO_GRIPPER_RS485_TORQUE, ER3PRO_GRIPPER_RS485_TORQUE_REG
from constants import ER3PRO_GRIPPER_THRESHOLD
from constants import ER3PRO_IP, ER3PRO_LOCAL_IP, ER3PRO_MOVE_VELOCITY, ER3PRO_MOVE_ZONE
from constants import ER3PRO_TELEOP_PRESET_JOINT_DEG, ER3PRO_UARM_RT_BIN
from constants import LEROBOT_FPS, LEROBOT_REPO_ID, LEROBOT_ROOT, LEROBOT_TASK
from constants import UARM_BAUDRATE, UARM_GRIPPER_CLOSE_DEG, UARM_GRIPPER_OPEN_DEG
from constants import UARM_JOINT_LIMIT_DEG_MAX, UARM_JOINT_LIMIT_DEG_MIN
from constants import UARM_JOINT_OFFSET_DEG, UARM_JOINT_SCALE, UARM_JOINT_SIGN
from constants import UARM_MAX_FRAME_DELTA_DEG, UARM_MAX_JOINT_ACCEL_DEG, UARM_MAX_JOINT_SPEED_DEG
from constants import UARM_RT_COMMAND_DELAY_US, UARM_RT_DEADBAND_DEG, UARM_RT_FILTER_ALPHA
from constants import UARM_RT_FILTER_FREQ, UARM_RT_INTERP_HZ, UARM_RT_INTERP_STEPS, UARM_RT_READ_TIMEOUT_US
from constants import UARM_RT_ROBOT_TARGET_DEADBAND_DEG, UARM_RT_ROBOT_TARGET_FILTER_HZ
from constants import UARM_RT_SERVO_PERIOD_MS, UARM_RT_SERVOJ_KP, UARM_RT_STALE_TIMEOUT
from constants import UARM_RT_STATUS_HZ, UARM_RT_STEP_DEADBAND_DEG, UARM_SERIAL_PORT
from constants import USE_KINOVA_WRIST_CAMERA, WRIST_CAMERA_DEVICE, WRIST_CAMERA_HEIGHT, WRIST_CAMERA_WIDTH


STATE_DIM = 15
ACTION_DIM = 8


class LatestBridgeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.ready = False
        self.last_error = None
        self.last_update_wall = 0.0
        self.measured_joints = np.deg2rad(ER3PRO_TELEOP_PRESET_JOINT_DEG.astype(np.float64))
        self.tcp_pos = np.zeros(3, dtype=np.float64)
        self.tcp_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self.target_joints = self.measured_joints.copy()
        self.gripper = 1.0
        self.uarm_deg = np.zeros(8, dtype=np.float64)
        self.uarm_age_ms = float("inf")
        self.uarm_period_ms = float("nan")
        self.uarm_frames = 0
        self.serial_errors = 0

    def update_from_line(self, line):
        items = line.split()
        if not items:
            return
        if items[0] == "READY":
            with self.lock:
                self.ready = True
            return
        if items[0] == "ERR":
            with self.lock:
                self.last_error = line
            return
        if items[0] != "STATE":
            return
        if len(items) != 36:
            with self.lock:
                self.last_error = f"bad STATE field count: {len(items)} line={line}"
            return

        vals = [float(v) for v in items[1:34]]
        with self.lock:
            self.measured_joints = np.asarray(vals[1:8], dtype=np.float64)
            self.tcp_pos = np.asarray(vals[8:11], dtype=np.float64)
            quat = np.asarray(vals[11:15], dtype=np.float64)
            norm = np.linalg.norm(quat)
            self.tcp_quat = quat / norm if norm > 1e-9 else np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            if self.tcp_quat[3] < 0.0:
                np.negative(self.tcp_quat, out=self.tcp_quat)
            self.target_joints = np.asarray(vals[15:22], dtype=np.float64)
            self.gripper = float(np.clip(vals[22], 0.0, 1.0))
            self.uarm_deg = np.asarray(vals[23:31], dtype=np.float64)
            self.uarm_age_ms = vals[31]
            self.uarm_period_ms = vals[32]
            self.uarm_frames = int(float(items[34]))
            self.serial_errors = int(float(items[35]))
            self.last_update_wall = time.time()

    def snapshot(self):
        with self.lock:
            return {
                "ready": self.ready,
                "last_error": self.last_error,
                "last_update_wall": self.last_update_wall,
                "measured_joints": self.measured_joints.copy(),
                "tcp_pos": self.tcp_pos.copy(),
                "tcp_quat": self.tcp_quat.copy(),
                "target_joints": self.target_joints.copy(),
                "gripper": self.gripper,
                "uarm_deg": self.uarm_deg.copy(),
                "uarm_age_ms": self.uarm_age_ms,
                "uarm_period_ms": self.uarm_period_ms,
                "uarm_frames": self.uarm_frames,
                "serial_errors": self.serial_errors,
            }


class BridgeProcess:
    def __init__(self, args):
        self.args = args
        self.state = LatestBridgeState()
        self.proc = None
        self.stdout_thread = None
        self.stderr_thread = None
        self.last_uarm_print_time = 0.0
        if not args.no_robot:
            self._start()
        else:
            self.state.ready = True

    def _start(self):
        bridge_path = Path(self.args.bridge_bin)
        if not bridge_path.is_absolute():
            bridge_path = Path(__file__).resolve().parent / bridge_path
        if not bridge_path.exists():
            raise RuntimeError(f"uarm_er3pro_rt not found: {bridge_path}")

        cmd = [
            str(bridge_path),
            "--robot-ip", self.args.robot_ip,
            "--uarm-port", self.args.uarm_port,
            "--uarm-baud", str(self.args.uarm_baud),
            "--read-timeout-us", str(self.args.read_timeout_us),
            "--uarm-command-delay-us", str(self.args.uarm_command_delay_us),
            "--servo-period-ms", str(self.args.servo_period_ms),
            "--status-hz", str(self.args.status_hz),
            "--stale-timeout", str(self.args.stale_timeout),
            "--max-frame-delta-deg", str(self.args.max_frame_delta_deg),
            "--filter-freq", str(self.args.filter_freq),
            "--servoj-kp", str(self.args.servoj_kp),
            "--uarm-deadband-deg", str(self.args.uarm_deadband_deg),
            "--uarm-filter-alpha", str(self.args.uarm_filter_alpha),
            "--uarm-step-deadband-deg", str(self.args.uarm_step_deadband_deg),
            "--uarm-interp-steps", str(self.args.uarm_interp_steps),
            "--uarm-interp-hz", str(self.args.uarm_interp_hz),
            "--robot-target-filter-hz", str(self.args.robot_target_filter_hz),
            "--robot-target-deadband-deg", str(self.args.robot_target_deadband_deg),
            "--speed", str(self.args.speed),
            "--zone", str(self.args.zone),
            "--preset-joints-deg", csv(ER3PRO_TELEOP_PRESET_JOINT_DEG),
            "--joint-sign", csv(UARM_JOINT_SIGN),
            "--joint-scale", csv(np.asarray(self.args.joint_scale, dtype=np.float64)),
            "--joint-offset-deg", csv(UARM_JOINT_OFFSET_DEG),
            "--joint-min-deg", csv(UARM_JOINT_LIMIT_DEG_MIN),
            "--joint-max-deg", csv(UARM_JOINT_LIMIT_DEG_MAX),
            "--max-speed-deg", csv(np.asarray(self.args.max_speed_deg, dtype=np.float64)),
            "--max-accel-deg", csv(np.asarray(self.args.max_accel_deg, dtype=np.float64)),
            "--gripper-open-deg", str(UARM_GRIPPER_OPEN_DEG),
            "--gripper-close-deg", str(UARM_GRIPPER_CLOSE_DEG),
            "--gripper-backend", ER3PRO_GRIPPER_BACKEND if ER3PRO_GRIPPER_BACKEND in ("di", "rs485_epg") else "rs485_epg",
            "--gripper-threshold", str(ER3PRO_GRIPPER_THRESHOLD),
            "--gripper-board", str(ER3PRO_GRIPPER_BOARD),
            "--gripper-di1-port", str(ER3PRO_GRIPPER_DI1_PORT),
            "--gripper-di2-port", str(ER3PRO_GRIPPER_DI2_PORT),
            "--gripper-rs485-slave-id", str(ER3PRO_GRIPPER_RS485_SLAVE_ID),
            "--gripper-rs485-init-reg", str(ER3PRO_GRIPPER_RS485_INIT_REG),
            "--gripper-rs485-init-value", str(ER3PRO_GRIPPER_RS485_INIT_VALUE),
            "--gripper-rs485-torque-reg", str(ER3PRO_GRIPPER_RS485_TORQUE_REG),
            "--gripper-rs485-pos-reg", str(ER3PRO_GRIPPER_RS485_POS_REG),
            "--gripper-rs485-open-pos", str(ER3PRO_GRIPPER_RS485_OPEN_POS),
            "--gripper-rs485-close-pos", str(ER3PRO_GRIPPER_RS485_CLOSE_POS),
            "--gripper-rs485-speed", str(ER3PRO_GRIPPER_RS485_SPEED),
            "--gripper-rs485-torque", str(ER3PRO_GRIPPER_RS485_TORQUE),
        ]
        if not ER3PRO_ENABLE_GRIPPER:
            cmd.append("--disable-gripper")
        if ER3PRO_GRIPPER_RS485_ENABLE_ON_START:
            cmd.append("--gripper-rs485-enable-on-start")
        if self.args.local_ip:
            cmd.extend(["--local-ip", self.args.local_ip])
        if self.args.dry_run:
            cmd.append("--dry-run")
        if self.args.skip_preset:
            cmd.append("--skip-preset")
        if self.args.use_preset:
            cmd.append("--use-preset")

        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.stdout_thread = threading.Thread(target=self._stdout_loop, daemon=True)
        self.stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self.stdout_thread.start()
        self.stderr_thread.start()
        self._wait_ready(timeout=20.0)

    def _stdout_loop(self):
        for line in self.proc.stdout:
            line = line.strip()
            self.state.update_from_line(line)
            if line.startswith("STATE"):
                self._maybe_print_uarm_angles()

    def _maybe_print_uarm_angles(self):
        if not self.args.print_uarm_angles:
            return
        now = time.time()
        period = 1.0 / max(float(self.args.uarm_print_hz), 0.1)
        if now - self.last_uarm_print_time < period:
            return
        self.last_uarm_print_time = now
        snap = self.state.snapshot()
        angles = " ".join(f"s{i}:{angle:.1f}" for i, angle in enumerate(snap["uarm_deg"]))
        print(
            f"UARM {angles} age_ms:{snap['uarm_age_ms']:.1f} period_ms:{snap['uarm_period_ms']:.1f}",
            flush=True,
        )

    def _stderr_loop(self):
        for line in self.proc.stderr:
            line = line.strip()
            if line:
                print(f"[uarm_er3pro_rt] {line}", file=sys.stderr, flush=True)
                if line.startswith("ERR"):
                    with self.state.lock:
                        self.state.last_error = line

    def _wait_ready(self, timeout):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"uarm_er3pro_rt exited with code {self.proc.returncode}")
            snap = self.state.snapshot()
            if snap["ready"]:
                return
            if snap["last_error"]:
                raise RuntimeError(snap["last_error"])
            time.sleep(0.02)
        raise RuntimeError("timeout waiting for uarm_er3pro_rt READY")

    def snapshot(self):
        if self.args.no_robot:
            t = time.time()
            joints = np.deg2rad(ER3PRO_TELEOP_PRESET_JOINT_DEG.astype(np.float64))
            with self.state.lock:
                self.state.last_update_wall = t
                self.state.measured_joints = joints
                self.state.target_joints = joints
                self.state.tcp_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                self.state.tcp_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
                self.state.gripper = 1.0
        return self.state.snapshot()

    def close(self):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def csv(values):
    return ",".join(str(float(v)) for v in values)


def make_camera(device, width, height, dummy=False):
    from cameras import DummyCamera, UVCCamera

    if dummy or str(device).strip() == "TODO":
        return DummyCamera(frame_width=width, frame_height=height)
    if str(device).strip().isdigit() or str(device).startswith("/dev/"):
        return UVCCamera(device, frame_width=width, frame_height=height)
    from cameras import LogitechCamera
    return LogitechCamera(device, frame_width=width, frame_height=height)


def import_lerobot_dataset():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset
    except Exception:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        return LeRobotDataset


def call_with_supported_kwargs(fn, **kwargs):
    sig = inspect.signature(fn)
    accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if accepts_kwargs:
        return fn(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)


def is_complete_lerobot_root(root):
    meta_dir = root / "meta"
    return (
        (meta_dir / "info.json").exists()
        and (meta_dir / "tasks.parquet").exists()
        and (meta_dir / "episodes.parquet").exists()
    )


def unique_lerobot_root(root):
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    candidate = root.with_name(f"{root.name}_{stamp}")
    idx = 1
    while candidate.exists():
        candidate = root.with_name(f"{root.name}_{stamp}_{idx}")
        idx += 1
    return candidate


def create_lerobot_dataset(args, base_shape, wrist_shape):
    LeRobotDataset = import_lerobot_dataset()
    features = {
        "observation.images.base": {
            "dtype": "image",
            "shape": tuple(base_shape),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": tuple(wrist_shape),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": [
                "joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6",
                "tcp_x", "tcp_y", "tcp_z", "tcp_qx", "tcp_qy", "tcp_qz", "tcp_qw", "gripper",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"],
        },
    }
    root = Path(args.root).expanduser()
    kwargs = {
        "repo_id": args.repo_id,
        "fps": args.fps,
        "root": root,
        "robot_type": "er3pro_uarm",
        "features": features,
        "use_videos": True,
        "image_writer_threads": 4,
        "image_writer_processes": 0,
    }
    if root.exists():
        if is_complete_lerobot_root(root):
            print(f"Loading existing LeRobot dataset root: {root}", flush=True)
            return call_with_supported_kwargs(LeRobotDataset, repo_id=args.repo_id, root=root)
        new_root = unique_lerobot_root(root)
        print(
            f"Existing LeRobot root is incomplete, creating a new root instead: {new_root}",
            flush=True,
        )
        kwargs["root"] = new_root
    return call_with_supported_kwargs(LeRobotDataset.create, **kwargs)


def add_frame(dataset, frame):
    dataset.add_frame(frame)


def save_episode(dataset, task):
    try:
        dataset.save_episode(task=task)
    except TypeError:
        dataset.save_episode()


def finalize_dataset(dataset):
    if hasattr(dataset, "finalize"):
        dataset.finalize()


def latest_image(camera, fallback, name):
    image = camera.get_image()
    if image is None:
        if fallback is None:
            raise RuntimeError(f"{name} camera has no frame yet")
        return fallback, fallback
    return image, image


def wait_initial_image(camera, name, timeout=5.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        image = camera.get_image()
        if image is not None:
            return image
        last = image
        time.sleep(0.02)
    raise RuntimeError(f"timeout waiting for {name} camera")


def key_ready():
    return bool(select.select([sys.stdin], [], [], 0.0)[0])


def read_key_line():
    if not key_ready():
        return None
    return sys.stdin.readline().rstrip("\n")


def build_frame(snap, base_image, wrist_image, task):
    state = np.concatenate((
        snap["measured_joints"],
        snap["tcp_pos"],
        snap["tcp_quat"],
        np.array([snap["gripper"]], dtype=np.float64),
    )).astype(np.float32)
    action = np.concatenate((
        snap["target_joints"],
        np.array([snap["gripper"]], dtype=np.float64),
    )).astype(np.float32)
    return {
        "observation.images.base": base_image,
        "observation.images.wrist": wrist_image,
        "observation.state": state,
        "action": action,
        "task": task,
    }


def print_status(prefix, frames, snap):
    print(
        f"{prefix} frames={frames} "
        f"uarm_age_ms={snap['uarm_age_ms']:.1f} "
        f"uarm_period_ms={snap['uarm_period_ms']:.1f} "
        f"serial_errors={snap['serial_errors']}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=LEROBOT_REPO_ID)
    parser.add_argument("--root", default=LEROBOT_ROOT)
    parser.add_argument("--task", default=LEROBOT_TASK)
    parser.add_argument("--fps", type=int, default=LEROBOT_FPS)
    parser.add_argument("--bridge-bin", default=ER3PRO_UARM_RT_BIN)
    parser.add_argument("--robot-ip", default=ER3PRO_IP)
    parser.add_argument("--local-ip", default=ER3PRO_LOCAL_IP)
    parser.add_argument("--uarm-port", default=UARM_SERIAL_PORT)
    parser.add_argument("--uarm-baud", type=int, default=UARM_BAUDRATE)
    parser.add_argument("--read-timeout-us", type=int, default=UARM_RT_READ_TIMEOUT_US)
    parser.add_argument("--uarm-command-delay-us", type=int, default=UARM_RT_COMMAND_DELAY_US)
    parser.add_argument("--servo-period-ms", type=float, default=UARM_RT_SERVO_PERIOD_MS)
    parser.add_argument("--status-hz", type=float, default=UARM_RT_STATUS_HZ)
    parser.add_argument("--stale-timeout", type=float, default=UARM_RT_STALE_TIMEOUT)
    parser.add_argument("--max-frame-delta-deg", type=float, default=UARM_MAX_FRAME_DELTA_DEG)
    parser.add_argument("--filter-freq", type=float, default=UARM_RT_FILTER_FREQ)
    parser.add_argument("--servoj-kp", type=float, default=UARM_RT_SERVOJ_KP)
    parser.add_argument("--uarm-deadband-deg", type=float, default=UARM_RT_DEADBAND_DEG)
    parser.add_argument("--uarm-filter-alpha", type=float, default=UARM_RT_FILTER_ALPHA)
    parser.add_argument("--uarm-step-deadband-deg", type=float, default=UARM_RT_STEP_DEADBAND_DEG)
    parser.add_argument("--uarm-interp-steps", type=int, default=UARM_RT_INTERP_STEPS)
    parser.add_argument("--uarm-interp-hz", type=float, default=UARM_RT_INTERP_HZ)
    parser.add_argument("--robot-target-filter-hz", type=float, default=UARM_RT_ROBOT_TARGET_FILTER_HZ)
    parser.add_argument("--robot-target-deadband-deg", type=float, default=UARM_RT_ROBOT_TARGET_DEADBAND_DEG)
    parser.add_argument("--speed", type=float, default=ER3PRO_MOVE_VELOCITY)
    parser.add_argument("--zone", type=float, default=ER3PRO_MOVE_ZONE)
    parser.add_argument("--joint-scale", type=float, nargs=7, default=UARM_JOINT_SCALE.tolist())
    parser.add_argument("--max-speed-deg", type=float, nargs=7, default=UARM_MAX_JOINT_SPEED_DEG.tolist())
    parser.add_argument("--max-accel-deg", type=float, nargs=7, default=UARM_MAX_JOINT_ACCEL_DEG.tolist())
    parser.add_argument("--dry-run", action="store_true", help="Start C++ bridge without connecting to robot")
    parser.add_argument("--no-robot", action="store_true", help="Do not start C++ bridge; record synthetic robot state")
    parser.add_argument("--skip-preset", action="store_true")
    parser.add_argument("--use-preset", action="store_true", help="Move ER3Pro to ER3PRO_TELEOP_PRESET_JOINT_DEG before realtime teleop")
    parser.add_argument("--print-uarm-angles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--uarm-print-hz", type=float, default=UARM_RT_STATUS_HZ)
    parser.add_argument("--dummy-cameras", action="store_true")
    parser.add_argument("--auto-seconds", type=float, default=0.0, help="Record one episode for N seconds, then save and exit")
    args = parser.parse_args()

    bridge = None
    base_camera = None
    wrist_camera = None
    dataset = None
    try:
        bridge = BridgeProcess(args)
        base_camera = make_camera(BASE_CAMERA_DEVICE, BASE_CAMERA_WIDTH, BASE_CAMERA_HEIGHT, dummy=args.dummy_cameras)
        wrist_camera = (
            __import__("cameras", fromlist=["KinovaCamera"]).KinovaCamera()
            if USE_KINOVA_WRIST_CAMERA and not args.dummy_cameras
            else make_camera(WRIST_CAMERA_DEVICE, WRIST_CAMERA_WIDTH, WRIST_CAMERA_HEIGHT, dummy=args.dummy_cameras)
        )
        base_image = wait_initial_image(base_camera, "base")
        wrist_image = wait_initial_image(wrist_camera, "wrist")
        dataset = create_lerobot_dataset(args, base_image.shape, wrist_image.shape)

        print("READY")
        print("Press Enter to start, s+Enter to save, d+Enter to discard, q+Enter to quit.", flush=True)

        recording = False
        episode_frames = 0
        next_frame_time = time.monotonic()
        auto_end_time = None
        last_status_time = 0.0

        while True:
            cmd = read_key_line()
            if cmd is not None:
                if cmd == "" and not recording:
                    recording = True
                    episode_frames = 0
                    next_frame_time = time.monotonic()
                    auto_end_time = time.monotonic() + args.auto_seconds if args.auto_seconds > 0 else None
                    print("EPISODE_STARTED", flush=True)
                elif cmd == "s" and recording:
                    save_episode(dataset, args.task)
                    recording = False
                    print(f"EPISODE_SAVED frames={episode_frames}", flush=True)
                elif cmd == "d" and recording:
                    if hasattr(dataset, "clear_episode_buffer"):
                        dataset.clear_episode_buffer()
                    else:
                        print("Warning: this LeRobot version has no clear_episode_buffer(); restart if discard is required.", flush=True)
                    recording = False
                    print(f"EPISODE_DISCARDED frames={episode_frames}", flush=True)
                elif cmd == "q":
                    break

            if not recording:
                time.sleep(0.01)
                continue

            now = time.monotonic()
            if auto_end_time is not None and now >= auto_end_time:
                save_episode(dataset, args.task)
                print(f"EPISODE_SAVED frames={episode_frames}", flush=True)
                break
            if now < next_frame_time:
                time.sleep(min(0.002, next_frame_time - now))
                continue

            snap = bridge.snapshot()
            if snap["last_error"]:
                raise RuntimeError(snap["last_error"])
            base_image, _ = latest_image(base_camera, base_image, "base")
            wrist_image, _ = latest_image(wrist_camera, wrist_image, "wrist")
            frame = build_frame(snap, base_image, wrist_image, args.task)
            add_frame(dataset, frame)
            episode_frames += 1

            if time.monotonic() - last_status_time > 1.0:
                print_status("RECORDING", episode_frames, snap)
                last_status_time = time.monotonic()

            next_frame_time += 1.0 / float(args.fps)
    finally:
        if dataset is not None:
            finalize_dataset(dataset)
        if base_camera is not None:
            base_camera.close()
        if wrist_camera is not None:
            wrist_camera.close()
        if bridge is not None:
            bridge.close()


if __name__ == "__main__":
    main()
