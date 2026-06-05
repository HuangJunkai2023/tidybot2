# Author: Jimmy Wu
# Date: October 2024

import pickle
import threading
import time
from datetime import datetime
from pathlib import Path
import inspect
import cv2 as cv
import numpy as np
from constants import LEROBOT_FPS, LEROBOT_REPO_ID, LEROBOT_ROOT, LEROBOT_TASK
from constants import POLICY_CONTROL_FREQ

LEROBOT_STATE_DIM = 15
LEROBOT_ACTION_DIM = 8

def write_frames_to_mp4(frames, mp4_path):
    height, width, _ = frames[0].shape
    codec_candidates = ['mp4v', 'avc1', 'H264']
    out = None
    chosen_codec = None

    for codec in codec_candidates:
        fourcc = cv.VideoWriter_fourcc(*codec)
        writer = cv.VideoWriter(str(mp4_path), fourcc, POLICY_CONTROL_FREQ, (width, height))
        if writer.isOpened():
            out = writer
            chosen_codec = codec
            break
        writer.release()

    if out is None:
        raise RuntimeError(f'Failed to open VideoWriter for {mp4_path}. Tried codecs: {codec_candidates}')

    for frame in frames:
        bgr_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        out.write(bgr_frame)
    out.release()
    print(f'Video saved with codec {chosen_codec}: {mp4_path.name}')

def read_frames_from_mp4(mp4_path):
    cap = cv.VideoCapture(str(mp4_path))
    frames = []
    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            break
        frames.append(cv.cvtColor(bgr_frame, cv.COLOR_BGR2RGB))
    cap.release()
    return frames

class EpisodeWriter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.episode_dir = self.output_dir / datetime.now().strftime('%Y%m%dT%H%M%S%f')
        assert not self.episode_dir.exists()

        # Episode data
        self.timestamps = []
        self.observations = []
        self.actions = []

        # Write to disk in separate thread to avoid blocking main thread
        self.flush_thread = None

    def step(self, obs, action):
        if len(self.observations) == 0 and not np.allclose(obs['base_pose'], 0.0, atol=0.01):
            raise Exception('Initial base pose should be zero. Did the base get pushed?')
        self.timestamps.append(time.time())
        self.observations.append(obs)
        self.actions.append(action)

    def __len__(self):
        return len(self.observations)

    def _flush(self):
        assert len(self) > 0

        # Create episode dir
        self.episode_dir.mkdir(parents=True)

        # Extract image observations
        frames_dict = {}
        for obs in self.observations:
            for k, v in obs.items():
                if v.ndim == 3:
                    if k not in frames_dict:
                        frames_dict[k] = []
                    frames_dict[k].append(v)
                    obs[k] = None

        # Write images as MP4 videos
        for k, frames in frames_dict.items():
            mp4_path = self.episode_dir / f'{k}.mp4'
            write_frames_to_mp4(frames, mp4_path)

        # Write rest of episode data
        with open(self.episode_dir / 'data.pkl', 'wb') as f:  # Note: Not secure. Only unpickle data you trust.
            pickle.dump({'timestamps': self.timestamps, 'observations': self.observations, 'actions': self.actions}, f)
        num_episodes = len([child for child in self.output_dir.iterdir() if child.is_dir()])
        print(f'Saved episode to {self.episode_dir} ({num_episodes} total)')

    def flush_async(self):
        print('Saving successful episode to disk...')
        # Note: Disk writes may cause latency spikes in low-level controllers
        self.flush_thread = threading.Thread(target=self._flush, daemon=True)
        self.flush_thread.start()

    def wait_for_flush(self):
        if self.flush_thread is not None:
            self.flush_thread.join()
            self.flush_thread = None


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
    return fn(**{k: v for k, v in kwargs.items() if k in sig.parameters})


def is_complete_lerobot_root(root):
    meta_dir = root / 'meta'
    episode_files = list((meta_dir / 'episodes').glob('chunk-*/*.parquet'))
    return (
        (meta_dir / 'info.json').exists()
        and (meta_dir / 'tasks.parquet').exists()
        and ((meta_dir / 'episodes.parquet').exists() or bool(episode_files))
    )


def unique_lerobot_root(root):
    stamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    candidate = root.with_name(f'{root.name}_{stamp}')
    idx = 1
    while candidate.exists():
        candidate = root.with_name(f'{root.name}_{stamp}_{idx}')
        idx += 1
    return candidate


def create_lerobot_dataset(repo_id, root, fps, base_shape, wrist_shape, video_codec='h264', resume_existing=True):
    LeRobotDataset = import_lerobot_dataset()
    base_h, base_w, base_c = base_shape
    wrist_h, wrist_w, wrist_c = wrist_shape
    features = {
        'observation.images.base': {
            'dtype': 'video',
            'shape': (base_c, base_h, base_w),
            'names': ['channels', 'height', 'width'],
        },
        'observation.images.wrist': {
            'dtype': 'video',
            'shape': (wrist_c, wrist_h, wrist_w),
            'names': ['channels', 'height', 'width'],
        },
        'observation.state': {
            'dtype': 'float32',
            'shape': (LEROBOT_STATE_DIM,),
            'names': [
                'joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6',
                'tcp_x', 'tcp_y', 'tcp_z', 'tcp_qx', 'tcp_qy', 'tcp_qz', 'tcp_qw', 'gripper',
            ],
        },
        'action': {
            'dtype': 'float32',
            'shape': (LEROBOT_ACTION_DIM,),
            'names': ['joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'gripper'],
        },
    }
    root = Path(root).expanduser()
    if root.exists() and resume_existing and is_complete_lerobot_root(root):
        print(f'Loading existing LeRobot dataset root: {root}', flush=True)
        return call_with_supported_kwargs(LeRobotDataset, repo_id=repo_id, root=root)
    if root.exists():
        new_root = unique_lerobot_root(root)
        print(f'Existing LeRobot root found, creating a new root instead: {new_root}', flush=True)
        root = new_root
    return call_with_supported_kwargs(
        LeRobotDataset.create,
        repo_id=repo_id,
        fps=fps,
        root=root,
        robot_type='er3pro_uarm',
        features=features,
        use_videos=True,
        image_writer_threads=4,
        image_writer_processes=0,
        vcodec=video_codec,
    )


def _as_vector(data, key, shape):
    value = np.asarray(data[key], dtype=np.float64).reshape(shape)
    if not np.all(np.isfinite(value)):
        raise ValueError(f'{key} contains non-finite values: {value}')
    return value


def build_lerobot_frame(obs, action, task):
    arm_joints = _as_vector(obs, 'arm_joints', (7,))
    arm_pos = _as_vector(obs, 'arm_pos', (3,))
    arm_quat = _as_vector(obs, 'arm_quat', (4,))
    gripper_pos = _as_vector(obs, 'gripper_pos', (1,))

    if 'arm_joints' in action:
        action_joints = _as_vector(action, 'arm_joints', (7,))
    else:
        # Cartesian gamepad teleop has no IK target joints in the action dict.
        # Keep the LeRobot/UArm 8D schema by using the latest observed joints.
        action_joints = arm_joints
    action_gripper = _as_vector(action, 'gripper_pos', (1,))

    state = np.concatenate((arm_joints, arm_pos, arm_quat, gripper_pos)).astype(np.float32)
    action_vec = np.concatenate((action_joints, action_gripper)).astype(np.float32)
    if state.shape != (LEROBOT_STATE_DIM,):
        raise ValueError(f'LeRobot state shape {state.shape} != {(LEROBOT_STATE_DIM,)}')
    if action_vec.shape != (LEROBOT_ACTION_DIM,):
        raise ValueError(f'LeRobot action shape {action_vec.shape} != {(LEROBOT_ACTION_DIM,)}')
    return {
        'observation.images.base': obs['base_image'],
        'observation.images.wrist': obs['wrist_image'],
        'observation.state': state,
        'action': action_vec,
        'task': task,
    }


def save_lerobot_episode(dataset, task):
    try:
        dataset.save_episode(task=task)
    except TypeError:
        dataset.save_episode()


def flush_lerobot_saved_episode(dataset):
    if hasattr(dataset, '_close_writer'):
        dataset._close_writer()
        if hasattr(dataset, '_writer_closed_for_reading'):
            dataset._writer_closed_for_reading = True
    meta = getattr(dataset, 'meta', None)
    if meta is not None and hasattr(meta, '_close_writer'):
        meta._close_writer()
        latest = getattr(meta, 'latest_episode', None)
        if latest is not None:
            chunks_size = int(getattr(meta, 'chunks_size', 1000))
            chunk_idx = int(latest['meta/episodes/chunk_index'][0])
            file_idx = int(latest['meta/episodes/file_index'][0]) + 1
            if file_idx >= chunks_size:
                chunk_idx += 1
                file_idx = 0
            latest['meta/episodes/chunk_index'][0] = chunk_idx
            latest['meta/episodes/file_index'][0] = file_idx


class LeRobotEpisodeWriter:
    def __init__(
        self,
        root=LEROBOT_ROOT,
        repo_id=LEROBOT_REPO_ID,
        task=LEROBOT_TASK,
        fps=LEROBOT_FPS,
        video_codec='h264',
        resume_existing=True,
    ):
        self.root = root
        self.repo_id = repo_id
        self.task = task
        self.fps = fps
        self.video_codec = video_codec
        self.resume_existing = resume_existing
        self.dataset = None
        self.episode_frames = 0
        self.flush_thread = None

    def step(self, obs, action):
        if self.dataset is None:
            self.dataset = create_lerobot_dataset(
                self.repo_id,
                self.root,
                self.fps,
                obs['base_image'].shape,
                obs['wrist_image'].shape,
                video_codec=self.video_codec,
                resume_existing=self.resume_existing,
            )
        self.dataset.add_frame(build_lerobot_frame(obs, action, self.task))
        self.episode_frames += 1

    def __len__(self):
        return self.episode_frames

    def _flush(self):
        assert self.dataset is not None
        assert len(self) > 0
        save_lerobot_episode(self.dataset, self.task)
        flush_lerobot_saved_episode(self.dataset)
        print(f'Saved LeRobot episode to {self.root} frames={self.episode_frames}')
        self.episode_frames = 0

    def flush_async(self):
        print('Saving successful LeRobot episode to disk...')
        self.flush_thread = threading.Thread(target=self._flush, daemon=True)
        self.flush_thread.start()

    def wait_for_flush(self):
        if self.flush_thread is not None:
            self.flush_thread.join()
            self.flush_thread = None

    def discard(self):
        if self.dataset is not None and hasattr(self.dataset, 'clear_episode_buffer'):
            self.dataset.clear_episode_buffer()
        self.episode_frames = 0

    def close(self):
        self.wait_for_flush()
        if self.dataset is not None and hasattr(self.dataset, 'finalize'):
            self.dataset.finalize()

class EpisodeReader:
    def __init__(self, episode_dir):
        self.episode_dir = episode_dir

        # Load data
        with open(episode_dir / 'data.pkl', 'rb') as f:  # Note: Not secure. Only unpickle data you trust.
            data = pickle.load(f)
        self.timestamps = data['timestamps']
        self.observations = data['observations']
        self.actions = data['actions']
        assert len(self.timestamps) > 0
        assert len(self.timestamps) == len(self.observations) == len(self.actions)

        # Restore image observations from MP4 videos
        frames_dict = {}
        for step_idx, obs in enumerate(self.observations):
            for k, v in obs.items():
                if v is None:  # Images are stored as MP4 videos
                    # Load images from MP4 file
                    if k not in frames_dict:
                        mp4_path = episode_dir / f'{k}.mp4'
                        frames_dict[k] = read_frames_from_mp4(mp4_path)

                    # Restore image for current step
                    obs[k] = frames_dict[k][step_idx]  # np.uint8

    def __len__(self):
        return len(self.observations)
