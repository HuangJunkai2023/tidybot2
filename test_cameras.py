import time
import sys
import types
import unittest
from unittest import mock

cv2_stub = types.SimpleNamespace(COLOR_BGR2RGB=1, cvtColor=lambda image, _: image)
sys.modules.setdefault('cv2', cv2_stub)
numpy_stub = types.SimpleNamespace(uint8='uint8', zeros=lambda shape, dtype=None: None)
sys.modules.setdefault('numpy', numpy_stub)
constants_stub = types.SimpleNamespace(
    BASE_RPC_HOST='localhost',
    BASE_RPC_PORT=50000,
    ARM_RPC_HOST='localhost',
    ARM_RPC_PORT=50001,
    RPC_AUTHKEY=b'',
    ENABLE_BASE=True,
    ENABLE_ARM=False,
    BASE_CAMERA_DEVICE='0',
    BASE_CAMERA_WIDTH=640,
    BASE_CAMERA_HEIGHT=480,
    WRIST_CAMERA_DEVICE='1',
    WRIST_CAMERA_WIDTH=640,
    WRIST_CAMERA_HEIGHT=480,
    USE_KINOVA_WRIST_CAMERA=False,
)
sys.modules.setdefault('constants', constants_stub)
sys.modules.setdefault('arm_server', types.SimpleNamespace(ArmManager=None))


class FakeBase:
    def close(self):
        pass


class FakeBaseManager:
    def __init__(self, *args, **kwargs):
        pass

    def connect(self):
        pass

    def Base(self, *args, **kwargs):
        return FakeBase()


sys.modules.setdefault('base_server', types.SimpleNamespace(BaseManager=FakeBaseManager))

from cameras import Camera, UVCCamera
from real_env import RealEnv


class FakeCamera(Camera):
    def __init__(self):
        self.cap = FakeCapture()
        super().__init__()


class FakeCapture:
    def __init__(self):
        self.read_count = 0
        self.release_count = 0
        self.read_after_release = False

    def read(self):
        if self.release_count > 0:
            self.read_after_release = True
        self.read_count += 1
        return False, None

    def release(self):
        self.release_count += 1


class TestCameraLifecycle(unittest.TestCase):
    def test_missing_by_id_path_does_not_guess_video_index(self):
        camera = UVCCamera.__new__(UVCCamera)
        hint = '/dev/v4l/by-id/usb-Missing_Camera-video-index0'

        self.assertEqual(camera._resolve_device(hint), hint)

    def test_close_stops_worker_before_releasing_capture(self):
        camera = FakeCamera()

        deadline = time.time() + 1.0
        while camera.cap.read_count == 0 and time.time() < deadline:
            time.sleep(0.01)

        self.assertGreater(camera.cap.read_count, 0)

        camera.close()
        reads_after_close = camera.cap.read_count
        time.sleep(0.08)

        self.assertEqual(camera.cap.release_count, 1)
        self.assertFalse(camera.cap.read_after_release)
        self.assertEqual(camera.cap.read_count, reads_after_close)


class ClosableThing:
    def __init__(self):
        self.close_count = 0

    def close(self):
        self.close_count += 1


class TestRealEnvLifecycle(unittest.TestCase):
    def test_init_failure_closes_already_created_camera(self):
        base_camera = ClosableThing()

        with mock.patch.object(RealEnv, '_create_base_camera', return_value=base_camera):
            with mock.patch('real_env.UVCCamera', side_effect=RuntimeError('camera busy')):
                with self.assertRaises(RuntimeError):
                    RealEnv()

        self.assertEqual(base_camera.close_count, 1)


if __name__ == '__main__':
    unittest.main()
