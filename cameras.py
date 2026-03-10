# Author: Jimmy Wu
# Date: October 2024

import threading
import time
from pathlib import Path
import re
import subprocess
import cv2 as cv
import numpy as np
from constants import BASE_CAMERA_DEVICE, BASE_CAMERA_WIDTH, BASE_CAMERA_HEIGHT
from constants import WRIST_CAMERA_DEVICE, WRIST_CAMERA_WIDTH, WRIST_CAMERA_HEIGHT, USE_KINOVA_WRIST_CAMERA

class Camera:
    def __init__(self):
        self.image = None
        self.last_read_time = time.time()
        threading.Thread(target=self.camera_worker, daemon=True).start()

    def camera_worker(self):
        # Note: We read frames at 30 fps but not every frame is necessarily
        # saved during teleop or used during policy inference
        while True:
            # Reading new frames too quickly causes latency spikes
            while time.time() - self.last_read_time < 0.0333:  # 30 fps
                time.sleep(0.0001)
            _, bgr_image = self.cap.read()
            self.last_read_time = time.time()
            if bgr_image is not None:
                self.image = cv.cvtColor(bgr_image, cv.COLOR_BGR2RGB)

    def get_image(self):
        return self.image

    def close(self):
        self.cap.release()

class DummyCamera:
    def __init__(self, frame_width=640, frame_height=480):
        self.image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    def get_image(self):
        return self.image

    def close(self):
        pass

class UVCCamera(Camera):
    def __init__(self, device_hint, frame_width=640, frame_height=480, autofocus=False):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.autofocus = autofocus
        self.cap = self.get_cap(device_hint)
        super().__init__()

    def _resolve_device(self, device_hint):
        hint = str(device_hint).strip()
        if hint.startswith('/dev/'):
            p = Path(hint)
            if p.exists():
                try:
                    return str(p.resolve())
                except Exception:
                    return hint

            # Fallback for missing /dev/v4l/by-id entries: extract video index from name
            # e.g. ...-video-index0 -> /dev/video0
            m = re.search(r'video-index(\d+)$', hint)
            if m is not None:
                fallback = Path(f'/dev/video{m.group(1)}')
                if fallback.exists():
                    return str(fallback)
            return hint
        if hint.isdigit():
            return int(hint)

        by_id_dir = Path('/dev/v4l/by-id')
        if by_id_dir.exists():
            matches = sorted(by_id_dir.glob(f'*{hint}*'))
            if matches:
                return str(matches[0])
        return hint

    def _candidate_devices(self, device_hint):
        hint = str(device_hint).strip()
        candidates = []

        if hint.startswith('/dev/v4l/by-id/'):
            by_id_path = Path(hint)
            by_id_dir = by_id_path.parent
            name = by_id_path.name
            m = re.match(r'^(.*)-video-index\d+$', name)
            if m is not None and by_id_dir.exists():
                prefix = m.group(1)
                siblings = sorted(by_id_dir.glob(f'{prefix}-video-index*'))
                candidates.extend(str(p) for p in siblings)

        candidates.append(device_hint)
        resolved_candidates = [self._resolve_device(c) for c in candidates]
        candidates.extend(resolved_candidates)

        # Expand to all /dev/video* nodes that belong to the same USB camera group
        # (RealSense often exposes color node that is not listed under the expected by-id index)
        for c in list(resolved_candidates):
            if isinstance(c, str) and c.startswith('/dev/video'):
                for peer in self._same_video_group_devices(c):
                    candidates.append(peer)

        for c in list(resolved_candidates):
            if isinstance(c, str) and c.startswith('/dev/video'):
                try:
                    candidates.append(int(c.replace('/dev/video', '')))
                except Exception:
                    pass

        # Keep order, remove duplicates
        ordered_unique = []
        seen = set()
        for c in candidates:
            key = str(c)
            if key not in seen:
                seen.add(key)
                ordered_unique.append(c)
        return ordered_unique

    def _video_group_key(self, device):
        device_str = str(device)
        m = re.match(r'^/dev/video(\d+)$', device_str)
        if m is None:
            return None
        sys_node = Path(f'/sys/class/video4linux/video{m.group(1)}/device')
        if not sys_node.exists():
            return None
        try:
            resolved = str(sys_node.resolve())
        except Exception:
            return None
        # Typical path includes .../<usb-node>:1.x/... ; use <usb-node> as camera group key
        m2 = re.search(r'(.+):\d+\.\d+/', resolved)
        if m2 is not None:
            return m2.group(1)
        return str(Path(resolved).parent)

    def _same_video_group_devices(self, device):
        key = self._video_group_key(device)
        if key is None:
            return []
        devices = []
        root = Path('/sys/class/video4linux')
        if not root.exists():
            return devices
        for node in sorted(root.glob('video*')):
            dev = f'/dev/{node.name}'
            if self._video_group_key(dev) == key:
                devices.append(dev)
        return devices

    def _v4l2_preference_score(self, device):
        device_str = str(device)
        if not device_str.startswith('/dev/video'):
            return 0
        try:
            fmt_result = subprocess.run(
                ['v4l2-ctl', '-d', device_str, '--list-formats-ext'],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
            all_result = subprocess.run(
                ['v4l2-ctl', '-d', device_str, '--all'],
                capture_output=True,
                text=True,
                timeout=1.5,
                check=False,
            )
        except Exception:
            return 0
        fmt_text = (fmt_result.stdout or '').upper()
        all_text = (all_result.stdout or '').upper()
        if not fmt_text:
            return 0

        has_yuyv = "'YUYV'" in fmt_text
        has_mjpg = "'MJPG'" in fmt_text
        has_rgb = "'RGB3'" in fmt_text or "'BGR3'" in fmt_text
        has_uyvy = "'UYVY'" in fmt_text
        has_depth = "'Z16 '" in fmt_text
        has_grey = "'GREY'" in fmt_text

        has_color = has_yuyv or has_mjpg or has_rgb or has_uyvy
        is_processing_pipe = 'PROCESSING' in all_text

        score = 0
        if is_processing_pipe:
            score += 6
        if has_color:
            score += 3
        if has_yuyv or has_mjpg or has_rgb:
            score += 3
        if has_uyvy and not (has_yuyv or has_mjpg or has_rgb):
            score -= 2
        if has_depth:
            score -= 4
        if has_grey and not (has_yuyv or has_mjpg or has_rgb):
            score -= 2
        return score

    def _configure_cap(self, cap):
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv.CAP_PROP_AUTOFOCUS, 1 if self.autofocus else 0)

    def _readable_after_warmup(self, cap, attempts=20):
        for _ in range(attempts):
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return True
        return False

    def get_cap(self, device_hint):
        attempted = []
        all_candidates = self._candidate_devices(device_hint)
        scored_candidates = []
        for i, candidate_hint in enumerate(all_candidates):
            resolved = self._resolve_device(candidate_hint)
            score = self._v4l2_preference_score(resolved)
            scored_candidates.append((score, i, candidate_hint))

        # Prefer likely color nodes first, then preserve original order
        scored_candidates.sort(key=lambda x: (-x[0], x[1]))

        for _, _, candidate_hint in scored_candidates:
            device = self._resolve_device(candidate_hint)

            # Try V4L2 backend first
            for backend_name, backend in [('v4l2', cv.CAP_V4L2), ('default', None)]:
                cap = cv.VideoCapture(device, backend) if backend is not None else cv.VideoCapture(device)
                if not cap.isOpened():
                    attempted.append(f'{candidate_hint} -> {device} ({backend_name}:open-fail)')
                    cap.release()
                    continue

                self._configure_cap(cap)

                # Warm up and ensure this node is actually readable
                if self._readable_after_warmup(cap):
                    return cap

                attempted.append(f'{candidate_hint} -> {device} ({backend_name}:read-fail)')
                cap.release()

        assert False, f"Unable to open camera: {device_hint}; attempts={'; '.join(attempted)}"

class LogitechCamera(Camera):
    def __init__(self, serial, frame_width=640, frame_height=360, focus=0):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.focus = focus  # Note: Set this to 100 when using fisheye lens attachment
        self.cap = self.get_cap(serial)
        super().__init__()

    def get_cap(self, serial):
        cap = cv.VideoCapture(f'/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_{serial}-video-index0')
        cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # Important - results in much better latency

        # Disable autofocus
        cap.set(cv.CAP_PROP_AUTOFOCUS, 0)

        # Read several frames to let settings (especially gain/exposure) stabilize
        for _ in range(30):
            cap.read()
            cap.set(cv.CAP_PROP_FOCUS, self.focus)  # Fixed focus

        # Check all settings match expected
        assert cap.get(cv.CAP_PROP_FRAME_WIDTH) == self.frame_width
        assert cap.get(cv.CAP_PROP_FRAME_HEIGHT) == self.frame_height
        assert cap.get(cv.CAP_PROP_BUFFERSIZE) == 1
        assert cap.get(cv.CAP_PROP_AUTOFOCUS) == 0
        assert cap.get(cv.CAP_PROP_FOCUS) == self.focus

        return cap

def find_fisheye_center(image):
    # Find contours
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    # Fit a minimum enclosing circle around all contours
    return cv.minEnclosingCircle(np.vstack(contours))

def check_fisheye_centered(image):
    height, width, _ = image.shape
    center, _ = find_fisheye_center(image)
    if center is None:
        return True
    return abs(width / 2 - center[0]) < 0.05 * width and abs(height / 2 - center[1]) < 0.05 * height

class KinovaCamera(Camera):
    def __init__(self):
        # GStreamer video capture (see https://github.com/Kinovarobotics/kortex/issues/88)
        # Note: max-buffers=1 and drop=true are added to reduce latency spikes
        self.cap = cv.VideoCapture('rtspsrc location=rtsp://192.168.1.10/color latency=0 ! decodebin ! videoconvert ! appsink sync=false max-buffers=1 drop=true', cv.CAP_GSTREAMER)
        # self.cap = cv.VideoCapture('rtsp://192.168.1.10/color', cv.CAP_FFMPEG)  # This stream is high latency but works with pip-installed OpenCV
        assert self.cap.isOpened(), 'Unable to open stream. Please make sure OpenCV was built from source with GStreamer support.'

        # Apply camera settings
        threading.Thread(target=self.apply_camera_settings, daemon=True).start()
        super().__init__()

        # Wait for camera to warm up
        image = None
        while image is None:
            image = self.get_image()

        # Make sure fisheye lens did not accidentally get bumped
        if not check_fisheye_centered(image):
            raise Exception('The fisheye lens on the Kinova wrist camera appears to be off-center')

    def apply_camera_settings(self):
        # Note: This function adds significant camera latency when it is called
        # directly in __init__, so we call it in a separate thread instead

        from kortex_api.autogen.client_stubs.DeviceManagerClientRpc import DeviceManagerClient
        from kortex_api.autogen.client_stubs.VisionConfigClientRpc import VisionConfigClient
        from kortex_api.autogen.messages import DeviceConfig_pb2, VisionConfig_pb2
        from kinova import DeviceConnection

        # Use Kortex API to set camera settings
        with DeviceConnection.createTcpConnection() as router:
            device_manager = DeviceManagerClient(router)
            vision_config = VisionConfigClient(router)

            # Get vision device ID
            device_handles = device_manager.ReadAllDevices()
            vision_device_ids = [
                handle.device_identifier for handle in device_handles.device_handle
                if handle.device_type == DeviceConfig_pb2.VISION
            ]
            assert len(vision_device_ids) == 1
            vision_device_id = vision_device_ids[0]

            # Check that resolution, frame rate, and bit rate are correct
            sensor_id = VisionConfig_pb2.SensorIdentifier()
            sensor_id.sensor = VisionConfig_pb2.SENSOR_COLOR
            sensor_settings = vision_config.GetSensorSettings(sensor_id, vision_device_id)
            try:
                assert sensor_settings.resolution == VisionConfig_pb2.RESOLUTION_640x480  # FOV 65 ± 3° (diagonal)
                assert sensor_settings.frame_rate == VisionConfig_pb2.FRAMERATE_30_FPS
                assert sensor_settings.bit_rate == VisionConfig_pb2.BITRATE_10_MBPS
            except:
                sensor_settings.sensor = VisionConfig_pb2.SENSOR_COLOR
                sensor_settings.resolution = VisionConfig_pb2.RESOLUTION_640x480
                sensor_settings.frame_rate = VisionConfig_pb2.FRAMERATE_30_FPS
                sensor_settings.bit_rate = VisionConfig_pb2.BITRATE_10_MBPS
                vision_config.SetSensorSettings(sensor_settings, vision_device_id)
                assert False, 'Incorrect Kinova camera sensor settings detected, please restart the camera to apply new settings'

            # Disable autofocus and set manual focus to infinity
            # Note: This must be called after the OpenCV stream is created,
            # otherwise the camera will still have autofocus enabled
            sensor_focus_action = VisionConfig_pb2.SensorFocusAction()
            sensor_focus_action.sensor = VisionConfig_pb2.SENSOR_COLOR
            sensor_focus_action.focus_action = VisionConfig_pb2.FOCUSACTION_SET_MANUAL_FOCUS
            sensor_focus_action.manual_focus.value = 0
            vision_config.DoSensorFocusAction(sensor_focus_action, vision_device_id)

if __name__ == '__main__':
    if BASE_CAMERA_DEVICE == 'TODO':
        base_camera = DummyCamera(frame_width=BASE_CAMERA_WIDTH, frame_height=BASE_CAMERA_HEIGHT)
    elif str(BASE_CAMERA_DEVICE).strip().isdigit() or str(BASE_CAMERA_DEVICE).startswith('/dev/'):
        base_camera = UVCCamera(BASE_CAMERA_DEVICE, frame_width=BASE_CAMERA_WIDTH, frame_height=BASE_CAMERA_HEIGHT)
    else:
        base_camera = LogitechCamera(BASE_CAMERA_DEVICE, frame_width=BASE_CAMERA_WIDTH, frame_height=BASE_CAMERA_HEIGHT)
    wrist_camera = (
        KinovaCamera()
        if USE_KINOVA_WRIST_CAMERA
        else UVCCamera(WRIST_CAMERA_DEVICE, frame_width=WRIST_CAMERA_WIDTH, frame_height=WRIST_CAMERA_HEIGHT)
    )
    try:
        while True:
            base_image = base_camera.get_image()
            wrist_image = wrist_camera.get_image()
            cv.imshow('base_image', cv.cvtColor(base_image, cv.COLOR_RGB2BGR))
            cv.imshow('wrist_image', cv.cvtColor(wrist_image, cv.COLOR_RGB2BGR))
            key = cv.waitKey(1)
            if key == ord('s'):  # Save image
                base_image_path = f'base-image-{int(10 * time.time()) % 100000000}.jpg'
                cv.imwrite(base_image_path, cv.cvtColor(base_image, cv.COLOR_RGB2BGR))
                print(f'Saved image to {base_image_path}')
                wrist_image_path = f'wrist-image-{int(10 * time.time()) % 100000000}.jpg'
                cv.imwrite(wrist_image_path, cv.cvtColor(wrist_image, cv.COLOR_RGB2BGR))
                print(f'Saved image to {wrist_image_path}')
    finally:
        base_camera.close()
        wrist_camera.close()
        cv.destroyAllWindows()
