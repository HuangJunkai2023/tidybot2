import numpy as np

################################################################################
# Backend and subsystem switches

ARM_BACKEND = 'er3pro'  # 'er3pro' or 'kinova'
ENABLE_BASE = True
ENABLE_ARM = True

################################################################################
# Mobile base

# Vehicle center to steer axis (m)
h_x, h_y = 0.190150 * np.array([1.0, 1.0, -1.0, -1.0]), 0.170150 * np.array([-1.0, 1.0, 1.0, -1.0])  # Kinova / Franka
# h_x, h_y = 0.140150 * np.array([1.0, 1.0, -1.0, -1.0]), 0.120150 * np.array([-1.0, 1.0, 1.0, -1.0])  # ARX5

# Encoder magnet offsets
ENCODER_MAGNET_OFFSETS = [0.0 / 4096, 0.0 / 4096, 0.0 / 4096, 0.0 / 4096]  # TODO

################################################################################
# Teleop and imitation learning

# Base and arm RPC servers
BASE_RPC_HOST = 'localhost'
BASE_RPC_PORT = 50000
ARM_RPC_HOST = 'localhost'
ARM_RPC_PORT = 50001
RPC_AUTHKEY = b'secret password'

# Mobile base backend
BASE_BACKEND = 'ros1_udp'  # 'ros1_udp' or 'native'

# ROS1 Docker bridge (see src/p500/docker_cmd_vel_server.py)
BASE_CMD_UDP_HOST = '127.0.0.1'
BASE_CMD_UDP_PORT = 15000
BASE_CMD_UDP_PUBLISH_RATE = 30.0
BASE_CMD_TIMEOUT = 0.25  # s

# Pose tracking and position-to-velocity control gains for ros1_udp backend
BASE_MAX_VEL_XY = 0.35      # m/s
BASE_MAX_VEL_THETA = 1.20   # rad/s
BASE_POS_KP_LINEAR = 1.60
BASE_POS_KP_ANGULAR = 2.00
BASE_DIFF_DRIVE_MODE = True  # True for two-wheel differential-drive base

# ER3Pro arm
ER3PRO_IP = '192.168.0.160'
ER3PRO_LOCAL_IP = '192.168.0.200'
ER3PRO_MOVE_VELOCITY = 300
ER3PRO_MOVE_ZONE = 10
ER3PRO_ENABLE_GRIPPER = True

# Gripper backend for ER3Pro: 'jodell_usb', 'rs485_epg', or 'di'
ER3PRO_GRIPPER_BACKEND = 'rs485_epg'

# Legacy DI-based gripper control
ER3PRO_GRIPPER_THRESHOLD = 0.5
ER3PRO_GRIPPER_BOARD = 2
ER3PRO_GRIPPER_DI1_PORT = 0
ER3PRO_GRIPPER_DI2_PORT = 1

# RS485 + Modbus RTU gripper control (Jodell EPG)
ER3PRO_GRIPPER_RS485_SLAVE_ID = 9
ER3PRO_GRIPPER_RS485_ENABLE_ON_START = True
ER3PRO_GRIPPER_RS485_INIT_REG = 0x03E8      # enableClamp writes 1 to 0x03E8
ER3PRO_GRIPPER_RS485_INIT_VALUE = 0x0001
ER3PRO_GRIPPER_RS485_TORQUE_REG = 0x03FD    # switchMode writes 0(serial)/0x55(IO)
ER3PRO_GRIPPER_RS485_POS_REG = 0x03E8       # runWithParam start register
ER3PRO_GRIPPER_RS485_SPEED_REG = 0x03E8     # runWithParam start register
ER3PRO_GRIPPER_RS485_POS_NOW_REG = 0x07D1   # input register: high byte is clamp position
ER3PRO_GRIPPER_RS485_OPEN_POS = 0
ER3PRO_GRIPPER_RS485_CLOSE_POS = 255
ER3PRO_GRIPPER_RS485_SPEED = 255
ER3PRO_GRIPPER_RS485_TORQUE = 80

# Gripper observation source for data recording:
# - 'state': always trust hardware-reported state
# - 'command': always use last commanded value
# - 'hybrid': use state when close to command, else fallback to command
ER3PRO_GRIPPER_OBS_SOURCE = 'command'
ER3PRO_GRIPPER_OBS_MAX_DEVIATION = 0.25

# Arm pose observation source for data recording (affects saved episodes only):
# - 'state': save measured arm pose from hardware state
# - 'command': save teleop commanded arm pose/quaternion
ER3PRO_ARM_POSE_OBS_SOURCE = 'command'

# USB-RS485 + Jodell Python SDK backend (uses src/jodell/python/JodellTool-0.1.5-py3-none-any.whl)
ER3PRO_GRIPPER_USB_PORT = '/dev/ttyUSB0'
ER3PRO_GRIPPER_USB_BAUD = 115200
ER3PRO_GRIPPER_USB_SLAVE_ID = 9
ER3PRO_GRIPPER_USB_SPEED = 255
ER3PRO_GRIPPER_USB_TORQUE = 80
ER3PRO_GRIPPER_USB_OPEN_POS = 0
ER3PRO_GRIPPER_USB_CLOSE_POS = 255
ER3PRO_GRIPPER_USB_MIN_CMD_INTERVAL = 0.08
ER3PRO_CPP_BRIDGE_BIN = '../xCoreSDK_cpp-v0.7.1/build/bin/arm_bridge'
ER3PRO_FOLLOW_SCALE = 0.8   # FollowPosition speed scale in [0,1], larger -> more responsive
ER3PRO_RT_FILTER_FREQ = 15.0 # Hz, larger -> less lag, smaller -> smoother
ER3PRO_MAX_POS_SPEED = 0.18   # m/s
ER3PRO_MAX_ROT_SPEED = 0.90   # rad/s
ER3PRO_MAX_POS_ACCEL = 0.90   # m/s^2
ER3PRO_MAX_ROT_ACCEL = 4.50   # rad/s^2
ER3PRO_CMD_TIMEOUT = 0.25     # s, equivalent to 2.5 * POLICY_CONTROL_PERIOD
ER3PRO_TCP_OFFSET_Z = 0.10    # m, TCP defined at gripper center 10 cm along flange +Z
ER3PRO_TELEOP_PRESET_JOINT_DEG = np.array([0.0, 30.0, 0.0, 60.0, 0.0, 90.0, 0.0], dtype=np.float64)

# Cameras
BASE_CAMERA_DEVICE = '/dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_435if_Intel_R__RealSense_TM__Depth_Camera_435if-video-index0'
BASE_CAMERA_WIDTH = 1280
BASE_CAMERA_HEIGHT = 720
# Backward-compatible alias: base camera is configured by device hint/path.
BASE_CAMERA_SERIAL = BASE_CAMERA_DEVICE
WRIST_CAMERA_DEVICE = '/dev/v4l/by-id/usb-GS02_1080P_CAMERA_GS02_1080P_CAMERA-video-index0'
WRIST_CAMERA_WIDTH = 1080
WRIST_CAMERA_HEIGHT = 720
USE_KINOVA_WRIST_CAMERA = False

# Policy
POLICY_SERVER_HOST = 'localhost'
POLICY_SERVER_PORT = 5555
POLICY_CONTROL_FREQ = 20
POLICY_CONTROL_PERIOD = 1.0 / POLICY_CONTROL_FREQ
POLICY_IMAGE_WIDTH = 84
POLICY_IMAGE_HEIGHT = 84
