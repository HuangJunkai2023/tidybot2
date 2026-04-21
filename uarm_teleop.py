import re
import time

import numpy as np
import serial

from constants import UARM_ARM_SERVO_IDS
from constants import UARM_BAUDRATE
from constants import UARM_GRIPPER_CLOSE_DEG
from constants import UARM_GRIPPER_OPEN_DEG
from constants import UARM_GRIPPER_SERVO_ID
from constants import UARM_MAX_FRAME_DELTA_DEG
from constants import UARM_SERIAL_PORT
from constants import UARM_SERVO_IDS


class UarmMasterReader:
    def __init__(self, port=UARM_SERIAL_PORT, baudrate=UARM_BAUDRATE):
        self.port = port
        self.baudrate = baudrate
        self.ser = serial.Serial(self.port, self.baudrate, timeout=0.03)
        self.zero_angles = np.zeros(len(UARM_SERVO_IDS), dtype=np.float64)
        self.last_angles = np.zeros(len(UARM_SERVO_IDS), dtype=np.float64)
        self.last_gripper_norm = 1.0
        self._initialize()

    def close(self):
        if getattr(self, 'ser', None) is not None and self.ser.is_open:
            self.ser.close()

    def _send_command(self, cmd):
        self.ser.write(cmd.encode('ascii'))
        time.sleep(0.008)
        return self.ser.read_all().decode('ascii', errors='ignore')

    def _pwm_to_angle(self, response_str, pwm_min=500, pwm_max=2500, angle_range=270.0):
        match = re.search(r'P(\d{4})', response_str)
        if not match:
            return None
        pwm_val = int(match.group(1))
        return (pwm_val - pwm_min) / (pwm_max - pwm_min) * angle_range

    def _read_servo_angle(self, servo_id):
        response = self._send_command(f'#{servo_id:03d}PRAD!')
        angle = self._pwm_to_angle(response.strip())
        return angle, response

    def _initialize(self):
        self._send_command('#000PVER!')
        self._send_command('#000PCSK!')
        for servo_id in UARM_SERVO_IDS:
            self._send_command(f'#{servo_id:03d}PULK!')
            angle, _ = self._read_servo_angle(servo_id)
            if angle is not None:
                self.zero_angles[servo_id] = angle
                self.last_angles[servo_id] = angle
        print(f'[uarm] connected {self.port} @ {self.baudrate}')
        print(f'[uarm] zero angles deg={np.round(self.zero_angles, 2).tolist()}')

    def recalibrate_zero(self):
        angles = self.read_servo_angles()
        self.zero_angles = angles.copy()
        print(f'[uarm] recalibrated zero deg={np.round(self.zero_angles, 2).tolist()}')

    def read_servo_angles(self):
        angles = self.last_angles.copy()
        for servo_id in UARM_SERVO_IDS:
            angle, response = self._read_servo_angle(servo_id)
            if angle is None:
                continue
            if abs(angle - self.last_angles[servo_id]) > UARM_MAX_FRAME_DELTA_DEG:
                print(
                    f'[uarm] ignore jump servo={servo_id} '
                    f'prev={self.last_angles[servo_id]:.2f} now={angle:.2f} raw="{response.strip()}"',
                    flush=True,
                )
                continue
            angles[servo_id] = angle
        self.last_angles = angles
        return angles

    def read(self):
        angles = self.read_servo_angles()
        arm_deg = angles[list(UARM_ARM_SERVO_IDS)] - self.zero_angles[list(UARM_ARM_SERVO_IDS)]

        grip_angle = float(angles[UARM_GRIPPER_SERVO_ID] - self.zero_angles[UARM_GRIPPER_SERVO_ID])
        denom = UARM_GRIPPER_OPEN_DEG - UARM_GRIPPER_CLOSE_DEG
        if abs(denom) < 1e-6:
            gripper_norm = self.last_gripper_norm
        else:
            gripper_norm = (grip_angle - UARM_GRIPPER_CLOSE_DEG) / denom
            gripper_norm = float(np.clip(gripper_norm, 0.0, 1.0))
        self.last_gripper_norm = gripper_norm

        return arm_deg.astype(np.float64), gripper_norm
