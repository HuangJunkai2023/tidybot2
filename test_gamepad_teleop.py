import sys
import types
import unittest

import numpy as np


cv2_stub = types.SimpleNamespace(
    COLOR_RGB2BGR=1,
    IMWRITE_JPEG_QUALITY=1,
    resize=lambda image, size: image,
    imencode=lambda ext, image, params=None: (True, np.array([1], dtype=np.uint8)),
)
sys.modules.setdefault('cv2', cv2_stub)
sys.modules.setdefault(
    'flask',
    types.SimpleNamespace(
        Flask=lambda name: types.SimpleNamespace(route=lambda path: (lambda fn: fn)),
        render_template=lambda name: '',
        request=types.SimpleNamespace(remote_addr='test'),
    ),
)
sys.modules.setdefault(
    'flask_socketio',
    types.SimpleNamespace(SocketIO=lambda app: types.SimpleNamespace(on=lambda event: (lambda fn: fn), run=lambda *args, **kwargs: None), emit=lambda *args, **kwargs: None),
)
sys.modules.setdefault('zmq', types.SimpleNamespace(Context=lambda: None, REQ=0, RCVTIMEO=0, SNDTIMEO=0))
sys.modules.setdefault('uarm_teleop', types.SimpleNamespace(UarmMasterReader=object))

from policies import GamepadTeleopPolicy


class FakeEvent:
    def __init__(self, on_pump=None):
        self.on_pump = on_pump
        self.pump_count = 0

    def pump(self):
        self.pump_count += 1
        if self.on_pump is not None:
            self.on_pump(self.pump_count)


class FakePygame:
    def __init__(self, on_pump=None):
        self.event = FakeEvent(on_pump)
        self.init_count = 0
        self.quit_count = 0

    def init(self):
        self.init_count += 1

    def quit(self):
        self.quit_count += 1


class FakeJoystick:
    def __init__(self):
        self.axes = {}
        self.buttons = {}

    def init(self):
        pass

    def get_axis(self, axis):
        return self.axes.get(axis, 0.0)

    def get_button(self, button):
        return self.buttons.get(button, False)


def make_obs():
    return {
        'base_pose': np.array([0.1, 0.2, 0.3], dtype=np.float64),
        'arm_pos': np.array([0.4, 0.0, 0.2], dtype=np.float64),
        'arm_quat': np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        'gripper_pos': np.array([0.5], dtype=np.float64),
    }


class TestGamepadTeleopPolicy(unittest.TestCase):
    def test_reset_waits_until_web_start_is_received(self):
        joystick = FakeJoystick()
        holder = {}

        def on_pump(pump_count):
            if pump_count == 3:
                holder['policy'].message_buffer.put({'state_update': 'episode_started'})

        pygame = FakePygame(on_pump=on_pump)
        policy = GamepadTeleopPolicy(joystick=joystick, pygame_module=pygame, sleep_fn=lambda _: None, start_web_server=False)
        holder['policy'] = policy

        policy.reset()

        self.assertGreaterEqual(pygame.event.pump_count, 3)

    def test_step_outputs_hold_action_without_lb_deadman(self):
        joystick = FakeJoystick()
        policy = GamepadTeleopPolicy(joystick=joystick, pygame_module=FakePygame(), sleep_fn=lambda _: None, start_web_server=False)
        policy.message_buffer.put({'state_update': 'episode_started'})
        policy.reset()

        action = policy.step(make_obs())

        np.testing.assert_allclose(action['arm_pos'], make_obs()['arm_pos'])
        np.testing.assert_allclose(action['arm_quat'], make_obs()['arm_quat'])
        np.testing.assert_allclose(action['gripper_pos'], make_obs()['gripper_pos'])

    def test_left_stick_moves_tcp_horizontally(self):
        joystick = FakeJoystick()
        joystick.buttons[7] = True
        joystick.axes[0] = 1.0
        joystick.axes[1] = -1.0
        policy = GamepadTeleopPolicy(joystick=joystick, pygame_module=FakePygame(), sleep_fn=lambda _: None, start_web_server=False)
        policy.message_buffer.put({'state_update': 'episode_started'})
        policy.reset()

        action = policy.step(make_obs())

        self.assertGreater(action['arm_pos'][0], make_obs()['arm_pos'][0])
        self.assertLess(action['arm_pos'][1], make_obs()['arm_pos'][1])
        np.testing.assert_allclose(action['base_pose'], make_obs()['base_pose'])

    def test_right_stick_vertical_and_roll_controls_tcp_pose(self):
        joystick = FakeJoystick()
        joystick.buttons[7] = True
        joystick.axes[3] = 1.0
        joystick.axes[4] = -1.0
        policy = GamepadTeleopPolicy(joystick=joystick, pygame_module=FakePygame(), sleep_fn=lambda _: None, start_web_server=False)
        policy.message_buffer.put({'state_update': 'episode_started'})
        policy.reset()

        action = policy.step(make_obs())

        self.assertGreater(action['arm_pos'][2], make_obs()['arm_pos'][2])
        self.assertFalse(np.allclose(action['arm_quat'], make_obs()['arm_quat']))

    def test_outputs_slew_limited_command_state_like_phone_teleop(self):
        joystick = FakeJoystick()
        joystick.axes[1] = -1.0
        policy = GamepadTeleopPolicy(joystick=joystick, pygame_module=FakePygame(), sleep_fn=lambda _: None, start_web_server=False)
        policy.message_buffer.put({'state_update': 'episode_started'})
        policy.reset()

        action = policy.step(make_obs())

        self.assertTrue(hasattr(policy, 'arm_cmd_pos'))
        np.testing.assert_allclose(action['arm_pos'], policy.arm_cmd_pos)
        np.testing.assert_allclose(action['gripper_pos'], policy.gripper_cmd_pos)

    def test_triggers_close_and_open_gripper_as_velocity_commands(self):
        joystick = FakeJoystick()
        policy = GamepadTeleopPolicy(joystick=joystick, pygame_module=FakePygame(), sleep_fn=lambda _: None, start_web_server=False)
        policy.message_buffer.put({'state_update': 'episode_started'})
        policy.reset()

        joystick.axes[2] = 1.0
        close_action = policy.step(make_obs())
        joystick.axes[2] = 0.0
        joystick.axes[5] = 1.0
        open_action = policy.step(make_obs())

        self.assertLess(close_action['gripper_pos'][0], make_obs()['gripper_pos'][0])
        self.assertGreater(open_action['gripper_pos'][0], close_action['gripper_pos'][0])

    def test_web_end_and_reset_states_control_episode(self):
        joystick = FakeJoystick()
        policy = GamepadTeleopPolicy(joystick=joystick, pygame_module=FakePygame(), sleep_fn=lambda _: None, start_web_server=False)
        policy.message_buffer.put({'state_update': 'episode_started'})
        policy.reset()

        policy.message_buffer.put({'state_update': 'episode_ended'})
        self.assertEqual(policy.step(make_obs()), 'end_episode')
        policy.message_buffer.put({'state_update': 'reset_env'})
        self.assertEqual(policy.step(make_obs()), 'reset_env')


if __name__ == '__main__':
    unittest.main()
