import argparse
import sys
import types
import unittest


sys.modules.setdefault('numpy', types.SimpleNamespace(asarray=lambda value, dtype=None: value, float64=float))
sys.modules.setdefault('constants', types.SimpleNamespace(
    POLICY_CONTROL_PERIOD=0.1,
    ENABLE_ARM=False,
    ARM_BACKEND='er3pro',
    ER3PRO_ARM_POSE_OBS_SOURCE='state',
    LEROBOT_FPS=10,
    LEROBOT_REPO_ID='local/test',
    LEROBOT_ROOT='data/test_lerobot',
    LEROBOT_TASK='test_task',
))
sys.modules.setdefault('episode_storage', types.SimpleNamespace(EpisodeWriter=None, LeRobotEpisodeWriter=object))
sys.modules.setdefault('policies', types.SimpleNamespace(
    TeleopPolicy=object,
    RemotePolicy=object,
    UarmTeleopPolicy=object,
    GamepadTeleopPolicy=object,
))

import main


class TestMainArgs(unittest.TestCase):
    def parse(self, *args):
        parser = main.build_arg_parser()
        parsed = parser.parse_args(args)
        main.validate_args(parser, parsed)
        return parsed

    def test_gamepad_requires_teleop(self):
        with self.assertRaises(SystemExit):
            self.parse('--gamepad')

    def test_gamepad_and_uarm_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            self.parse('--teleop', '--gamepad', '--uarm')

    def test_gamepad_teleop_args_are_valid(self):
        args = self.parse('--teleop', '--gamepad')

        self.assertTrue(args.teleop)
        self.assertTrue(args.gamepad)


class TestWriterSelection(unittest.TestCase):
    def test_gamepad_save_uses_lerobot_writer(self):
        args = argparse.Namespace(save=True, gamepad=True, output_dir='unused', lerobot_root='root', lerobot_repo_id='repo', lerobot_task='task')
        original_lerobot_writer = main.LeRobotEpisodeWriter
        class FakeLeRobotEpisodeWriter:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
        main.LeRobotEpisodeWriter = FakeLeRobotEpisodeWriter
        self.addCleanup(lambda: setattr(main, 'LeRobotEpisodeWriter', original_lerobot_writer))

        writer = main.create_episode_writer(args)

        self.assertIsInstance(writer, main.LeRobotEpisodeWriter)

    def test_phone_save_keeps_pickle_episode_writer(self):
        args = argparse.Namespace(save=True, gamepad=False, output_dir='data/demos')
        original_episode_writer = main.EpisodeWriter
        created = object()
        main.EpisodeWriter = lambda output_dir: created
        self.addCleanup(lambda: setattr(main, 'EpisodeWriter', original_episode_writer))

        writer = main.create_episode_writer(args)

        self.assertIs(writer, created)


if __name__ == '__main__':
    unittest.main()
