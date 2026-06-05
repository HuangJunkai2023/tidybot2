import signal
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


class Closable:
    def __init__(self):
        self.close_count = 0

    def close(self):
        self.close_count += 1


class TestShutdownHandlers(unittest.TestCase):
    def test_sigtstp_closes_resources_before_exiting(self):
        env = Closable()
        policy = Closable()
        restore = main.install_shutdown_handlers(lambda: (env, policy))
        try:
            handler = signal.getsignal(signal.SIGTSTP)
            with self.assertRaises(SystemExit):
                handler(signal.SIGTSTP, None)
        finally:
            restore()

        self.assertEqual(env.close_count, 1)
        self.assertEqual(policy.close_count, 1)


if __name__ == '__main__':
    unittest.main()
