import io
import unittest
from unittest import mock

import arm_server


class FakeProcess:
    def __init__(self):
        self.stdin = io.StringIO()
        self.stdout = io.StringIO('ERR bad_args\n')
        self.stderr = io.StringIO('usage: unsupported startup option\n')

    def poll(self):
        return None


class TestER3ProBridgeStartup(unittest.TestCase):
    def test_startup_error_preserves_stderr_before_background_reader_starts(self):
        thread_starts = []

        class FakeThread:
            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                thread_starts.append(True)

        with mock.patch.object(arm_server.Path, 'exists', return_value=True), \
             mock.patch.object(arm_server.subprocess, 'Popen', return_value=FakeProcess()), \
             mock.patch.object(arm_server.threading, 'Thread', FakeThread):
            with self.assertRaisesRegex(RuntimeError, 'ERR bad_args.*unsupported startup option'):
                arm_server.ER3ProCppBridgeArm()

        self.assertEqual(thread_starts, [])


if __name__ == '__main__':
    unittest.main()
