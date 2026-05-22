import argparse
import time

import numpy as np

from arm_server import ER3ProCppBridgeArm
from constants import ER3PRO_JOINT_IMPEDANCE, ER3PRO_TELEOP_PRESET_JOINT_DEG


def main():
    parser = argparse.ArgumentParser(
        description='Hold ER3Pro at the preset pose under joint impedance control.'
    )
    parser.add_argument('--duration', type=float, default=60.0)
    parser.add_argument('--print-period', type=float, default=1.0)
    args = parser.parse_args()

    arm = ER3ProCppBridgeArm()
    try:
        print(
            'Moving to ER3PRO_TELEOP_PRESET_JOINT_DEG '
            f'{np.round(ER3PRO_TELEOP_PRESET_JOINT_DEG, 3).tolist()}'
        )
        print(
            'Using ER3PRO_JOINT_IMPEDANCE '
            f'{np.round(ER3PRO_JOINT_IMPEDANCE, 3).tolist()}'
        )
        arm.reset()
        arm.move_to_teleop_preset()
        print('Holding still with joint impedance control. Push the arm gently to test compliance.')

        start = time.monotonic()
        next_print = start
        while time.monotonic() - start < args.duration:
            now = time.monotonic()
            if now >= next_print:
                state = arm.get_state()
                print(
                    f't={now - start:6.2f}s '
                    f'pos={np.round(state["arm_pos"], 4).tolist()} '
                    f'joints_deg={np.round(np.rad2deg(state["arm_joints"]), 2).tolist()} '
                    f'gripper={float(state["gripper_pos"][0]):.3f} '
                    f'force={float(state["gripper_force"][0]):.3f}',
                    flush=True,
                )
                next_print = now + args.print_period
            time.sleep(0.01)
    finally:
        arm.close()


if __name__ == '__main__':
    main()
