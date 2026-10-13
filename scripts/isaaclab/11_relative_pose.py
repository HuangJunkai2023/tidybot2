import argparse
import math

from isaaclab.app import AppLauncher

# 1. 解析启动参数，例如 --headless
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# 2. 启动 Isaac Sim 应用
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 应用启动后，再导入仿真相关模块
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import Articulation
from isaaclab_assets import KINOVA_GEN3_N7_CFG
from isaaclab.utils.math import subtract_frame_transforms


# 3. 创建物理仿真上下文
sim_cfg = SimulationCfg(dt=0.01)
sim = SimulationContext(sim_cfg)
sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])


# 配置并创建地面
ground_cfg = sim_utils.GroundPlaneCfg(
    color=(0.5, 0.5, 0.5),
    size=(10.0, 10.0),
)
ground_cfg.func("/World/Ground", ground_cfg)

# 配置并创建环境灯光
light_cfg = sim_utils.DomeLightCfg(
    intensity=2000.0,
    color=(1.0, 1.0, 1.0),
)
light_cfg.func("/World/Light", light_cfg)



robot_cfg = KINOVA_GEN3_N7_CFG.copy()
robot_cfg.prim_path = "/World/Robot"

# 把机械臂底座固定在世界坐标系中
robot_cfg.spawn.articulation_props.fix_root_link = True

# 本次先关闭机械臂的重力，便于观察关节和控制接口
robot_cfg.spawn.rigid_props.disable_gravity = True

robot = Articulation(cfg=robot_cfg)


sim.reset()

ee_index = robot.body_names.index("end_effector_link")


# 设置初始关节状态
joint_pos = robot.data.default_joint_pos.clone()
joint_vel = robot.data.default_joint_vel.clone()
robot.write_joint_state_to_sim(joint_pos, joint_vel)

# 让执行器保持初始关节角度
robot.set_joint_position_target(joint_pos)

print("关节名称：", robot.joint_names)
print("初始关节角度：", robot.data.joint_pos[0])


sim_dt = sim.get_physics_dt()
step_count = 0

# 按名称找到 joint_1，避免假设关节的排列顺序
joint_index = robot.joint_names.index("joint_1")

target = joint_pos.clone()

while simulation_app.is_running():
    t = step_count * sim_dt

    # 其余关节保持原目标，只修改 joint_1

    target[0, joint_index] = (
        joint_pos[0, joint_index]
        + 0.3 * math.sin(2 * math.pi * 0.2 * t)
    )

    robot.set_joint_position_target(target)
    robot.write_data_to_sim()

    sim.step()
    robot.update(sim_dt)

    step_count += 1

    if step_count % 100 == 0:
        # 基座在世界中的位姿
        base_pos_w = robot.data.root_link_pos_w
        base_quat_w = robot.data.root_link_quat_w

        # 末端在世界中的位姿
        ee_pos_w = robot.data.body_link_pos_w[:, ee_index]
        ee_quat_w = robot.data.body_link_quat_w[:, ee_index]

        # 计算末端相对基座的位姿
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            base_pos_w,
            base_quat_w,
            ee_pos_w,
            ee_quat_w,
        )

        print("末端世界位置：", ee_pos_w[0].tolist())
        print("末端基座位置：", ee_pos_b[0].tolist())
        print("末端基座姿态：", ee_quat_b[0].tolist())

simulation_app.close()