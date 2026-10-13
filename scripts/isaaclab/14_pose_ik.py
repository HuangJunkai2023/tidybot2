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
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
import torch
from isaaclab.utils.math import quat_error_magnitude


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
# 当前机械臂底座固定，PhysX 的雅可比不包含底座这一项
jacobian_index = ee_index - 1


# 设置初始关节状态
joint_pos = robot.data.default_joint_pos.clone()
joint_vel = robot.data.default_joint_vel.clone()
robot.write_joint_state_to_sim(joint_pos, joint_vel)

# 让执行器保持初始关节角度
robot.set_joint_position_target(joint_pos)

print("关节名称：", robot.joint_names)
print("初始关节角度：", robot.data.joint_pos[0])


sim_dt = sim.get_physics_dt()

# 创建 IK 控制器
ik_cfg = DifferentialIKControllerCfg(
    command_type="pose",
    use_relative_mode=False,
    ik_method="dls",
)

ik = DifferentialIKController(
    cfg=ik_cfg,
    num_envs=1,
    device=robot.device,
)

# 先推进一步，让初始关节状态对应的连杆位姿得到更新
robot.write_data_to_sim()
sim.step()
robot.update(sim_dt)

# 构建固定的末端目标：初始位置向上 3 厘米
initial_ee_pos = robot.data.body_link_pos_w[:, ee_index].clone()
initial_ee_quat = robot.data.body_link_quat_w[:, ee_index].clone()

goal = initial_ee_pos.clone()
goal[:, 2] += 0.03

goal_pose = torch.cat([goal, initial_ee_quat], dim=-1)
ik.set_command(goal_pose)

print("初始位置：", initial_ee_pos[0].tolist())
print("目标位置：", goal[0].tolist())

step_count = 0

while simulation_app.is_running():
    # 读取当前实际状态
    ee_pos = robot.data.body_link_pos_w[:, ee_index]
    ee_quat = robot.data.body_link_quat_w[:, ee_index]
    jacobian = robot.root_physx_view.get_jacobians()[:, jacobian_index]
    current_joint_pos = robot.data.joint_pos

    # IK 根据末端误差，计算本次关节目标
    joint_target = ik.compute(
        ee_pos,
        ee_quat,
        jacobian,
        current_joint_pos,
    )

    robot.set_joint_position_target(joint_target)
    robot.write_data_to_sim()

    sim.step()
    robot.update(sim_dt)

    step_count += 1

    if step_count % 50 == 0:
        actual_pos = robot.data.body_link_pos_w[:, ee_index]
        actual_quat = robot.data.body_link_quat_w[:, ee_index]

        error_mm = (goal - actual_pos).norm(dim=-1)[0].item() * 1000

        error_rad = quat_error_magnitude(
            actual_quat,
            initial_ee_quat,
        )[0].item()

        error_deg = math.degrees(error_rad)

        print(
            f"位置误差：{error_mm:.2f} mm，"
            f"朝向误差：{error_deg:.2f}°"
        )

simulation_app.close()