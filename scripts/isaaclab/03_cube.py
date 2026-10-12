import argparse

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


# 加入方块
cube_cfg = sim_utils.CuboidCfg(
    size=(0.1, 0.1, 0.1),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.8, 0.2, 0.2),
    ),
)

cube_cfg.func(
    "/World/Cube",
    cube_cfg,
    translation=(0.0, 0.0, 1.0),
)


sim.reset()

print("Isaac Lab 场景启动成功")

# 4. 持续推进仿真
while simulation_app.is_running():
    sim.step()

simulation_app.close()