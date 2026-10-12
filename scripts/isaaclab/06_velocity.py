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
import omni.ui as ui
from isaaclab.assets import RigidObject, RigidObjectCfg

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

cube = RigidObject(
    cfg=RigidObjectCfg(
        prim_path="/World/Cube",
        spawn=cube_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            lin_vel=(3,0,0),
        ),
    )
)


sim.reset()

initial_state = cube.data.default_root_state.clone()
cube.write_root_state_to_sim(initial_state)

sim_dt = sim.get_physics_dt()
step_count = 0

while simulation_app.is_running():
    cube.write_data_to_sim()
    sim.step()
    cube.update(sim_dt)

    step_count += 1

    # 每 10 个物理步打印一次
    if step_count % 10 == 0:
        position = cube.data.root_pos_w[0]
        velocity = cube.data.root_lin_vel_w[0]

        print(
            f"时间={step_count * sim_dt:.2f}s, "
            f"x={position[0].item():.3f}m, "
            f"z={position[2].item():.3f}m, "
            f"vx={velocity[0].item():.3f}m/s, "
            f"vz={velocity[2].item():.3f}m/s"
        )

simulation_app.close()