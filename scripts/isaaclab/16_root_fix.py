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
from pxr import UsdPhysics, PhysxSchema

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


model_cfg = sim_utils.UsdFileCfg(
    usd_path="/home/huang/embodied_ai/tidybot2/models/isaaclab/gen3_2f85/gen3_2f85.usd",
)
model_cfg.func("/World/Robot", model_cfg)

# 清除导入器留下的两个根标记
for path in (
    "/World/Robot/worldBody",
    "/World/Robot/base_link/base_link",
):
    prim = sim.stage.GetPrimAtPath(path)
    prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

# 将整个机器人节点标记为一个关节系统
root = sim.stage.GetPrimAtPath("/World/Robot")
UsdPhysics.ArticulationRootAPI.Apply(root)
PhysxSchema.PhysxArticulationAPI.Apply(
    root
).CreateEnabledSelfCollisionsAttr(False)

# 底座固定关节直接连接世界
root_joint = sim.stage.GetPrimAtPath(
    "/World/Robot/joints/rootJoint_base_link"
)
root_joint.GetRelationship("physics:body0").ClearTargets(True)

# 检查场景中的关节系统根
for prim in sim.stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        print("关节系统根：", prim.GetPath())



# 4. 持续推进仿真
while simulation_app.is_running():
    sim.render()

simulation_app.close()