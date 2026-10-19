import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))   # 让 core/ 里的模块可被导入
"""交互式可视化：真实零件网格 + 七轴机械臂 + 后桥摆动悬挂，可随时开/关悬挂。
运行：python view_sim.py [速度m/s] [姿态] [悬挂 on|off] [台阶高度mm]
  姿态：ready reach side pick lift stow rear zero
  例：python view_sim.py 0.5 ready on 40      （默认值）
窗口里按键：
  S  切换悬挂 开启(摆动桥自由摆动) / 关闭(摆动桥锁死，相当于刚性车架)
  R  或 Backspace  重置     空格  暂停
  车顶上的小球：绿色 = 悬挂开启，红色 = 悬挂关闭
前方 1.5 m 处有一个只在左侧的斜坡台：悬挂开启时后桥会摆动，四个轮子都着地；
关闭时一个轮子会被顶离地面。
"""
import sys, time
import numpy as np
import mujoco, mujoco.viewer
from chassis_sim import CONFIG, compile_model, cfg_for_pose

v = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
pose = sys.argv[2] if len(sys.argv) > 2 else "ready"
susp = sys.argv[3] if len(sys.argv) > 3 else "on"
bump = float(sys.argv[4]) / 1000 if len(sys.argv) > 4 else 0.04
payload = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
cfg = cfg_for_pose(dict(CONFIG, suspension=susp, bump=bump), pose, payload, mesh=True)
print("整机质心 (x,y,z) =", [round(c, 3) for c in cfg["body_com"]], " 质量", cfg["body_mass"])
m = compile_model(cfg)
d = mujoco.MjData(m)
d.ctrl[:] = v / cfg["wheel_radius"]
EQ = m.eq_active0[0] and 1 or 0
state = {"lock": susp == "off", "reset": False}


def on_key(key):
    if key in (ord("S"), ord("L")):
        state["lock"] = not state["lock"]
        print("悬挂", "关闭（锁死）" if state["lock"] else "开启（自由摆动）")
    elif key in (ord("R"), 259):          # R / Backspace
        state["reset"] = True


with mujoco.viewer.launch_passive(m, d, key_callback=on_key) as viewer:
    viewer.cam.lookat[:] = [0.4, 0, 0.9]; viewer.cam.distance = 4.5; viewer.cam.azimuth = 140; viewer.cam.elevation = -20
    viewer.user_scn.ngeom = 1
    adr = m.joint("swing").qposadr[0]
    while viewer.is_running():
        t0 = time.time()
        if state["reset"]:
            mujoco.mj_resetData(m, d); d.ctrl[:] = v / cfg["wheel_radius"]; state["reset"] = False
        d.eq_active[0] = 1 if state["lock"] else 0
        mujoco.mj_step(m, d)
        base = d.qpos[:3]
        viewer.cam.lookat[:2] = base[:2] + [0.4, 0]
        col = [0.9, 0.2, 0.2, 1] if state["lock"] else [0.2, 0.8, 0.3, 1]
        mujoco.mjv_initGeom(viewer.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE, [0.06, 0, 0],
                            [base[0] - 0.2, base[1], base[2] + 1.9], np.eye(3).flatten(), col)
        viewer.sync()
        time.sleep(max(0, m.opt.timestep - (time.time() - t0)))
