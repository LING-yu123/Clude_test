import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))   # 让 core/ 里的模块可被导入
import sys, mujoco
from PIL import Image
from chassis_sim import CONFIG, compile_model
pose = sys.argv[1] if len(sys.argv) > 1 else "ready"
m = compile_model(dict(CONFIG, mesh=True, arm_pose=pose))
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 720, 1280)
for n, (az, el, dist, look) in {"iso": (140, -20, 4.5, (0.4, 0, 0.9)), "side": (90, 0, 5.0, (0.4, 0, 0.9))}.items():
    cam = mujoco.MjvCamera(); cam.lookat[:] = look; cam.azimuth = az; cam.elevation = el; cam.distance = dist
    r.update_scene(d, cam)
    Image.fromarray(r.render()).save(f"output/images/view_{pose}_{n}.png")
print("ok")
