import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""台阶通过的稳健性：起点 x 在 ±3 cm 内取 5 个，看通过率。python tools/step_robust.py [台阶mm=20] [摩擦=0.6] [力矩...]"""
import mujoco
from capability_check import fresh, drive
h = float(sys.argv[1]) / 1000 if len(sys.argv) > 1 else 0.02
mu = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
torques = [float(x) for x in sys.argv[3:]] or [120.0, 300.0]
for T in torques:
    res = []
    for dx in (-0.03, -0.015, 0.0, 0.015, 0.03):
        d = fresh(susp="on", payload=50.0, motor_torque=T, speed_gain=3000.0, friction=mu, floor_friction=mu, step_h=h, step_x=1.0)
        d.d.qpos[0] = -1.0 + dx; mujoco.mj_forward(d.m, d.d)
        r = drive(d, 0.1, 70.0, a=0.1, stop_x=2.2)
        res.append("通过%.0fs" % r["t"] if r["x"] > 2.2 else "卡住")
    print("台阶 %.0f mm μ=%.1f 单轮 %3.0f N·m：%s" % (h * 1000, mu, T, " ".join(res)), flush=True)
