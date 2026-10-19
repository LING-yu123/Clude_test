import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""底盘质心 / 配重对 前后载荷、10° 坡倾翻角、20 mm 台阶（μ=0.6）的影响。运行：python tools/chassis_tune.py"""
import numpy as np, mujoco
import chassis_sim as C
import unload_plan as U, arm_kin as K
from capability_check import fresh, drive

BASE = dict(C.CONFIG)


def variant(name, com_x=None, ballast=0.0, bx=-0.40, bz=0.25, torque=300.0):
    m0, c0 = BASE["chassis_mass"], np.array(BASE["chassis_com"], float)
    if com_x is not None:
        c0 = np.array([com_x, c0[1], c0[2]])
    M, com = m0, c0
    if ballast:
        pb = np.array([bx, 0.0, bz])
        M = m0 + ballast
        com = (m0 * c0 + ballast * pb) / M
    ex = dict(chassis_mass=M, chassis_com=tuple(com))
    C.CONFIG.update(ex)
    try:
        tip = min(U.static_tip(K.preset_theta(p), 50) for p in ("stow", "ready"))     # 悬挂锁死的矩形支承，最危险坡向
        d = fresh(susp="on", payload=50.0, motor_torque=torque, speed_gain=3000.0, step_h=0.02, step_x=1.0, **ex)
        for _ in range(500): d.step()
        L = d.loads()
        front = (L["fl"] + L["fr"]) / 4
        d.d.qpos[0] = -1.0; mujoco.mj_forward(d.m, d.d)
        r = drive(d, 0.1, 60.0, a=0.1, stop_x=2.2)
        ok = r["x"] > 2.2
        print("%-26s 整车 %4.0f kg  前轮载荷占比 %2.0f%%  最小倾翻角 %4.1f°  20mm台阶(μ0.6,%3.0fN·m)：%s" % (
            name, M + 820, front * 100, tip, torque, ("通过 %.0f s" % r["t"]) if ok else "未通过"), flush=True)
    finally:
        C.CONFIG.update({k: BASE[k] for k in ex})


if __name__ == "__main__":
    variant("当前(质心 x=-0.05)")
    variant("质心后移 x=-0.25", com_x=-0.25)
    variant("质心后移 x=-0.45", com_x=-0.45)
    variant("后部配重 +150 kg", ballast=150.0)
    variant("后部配重 +300 kg", ballast=300.0)
    variant("配重+300 且 600 N·m", ballast=300.0, torque=600.0)
