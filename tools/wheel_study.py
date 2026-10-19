import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""设计优化 第 2 步：轮径对 越障（台阶）/ 爬坡力矩 / 轮速 的影响。车体和质心的地面高度保持不变（轮子装在轮罩里，轴心随轮径抬高）。
运行：python tools/wheel_study.py       结果：output/tables/design_study_wheel.csv
"""
import csv, math
import mujoco
from capability_check import fresh, drive

RADII = (0.075, 0.100, 0.125)
STEPS = (0.020, 0.030, 0.040)
TORQUES = (100.0, 200.0, 400.0)


def step_pass(r, h, T):
    d = fresh(susp="on", payload=50.0, motor_torque=T, wheel_radius=r, step_h=h, step_x=1.0)
    d.d.qpos[0] = -1.0; mujoco.mj_forward(d.m, d.d)
    out = drive(d, 0.1, 60.0, a=0.1, stop_x=2.2)
    return out["x"] > 2.2


def climb_torque(r):
    d = fresh(alpha=10.0, phi=180.0, susp="on", payload=50.0, wheel_radius=r)
    for _ in range(500): d.step()
    out = drive(d, 0.3, 8.0, a=0.2)
    return out["torque"]


def main():
    rows = []
    for r in RADII:
        tq = climb_torque(r)
        print("轮径 %3.0f mm：10° 坡 0.3 m/s 爬升最大单轮力矩 %.0f N·m，0.9 m/s 时轮速 %.0f rpm" % (2000 * r, tq, 0.9 / r * 60 / (2 * math.pi)), flush=True)
        for h in STEPS:
            res = []
            for T in TORQUES:
                ok = step_pass(r, h, T)
                res.append(ok)
                rows.append(dict(diameter_mm=2000 * r, step_mm=1000 * h, torque=T, passed=int(ok), climb_torque=round(tq)))
            print("   台阶 %2.0f mm：" % (h * 1000) + "  ".join("%3.0f N·m:%s" % (T, "过" if ok else "卡") for T, ok in zip(TORQUES, res)), flush=True)
    with open("output/tables/design_study_wheel.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)


if __name__ == "__main__":
    main()
