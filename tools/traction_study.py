import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""设计优化 第 3 步：台阶是摩擦牵引受限，看 质心位置（前后载荷）和 摩擦系数 对 20/30 mm 台阶 的影响。
运行：python tools/traction_study.py     结果：output/tables/design_study_traction.csv
"""
import csv
import mujoco
from capability_check import fresh, drive

rows = []
for cx in (-0.15, -0.35, -0.55):
    for mu in (0.6, 0.9):
        for h in (0.020, 0.030):
            res = []
            for T in (200.0, 400.0):
                d = fresh(susp="on", payload=50.0, motor_torque=T, friction=mu, floor_friction=mu, chassis_com=(cx, 0.0, 0.40), step_h=h, step_x=1.0)
                for _ in range(300): d.step()
                L = d.loads(); front = (L["fl"] + L["fr"]) / 4
                d.d.qpos[0] = -1.0; mujoco.mj_forward(d.m, d.d)
                ok = drive(d, 0.1, 60.0, a=0.1, stop_x=2.2)["x"] > 2.2
                res.append(ok)
                rows.append(dict(com_x=cx, mu=mu, step_mm=1000 * h, torque=T, front_load=round(front, 2), passed=int(ok)))
            print("质心x %+.2f μ=%.1f 台阶 %2.0f mm 前轮载荷 %2.0f%%：200 N·m:%s  400 N·m:%s" % (cx, mu, h * 1000, front * 100, "过" if res[0] else "卡", "过" if res[1] else "卡"), flush=True)
with open("output/tables/design_study_traction.csv", "w", newline="", encoding="utf-8-sig") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
