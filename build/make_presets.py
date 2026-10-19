import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))   # 让 core/ 里的模块可被导入
"""用 IK 生成 GP110B 的预设姿态，存到 abb_presets.json（单位：度）。运行一次即可。"""
import json, math
import numpy as np
import arm_kin as K
import unload_plan as U

Z0 = K.Z_AXLE
def Rz(deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def flip(s):
    t = np.array(s, float).copy(); t[0] = math.pi; return t


def solve(center_world, R, station=0.0, holding=False, contact=False, wall=1.402):
    c = np.array(center_world, float) - np.array([0, station, Z0])
    p = K.j7_for_box_center(c, R)
    q, e = K.ik_tidy(p, R, K.SEEDS + [flip(s) for s in K.SEEDS], n_random=40,
                     check=lambda qq: K.keepout_ok(qq, station, holding=holding, grip_contact=contact, wall_front=wall)[0])
    if q is None:
        return np.zeros(7), 1.0, False, '无无碰撞解'
    ok, why = K.keepout_ok(q, station, holding=holding, grip_contact=contact, wall_front=wall)
    return q, e, ok, why

out = {"zero": [0.0] * 7}
tests = {
    "pick": ((1.552, 0.0, 1.0), K.R_FACE_FWD, dict(contact=True)),
    "reach": ((1.9 + 0.15, 0.0, 1.2), K.R_FACE_FWD, dict(contact=True, wall=1.9)),
    "lift": ((1.552, 0.0, 1.875), K.R_FACE_FWD, dict(contact=True)),
    "side": ((0.2, 1.1, 1.0), Rz(90) @ K.R_FACE_FWD, dict()),
    "rear": ((-0.85, 0.0, 0.855), K.R_FACE_BACK, dict(holding=True, contact=True)),
}
for name, (c, R, kw) in tests.items():
    q, e, ok, why = solve(c, R, **kw)
    print(f"{name:6s} err {e*1000:.2f} mm  keepout {ok} {why}  θ°={np.degrees(q).round(0)}")
    out[name] = np.degrees(q).tolist()
# 行驶姿态：在关节空间里搜索整机最稳（静态倾翻角最大）且不碰东西的收拢姿态
best = {"stow": (0, None), "ready": (0, None)}
for j2 in range(-45, 156, 15):
    for j3 in range(-45, 121, 30):
        for j4 in range(-70, 91, 20):
            for j6 in range(-120, 121, 40):
                q = np.radians([0, j2, j3, j4, 0, j6, 0])
                ok, _ = K.keepout_ok(q, 0.0, holding=False, grip_contact=False, margin=0.05)
                if not ok:
                    continue
                tip = U.static_tip(q, 0.0)
                zg = K.fk(q)[:3, 2]                       # 吸盘朝向
                if tip > best["stow"][0]:
                    best["stow"] = (tip, q)
                if zg[0] > 0.7 and tip > best["ready"][0]:  # ready：吸盘朝前，随时可以伸出去
                    best["ready"] = (tip, q)
for n in ("stow", "ready"):
    tip, q = best[n]
    out[n] = np.degrees(q).tolist() if q is not None else [0.0] * 7
    print(n, "静态倾翻角 %.1f°" % tip, np.degrees(q).round(0) if q is not None else None)
json.dump(out, open("data/abb_presets.json", "w"))
