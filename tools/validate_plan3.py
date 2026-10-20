import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))   # 让 core/ 里的模块可被导入
"""用 MuJoCo 动力学把成组卸货计划（unload_plan3.json）完整跑一遍：倾覆裕度、臂跟踪误差、实际姿态碰撞复查、节拍。
运行：python tools/validate_plan3.py [坡度°=0] [坡向°=0] [组数=全部] [drive=1] [加速时间=0.8] [速度比例=0.9] [悬挂 on/off=off]
"""
import json, math, sys, time
import numpy as np
from tip_view import TipDemo
from unload_player3 import UnloadPlayer3
from cycle_test import margin_cm
import box_spec as BS


def run(alpha=0.0, phi=0.0, nparts=None, drive=True, acc=None, scale=None, susp="off", quiet=False, plan_file="data/unload_plan3.json"):
    plan = json.load(open(plan_file, encoding="utf-8"))
    d = TipDemo("stow", 0, susp, phi, alpha)
    P = UnloadPlayer3(d, plan, drive=drive, acc_time=acc, speed_scale=scale, max_parts=nparts)
    from collision import Collider
    col = Collider()
    P.start()
    dt = d.m.opt.timestep
    minm, maxtilt, maxerr, ncol, ncheck = 1e9, 0.0, 0.0, 0, 0
    worst = {}
    k, t0 = 0, time.time()
    removed = set()
    while P.active and d.d.time < 6000:
        P.update(dt); d.step(); k += 1
        if k % 50 == 0:
            minm = min(minm, margin_cm(d)); maxtilt = max(maxtilt, d.tilt_deg())
            qa = np.array([d.d.qpos[a] for a in d.arm_qadr])
            maxerr = max(maxerr, float(np.max(np.abs(qa - d.q_cmd))) * 180 / math.pi)
        if k % 500 == 0 and P.cur is not None and 0 <= P.k < len(P.parts):
            p = P.parts[P.k]
            qa = np.array([d.d.qpos[a] for a in d.arm_qadr])
            pi, rho = d.grip_state()
            holding = d.payload > 0
            rem = set((c, l) for (c, l) in P.order[:P.done])
            if not holding:
                rem = rem                                  # 已取走的箱子
            else:
                rem = rem | set((c, p["layer"]) for c in p["cols"])
            gs = dict(width=p["width"], top=(p["mode"] == "top"), pi=pi, rho=rho, yoff=p["yoff"], pw=(p.get("pw") or BS.PLATE_HALF_W), tray=not plan.get("simple", False))
            y_now = P._plane_y(d.d.xpos[d.base]) if P.drive_enabled else 0.0
            pr = col.check(qa, station_y=y_now, holding=holding, mode="pick", removed=rem, grip=gs)
            # 只看：臂连杆之间/连杆与环境的贯穿 > 6 mm；夹具与箱墙/辊床的接触是允许的
            bad = []
            for a, b, dist in pr:
                if dist > -0.006:
                    continue
                grip_a, grip_b = a.startswith("g_"), b.startswith("g_")
                env_ok = lambda n: n.startswith("wb_") or n == "wall_back" or n.startswith("xf_")
                if (grip_a and env_ok(b)) or (grip_b and env_ok(a)):
                    continue
                bad.append((a, b, dist))
            ncheck += 1
            if bad:
                ncol += 1
                a, b, dist = bad[0]
                w0 = worst.get((a, b), (0, 0.0))
                worst[(a, b)] = (w0[0] + 1, min(w0[1], dist * 1000))
    T = d.d.time
    if not quiet:
        print(f"坡度 {alpha:.0f}° 坡向 {phi:.0f}°，卸 {P.done}/{len(P.order)} 箱（{len(P.parts)} 组），仿真 {T:.0f} s（实际 {time.time()-t0:.0f} s），平均 {T / max(P.done, 1):.1f} s/箱")
        print(f"最小倾覆裕度 {minm:.1f} cm   最大倾斜 {maxtilt:.2f}°   臂关节最大跟踪误差 {maxerr:.2f}°")
        print(f"实际姿态碰撞复查：{ncheck} 次采样，其中 {ncol} 次有 >6 mm 的贯穿 " + (" ".join(f"{a}-{b}: {n}次 最深 {-m:.1f} mm" for (a, b), (n, m) in worst.items()) if worst else ""))
        print("车体最终位置 y=%.3f m" % P._plane_y(d.d.xpos[d.base]))
    return dict(done=P.done, boxes=len(P.order), T=T, margin=minm, tilt=maxtilt, err=maxerr, ncol=ncol, ncheck=ncheck, per_box=T / max(P.done, 1))


if __name__ == "__main__":
    a = sys.argv[1:]
    alpha = float(a[0]) if len(a) > 0 else 0.0
    phi = float(a[1]) if len(a) > 1 else 0.0
    nparts = int(a[2]) if len(a) > 2 and int(a[2]) > 0 else None
    drive = (a[3] != "0") if len(a) > 3 else True
    acc = float(a[4]) if len(a) > 4 else None
    scale = float(a[5]) if len(a) > 5 else None
    susp = a[6] if len(a) > 6 else "off"
    run(alpha, phi, nparts, drive, acc, scale, susp)
