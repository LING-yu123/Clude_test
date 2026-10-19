import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))   # 让 core/ 里的模块可被导入
"""用 MuJoCo 动力学把规划好的卸货动作完整跑一遍，检查：倾覆裕度、臂的跟踪误差、车体横移精度、实际姿态下的碰撞。
运行：python validate_plan.py [坡度°=10] [坡向°=0] [箱子数=8] [drive=1]
"""
import json, math, sys, time
import numpy as np
import mujoco
from tip_view import TipDemo
from unload_player import UnloadPlayer
from cycle_test import margin_cm


def main():
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    phi = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    nbox = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    drive = (sys.argv[4] != "0") if len(sys.argv) > 4 else True
    plan = json.load(open("data/unload_plan2.json", encoding="utf-8"))
    plan["order"] = plan["order"][:nbox]
    d = TipDemo("stow", 0, "off", phi, alpha)
    P = UnloadPlayer(d, plan, drive=drive)
    from collision import Collider
    col = Collider()
    P.start()
    dt = d.m.opt.timestep
    minm, maxtilt, maxerr, ncol, ncheck = 1e9, 0.0, 0.0, 0, 0
    worst_pairs = {}
    k, t0 = 0, time.time()
    while P.active and d.d.time < 4000:
        P.update(dt); d.step(); k += 1
        if k % 50 == 0:
            minm = min(minm, margin_cm(d)); maxtilt = max(maxtilt, d.tilt_deg())
            qa = np.array([d.d.qpos[a] for a in d.arm_qadr])
            maxerr = max(maxerr, float(np.max(np.abs(qa - d.q_cmd))) * 180 / math.pi)
        if k % 100 == 0 and P.cur is not None and P.cur[0] in ("path", "wait"):          # 用实际关节角复查碰撞
            qa = np.array([d.d.qpos[a] for a in d.arm_qadr])
            holding = d.payload > 0
            st = P.picks["%d,%d" % P.box]["station"] if P.drive_enabled else 0.0
            # 车体实际位置相对工位的偏差，用来平移场景
            y_now = P._plane_y(d.d.xpos[d.base]) if P.drive_enabled else 0.0
            extracting = P.grabbed and P.paths_since_grab <= 1                  # 吸着后沿箱面拔出的那一段：箱子/夹具贴着箱墙是正常的
            mode = "pick" if (P.cur[0] == "wait" or extracting) else "free"
            pr = col.check(qa, station_y=y_now, holding=holding, mode="pick" if mode == "pick" else "free",
                           removed=set(tuple(b) for b in P.order[:P.k]))
            pr = [x for x in pr if x[2] < -0.004 and not (x[0].startswith('wb_') and x[1] in ('g_plate', 'g_tray', 'g_box'))]   # 贯穿 > 4 mm；夹具贴箱墙（吸取/拔出）是允许的
            ncheck += 1
            if pr:
                ncol += 1
                for a, b, dist in pr[:1]:
                    w0 = worst_pairs.get((a, b), (0, 0.0))
                    worst_pairs[(a, b)] = (w0[0] + 1, min(w0[1], dist * 1000))     # (次数, 最大贯穿 mm)
    print(f"坡度 {alpha:.0f}° 坡向 {phi:.0f}°，卸 {P.done}/{len(plan['order'])} 个，仿真 {d.d.time:.0f} s（实际 {time.time()-t0:.0f} s）")
    print(f"最小倾覆裕度 {minm:.1f} cm   最大倾斜 {maxtilt:.2f}°   臂关节最大跟踪误差 {maxerr:.2f}°")
    print(f"实际姿态碰撞复查：{ncheck} 次采样，其中 {ncol} 次有 >4 mm 的贯穿 " + (" ".join(f"{k[0]}-{k[1]}: {n}次 最深 {-m:.1f} mm" for k, (n, m) in worst_pairs.items()) if worst_pairs else ""))
    print("车体最终位置 y=%.3f m" % (P._plane_y(d.d.xpos[d.base])))


if __name__ == "__main__":
    main()
