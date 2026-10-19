import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""卸货节拍优化：扫 (加速时间, 速度比例)，在最差坡向 10° 坡上看 节拍 / 倾覆裕度 / 倾斜 / 跟踪误差。
运行：python tools/cycle_opt.py [箱数=6]"""
import json, math
import numpy as np
import unload_player as U
from tip_view import TipDemo
from cycle_test import margin_cm

NB = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def run(acc, scale, phi, alpha=10.0):
    U.ACC_TIME = acc
    U.AMAX = U.VMAX / acc
    plan = json.load(open("data/unload_plan2.json", encoding="utf-8")); plan["order"] = plan["order"][:NB]
    d = TipDemo("stow", 0, "off", phi, alpha)
    P = U.UnloadPlayer(d, plan, True); P.speed_scale = scale; P.start()
    dt = d.m.opt.timestep; k = 0; mm, mt, me = 1e9, 0.0, 0.0
    while P.active and d.d.time < 4000:
        P.update(dt); d.step(); k += 1
        if k % 50 == 0:
            mm = min(mm, margin_cm(d)); mt = max(mt, d.tilt_deg())
            qa = np.array([d.d.qpos[a] for a in d.arm_qadr]); me = max(me, float(np.max(np.abs(qa - d.q_cmd))) * 180 / math.pi)
    return d.d.time / max(P.done, 1), P.done, mm, mt, me


if __name__ == "__main__":
    print("加速时间 速度比例 | 坡向 | 节拍 s/个 | 卸出 | 最小裕度 cm | 最大倾斜° | 跟踪误差°", flush=True)
    for acc, sc in ((1.5, 0.6), (1.2, 0.7), (1.0, 0.8), (0.8, 0.9), (0.6, 1.0)):
        for phi in (90.0, 180.0):
            cyc, n, mm, mt, me = run(acc, sc, phi)
            print("  %.1f s   %.1f   | %3.0f° | %6.1f | %d/%d | %6.1f | %5.2f | %5.2f" % (acc, sc, phi, cyc, n, NB, mm, mt, me), flush=True)
