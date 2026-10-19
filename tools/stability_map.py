import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""卸货全过程里每个姿态的静态倾翻角（悬挂锁死，四轮矩形支承）按坡向取最小值，并比较 质心/轮距/轴距 的设计变化。
运行：python tools/stability_map.py
"""
import json, math
import numpy as np, mujoco
import chassis_sim as C
from tip_view import TipDemo

PLAN = json.load(open("data/unload_plan2.json", encoding="utf-8"))
HEADINGS = list(range(0, 360, 15))


def poses():
    out = []                                                    # (关节角, 负载 kg)
    for g in PLAN["picks"].values():
        L = g["legs"]
        for leg, pl in (("approach", 0.0), ("in", 0.0), ("out", 50.0), ("carry", 50.0)):
            out += [(np.array(q), pl) for q in L[leg]]
    for key in ("place_down", "place_up"):
        v = PLAN[key]
        for seq in (v.values() if isinstance(v, dict) else [v]):
            out += [(np.array(q), 50.0) for q in seq]
    return out


def ray_dist(poly, p, u):
    """从 p 沿 u 到凸多边形边界的距离"""
    best = 1e9
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        den = u[0] * e[1] - u[1] * e[0]
        if abs(den) < 1e-12:
            continue
        w = a - p
        t = (w[0] * e[1] - w[1] * e[0]) / den
        s = (w[0] * u[1] - w[1] * u[0]) / den
        if t > 0 and -1e-9 <= s <= 1 + 1e-9:
            best = min(best, t)
    return best


def evaluate(extra, plist):
    d = TipDemo("stow", 0, "off", 0.0, 0.0, cfg_extra=extra)
    m, dd = d.m, d.d
    worst = np.full(len(HEADINGS), 90.0)
    for q, pl in plist:
        d.set_payload(pl)
        for adr, qq in zip(d.arm_qadr, q):
            dd.qpos[adr] = qq
        mujoco.mj_forward(m, dd)
        total = float(m.body_subtreemass[d.base])
        com = dd.subtree_com[d.base].copy()
        pts = np.array([dd.xpos[b][:2] for b in d.wb])
        # 矩形支承：按绕质心极角排序
        c0 = pts.mean(axis=0)
        order = np.argsort([math.atan2(p[1] - c0[1], p[0] - c0[0]) for p in pts])
        poly = pts[order]
        h = com[2]
        for i, phi in enumerate(HEADINGS):
            u = np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi))])
            s = ray_dist(poly, com[:2], u)
            worst[i] = min(worst[i], math.degrees(math.atan2(s, h)))
    return worst


if __name__ == "__main__":
    plist = poses()
    seen, uniq = set(), []
    for q, pl in plist:                                          # 去重，减少计算
        k = (tuple(np.round(q, 2)), pl)
        if k not in seen:
            seen.add(k); uniq.append((q, pl))
    print("姿态数 %d（去重后）" % len(uniq), flush=True)
    variants = [("当前 质心x=-0.25 轴距0.89 轮距0.73", {})]
    for cx in (-0.05, -0.15):
        variants.append(("质心 x=%.2f" % cx, dict(chassis_com=(cx, 0.0, 0.40))))
    for ts in (1.25, 1.5):
        variants.append(("轮距 x%.2f（%.2f m）" % (ts, 0.73 * ts), dict(track_scale=ts)))
    for ws in (1.25, 1.5):
        variants.append(("轴距 x%.2f（%.2f m）" % (ws, 0.89 * ws), dict(wheelbase_scale=ws)))
    variants.append(("轮距x1.25 + 轴距x1.25", dict(track_scale=1.25, wheelbase_scale=1.25)))
    for name, ex in variants:
        w = evaluate(ex, uniq)
        by = {phi: w[i] for i, phi in enumerate(HEADINGS)}
        print("%-34s 最小 %4.1f°（坡向 %3d°） | 0°:%4.1f 90°:%4.1f 180°:%4.1f 270°:%4.1f" % (
            name, w.min(), HEADINGS[int(w.argmin())], by[0], by[90], by[180], by[270]), flush=True)
