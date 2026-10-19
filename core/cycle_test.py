"""装卸货动作循环在 10° 坡上的倾覆仿真（MuJoCo，机械臂为真实关节，动力学包含臂的运动）

动作序列取自 chassis_A_3d.html 的"自动演示"：准备 → 前伸 → 吸取 → 举高 → 侧向 → 准备 → 收纳，
每个姿态停留 2.8 s。吸取完成后吸着 50 kg 的箱子，经 举高、侧向 搬运，回到准备姿态时放下。
车体停在坡上不动（轮子抱死），在不同坡向（车头相对下坡方向）下各跑一遍。

指标：
  裕度 = 质心沿重力方向在坡面上的投影，到支承多边形最近边的距离 [cm]（<0 即已越出，会翻）
  最大倾斜 = 车体相对坡面的额外倾斜角 [°]（>3° 认为已倾翻）

运行：python cycle_test.py          （多进程，约一两分钟）
"""
import math, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

SEQ = ["ready", "reach", "pick", "lift", "side", "ready", "stow"]
DWELL = 4.0
CARRY_ON, CARRY_OFF = "lift", 5          # 到 举高 时吸起箱子；序列第 5 步(回到 准备)时放下
SLOPE = 10.0
PAYLOAD = 50.0
AZS = list(range(0, 360, 45))
BALLAST_POS = (-0.55, 0.0, 0.20)


def margin_cm(demo):
    """质心投影到最近支承边的距离（cm），负值 = 已越出"""
    n = demo.normal()
    com = demo.d.subtree_com[demo.base]
    proj = com - np.array([0, 0, float(n @ com) / n[2]])
    pts = {w: demo.d.xpos[b] - n * demo.r for w, b in zip(("fl", "fr", "rl", "rr"), demo.wb)}
    poly = ([pts["fl"], pts["fr"], (pts["rl"] + pts["rr"]) / 2] if demo.susp
            else [pts["fl"], pts["fr"], pts["rr"], pts["rl"]])
    ctr = sum(poly) / len(poly)
    best = 1e9
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        e = b - a
        nn = np.cross(n, e); nn /= np.linalg.norm(nn)
        if nn @ (ctr - a) > 0:
            nn = -nn                       # 朝外
        best = min(best, -float(nn @ (proj - a)))   # 内侧为正
    return best * 100


def run(job):
    import tip_view
    from tip_view import TipDemo
    import chassis_sim
    susp, az, ballast, rate = job
    cfg = chassis_sim.CONFIG
    if ballast:                              # 后部低位配重并入底盘（质量、质心）
        mb = cfg["chassis_mass"] - 4 * (cfg["wheel_mass"] + cfg["steer_mass"])
        com = (mb * np.array(cfg["chassis_com"]) + ballast * np.array(BALLAST_POS)) / (mb + ballast)
        chassis_sim.CONFIG = dict(cfg, chassis_mass=cfg["chassis_mass"] + ballast, chassis_com=tuple(com))
        tip_view.CONFIG = chassis_sim.CONFIG
    tip_view.ARM_RATE = math.radians(rate); tip_view.ARM_ACC = math.radians(rate * 2.0)   # 0.5 s 加速到最大转速
    d = TipDemo("ready", 0.0, susp, az, SLOPE)
    mind, maxtilt, where = 1e9, 0.0, ""
    step_i = 0
    t_end = DWELL * len(SEQ)
    n = int(t_end / d.m.opt.timestep)
    carrying = False
    cur = -1
    for k in range(n):
        idx = int(k * d.m.opt.timestep / DWELL)
        if idx != cur:
            cur = idx
            d.set_pose(tip_view.ORDER.index(SEQ[idx]))
            if SEQ[idx] == CARRY_ON and not carrying:
                d.set_payload(PAYLOAD); carrying = True
            if idx == CARRY_OFF and carrying:
                d.set_payload(0.0); carrying = False
        d.step()
        if k % 50 == 0 and k > 1000:
            m_ = margin_cm(d)
            tl = d.tilt_deg()
            maxtilt = max(maxtilt, tl)
            if m_ < mind:
                mind, where = m_, SEQ[idx] + ("(吸着)" if carrying else "")
            if tl > 10:
                break
    return job, mind, maxtilt, where


def main():
    rates = [10, 20, 30, 45]
    cfgs = [("off", 0.0), ("on", 0.0), ("off", 100.0)]
    jobs = [(s, az, b, r) for (s, b) in cfgs for r in rates for az in AZS]
    with ProcessPoolExecutor() as ex:
        res = {(j[0], j[1], j[2], j[3]): r for j, *r in ex.map(run, jobs)}
    names = {("off", 0.0): "悬挂锁死，无配重", ("on", 0.0): "悬挂开启，无配重", ("off", 100.0): "悬挂锁死，后部配重 100 kg"}
    print(f"装卸货循环（{' → '.join(SEQ)}，每姿态 {DWELL:.0f} s，吸着 {PAYLOAD:.0f} kg），坡度 {SLOPE:.0f}°，8 个坡向")
    print("关节最大转速 °/s ：最小裕度(cm，最危险坡向) / 是否有坡向倾翻")
    for key, nm in names.items():
        print(f"\n【{nm}】")
        for r in rates:
            rows = [(res[(key[0], az, key[1], r)], az) for az in AZS]
            worst = min(rows, key=lambda x: x[0][0])
            tipped = [az for (m_, t_, w_), az in rows if m_ <= 0 or t_ >= 3]
            ws = worst[0]
            print(f"  {r:>3}°/s: 最小裕度 {ws[0]:6.1f} cm @坡向 {worst[1]:>3}° 姿态 {ws[2]:<10} "
                  + ("→ 稳定" if not tipped else f"→ 倾翻（坡向 {','.join(map(str, tipped))}）"))


if __name__ == "__main__":
    main()
