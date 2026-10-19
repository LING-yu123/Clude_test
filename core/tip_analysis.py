"""倾覆分析：七轴臂 800 kg + 底盘 1000 kg + 末端负载 0–50 kg（MuJoCo 仿真 + 支承多边形解析对照）

做三件事：
  1. 静态倾覆角：对每个 机械臂姿态 × 负载 × 悬挂(开/关)，在斜坡(用倾斜重力等效)上二分搜索倾翻坡度。
     坡向 φ = 下坡方向与车头方向的夹角：0° 车头朝下坡，180° 车头朝上坡，90° 左侧朝下坡，270° 右侧朝下坡。
     同时给出解析最危险坡向（支承多边形 + 质心高度），并在该坡向上再跑一遍仿真。
  2. 抗倾覆系数 K = 稳定力矩 / 倾覆力矩（绕支承多边形各边，取最小）：底盘重力为稳定力矩，
     机械臂+负载若质心越过边线则为倾覆力矩。页面里的目标是 K ≥ 1.5；K = ∞ 表示质心都在边线内侧。
  3. 动态：平地制动，看后轮载荷和俯仰（前向倾覆的临界减速度）。

运行：python tip_analysis.py            （约几分钟，多进程）
输出：tip_results.csv、tip_polar.png（如果装了 matplotlib）
"""
import csv, math, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

POSES = ["ready", "stow", "reach", "pick", "lift", "side", "rear", "zero"]
POSE_CN = {"ready": "准备", "stow": "收纳", "reach": "前伸", "pick": "吸取",
           "lift": "举高", "side": "侧向", "rear": "朝后扫描", "zero": "零位"}
PAYLOADS = [0.0, 50.0]
SUSP = ["on", "off"]                 # on = 后桥自由摆动(三点支承)，off = 锁死(四点支承)
PRINCIPAL = [0, 90, 180, 270]
HI, ITER = 60.0, 8


# ---------------- 解析：支承多边形 ----------------
def support_polygon(cfg, susp):
    r = cfg["wheel_radius"]
    L, B = cfg["wheelbase"], cfg["track"]
    fl, fr = (L / 2, B / 2), (L / 2, -B / 2)
    rl, rr = (-L / 2, B / 2), (-L / 2, -B / 2)
    if susp == "on":                  # 前两轮 + 后桥等效支承点（三点支承）
        return np.array([fl, fr, (-L / 2, 0.0)])   # 后桥只能绕 x 轴转，对俯仰是刚性的，等效支承点在后轮轴线中点
    return np.array([fl, rl, rr, fr])  # 逆时针不重要，下面按质心自行判断内外


def _edges(poly):
    n = len(poly)
    return [(poly[i], poly[(i + 1) % n]) for i in range(n)]


def com_total(cfg):
    from chassis_sim import composite_com
    M, x, y, h = composite_com(cfg)
    return M, np.array([x, y]), h


def analytic_tip(cfg, susp, phi_deg):
    """质心沿下坡方向射线与支承多边形边界相交的距离 s，倾翻角 = atan(s/h)"""
    poly = support_polygon(cfg, susp)
    _, c, h = com_total(cfg)
    u = np.array([math.cos(math.radians(phi_deg)), math.sin(math.radians(phi_deg))])
    ctr = poly.mean(0)
    best = None
    for a, b in _edges(poly):
        e = b - a
        n = np.array([e[1], -e[0]]); n /= np.linalg.norm(n)
        if n @ (ctr - a) > 0:          # 让法向朝外
            n = -n
        nu = n @ u
        if nu <= 1e-9:
            continue
        t = n @ (a - c) / nu
        if t >= 0 and (best is None or t < best):
            best = t
    if best is None:
        return 90.0
    return math.degrees(math.atan2(best, h))


def worst_az(cfg, susp):
    angs = [(analytic_tip(cfg, susp, p), p) for p in range(0, 360, 5)]
    return min(angs)[1], min(angs)[0]


def stability_K(cfg, susp):
    """K = Σ稳定力矩 / Σ倾覆力矩，绕支承多边形各边，取最小；机械臂+负载越过边线算倾覆"""
    r, mw = cfg["wheel_radius"], cfg["wheel_mass"]
    L, B = cfg["wheelbase"], cfg["track"]
    arm_c = np.array(cfg["_arm_xy"]); tool_c = np.array(cfg["_tool_xy"])
    ch_c = np.array(cfg["_chassis_xy"])
    groups = [(cfg["chassis_mass"], ch_c), (cfg["arm_mass"], arm_c), (cfg["payload"], tool_c)]
    poly = support_polygon(cfg, susp)
    ctr = poly.mean(0)
    Kmin = math.inf
    for a, b in _edges(poly):
        e = b - a
        n = np.array([e[1], -e[0]]); n /= np.linalg.norm(n)
        if n @ (ctr - a) < 0:
            n = -n                                  # 法向朝内
        rest = over = 0.0
        for m, c in groups:
            mom = m * G * (n @ (c - a))
            if mom >= 0:
                rest += mom
            else:
                over += -mom
        if over > 0:
            Kmin = min(Kmin, rest / over)
    return Kmin


G = 9.81


# ---------------- 仿真：静态倾覆 ----------------
def tipped(cfg, alpha, phi):
    from chassis_sim import Sim, WHEELS
    sim = Sim(cfg)
    a, p = math.radians(alpha), math.radians(phi)
    sim.reset((math.sin(a) * math.cos(p), math.sin(a) * math.sin(p), -math.cos(a)))
    for n in WHEELS:                                # 轮子抱死
        sim.m.dof_damping[sim.m.jnt_dofadr[sim.m.joint(f"j_{n}").id]] = 1e6
    sim.step(1200)
    tilt, front = 0.0, []
    for _ in range(200):
        sim.step()
        r, pt = sim.roll_pitch()
        tilt = max(tilt, abs(r), abs(pt))
        L = sim.wheel_loads()
        front.append(min(L["fl"], L["fr"]) / sim.weight)
    if tilt > 3.0:
        return True
    # 悬挂开启时是三点支承：前轮开始离地就是越过了支承边界（之后车体只是被摆角限位慢慢挡住，不能算稳定）
    return cfg.get("suspension") == "on" and float(np.mean(front[-100:])) < 0.02


def sim_tip_angle(cfg, phi):
    c = dict(cfg, friction=3.0)                    # 摩擦取大，排除侧滑，只看几何倾覆
    lo, hi = 0.0, HI
    for _ in range(ITER):
        mid = (lo + hi) / 2
        if tipped(c, mid, phi):
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def job(args):
    key, cfg, phi = args
    return key, phi, sim_tip_angle(cfg, phi)


# ---------------- 主流程 ----------------
def build_cases():
    from chassis_sim import CONFIG, cfg_for_pose
    from a_model import get_visuals
    cases = []
    for pose in POSES:
        v = get_visuals(pose, detail="full")
        for pl in PAYLOADS:
            for susp in SUSP:
                cfg = cfg_for_pose(dict(CONFIG, suspension=susp), pose, pl)
                # 供解析 K 用的各部分水平位置
                mw, L, B = cfg["wheel_mass"], cfg["wheelbase"], cfg["track"]
                wheel_sum = np.array([0.0, 0.0])
                cfg["_chassis_xy"] = tuple(np.array(CONFIG["chassis_com"][:2]))
                cfg["_arm_xy"] = tuple(v["arm_com"][:2])
                cfg["_tool_xy"] = tuple(v["tool_pos"][:2])
                cases.append(((pose, pl, susp), cfg))
    return cases


def dynamic_brake(cases):
    from chassis_sim import brake_test, cfg_for_pose, CONFIG
    print("\n[3] 平地制动（初速 1.0 m/s，悬挂开启）：后轮最低载荷占比 / 最大俯仰")
    print(f"  {'姿态':<8}{'负载':>5} {'临界减速度(解析)':>16}  " + "".join(f"{lab:>16}" for lab in ("斜坡 1.0 m/s²", "斜坡 2.0 m/s²", "斜坡 3.0 m/s²", "满力矩硬刹")))
    rows = []
    for pose in ("ready", "reach", "pick", "lift", "side"):
        for pl in (0.0, 50.0):
            cfg = [c for k, c in cases if k == (pose, pl, "on")][0]
            M, com, h = com_total(cfg)
            a_crit = G * (cfg["wheelbase"] / 2 - com[0]) / h
            out = []
            for dec in (1.0, 2.0, 3.0, None):
                b = brake_test(dict(cfg, friction=0.8), v=1.0, decel=dec)
                out.append(f"{b['rear_load_min']*100:4.0f}% / {b['pitch_max']:.2f}°")
            print(f"  {POSE_CN[pose]:<8}{pl:>4.0f}  {a_crit:>13.2f}   " + "".join(f"{o:>16}" for o in out))
            rows.append((pose, pl, a_crit, out))
    return rows


def main():
    cases = build_cases()
    cfgs = dict(cases)
    # 解析最危险坡向
    jobs, wa = [], {}
    for key, cfg in cases:
        pose, pl, susp = key
        phi_w, ang_w = worst_az(cfg, susp)
        wa[key] = (phi_w, ang_w)
        phis = list(PRINCIPAL) + ([] if phi_w in PRINCIPAL else [phi_w])
        for phi in phis:
            jobs.append((key, cfg, phi))
    print(f"提交 {len(jobs)} 个二分搜索（每个约 {ITER} 次仿真），进程数 {os.cpu_count()} ...", flush=True)
    res = {}
    with ProcessPoolExecutor() as ex:
        for done, (key, phi, ang) in enumerate(ex.map(job, jobs, chunksize=2), 1):
            res[(key, phi)] = ang
            if done % 20 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)

    rows = []
    print("\n[1][2] 静态倾覆（仿真，单位 °）  坡向：0°=车头朝下坡  180°=车头朝上坡  90°=左侧下坡  270°=右侧下坡")
    print(f"{'姿态':<8}{'负载':>4} {'悬挂':<3}{'整机质心(x,y,z)m':>20}{'0°':>7}{'90°':>7}{'180°':>7}{'270°':>7}"
          f"{'最危险坡向':>10}{'仿真':>7}{'解析':>7}{'K':>7}")
    for key, cfg in cases:
        pose, pl, susp = key
        M, com, h = com_total(cfg)
        phi_w, ang_w = wa[key]
        sim_w = res[(key, phi_w)]
        K = stability_K(cfg, susp)
        ks = "∞" if math.isinf(K) else f"{K:.2f}"
        print(f"{POSE_CN[pose]:<8}{pl:>4.0f} {'开' if susp=='on' else '关':<3}"
              f"{com[0]:>7.2f},{com[1]:>5.2f},{h:>5.2f}"
              + "".join(f"{res[(key, p)]:>7.1f}" for p in PRINCIPAL)
              + f"{phi_w:>9d}°{sim_w:>7.1f}{ang_w:>7.1f}{ks:>7}")
        rows.append([POSE_CN[pose], pl, susp, round(M, 1), round(com[0], 3), round(com[1], 3), round(h, 3)]
                    + [round(res[(key, p)], 1) for p in PRINCIPAL] + [phi_w, round(sim_w, 1), round(ang_w, 1), ks])

    with open("output/tables/tip_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["姿态", "负载kg", "悬挂", "整机质量kg(含轮)", "质心x", "质心y", "质心高", "倾翻角0°", "倾翻角90°", "倾翻角180°",
                    "倾翻角270°", "最危险坡向°", "最危险倾翻角_仿真", "最危险倾翻角_解析", "抗倾覆系数K"])
        w.writerows(rows)

    dynamic_brake(cases)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(projection="polar"))
        for ax, susp in zip(axs, SUSP):
            for pose, color in (("ready", "#1f77b4"), ("reach", "#d62728"), ("side", "#2ca02c"), ("pick", "#ff7f0e")):
                cfg = cfgs[(pose, 50.0, susp)]
                ph = np.arange(0, 361, 5)
                ax.plot(np.radians(ph), [analytic_tip(cfg, susp, p) for p in ph], color=color, label=f"{POSE_CN[pose]} 50kg")
            ax.set_title("悬挂开启(三点支承)" if susp == "on" else "悬挂关闭(四点支承)")
            ax.set_theta_zero_location("E"); ax.set_ylim(0, 50)
            ax.legend(loc="lower right", fontsize=8)
        fig.suptitle("倾覆坡度（半径，°）随下坡方向的变化（0° = 车头朝下坡）")
        fig.savefig("output/images/tip_polar.png", dpi=110, bbox_inches="tight")
        print("\n已保存 tip_results.csv, tip_polar.png")
    except Exception as e:
        print("\n已保存 tip_results.csv（未生成图：%s）" % e)


if __name__ == "__main__":
    main()
