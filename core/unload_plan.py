"""烟箱卸货动作规划与优化（箱墙在车厢里，卸下的箱子放到机器人后面的理料机构上）

场景（取自 chassis_A_3d.html，单位 m，车厢截面 3 × 3，长 5.5）：
  车厢：x ∈ [-2.5, 3.0]，y ∈ [-1.5, 1.5]，z ∈ [0, 3.0]；机器人停在车厢里，原点 x=0
  箱墙：前表面 x=1.402；前排 8 列（宽 0.3/0.4×6/0.3，交错码放）× 8 层（每层 0.25 高，总高 2.0），烟箱 0.30(深)×0.40(宽)×0.25(高)，约 50 kg
  理料机构：在机器人后面，出箱位辊床 x∈[-1.05,-0.65]，上表面高 0.73；箱子放在它的中心 (-0.85, 0)
  机器人可以沿 y 方向在车厢里横移（舵轮），在不同工位取不同列。

规划内容：
  1. 逆运动学：每个箱子 → 吸取姿态；放箱姿态；"carry 姿态"（臂收拢、质心低）
  2. 一个周期：carry → 预取 → 吸取 → 退回 → carry → 转 J1 到 180° → 预放 → 放箱 → 抬起 → 转回 carry
  3. 取箱顺序：行优先（扫描式）vs 列优先（一列从上到下，工位少）vs 贪心（最小耗时）
  4. 速度优化：在 10° 坡上用 MuJoCo 动力学仿真，搜索各阶段关节转速，使整个周期里质心投影到支承边的裕度 ≥ 3 cm，且周期最短
运行：python unload_plan.py      （多进程，几分钟）
"""
import json, math, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import arm_kin as K

# ---------------- 场景 ----------------
WALL_FRONT = 1.402
BOX_D, BOX_W, BOX_H = K.BOX                     # 0.30 × 0.40 × 0.25
COL_EDGES = [-1.5, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.5]
COLS = [(COL_EDGES[i] + COL_EDGES[i + 1]) / 2 for i in range(8)]
LAYERS = [BOX_H / 2 + BOX_H * k for k in range(8)]          # 箱中心高度
BED_CENTER = np.array([-0.85, 0.0, 0.73 + BOX_H / 2 + 0.03])  # 放箱：箱子中心（世界），高出辊床 3 cm 再放下
STATIONS = [round(y, 2) for y in np.linspace(-0.9, 0.9, 7)]
PLATE_HALF_W = 0.29
Z0 = K.Z_AXLE

SLOPE = 10.0
MARGIN_MIN_CM = 3.0
SUCTION_S, RELEASE_S = 0.6, 0.5                  # 吸附 / 释放 停留时间 [s]
STATION_SPEED = 0.3                              # 机器人横移速度 [m/s]
VMAX = np.radians([140, 110, 110, 130, 175, 175, 255])  # GP110B 各轴最大速度 S/L/E/U/R/B/T（安川样本）[rad/s]


def box_center_world(col, layer):
    return np.array([WALL_FRONT + BOX_D / 2, COLS[col], LAYERS[layer]])


def plate_y(col):
    """吸盘板中心的 y（两端的窄箱让吸盘板向车厢内侧偏一点，避免板子撞侧壁）"""
    return float(np.clip(COLS[col], -(1.5 - PLATE_HALF_W - 0.01), 1.5 - PLATE_HALF_W - 0.01))


# ---------------- 取箱 IK ----------------
def pick_theta(col, layer, station, standoff=0.0, seed=None):
    """station：车体中心 y。standoff：吸盘面离箱前表面的距离（预取点 > 0）。返回 (θ, 残差) 或 (None, ..)"""
    c = np.array([WALL_FRONT + BOX_D / 2 - standoff, plate_y(col), LAYERS[layer]])
    c_b = c - np.array([0, station, Z0])
    p = K.j7_for_box_center(c_b, K.R_FACE_FWD)
    seeds = ([seed] if seed is not None else []) + K.SEEDS
    q, err = K.ik(p, K.R_FACE_FWD, seeds)
    if err > 5e-4:
        return None, err
    ok, why = K.keepout_ok(q, station, holding=False, grip_contact=standoff == 0.0)
    return (q if ok else None), (err if ok else 9.9)


def best_station(col, layer, prev_station=None):
    """能取到这个箱子的工位里，偏移最小的（臂尽量对称）；若 prev_station 也可行就优先用它（少横移）"""
    cands = []
    for s in STATIONS:
        q, e = pick_theta(col, layer, s)
        q0, e0 = pick_theta(col, layer, s, standoff=0.25) if q is not None else (None, 1)
        if q is not None and q0 is not None:
            cands.append((abs(plate_y(col) - s), s, q, q0))
    if not cands:
        return None
    if prev_station is not None:
        for off, s, q, q0 in cands:
            if s == prev_station and off <= 0.65:
                return s, q, q0
    off, s, q, q0 = min(cands)
    return s, q, q0


# ---------------- carry 姿态：箱子抓在手里时，质心最低、倾覆裕度最大的收拢姿态 ----------------
def static_tip(theta, payload):
    """该臂姿态（锁死悬挂）最危险坡向的倾翻角 [°]"""
    from chassis_sim import CONFIG
    import tip_analysis as T
    r, mw = CONFIG["wheel_radius"], CONFIG["wheel_mass"]
    M_arm, c_arm = K.arm_com(theta, payload)
    import arm_abb as AB
    ch_m = CONFIG["chassis_mass"] - 4 * (mw + CONFIG["steer_mass"])
    mb = AB.base_mass(CONFIG["arm_mass"])                                # 固定底座并入底盘
    pb = np.array([AB.BASE_COM[0], AB.BASE_COM[1], 0.79 + AB.BASE_COM[2]])
    com = (ch_m * np.array(CONFIG["chassis_com"]) + mb * pb + M_arm * (c_arm + np.array([0, 0, r]))) / (ch_m + mb + M_arm)
    cfg = dict(CONFIG, suspension="off", body_mass=float(ch_m + mb + M_arm), body_com=tuple(com))
    return min(T.analytic_tip(cfg, "off", p) for p in range(0, 360, 10))


def find_carry():
    best = (0, None, None)
    q_prev = K.preset_theta("ready")
    for x in (0.25, 0.35, 0.45, 0.55, 0.65):
        for z in (0.95, 1.10, 1.25, 1.40, 1.55):
            c_b = np.array([x, 0.0, z - Z0])
            p = K.j7_for_box_center(c_b, K.R_FACE_FWD)
            q, err = K.ik_tidy(p, K.R_FACE_FWD, K.SEEDS, n_random=10, j1_max=0.6)
            if err > 5e-4:
                continue
            ok, why = K.keepout_ok(q, 0.0, holding=True)
            if not ok:
                continue
            # 转到 180° 后同样要不碰：J1 加 π
            q2 = q.copy(); q2[0] = q[0] + math.pi if q[0] < 0 else q[0] - math.pi
            if q2[0] < K.LIM_TH[0, 0] or q2[0] > K.LIM_TH[0, 1]:
                continue
            ok2, _ = K.keepout_ok(q2, 0.0, holding=True)
            if not ok2:
                continue
            tip = min(static_tip(q, 50.0), static_tip(q2, 50.0))
            if tip > best[0]:
                best = (tip, q, (x, z))
    if best[1] is None:      # GP110B 前臂长，吸盘朝前时没法收到车体上方，用 ready 姿态代替
        q = K.preset_theta("ready")
        return static_tip(q, 50.0), q, (0.0, 0.0)
    return best


def place_theta(dz=0.0):
    c = BED_CENTER + np.array([0, 0, dz]) - np.array([0, 0, Z0])
    p = K.j7_for_box_center(c, K.R_FACE_BACK)
    seeds = [K.preset_theta("rear")] + K.SEEDS
    q, err = K.ik_tidy(p, K.R_FACE_BACK, seeds, n_random=40, check=lambda qq: K.keepout_ok(qq, 0.0, holding=True, grip_contact=True)[0])
    if q is None:
        return None
    ok, why = K.keepout_ok(q, 0.0, holding=True, grip_contact=True)
    return (q if (err < 5e-4 and ok) else None)


# ---------------- 时间模型 ----------------
def seg_time(q0, q1, rate_scale=1.0, acc_time=0.5):
    """梯形速度曲线：每关节限速 VMAX*rate_scale，加速时间 acc_time"""
    d = np.abs(np.asarray(q1) - np.asarray(q0))
    v = VMAX * rate_scale
    a = v / acc_time
    t = []
    for di, vi, ai in zip(d, v, a):
        if di < 1e-6:
            t.append(0.0)
        elif di >= vi * vi / ai:
            t.append(di / vi + vi / ai)
        else:
            t.append(2 * math.sqrt(di / ai))
    return max(t)


def cycle_waypoints(q_carry, q_pre, q_pick, q_place_hi, q_place):
    """一个周期的路点：(θ, 阶段名, 停留 s, 动作)"""
    q_carry_back = q_carry.copy(); q_carry_back[0] = (q_carry[0] + math.pi) if q_carry[0] < 0 else (q_carry[0] - math.pi)
    return [
        (q_pre, "ext", 0.0, None),
        (q_pick, "ext", SUCTION_S, "grab"),
        (q_pre, "ext", 0.0, None),
        (q_carry, "carry", 0.0, None),
        (q_carry_back, "swing", 0.0, None),
        (q_place_hi, "place", 0.0, None),
        (q_place, "place", RELEASE_S, "release"),
        (q_place_hi, "place", 0.0, None),
        (q_carry_back, "place", 0.0, None),
        (q_carry, "swing", 0.0, None),
    ]


def cycle_time(wps, q_start, rates):
    t, q = 0.0, q_start
    for qq, phase, dwell, act in wps:
        t += seg_time(q, qq, rates[phase]) + dwell
        q = qq
    return t, q


if __name__ == "__main__":
    t0 = time.time()
    tip_c, q_carry, pos_c = find_carry()
    print(f"carry 姿态：箱心 x={pos_c[0]}, z={pos_c[1]}（吸着 50 kg 时最危险坡向倾翻角 {tip_c:.1f}°）  θ°={np.degrees(q_carry).round(0)}")
    qp_hi, qp = place_theta(0.12), place_theta(0.0)
    print("放箱姿态：", None if qp is None else np.degrees(qp).round(0))
    # 箱墙可达性
    print("\n箱墙可达性（每格：能取到时的最小横向偏移 cm，× = 取不到）：行=层(上→下)，列=从左到右")
    reach = {}
    for L in range(7, -1, -1):
        row = []
        for C in range(8):
            r = best_station(C, L)
            reach[(C, L)] = r
            row.append("×" if r is None else f"{abs(plate_y(C) - r[0]) * 100:3.0f}")
        print(f"  层{L + 1} z={LAYERS[L]:.2f}: " + " ".join(f"{x:>3}" for x in row))
    n_ok = sum(v is not None for v in reach.values())
    print(f"可取箱子 {n_ok}/64   用时 {time.time() - t0:.0f} s")
    json.dump({"carry": q_carry.tolist(), "place": qp.tolist(), "place_hi": qp_hi.tolist()}, open("data/unload_poses.json", "w"))
