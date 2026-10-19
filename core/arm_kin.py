"""七轴臂正/逆运动学（取自 chassis_A_3d.html 的关节链），用于规划装卸货动作。

坐标：以车体 base 系（原点在轮轴高度的车体中心，x 向前、y 向左、z 向上，单位 m）。
关节角 θ = 页面里的 q + 零位偏置 off（弧度），与 MuJoCo 里 arm_j1..arm_j7 的关节角一致。
J7 坐标系：z 轴 = 吸盘朝向（吸嘴在 +z 侧），x 轴 = 托板方向（底托板在 +x 侧，吸取姿态时 x 朝下）。
"""
import math
import numpy as np
import a_model as A

_, ROB = A._load()
JOINTS = ROB["joints"]
set_j2_shift = A.set_j2_shift      # 改 J2 在转盘上的前后位置（m），FK/IK 立即生效
OFF = np.radians(ROB["off"])
BASE = np.array(ROB["base"], float) / 1000.0 - np.array([0, 0, A.Z0 / 1000.0])   # 臂座在 base 系的位置
LIM = np.array([[ROB["jn"][i][1], ROB["jn"][i][2]] for i in range(7)], float)    # 页面里的 q 范围 [°]
LIM_TH = np.radians(LIM + np.array(ROB["off"])[:, None])                          # 关节角 θ 的范围 [rad]
PLATE_FACE = 0.222          # J7 原点到吸嘴面的距离 [m]
BOX = (0.30, 0.40, 0.25)    # 箱子 深(沿吸盘方向) × 宽 × 高 [m]
BOX_CENTER_J7 = np.array([0.069, 0.0, PLATE_FACE + BOX[0] / 2])   # 箱子中心在 J7 系的位置：坐在底托板上（托板在 x=+0.194），贴着吸嘴面


def preset_theta(name):
    pre = ROB["pre"][A.POSES.get(name, name)]
    return np.radians(pre[:7]) + OFF


def _rot(axis, ang):
    c, s = math.cos(ang), math.sin(ang)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def fk(theta, all_links=False):
    """返回 J7 坐标系在 base 系的 4×4 变换（all_links=True 时返回 7 个关节坐标系的列表）"""
    T = np.eye(4); T[:3, 3] = BASE
    Ts = []
    for j, th in zip(JOINTS, theta):
        Tl = np.eye(4)
        Tl[:3, :3] = _rot(j["a"], j["s"] * th)
        Tl[:3, 3] = np.array(j["o"], float) / 1000.0
        T = T @ Tl
        Ts.append(T.copy())
    return Ts if all_links else T


def pose_err(T, p_t, R_t):
    e_p = p_t - T[:3, 3]
    Rc = T[:3, :3]
    Re = R_t @ Rc.T
    e_r = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
    return np.concatenate([e_p, e_r])


Q_NOM = np.zeros(7)        # IK 零空间的中性姿态
LOCKED = []                # 锁死的关节（雅可比列清零）；ABB 臂里是 J4


def _wrap(q):
    q = q.copy()
    for i in (0, 4, 6):          # 可无限转的关节折回 ±π（姿态不变）
        w = (q[i] + math.pi) % (2 * math.pi) - math.pi
        if LIM_TH[i, 0] <= w <= LIM_TH[i, 1]:
            q[i] = w
    return q


def _solve_from(q, p_t, R_t, iters=200, tol=1e-4):
    q = np.array(q, float).copy()
    for it in range(iters):
        T = fk(q)
        e = pose_err(T, p_t, R_t)
        en = np.linalg.norm(e)
        if en < tol:
            break
        if it == 60 and en > 0.05:          # 60 步还没靠近：这个起点基本是发散/卡在限位，放弃，省时间
            break
        J = np.zeros((6, 7))
        for i in range(7):
            dq = q.copy(); dq[i] += 1e-6
            J[:, i] = (pose_err(fk(dq), p_t, R_t) - e) / 1e-6 * -1.0
        J[:, LOCKED] = 0
        lam = 0.05 if np.linalg.norm(e) > 0.02 else 0.005
        Jp = J.T @ np.linalg.inv(J @ J.T + lam ** 2 * np.eye(6))
        dq = Jp @ e + (np.eye(7) - Jp @ J) @ (0.25 * (Q_NOM - q))
        q = np.clip(q + 0.7 * dq, LIM_TH[:, 0], LIM_TH[:, 1])
    return q, float(np.linalg.norm(pose_err(fk(q), p_t, R_t)))


def ik_tidy(p_t, R_t, seeds, n_random=30, w=(0.4, 1, 1, 1, 1, 1, 1), j1_max=None, check=None, max_valid=None):
    """收集多个起点的所有收敛解，挑"离中性姿态最近"的（关节角小、不绕圈、不折叠）。预设姿态用它。"""
    rng = np.random.default_rng(1)
    starts = [np.array(s, float) for s in seeds] + [rng.uniform(LIM_TH[:, 0], LIM_TH[:, 1]) for _ in range(n_random)]
    best, best_c = None, 1e18
    nvalid = 0
    for s in starts:
        q, err = _solve_from(s, p_t, R_t)
        if err > 5e-4:
            continue
        q = _wrap(q)
        if j1_max is not None and abs(q[0]) > j1_max:
            continue
        if check is not None and not check(q):          # 只在"不碰东西"的解里挑
            continue
        c = float(np.sum(np.array(w) * np.abs(q - Q_NOM)))
        nvalid += 1
        if c < best_c:
            best, best_c = q, c
        if max_valid is not None and nvalid >= max_valid:
            break
    return (best, 0.0) if best is not None else (None, 1.0)


def ik(p_t, R_t, seeds, iters=200, tol=1e-4):
    """阻尼最小二乘 IK：J7 位置 p_t、姿态 R_t（base 系）。多初值，返回 (θ, 残差) 最好的一个。
    冗余自由度（7 轴 vs 6 维）用靠近初值 + 关节限位处理。"""
    best = (None, 1e9)
    for q0 in seeds:
        q = np.array(q0, float).copy()
        for _ in range(iters):
            T = fk(q)
            e = pose_err(T, p_t, R_t)
            if np.linalg.norm(e) < tol:
                break
            J = np.zeros((6, 7))
            for i in range(7):
                dq = q.copy(); dq[i] += 1e-6
                J[:, i] = (pose_err(fk(dq), p_t, R_t) - e) / 1e-6 * -1.0
            J[:, LOCKED] = 0
            lam = 0.05 if np.linalg.norm(e) > 0.02 else 0.005
            Jp = J.T @ np.linalg.inv(J @ J.T + lam ** 2 * np.eye(6))
            dq = Jp @ e + (np.eye(7) - Jp @ J) @ (0.25 * (Q_NOM - q))   # 零空间：把冗余自由度拉向中性姿态
            q = np.clip(q + 0.7 * dq, LIM_TH[:, 0], LIM_TH[:, 1])
        err = np.linalg.norm(pose_err(fk(q), p_t, R_t))
        if err < best[1]:
            best = (q, err)
        if best[1] < 5e-4:
            break
    if best[1] >= 5e-4:                      # 给定初值都没解出来：随机多起点补救
        rng = np.random.default_rng(0)
        for _ in range(40):
            q = rng.uniform(LIM_TH[:, 0], LIM_TH[:, 1])
            for _ in range(iters):
                T = fk(q)
                e = pose_err(T, p_t, R_t)
                if np.linalg.norm(e) < tol:
                    break
                J = np.zeros((6, 7))
                for i in range(7):
                    dq = q.copy(); dq[i] += 1e-6
                    J[:, i] = (pose_err(fk(dq), p_t, R_t) - e) / 1e-6 * -1.0
                J[:, LOCKED] = 0
                lam = 0.05 if np.linalg.norm(e) > 0.02 else 0.005
                Jp = J.T @ np.linalg.inv(J @ J.T + lam ** 2 * np.eye(6))
                dq = Jp @ e + (np.eye(7) - Jp @ J) @ (0.25 * (Q_NOM - q))
                q = np.clip(q + 0.7 * dq, LIM_TH[:, 0], LIM_TH[:, 1])
            err = np.linalg.norm(pose_err(fk(q), p_t, R_t))
            if err < best[1]:
                best = (q, err)
            if best[1] < 5e-4:
                break
    return (_wrap(best[0]) if best[0] is not None else None), best[1]


# 目标姿态
R_FACE_FWD = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)      # 吸盘朝前(+x)，托板朝下（吸取姿态）
R_FACE_BACK = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]], float)   # 吸盘朝后(-x)，托板朝下（J1=180° 放箱）


def j7_for_box_center(center, R):
    """让箱子中心落在 center 时，J7 原点的位置"""
    return center - R @ BOX_CENTER_J7


SEEDS = [preset_theta(n) for n in ("吸取", "前伸", "朝后扫描", "准备", "举高", "侧向")]




# ---------------- 臂质心（与 MuJoCo live 模型相同的质量分摊）----------------
def _link_model(arm_mass=800.0, mast_kg=20.0, grip_kg=30.0):
    def stat_of(lst, shift=None):
        Ar, C = 0.0, np.zeros(3)
        for p in lst:
            v, f = A._arrays(p)
            if len(f) < 4:
                continue
            a_, c_ = A._area_centroid(v, f)
            Ar += a_; C += a_ * (c_ + (shift if shift is not None else 0))
        return Ar, C / max(Ar, 1e-9)
    st = {i: stat_of(ROB["links"].get(str(i), [])) for i in range(1, 8)}
    tot = sum(a for a, _ in st.values())
    body = max(arm_mass - mast_kg - grip_kg, 0.0)
    mA, mC = stat_of(list(ROB["mastFixed"]))
    rA, rC = stat_of(list(ROB["mastRot"]), np.array(ROB["mastO"]))
    gA, gC = stat_of(list(ROB["slide"]) + list(ROB["tray"]))
    out = {}
    for i, (a, c) in st.items():
        ms = [(body * a / tot, c)]
        if i == 1:
            ms.append((mast_kg, (mA * mC + rA * rC) / max(mA + rA, 1e-9)))
        if i == 7:
            ms.append((grip_kg, gC))
        m = sum(x for x, _ in ms)
        out[i] = (m, sum(x * cc for x, cc in ms) / max(m, 1e-9) / 1000.0)
    return out


_LM = _link_model()


def arm_com(theta, payload=0.0):
    """整机里"臂+负载"的质量和质心（base 系，z 相对轮轴）；负载按箱子中心计入"""
    Ts = fk(theta, all_links=True)
    M, c = 0.0, np.zeros(3)
    for i, T in enumerate(Ts, start=1):
        m, cl = _LM[i]
        M += m; c += m * (T @ np.append(cl, 1))[:3]
    if payload > 0:
        T7 = fk(theta)
        M += payload; c += payload * (T7 @ np.append(BOX_CENTER_J7, 1))[:3]
    return M, c / M


# ---------------- 干涉检查（粗略：连杆按折线 + 半径）----------------
Z_AXLE = 0.075
VEH_X = (-1.10, 0.62)        # 车体 + 后部理料机构 的 x 范围（base 系）
VEH_Y = 0.55
DECK_Z = 0.74                # 车体/理料机构上表面之上，臂不得低于这个高度（base 系 z + 轴高 = 世界高度）


def keepout_ok(theta, y_station=0.0, wall_front=1.402, holding=False, grip_contact=False, margin=0.15):
    """返回 (是否通过, 说明)。世界系：地面 z=0，车厢 y∈[-1.5,1.5]、z∈[0,3]，箱墙前表面 x=wall_front（前排高 2.0 m）"""
    Ts = fk(theta, all_links=True)
    pts = [(T[:3, 3], i) for i, T in enumerate(Ts)]
    samples = []
    for (a, ia), (b, ib) in zip(pts[:-1], pts[1:]):
        for t in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) / 0.08))):
            samples.append((a + (b - a) * t, ia + 1))
    T7 = fk(theta)
    # 夹具（吸盘板 + 箱子）包围点
    grip = []
    for dx in (-0.15, 0.15):
        for dy in (-0.29, 0.29):
            for dz in (0.0, PLATE_FACE):
                grip.append(T7 @ np.array([dx, dy, dz, 1.0]))
    if holding:
        for dx in (-0.056, 0.194):
            for dy in (-0.2, 0.2):
                for dz in (PLATE_FACE, PLATE_FACE + BOX[0]):
                    grip.append(T7 @ np.array([dx, dy, dz, 1.0]))
    for p, link in samples:
        w = np.array([p[0], p[1] + y_station, p[2] + Z_AXLE])
        if link >= 2:
            if w[2] + margin > 3.0 or abs(w[1]) + margin > 1.5:
                return False, f"连杆{link}碰车厢顶/侧壁"
            if w[2] < 2.0 and w[0] + margin > wall_front:
                return False, f"连杆{link}碰箱墙"
            if VEH_X[0] < p[0] < VEH_X[1] and abs(p[1]) < VEH_Y and w[2] < DECK_Z + 0.02 and link >= 3:
                return False, f"连杆{link}碰车体"
    for g in grip:
        w = np.array([g[0], g[1] + y_station, g[2] + Z_AXLE])
        if w[2] > 3.0 or abs(w[1]) > 1.5 or w[2] < 0.0:
            return False, "夹具/箱子出车厢"
        if not grip_contact and w[2] < 2.0 and w[0] > wall_front + 0.005:
            return False, "夹具碰箱墙"
        if grip_contact and w[0] > wall_front + 0.03:
            return False, "夹具插进箱墙"
        if VEH_X[0] < g[0] < VEH_X[1] and abs(g[1]) < VEH_Y and w[2] < DECK_Z - 0.03 and not grip_contact:
            return False, "夹具碰车体"
    return True, ""




# ---------------- 安川 GP110B 臂（尺寸图）：覆盖上面页面七轴臂的定义 ----------------
if A.ARM_TYPE == "abb":
    import arm_abb as AB
    from a_model_abb import grip_mast_locs
    JOINTS = AB.JOINTS
    OFF = np.zeros(7)
    LIM = np.array(AB.LIM_DEG, float)
    LIM_TH = np.radians(LIM)
    set_j2_shift = A.set_j2_shift

    def preset_theta(name):
        return AB.preset_theta(name)

    def fk(theta, all_links=False):
        Ts, Te = AB.fk_chain(theta, BASE)
        return Ts if all_links else Te

    _TL, _ML = grip_mast_locs()
    _LM = AB.link_masses(820.0, _TL, _ML)

    def arm_com(theta, payload=0.0):
        Ts = fk(theta, all_links=True)
        M, c = 0.0, np.zeros(3)
        for i, T in enumerate(Ts, start=1):
            m, cl = _LM[i]
            M += m; c += m * (T @ np.append(cl, 1))[:3]
        if payload > 0:
            Te = fk(theta)
            M += payload; c += payload * (Te @ np.append(BOX_CENTER_J7, 1))[:3]
        return M, c / M

    SEEDS = [np.zeros(7)] + [AB.preset_theta(n) for n in AB.PRESETS_DEG]
    LOCKED = [i for i in range(7) if LIM_TH[i, 1] - LIM_TH[i, 0] < 1e-6]


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    T = fk(preset_theta("吸取")); print("吸取 J7:", T[:3, 3], "\n", T[:3, :3])
    # 前方箱墙前表面 x=1.402（世界系，车在原点）
    for (y, z) in [(0, 1.0), (0.6, 1.0), (-0.6, 1.5), (0, 1.9), (1.0, 0.3), (0, 0.15)]:
        c = np.array([1.402 + BOX[0] / 2 - BOX[0], y, z - A.Z0 / 1000])   # 吸附时箱子前表面在 x=1.402 → 箱中心 x=1.402+0.15
        c = np.array([1.402 + BOX[0] / 2, y, z - 0.075])
        p = j7_for_box_center(c, R_FACE_FWD)
        q, err = ik(p, R_FACE_FWD, SEEDS)
        print(f"pick y={y:+.1f} z={z:.2f}: err {err*1000:.2f} mm  θ°={np.degrees(q).round(0)}")
    # 放箱：后部辊床出箱位，箱心 (-0.85, 0, 0.855)
    c = np.array([-0.85, 0.0, 0.855 - 0.075])
    p = j7_for_box_center(c, R_FACE_BACK)
    q, err = ik(p, R_FACE_BACK, SEEDS)
    print("place rear: err %.2f mm" % (err * 1000), np.degrees(q).round(0), "J7", p.round(3))
