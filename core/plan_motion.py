"""烟箱卸货的无碰撞动作规划（GP110B，真实碰撞外形，真实关节范围）。

场景：车厢截面 3×3 m；前方箱墙前表面 x=1.402（前排 8 列 × 8 层）；卸下的箱子放到后面理料机构的辊床上。
流程（每个箱子）：
  上一个箱子放完的"预放"姿态 → 预取点（离箱面 12 cm）→ 吸取点（贴箱面）→ 退回预取点 → 转到后面（S 轴转到 ~146°）
  → 预放点 → 放箱点 → 抬起回到预放点
  取箱：吸盘朝前；放箱：S 轴转到车尾，吸盘朝侧面（±y），箱子长边沿 x 躺在辊床上（和页面理料机构的 400×300 工位一致）。
每一段都用碰撞检测（臂 7 个连杆、夹具、箱子 vs 车体/理料机构/箱墙/车厢/相机立柱/自身），
长距离的转移用 RRT-Connect 在 7 维关节空间里找无碰撞路径，再做捷径平滑。
运行：python plan_motion.py          输出：unload_plan2.json
"""
import json, math, os, sys, time
import numpy as np
import arm_kin as K
from collision import Collider

Z0 = K.Z_AXLE
BOX_D, BOX_W, BOX_H = K.BOX
WALL_FRONT = 1.402
COL_EDGES = [-1.5, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.5]
COLS = [(COL_EDGES[i] + COL_EDGES[i + 1]) / 2 for i in range(8)]
LAYERS = [BOX_H / 2 + BOX_H * k for k in range(8)]
STATIONS = [round(y, 2) for y in np.linspace(-0.9, 0.9, 7)]
PLATE_HALF_W = 0.29
PULL_OUT = 0.35                              # 吸着箱子拔出的距离（箱深 0.30 + 余量）
STANDOFF = 0.15                              # 预取点离箱面的距离（托板此时缩回，见 collision.py）
PLACE_Y, PLACE_X = 0.0, -0.85                 # 辊床中心工位
MAX_STEP = 0.012                              # 段内碰撞采样间隔 [rad]（各关节中的最大角差）；臂端 ~1.5 m 时约 18 mm，小于 20 mm 的安全距离，不会漏掉薄壁穿越

col = Collider()
PLACE_OK = set(str(s) for s in STATIONS)          # 放箱下放/抬起路径可用的工位（主进程算完后通过环境传给子进程）


def plate_y(c):
    return float(np.clip(COLS[c], -(1.5 - PLATE_HALF_W - 0.06), 1.5 - PLATE_HALF_W - 0.06))


def Rz(deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


R_SIDE = Rz(-90) @ K.R_FACE_FWD          # 吸盘朝 -y（朝右），托板朝下：放箱姿态


# 取箱顺序：从上往下逐层，相邻两层左右蛇形（一层取完再取下一层，上面的箱子都已取走，臂不会碰到旁边高处的箱子）
ORDER = [(c, l) for i, l in enumerate(range(7, -1, -1)) for c in (range(8) if i % 2 == 0 else range(7, -1, -1))]
IDX = {b: k for k, b in enumerate(ORDER)}


REACH = None        # 实际会被取走的箱子集合（None = 假设全部都会取走）。够不到的箱子留在墙上，规划时必须当障碍


def removed_before(c, l):
    """取这个箱子之前已经被取走的箱子"""
    return set(b for b in ORDER[:IDX[(c, l)]] if REACH is None or b in REACH)


def seg_ok(q0, q1, **kw):
    n = max(2, int(np.ceil(np.max(np.abs(q1 - q0)) / MAX_STEP)) + 1)
    for t in np.linspace(0, 1, n):
        if not col.ok(q0 + (q1 - q0) * t, **kw):
            return False
    return True


def rrt_connect(qs, qg, max_iter=6000, step=0.30, seed=0, **kw):
    """7 维关节空间的 RRT-Connect；返回路径（含起点终点）或 None"""
    rng = np.random.default_rng(seed)
    lo, hi = K.LIM_TH[:, 0], K.LIM_TH[:, 1]
    lock = np.array([hi[i] - lo[i] < 1e-6 for i in range(7)])
    Ta, Tb = [qs], [qg]
    Pa, Pb = [-1], [-1]
    swapped = False

    def nearest(T, q):
        return int(np.argmin(np.linalg.norm(np.array(T) - q, axis=1)))

    def extend(T, P, qt):
        i = nearest(T, qt)
        d = qt - T[i]
        n = np.linalg.norm(d)
        qn = qt if n <= step else T[i] + d / n * step
        if seg_ok(T[i], qn, **kw):
            T.append(qn); P.append(i)
            return "reached" if n <= step else "advanced"
        return "trapped"

    for it in range(max_iter):
        qr = rng.uniform(lo, hi) if rng.random() > 0.15 else (qg if not swapped else qs)
        qr = np.where(lock, qs, qr)
        r = extend(Ta, Pa, qr)
        if r != "trapped":
            while True:
                r2 = extend(Tb, Pb, Ta[-1])
                if r2 != "advanced":
                    break
            if r2 == "reached":
                pa, i = [], len(Ta) - 1
                while i >= 0:
                    pa.append(Ta[i]); i = Pa[i]
                pb, j = [], len(Tb) - 1
                while j >= 0:
                    pb.append(Tb[j]); j = Pb[j]
                path = pa[::-1] + pb
                if swapped:
                    path = path[::-1]
                return path
        Ta, Tb, Pa, Pb = Tb, Ta, Pb, Pa
        swapped = not swapped
    return None


# ---------------- RRT*：以"运动时间"为代价，渐近最优 ----------------
VMAX = np.radians([140, 110, 110, 130, 175, 175, 255])        # GP110B 各轴最大速度 [rad/s]
SEG_STEP = 0.02                                              # RRT* 里边的碰撞采样间隔 [rad]


def tmetric(a, b):
    """两个构型之间同步运动的最短时间下界 [s]：各关节 |Δq|/vmax 的最大值（路径代价 = 各段之和 ≈ 运动时间）"""
    return float(np.max(np.abs(np.asarray(b) - np.asarray(a)) / VMAX))


def _seg_ok_c(q0, q1, step=SEG_STEP, **kw):
    n = max(2, int(np.ceil(np.max(np.abs(q1 - q0)) / step)) + 1)
    for t in np.linspace(0, 1, n):
        if not col.ok(q0 + (q1 - q0) * t, **kw):
            return False
    return True


def rrt_star(qs, qg, time_budget=4.0, max_iter=4000, step_t=0.40, rewire_t=0.9, goal_bias=0.08, k_near=8, seed=0,
             init_path=None, **kw):
    """RRT*：在 7 维关节空间采样，用 tmetric 作代价；先连通，再在预算时间内持续"重选父节点 + 重连"降低代价。
    init_path：可给一条已有可行路径（如 RRT-Connect 的结果）作为初始解，RRT* 在它的基础上继续优化。
    返回 (路径, 代价[s]) 或 (None, inf)。"""
    import time as _t
    rng = np.random.default_rng(seed)
    lo, hi = K.LIM_TH[:, 0], K.LIM_TH[:, 1]
    lock = np.array([hi[i] - lo[i] < 1e-6 for i in range(7)])
    N_MAX = max_iter + (len(init_path) if init_path else 0) + 5
    X = np.zeros((N_MAX, 7)); par = [-1] * N_MAX; cost = np.zeros(N_MAX); kids = [[] for _ in range(N_MAX)]
    X[0] = qs; n = 1
    goal_idx, goal_cost = -1, np.inf

    def add(q, p, c):
        nonlocal n
        X[n] = q; par[n] = p; cost[n] = c; kids[p].append(n); n += 1
        return n - 1

    def propagate(i):
        for j in kids[i]:
            cost[j] = cost[i] + tmetric(X[i], X[j]); propagate(j)

    def try_goal(i):
        nonlocal goal_idx, goal_cost
        if tmetric(X[i], qg) <= step_t * 1.5 and cost[i] + tmetric(X[i], qg) < goal_cost - 1e-6 and _seg_ok_c(X[i], qg, **kw):
            c = cost[i] + tmetric(X[i], qg)
            if goal_idx < 0:
                goal_idx = add(qg, i, c)
            else:
                kids[par[goal_idx]].remove(goal_idx); par[goal_idx] = i; kids[i].append(goal_idx); cost[goal_idx] = c
            goal_cost = c

    if init_path:                                          # 把 RRT-Connect 的路径放进树里作为初始解
        prev = 0
        for q in init_path[1:-1]:
            prev = add(np.asarray(q, float), prev, cost[prev] + tmetric(X[prev], q))
        try_goal(prev)
    t0 = _t.time()
    for it in range(max_iter):
        if _t.time() - t0 > time_budget:
            break
        qr = qg if rng.random() < goal_bias else rng.uniform(lo, hi)
        qr = np.where(lock, qs, qr)
        d = np.max(np.abs(X[:n] - qr) / VMAX, axis=1)
        i = int(np.argmin(d))
        dt = d[i]
        qn = qr if dt <= step_t else X[i] + (qr - X[i]) * (step_t / dt)
        if not _seg_ok_c(X[i], qn, **kw):
            continue
        dn = np.max(np.abs(X[:n] - qn) / VMAX, axis=1)
        near = np.where(dn <= rewire_t)[0]
        if len(near) > k_near:
            near = near[np.argsort(dn[near])[:k_near]]
        best_p, best_c = i, cost[i] + tmetric(X[i], qn)
        for j in near:                                      # 选代价最小的父节点
            c = cost[j] + dn[j]
            if c < best_c - 1e-9 and _seg_ok_c(X[j], qn, **kw):
                best_p, best_c = int(j), c
        new = add(qn, best_p, best_c)
        for j in near:                                      # 重连：经 new 更便宜就改父节点
            if j == best_p or j == 0:
                continue
            c = best_c + dn[j]
            if c < cost[j] - 1e-9 and _seg_ok_c(qn, X[j], **kw):
                kids[par[j]].remove(int(j)); par[j] = new; kids[new].append(int(j)); cost[j] = c; propagate(int(j))
        try_goal(new)
        if goal_idx >= 0:
            goal_cost = cost[goal_idx]
    if goal_idx < 0:
        return None, np.inf
    path, i = [], goal_idx
    while i >= 0:
        path.append(X[i].copy()); i = par[i]
    return path[::-1], float(cost[goal_idx])


def path_time(path):
    return float(sum(tmetric(a, b) for a, b in zip(path[:-1], path[1:]))) if path else float("inf")


def shortcut(path, iters=400, seed=1, **kw):
    rng = np.random.default_rng(seed)
    p = [np.array(q) for q in path]
    for _ in range(iters):
        if len(p) <= 2:
            break
        i, j = sorted(rng.choice(len(p), 2, replace=False))
        if j - i < 2:
            continue
        if seg_ok(p[i], p[j], **kw):
            p = p[:i + 1] + p[j:]
    return p


PLANNER = "rrt-connect"          # "rrt*"：RRT-Connect 先找可行路径，再用 RRT* 优化；"rrt-connect"：只用 RRT-Connect + 捷径
STAR_BUDGET = 3.0         # 每条长距离路径给 RRT* 的优化时间 [s]


def plan_leg(q0, q1, star=None, **kw):
    """先试直线（关节空间），不行再规划；规划器见 PLANNER（star=True 强制用 RRT* 优化）"""
    use_star = (PLANNER == "rrt*") if star is None else star
    if seg_ok(q0, q1, **kw):
        return [q0, q1]
    path = rrt_connect(q0, q1, **kw)
    if path is not None:
        path = shortcut(path, **kw)
    if use_star:
        p2, c2 = rrt_star(q0, q1, time_budget=STAR_BUDGET, init_path=path, **kw)
        if p2 is not None and (path is None or path_time(p2) < path_time(path)):
            path = shortcut(p2, iters=40, **kw)
    return path


# ---------------- 取箱 / 放箱 IK ----------------
def pick_ik(c, l, station, standoff=0.0, seed=None, n_random=25, holding=False):
    center = np.array([WALL_FRONT + BOX_D / 2 - standoff, plate_y(c), LAYERS[l]]) - np.array([0, station, Z0])
    p = K.j7_for_box_center(center, K.R_FACE_FWD)
    rem = removed_before(c, l)
    mode = "pick" if standoff == 0.0 else "free"
    rem_chk = rem | {(c, l)} if holding else rem          # 吸着的箱子已经离开箱墙
    chk = lambda q: col.ok(q, station_y=station, holding=holding, mode=mode, removed=rem_chk)
    seeds = ([seed] if seed is not None else []) + K.SEEDS
    q, e = K.ik_tidy(p, K.R_FACE_FWD, seeds, n_random=n_random, check=chk, j1_max=0.9, max_valid=4)
    return q


def pick_ik_r(c, l, station, standoff=0.0, seed=None, holding=False):
    """先用少量随机起点，找不到再加到 70 个（只有"难"的箱子才多花时间）"""
    for n in (25, 70):
        q = pick_ik(c, l, station, standoff=standoff, seed=seed, n_random=n, holding=holding)
        if q is not None:
            return q
    return None


def place_ik(dz=0.0, seed=None, z=1.0, n_random=200):
    center = np.array([PLACE_X, PLACE_Y, z + dz]) - np.array([0, 0, Z0])
    p = K.j7_for_box_center(center, R_SIDE)
    mode = "place" if dz == 0.0 else "free"
    chk = lambda q: abs(q[0]) >= math.radians(110) and col.ok(q, holding=True, mode=mode)
    seeds = ([seed] if seed is not None else []) + [K.preset_theta("rear")] + K.SEEDS
    q, e = K.ik_tidy(p, R_SIDE, seeds, n_random=n_random, check=chk, max_valid=3)
    return q


def plan_box(c, l, q_start, station, q_place_hi, q_place):
    """单个箱子的全部分段路径；任何一步失败返回 None"""
    rem = removed_before(c, l)
    qp = pick_ik_r(c, l, station)
    if qp is None:
        return None
    qpre = pick_ik_r(c, l, station, standoff=STANDOFF, seed=qp)
    if qpre is None:
        return None
    kw0 = dict(station_y=station, holding=False, mode="free", removed=rem)
    kwH = dict(station_y=station, holding=True, mode="free", removed=rem | {(c, l)})
    legs = {}
    legs["approach"] = plan_leg(q_start, qpre, **kw0)            # 空手：上一个放箱点 → 预取点
    if legs["approach"] is None:
        return None
    if not seg_ok(qpre, qp, station_y=station, holding=False, mode="pick", removed=rem):
        return None
    legs["in"] = [qpre, qp]
    # 吸着的箱子深 0.30 m，要先沿垂直箱面方向拔出 ≥ 0.35 m 才能离开箱墙；拔出过程中箱子/夹具允许贴着箱墙（pick 模式）
    qout = pick_ik_r(c, l, station, standoff=PULL_OUT, seed=qp, holding=True)
    if qout is None:
        return None
    if not seg_ok(qp, qout, station_y=station, holding=True, mode="pick", removed=rem | {(c, l)}):
        return None
    legs["out"] = [qp, qout]
    legs["carry"] = plan_leg(qout, q_place_hi, **kwH)             # 吸着箱子：拔出点 → 预放点
    if legs["carry"] is None:
        return None
    return dict(station=station, pick=qp, pre=qpre, legs=legs)


def _plan_one(args):
    """子进程：规划一个箱子（依次试几个工位）。每个进程各自持有一份碰撞模型（模块导入时创建）"""
    global REACH
    c, l, q_hi, q_place, ok_st, reach = args
    REACH = None if reach is None else set(map(tuple, reach))
    q_hi, q_place = np.array(q_hi), np.array(q_place)
    order = sorted([st for st in STATIONS if str(st) in ok_st], key=lambda st: abs(plate_y(c) - st))
    for st in order[:4]:
        r = plan_box(c, l, q_hi, st, q_hi, q_place)
        if r is not None:
            return (c, l, dict(station=r["station"], pick=r["pick"].tolist(), pre=r["pre"].tolist(),
                               legs={k: [np.asarray(q).tolist() for q in v] for k, v in r["legs"].items()}))
    return (c, l, None)


def main():
    t0 = time.time()
    print("碰撞几何体:", col.m.ngeom)
    # 放箱位：找能无碰撞放下的最低高度（结果缓存到 place_cache.json，几何/碰撞模型没变就不用重算）
    cache_f = "data/place_cache.json"
    key = [PLACE_X, PLACE_Y, round(float(np.sum(K.LIM_TH)), 4), col.m.ngeom]
    cached = json.load(open(cache_f)) if os.path.exists(cache_f) else None
    if cached and cached["key"] == key:
        zp, q_place, q_hi = cached["z"], np.array(cached["q"]), np.array(cached["hi"])
        print("放箱位：读取缓存")
    else:
        best = None
        for z in (0.96, 0.98, 1.00, 1.03, 1.06):
            q = place_ik(z=z, n_random=250)
            if q is not None:
                best = (z, q); break
        if best is None:
            print("放箱位无解"); return
        zp, q_place = best
        q_hi = place_ik(dz=0.10, seed=q_place, z=zp, n_random=250)
        if q_hi is not None:
            json.dump(dict(key=key, z=zp, q=q_place.tolist(), hi=q_hi.tolist()), open(cache_f, "w"))
    print(f"放箱点：箱心 (x={PLACE_X}, y={PLACE_Y}, z={zp}) → 箱底离辊床上表面 {(zp - BOX_H / 2 - 0.73) * 100:.0f} cm；"
          f"S 轴 {math.degrees(q_place[0]):.0f}°，θ°={np.degrees(q_place).round(0).tolist()}")
    if q_hi is None:
        print("预放点无解"); return
    # 放箱的"下放"和"抬起"也要无碰撞：预放点 ↔ 放箱点（下放时吸着箱子，可压辊床；抬起时已松手）
    # 车厢侧壁相对机器人的位置随工位变化，所以每个工位各自规划下放/抬起路径（RRT* 缩短、去掉绕弯）
    place_down, place_up = {}, {}
    for st in STATIONS:
        down = plan_leg(q_hi, q_place, star=True, station_y=st, holding=True, mode="place", removed=set())
        up = plan_leg(q_place, q_hi, star=True, station_y=st, holding=False, mode="free", removed=set())
        if down is not None and up is not None:
            place_down[str(st)] = [np.asarray(q).tolist() for q in down]
            place_up[str(st)] = [np.asarray(q).tolist() for q in up]
    print("放箱下放/抬起路径可用的工位：", sorted(map(float, place_down)))
    if not place_down:
        print("放箱点的下放/抬起路径规划失败"); return
    out = dict(place=q_place.tolist(), place_hi=q_hi.tolist(), place_z=zp, picks={}, place_down=place_down, place_up=place_up)
    from concurrent.futures import ProcessPoolExecutor
    workers = max(1, min((os.cpu_count() or 4) - 2, 20))
    print(f"并行规划：{workers} 个进程", flush=True)
    # 够不到的箱子（顶排、边列、底层）会一直留在墙上，必须当障碍。可取集合依赖"哪些已取走"，所以迭代到集合不再变化
    reach = None
    for it in range(5):
        jobs = [(c, l, q_hi.tolist(), q_place.tolist(), sorted(place_down), None if reach is None else sorted(reach)) for (c, l) in ORDER]
        picks, fail, done = {}, [], 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for c, l, r in ex.map(_plan_one, jobs, chunksize=1):
                done += 1
                if r is None:
                    fail.append((c + 1, l + 1))
                else:
                    picks[f"{c},{l}"] = r
                if done % 16 == 0:
                    print(f"  第 {it + 1} 轮 已处理 {done}/64，可取 {len(picks)}，用时 {time.time() - t0:.0f}s", flush=True)
        new_reach = set(tuple(map(int, k.split(","))) for k in picks)
        print(f"第 {it + 1} 轮：可取 {len(new_reach)}/64", flush=True)
        out["picks"] = picks
        if new_reach == reach:
            break
        reach = new_reach if reach is None else (new_reach & reach)
    ok_n = len(out["picks"])
    out["fail"] = fail
    out["order"] = [list(b) for b in ORDER if f"{b[0]},{b[1]}" in out["picks"]]
    json.dump(out, open("data/unload_plan2.json", "w"))
    print(f"完成：{ok_n}/64 可取；够不到/规划失败：{fail}；总用时 {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
