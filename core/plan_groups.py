"""成组取箱（最多 3 箱）+ 前吸/顶吸 + 吸盘板/底托联动 的卸货动作规划（GP110B，真实碰撞外形）。

取箱流程（每组）：
  上一组放完的"预放"姿态 → 预取姿态（吸盘板缩回，离箱面/箱顶 STROKE）→ [吸盘板伸出 STROKE 贴到箱子，同时底托后退] → 吸合
  → [吸盘板缩回，底托同步前伸托住箱底] → 臂后退/上提把箱子拉出箱墙 → 转到车尾（J1 ~180°，夹具不翻转，底托始终在下面）
  → 预放姿态 → 下放到辊床 → [吸盘板前推，底托同步回缩，箱子滑到辊床上] / 顶吸则直接放下 → 释放 → 抬起。
前吸（上半部各层）：吸盘朝前，底托托着箱底；顶吸（下半部各层）：吸盘朝下贴箱顶，没有底托。
每层先取中间一组，再两边；上一层取完才取下一层（顶层先取）。
运行：python core/plan_groups.py        输出：data/unload_plan3.json
"""
import json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import box_spec as BS
import arm_kin as K
import plan_motion as PM

col = PM.col
_orig_rrt = PM.rrt_connect
PM.rrt_connect = lambda qs, qg, max_iter=2500, **kw: _orig_rrt(qs, qg, max_iter=max_iter, **kw)      # 规划失败的情形占大头，早点放弃
Z0 = K.Z_AXLE
R_FRONT = K.R_FACE_FWD                                       # 吸盘朝前，托板朝下
R_TOP = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], float)  # 吸盘朝下（顶吸）
R_BACK = K.R_FACE_BACK                                       # 吸盘朝后，托板朝下（J1 ≈ 180° 放箱，不翻转夹具）
STATIONS = PM.STATIONS
S = BS.STROKE
PLATE_Y_MAX = 1.5 - BS.PLATE_HALF_W - 0.06                   # 吸盘板中心离车厢侧壁的最小距离（容器系）
BOXC_FRONT = np.array([0.0, 0.0, 0.222 + BS.BOX_D / 2])      # 箱组中心在 J7 系（前吸，吸盘板缩回时）
BOXC_TOP = np.array([0.0, 0.0, 0.222 + BS.BOX_H / 2])
TRAY_T = BS.TRAY_T
PLACE_Z = {"front": BS.BED_TOP + BS.BOX_H / 2 + 0.012, "top": BS.BED_TOP + 0.005 + BS.BOX_H / 2}      # 纯吸盘：箱底离辊床 4~5 mm 释放
PLACE_HI = {"front": 0.06, "top": 0.12}                      # 预放点比放箱点高多少（前吸再高 IK 无解）


def boxc(mode, yoff=0.0):
    c = (BOXC_TOP if mode == "top" else BOXC_FRONT).copy()
    c[1] = yoff
    return c


def R_of(mode, place=False):
    if place:
        return R_BACK if mode == "front" else R_TOP
    return R_TOP if mode == "top" else R_FRONT


def grip_state(mode, width, yoff, pi=0.0, rho=None, empty=False):
    """碰撞用的夹具状态。rho：底托回缩量（前吸托着箱子=0；空手时缩回）"""
    top = mode == "top"
    if rho is None:
        rho = 0.0 if (not empty and not top) else BS.TRAY_RETRACT
    return dict(width=width, top=top, pi=pi, rho=rho, yoff=yoff)


def group_info(layer, cols, station=0.0):
    """返回 (吸盘板中心 y [容器系], 箱组相对吸盘板中心的偏移 yoff, 箱组总宽, 各箱宽)"""
    yc, w = BS.group_geometry(cols)
    ypl = float(np.clip(yc, -PLATE_Y_MAX, PLATE_Y_MAX))
    return ypl, yc - ypl, w, [BS.COL_W[c] for c in sorted(cols)]


def pick_center0(mode, layer, cols, extra=0.0):
    """吸盘板缩回(π=0)时箱组中心在容器系的位置；extra：额外退出量（取出箱子后臂再后退/上提）"""
    ypl, yoff, w, _ = group_info(layer, cols)
    z = BS.LAYERS[layer]
    if mode == "top":
        return np.array([BS.WALL_FRONT + BS.BOX_D / 2, ypl + yoff, z + S + extra])     # 箱组中心；吸盘在上方 S
    return np.array([BS.WALL_FRONT + BS.BOX_D / 2 - S - extra, ypl + yoff, z])


def flange_target(center_world, station, mode, yoff, place=False):
    """箱组中心(容器/世界系，含横向偏移)→ J7 位置(base 系)和姿态"""
    R = R_of(mode, place)
    c = np.array(center_world, float) - np.array([0.0, station, Z0])
    return c - R @ boxc(mode, yoff), R


def removed_set(removed_boxes):
    return set(removed_boxes)


def pick_ik(layer, cols, station, extra, seed, n_random, removed, holding, mode=None):
    mode = mode or BS.mode_for_layer(layer)
    top_ = mode == "top"
    ypl, yoff, w, _ = group_info(layer, cols)
    ctr = pick_center0(mode, layer, cols, extra)
    ctr[1] = ypl
    # 箱组中心 = 吸盘板中心 + 偏移；flange_target 把偏移当作 ee 系偏移，所以世界中心 y 用板中心
    p, R = flange_target(ctr, station, mode, yoff)
    rem = set(removed) | (set((c, layer) for c in cols) if holding else set())
    gs = grip_state(mode, w, yoff, pi=0.0, rho=(None if holding else (BS.TRAY_PLACE_RETRACT if mode == "front" else 0.0)), empty=not holding)
    cm = "pick" if top_ else "free"                          # 顶吸时吸盘板紧贴箱墙后面那堵整墙（允许贴着），臂连杆仍要保持距离
    chk = lambda q: col.ok(q, station_y=station, holding=holding, mode=cm, removed=rem, grip=gs)
    seeds = ([seed] if seed is not None else []) + K.SEEDS
    q, e = K.ik_tidy(p, R, seeds, n_random=n_random, check=chk, j1_max=0.9, max_valid=4)
    return q


def pick_ik_r(layer, cols, station, extra=0.0, seed=None, removed=(), holding=False, mode=None):
    for n in (25, 70):
        q = pick_ik(layer, cols, station, extra, seed, n, removed, holding, mode)
        if q is not None:
            return q
    return None


def contact_ok(q, layer, cols, station, removed, mode=None):
    """吸盘板伸出 STROKE 贴到箱子（pick 模式允许夹具碰箱墙），其余连杆不得碰"""
    mode = mode or BS.mode_for_layer(layer)
    ypl, yoff, w, _ = group_info(layer, cols)
    gs = grip_state(mode, w, yoff, pi=S, rho=BS.TRAY_RETRACT, empty=True)
    return col.ok(q, station_y=station, holding=False, mode="pick", removed=removed, grip=gs)


def place_ik(mode, hi=False, seed=None, n_random=200):
    """放箱姿态（吸盘朝后/朝下，J1 转到车尾）。箱组按 3 箱宽度保守检查"""
    z = PLACE_Z[mode] + (PLACE_HI[mode] if hi else 0.0)
    x = BS.PLACE_X + (BS.PLACE_PUSH if mode == "front" else 0.0)    # 前吸：先放在离出箱位 PLACE_PUSH 处，再由吸盘板前推、底托回缩
    p, R = flange_target(np.array([x, BS.PLACE_Y, z]), 0.0, mode, 0.0, place=True)
    gs = grip_state(mode, BS.BOX_W * BS.MAX_GROUP, 0.0, pi=0.0, rho=0.0)
    cm = "free" if hi else "place"
    chk = lambda q: abs(q[0]) >= math.radians(110) and col.ok(q, holding=True, mode=cm, grip=gs)
    seeds = ([seed] if seed is not None else []) + [K.preset_theta("rear")] + K.SEEDS
    q, e = K.ik_tidy(p, R, seeds, n_random=n_random, check=chk, max_valid=3)
    return q


def above(layer):
    """层号比 layer 高的所有箱子（按顶层先取的顺序，已取走）"""
    return set((c, l) for l in range(layer + 1, BS.N_LAYERS) for c in range(BS.N_COLS))


# ---------------------------------------------------------------- 放箱姿态与站位相关的下放/抬起路径
def place_poses(mode):
    q = place_ik(mode, hi=False)
    if q is None:
        return None, None
    qh = place_ik(mode, hi=True, seed=q)
    if qh is not None:
        qh = _unwrap_like(qh, q)                 # 预放点与放箱点的 J1/J5/J7 取同一圈，避免拼接路径时跳 360°
    return q, qh


def _unwrap_like(q, ref):
    """J1/J5/J7 可以绕圈：把 q 的这几个关节换成和参考构型最近的等价角（避免线性插值绕 360°）"""
    q = np.array(q, float)
    for i in (0, 4, 6):
        d = (q[i] - ref[i] + math.pi) % (2 * math.pi) - math.pi
        q[i] = ref[i] + d
    return q


def place_descent(mode, q_place, q_hi):
    """从预放点到放箱点的竖直下放路径：中间每隔 2 cm 求一次 IK（用上一个解作初值），得到近似竖直的关节路径"""
    dz_total = PLACE_HI[mode]
    n = max(2, int(round(dz_total / 0.02)))
    wps = [_unwrap_like(q_hi, q_place)]
    prev = wps[0]
    for k in range(1, n):
        dz = dz_total * (1 - k / n)
        x = BS.PLACE_X + (BS.PLACE_PUSH if mode == "front" else 0.0)
        z = PLACE_Z[mode] + dz
        p, R = flange_target(np.array([x, BS.PLACE_Y, z]), 0.0, mode, 0.0, place=True)
        q, e = K.ik_tidy(p, R, [prev, q_place], n_random=0, check=None, max_valid=1)
        if q is None:
            return None
        q = _unwrap_like(q, q_place)
        wps.append(q); prev = q
    wps.append(np.asarray(q_place, float))
    return wps


def empty_grip(mode):
    return grip_state(mode, BS.BOX_W * BS.MAX_GROUP, 0.0, pi=0.0, rho=BS.TRAY_RETRACT, empty=True)


def full_grip(mode):
    return grip_state(mode, BS.BOX_W * BS.MAX_GROUP, 0.0, pi=0.0, rho=0.0)


def _place_leg_job(args):
    mode, st, q_hi, q_pl = args
    q_hi, q_pl = np.array(q_hi), np.array(q_pl)
    gf = full_grip(mode)
    gu = grip_state(mode, BS.BOX_W * BS.MAX_GROUP, 0.0, pi=0.0, rho=BS.TRAY_PLACE_RETRACT, empty=True)
    wps = place_descent(mode, q_pl, q_hi)
    down = up = None
    if wps is not None:
        ok_dn = all(PM.seg_ok(a, b, station_y=st, holding=True, mode="place", removed=set(), grip=gf) for a, b in zip(wps[:-1], wps[1:]))
        ok_up = all(PM.seg_ok(b, a, station_y=st, holding=False, mode="place", removed=set(), grip=gu) for a, b in zip(wps[:-1], wps[1:]))
        if ok_dn:
            down = wps
        if ok_up:
            up = wps[::-1]
    if down is None:
        down = PM.plan_leg(q_hi, q_pl, star=True, station_y=st, holding=True, mode="place", removed=set(), grip=gf)
    if up is None:
        up = PM.plan_leg(q_pl, q_hi, star=True, station_y=st, holding=False, mode="place", removed=set(), grip=gu)
    if down is None or up is None:
        return mode, st, None
    return mode, st, dict(down=[np.asarray(q).tolist() for q in down], up=[np.asarray(q).tolist() for q in up])


# ---------------------------------------------------------------- 单组取箱
def partitions(cols):
    c = sorted(cols)
    if len(c) == 3:
        return [[c], [[c[0], c[1]], [c[2]]], [[c[0]], [c[1], c[2]]], [[c[0]], [c[1]], [c[2]]]]
    if len(c) == 2:
        return [[c], [[c[0]], [c[1]]]]
    return [[c]]


Q_HI_CACHE = {}


def pick_ik_r_back(layer, cols, station, seed, removed):
    """顶吸：提起后再沿 -x 后退 0.12 m，让箱子离开后面那堵整墙"""
    mode = "top"
    ypl, yoff, w, _ = group_info(layer, cols)
    ctr = pick_center0(mode, layer, cols, BS.TOP_LIFT - S)
    ctr[1] = ypl
    ctr[0] -= 0.12
    p, R = flange_target(ctr, station, mode, yoff)
    gs = grip_state(mode, w, yoff, pi=0.0, rho=0.0)
    chk = lambda q: col.ok(q, station_y=station, holding=True, mode="free", removed=removed, grip=gs)
    q, e = K.ik_tidy(p, R, [seed] + K.SEEDS, n_random=40, check=chk, j1_max=0.9, max_valid=4)
    return q


def pull_path(layer, cols, station, mode, qpre, removed, boxes):
    """吸住箱子后沿笛卡尔直线把箱子拔出：每 5 cm 求一次 IK（用上一个解作初值，保持同一分支）。
    前吸：沿 -x 后退；顶吸：先竖直提起，再沿 -x 退 12 cm。返回关节路点（含 qpre）或 None"""
    ypl, yoff, w, _ = group_info(layer, cols)
    gh = grip_state(mode, w, yoff, pi=0.0, rho=0.0)
    steps = []
    if mode == "front":
        tot = BS.PULL_OUT - S
        n = max(1, int(math.ceil(tot / 0.05)))
        steps = [(tot * k / n, 0.0) for k in range(1, n + 1)]
    else:
        tot = BS.TOP_LIFT - S
        n = max(1, int(math.ceil(tot / 0.05)))
        steps = [(tot * k / n, 0.0) for k in range(1, n + 1)] + [(tot, 0.12 * k / 3) for k in range(1, 4)]
    rem = set(removed) | set(boxes)
    wps, prev = [np.asarray(qpre, float)], np.asarray(qpre, float)
    for i, (ex, bk) in enumerate(steps):
        last = i == len(steps) - 1
        ctr = pick_center0(mode, layer, cols, ex)
        ctr[1] = ypl
        ctr[0] -= bk
        p, R = flange_target(ctr, station, mode, yoff)
        cm = "free" if last else "pick"
        chk = lambda q, cm=cm: col.ok(q, station_y=station, holding=True, mode=cm, removed=rem, grip=gh)
        q, e = K.ik_tidy(p, R, [prev], n_random=0, check=chk, max_valid=1)
        if q is None:
            return None
        q = _unwrap_like(q, prev)
        if not PM.seg_ok(prev, q, station_y=station, holding=True, mode="pick", removed=rem, grip=gh):
            return None
        wps.append(q); prev = q
    return wps


def plan_part(layer, cols, removed, stations_ok, mode=None):
    """规划一组箱子的取出+搬运到预放点；成功返回 dict，失败返回 None"""
    mode = mode or BS.mode_for_layer(layer)
    ypl, yoff, w, widths = group_info(layer, cols)
    boxes = set((c, layer) for c in cols)
    sts = [st for st in STATIONS if str(st) in stations_ok[mode]]
    sts = sorted(sts, key=lambda st: abs(ypl - st))                     # 所有站位，按离箱组中心的远近依次试
    for st in sts:
        qpre = pick_ik_r(layer, cols, st, extra=0.0, removed=removed, holding=False, mode=mode)
        if qpre is None or not contact_ok(qpre, layer, cols, st, removed, mode):
            continue
        gh = grip_state(mode, w, yoff, pi=0.0, rho=0.0)                       # 取出后：吸盘板缩回，底托托住
        pull = pull_path(layer, cols, st, mode, qpre, removed, boxes)
        if pull is None:
            continue
        qout = pull[-1]
        carry = PM.plan_leg(qout, np.array(Q_HI_CACHE[mode]), station_y=st, holding=True, mode="free", removed=removed | boxes, grip=gh)
        if carry is None:
            continue
        return dict(layer=layer, cols=sorted(cols), mode=mode, station=st, ypl=ypl, yoff=yoff, width=w, widths=widths,
                    qpre=qpre.tolist(), pull=[np.asarray(q).tolist() for q in pull], carry=[np.asarray(q).tolist() for q in carry])
    return None


def _cand_job(args):
    """子进程：规划一个候选组（先整组，不行再拆成小组）。返回 (序号, [part, ...])"""
    global Q_HI_CACHE
    idx, layer, cols, before, q_hi, st_ok = args          # before：本组之前已取走的箱子（列表）
    Q_HI_CACHE = {m: np.asarray(q) for m, q in q_hi.items()}
    removed = set(map(tuple, before))
    best = []
    for part in partitions(cols):
        got, rem = [], set(removed)
        for sub in part:
            pref = BS.mode_for_layer(layer)
            modes = [pref] + [m for m in ("front", "top") if m != pref]
            modes = [m for m in modes if not (m == "front" and layer == 0) and not (m == "top" and layer > 2)]   # 最底层前吸不行；顶吸只在下面几层够得着
            r = None
            for m in modes:
                r = plan_part(layer, sub, rem, st_ok, mode=m)
                if r is not None:
                    break
            if r is not None:
                got.append(r)
                rem |= set((c, layer) for c in sub)
        if sum(len(g["cols"]) for g in got) > sum(len(g["cols"]) for g in best):
            best = got
        if sum(len(g["cols"]) for g in best) == len(cols):
            break
    return idx, best


def _approach_job(args):
    """子进程：上一组放完（预放点）→ 这一组的预取点"""
    k, q_from, mode_from, part, removed = args
    q_from = np.array(q_from)
    qpre = np.array(part["qpre"])
    mode = part["mode"]
    ge = grip_state(mode, part["width"], part["yoff"], pi=0.0, rho=(BS.TRAY_PLACE_RETRACT if mode == "front" else 0.0), empty=True)
    leg = PM.plan_leg(q_from, qpre, station_y=part["station"], holding=False, mode=("pick" if mode == "top" else "free"), removed=set(map(tuple, removed)), grip=ge)   # 顶吸的吸盘板贴着后面整墙，允许夹具碰墙
    return k, (None if leg is None else [np.asarray(q).tolist() for q in leg])


def main():
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    workers = max(1, min((os.cpu_count() or 4) - 2, 6))      # 每个进程要载入页面网格+碰撞模型，约 1 GB 内存
    print(f"碰撞几何体 {col.m.ngeom}；并行 {workers} 进程", flush=True)
    # ---- 放箱姿态
    Qp, Qh = {}, {}
    for mode in ("front", "top"):
        q, qh = place_poses(mode)
        if q is None or qh is None:
            print("放箱姿态无解:", mode); return
        Qp[mode], Qh[mode] = q, qh
        print(f"放箱姿态 {mode}: J1={math.degrees(q[0]):.0f}° θ°={np.degrees(q).round(0).tolist()}", flush=True)
    jobs = [(m, st, Qh[m].tolist(), Qp[m].tolist()) for m in ("front", "top") for st in STATIONS]
    place_legs = {"front": {}, "top": {}}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for mode, st, r in ex.map(_place_leg_job, jobs, chunksize=1):
            if r is not None:
                place_legs[mode][str(st)] = r
    st_ok = {m: sorted(place_legs[m]) for m in place_legs}
    print("放箱下放/抬起路径可用的工位:", {m: sorted(map(float, v)) for m, v in place_legs.items()}, f"{time.time()-t0:.0f}s", flush=True)
    if not st_ok["front"] or not st_ok["top"]:
        print("放箱路径规划失败"); return
    q_hi = {m: Qh[m].tolist() for m in Qh}
    # ---- 候选组（顶层先取；每层先中间再两边）
    cands = [(layer, cols) for layer in range(BS.N_LAYERS - 1, -1, -1) for cols in BS.groups_for_layer(layer)]
    reach = None
    parts_final = None
    prev_before, prev_res = {}, {}
    for it in range(5):
        jobs, reuse = [], {}
        for idx, (layer, cols) in enumerate(cands):
            before = [b for (l2, c2) in cands[:idx] for b in [(c, l2) for c in c2] if reach is None or b in reach]
            if prev_before.get(idx) == before and idx in prev_res:          # 前面取走的箱子没变，结果不变，直接沿用
                reuse[idx] = prev_res[idx]
            else:
                jobs.append((idx, layer, cols, before, q_hi, st_ok))
            prev_before[idx] = before
        res = dict(reuse)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for idx, parts in ex.map(_cand_job, jobs, chunksize=1):
                res[idx] = parts
        prev_res = res
        parts_all = [p for idx in range(len(cands)) for p in res[idx]]
        new_reach = set((c, p["layer"]) for p in parts_all for c in p["cols"])
        print(f"第 {it + 1} 轮：{len(jobs)} 组重算，{len(parts_all)} 组，可取 {len(new_reach)}/{BS.N_COLS * BS.N_LAYERS} 箱，{time.time()-t0:.0f}s", flush=True)
        parts_final = parts_all
        if new_reach == reach:
            break
        reach = new_reach if reach is None else (new_reach & reach)
    # ---- 组与组之间的接近路径（并行）
    jobs, removed_run = [], set()
    prev_q, prev_mode = Qh["front"].tolist(), "front"
    for k, p in enumerate(parts_final):
        jobs.append((k, prev_q, prev_mode, p, sorted(removed_run)))
        removed_run |= set((c, p["layer"]) for c in p["cols"])
        prev_q, prev_mode = Qh[p["mode"]].tolist(), p["mode"]
    appr = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, leg in ex.map(_approach_job, jobs, chunksize=1):
            appr[k] = leg
    ok_parts, dropped = [], []
    for k, p in enumerate(parts_final):
        if appr[k] is None:
            dropped.append(p)
            continue
        p = dict(p); p["approach"] = appr[k]; p["from_mode"] = jobs[k][2]
        ok_parts.append(p)
    if dropped:
        print("接近路径规划失败而丢弃的组:", [(d["layer"], d["cols"]) for d in dropped], flush=True)
    reach_boxes = set((c, p["layer"]) for p in ok_parts for c in p["cols"])
    out = dict(version=3, box=dict(D=BS.BOX_D, W=BS.BOX_W, H=BS.BOX_H, mass=BS.BOX_MASS),
               place={m: dict(down=Qp[m].tolist(), hi=Qh[m].tolist()) for m in Qp}, place_legs=place_legs,
               start=Qh["front"].tolist(), parts=ok_parts,
               unreachable=[[c, l] for l in range(BS.N_LAYERS) for c in range(BS.N_COLS) if (c, l) not in reach_boxes])
    json.dump(out, open("data/unload_plan3_raw.json", "w"))
    print(f"完成：{len(ok_parts)} 组 / {len(reach_boxes)} 箱可取（共 {BS.N_COLS * BS.N_LAYERS} 箱）；用时 {time.time()-t0:.0f}s；未取走：{out['unreachable']}", flush=True)


if __name__ == "__main__":
    main()
