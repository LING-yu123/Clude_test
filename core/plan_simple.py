"""纯吸盘（没有底托、没有吸盘与底托的相对运动）的成组卸货规划。目标：按层从上到下、每层先中间后两边，把箱子卸完、不漏。

底盘尽量居中不动：站位从中间(0)往两边试，能在中间完成的箱子都不动车；主要靠机械臂动作。
转到车尾时左转/右转都可以：J1 目标取 +180°（向左转）或 -180°（向右转），箱子在左边先试左转、在右边先试右转，不行再试另一边。
取箱（每组，同层相邻最多 3 箱）：
  预取姿态（离箱面/箱顶 STANDOFF）→ 臂笛卡尔直线推进到贴上箱子（吸盘板 pick 模式允许贴墙、底层箱子允许贴地）→ 吸合
  → 沿笛卡尔直线拔出箱墙（前吸：沿 -x 退出一个箱深；顶吸：竖直提起再后退；必要时再向中心线横移）→ RRT 转到车尾 → 预放 → 下放到辊床 → 释放 → 抬起。
前吸（吸盘朝前）和顶吸（吸盘朝下）两种方式都试；整组不行就拆成小组，直到单个箱子。
运行：python core/plan_simple.py        输出：data/unload_plan3_raw.json（再运行 core/smooth_plan.py）
"""
import json, math, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import box_spec as BS
import arm_kin as K
import plan_groups as G
import plan_motion as PM

col = G.col
STATIONS2 = [0.0, 0.15, -0.15, 0.3, -0.3, 0.45, -0.45, 0.6, -0.6, 0.75, -0.75, 0.92, -0.92]
S = 0.15                                      # 预取点离箱面/箱顶的距离（臂直线推进的距离）
STEP = 0.05                                   # 笛卡尔推进/拔出每步 5 cm
Q_HI = {}                                     # {"front_L": q, "front_R": q, "top_L": q, "top_R": q}


def group_info2(layer, cols):
    yc, w = BS.group_geometry(cols)
    pw = max(w / 2 + 0.02, min(w / 2 + 0.05, 1.475 - abs(yc)))     # 吸盘板半宽：两侧各留 5 cm，贴侧壁时留 2.5 cm 间隙
    return yc, w, [BS.COL_W[c] for c in sorted(cols)], pw


def gstate(mode, w, pw):
    return dict(width=w, top=(mode == "top"), pi=0.0, rho=0.0, yoff=0.0, tray=False, pw=pw)


def center_of(mode, layer, yc, extra, back=0.0):
    """箱组中心(容器系)：extra=0 为预取点；extra=-S 为贴上箱子；extra 增大 = 拔出（前吸沿 -x，顶吸向上）；back = 顶吸提起后再沿 -x 后退"""
    z = BS.LAYERS[layer]
    if mode == "top":
        return np.array([BS.WALL_FRONT + BS.BOX_D / 2 - back, yc, z + S + extra])
    return np.array([BS.WALL_FRONT + BS.BOX_D / 2 - S - extra - back, yc, z])


def ik_at(center, st, mode, seeds, n_random, check):
    p, R = G.flange_target(center, st, mode, 0.0)
    q, e = K.ik_tidy(p, R, seeds, n_random=n_random, check=check, j1_max=0.9, max_valid=4)
    return q


def cart_path(q0, steps, st, mode, holding, rem, g):
    """steps: [(中心, 碰撞模式)]；逐步 IK（上一个解作初值，保持同一分支），返回关节路点（含 q0）或 None"""
    wps, prev = [np.asarray(q0, float)], np.asarray(q0, float)
    for ctr, cm in steps:
        chk = lambda q, cm=cm: col.ok(q, station_y=st, holding=holding, mode=cm, removed=rem, grip=g)
        p, R = G.flange_target(ctr, st, mode, 0.0)
        q, e = K.ik_tidy(p, R, [prev], n_random=0, check=chk, max_valid=1)
        if q is None:
            return None
        q = G._unwrap_like(q, prev)
        if not PM.seg_ok(prev, q, station_y=st, holding=holding, mode="pick", removed=rem, grip=g):
            return None
        wps.append(q); prev = q
    return wps


def leg_robust(q0, q1, **kw):
    """先试直线，再 RRT-Connect（多个随机种子、6000 次迭代）+ 捷径"""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if PM.seg_ok(q0, q1, **kw):
        return [q0, q1]
    for seed in range(3):
        path = G._orig_rrt(q0, q1, max_iter=6000, seed=seed, **kw)
        if path is not None:
            return PM.shortcut(path, **kw)
    return None


def pull_steps(mode, layer, yc, st, lateral=False):
    """拔出路径的笛卡尔步骤：前吸沿 -x 退出一个箱深；顶吸先提起再后退；底层再抬高 3 cm 离开地板；lateral=True 时最后向机器人中心线横移一段"""
    steps = []
    if mode == "front":
        tot = BS.BOX_D + 0.04
        n = max(1, int(math.ceil(tot / STEP)))
        for k in range(1, n + 1):
            steps.append((center_of(mode, layer, yc, -S + tot * k / n), "free" if (k == n and layer > 0) else "pick"))
        last = lambda y: center_of(mode, layer, y, -S + tot)
        if layer == 0:                                        # 底层：拔出后先抬高 3 cm 让箱底离开地板
            steps.append((center_of(mode, layer, yc, -S + tot) + np.array([0.0, 0.0, 0.03]), "free"))
            last = lambda y: center_of(mode, layer, y, -S + tot) + np.array([0.0, 0.0, 0.03])
    else:
        tot = BS.BOX_H + 0.05
        n = max(1, int(math.ceil(tot / STEP)))
        for k in range(1, n + 1):
            steps.append((center_of(mode, layer, yc, -S + tot * k / n), "pick"))
        for k in range(1, 4):
            steps.append((center_of(mode, layer, yc, -S + tot, back=0.12 * k / 3), "free" if k == 3 else "pick"))
        last = lambda y: center_of(mode, layer, y, -S + tot, back=0.12)
    off = yc - st
    if lateral and abs(off) > 0.2:
        move = min(abs(off) - 0.05, 0.6)
        n = int(math.ceil(move / STEP))
        for k in range(1, n + 1):
            steps.append((last(yc - math.copysign(move * k / n, off)), "free"))
    return steps


def iter_part2(layer, cols, removed, st_ok, mode):
    yc, w, widths, pw = group_info2(layer, cols)
    boxes = set((c, layer) for c in cols)
    g = gstate(mode, w, pw)
    sts = sorted([st for st in STATIONS2 if str(st) in st_ok[mode + "_L"] and str(st) in st_ok[mode + "_R"]],
                 key=lambda st: (abs(st), abs(yc - st)))                     # 底盘尽量居中不动：站位从中间往两边试
    cm0 = "pick" if mode == "top" else "free"
    for st in sts:
        chk = lambda q: col.ok(q, station_y=st, holding=False, mode=cm0, removed=removed, grip=g)
        qpre = None
        for n in (25, 70):
            qpre = ik_at(center_of(mode, layer, yc, 0.0), st, mode, K.SEEDS, n, chk)
            if qpre is not None:
                break
        if qpre is None:
            continue
        n = max(1, int(math.ceil(S / STEP)))
        cont = cart_path(qpre, [(center_of(mode, layer, yc, -S * k / n), "pick") for k in range(1, n + 1)], st, mode, False, removed, g)
        if cont is None:
            continue
        rem = removed | boxes
        dirs = ("L", "R") if yc - st >= 0 else ("R", "L")                   # 箱子在左边先试左转，在右边先试右转
        for lat in (False, True):
            pull = cart_path(cont[-1], pull_steps(mode, layer, yc, st, lat), st, mode, True, rem, g)
            if pull is None:
                continue
            for d in dirs:
                carry = leg_robust(pull[-1], np.array(Q_HI[mode + "_" + d]), station_y=st, holding=True, mode="free", removed=rem, grip=g)
                if carry is not None:
                    yield dict(layer=layer, cols=sorted(cols), mode=mode, dir=d, station=st, ypl=yc, yoff=0.0, width=w, widths=widths, pw=pw,
                                qpre=qpre.tolist(), contact=[np.asarray(q).tolist() for q in cont],
                                pull=[np.asarray(q).tolist() for q in pull], carry=[np.asarray(q).tolist() for q in carry])


def plan_part2(layer, cols, removed, st_ok, mode):
    for r in iter_part2(layer, cols, removed, st_ok, mode):
        return r
    return None


def _cand_job(args):
    global Q_HI
    idx, layer, cols, before, q_hi, st_ok = args
    Q_HI = {m: np.asarray(q) for m, q in q_hi.items()}
    removed = set(map(tuple, before))
    best = []
    for part in G.partitions(cols):
        got, rem = [], set(removed)
        for sub in part:
            pref = BS.mode_for_layer(layer)
            modes = [pref] + [m for m in ("front", "top") if m != pref]
            r = None
            for m in modes:
                r = plan_part2(layer, sub, rem, st_ok, m)
                if r is not None:
                    break
            if r is not None:
                got.append(r)
                rem |= set((c, layer) for c in sub)
        if sum(len(x["cols"]) for x in got) > sum(len(x["cols"]) for x in best):
            best = got
        if sum(len(x["cols"]) for x in best) == len(cols):
            break
    return idx, best


def _approach_job(args):
    k, q_from, part, removed = args
    q_from = np.array(q_from)
    g = gstate(part["mode"], part["width"], part["pw"])
    leg = leg_robust(q_from, np.array(part["qpre"]), station_y=part["station"], holding=False,
                     mode=("pick" if part["mode"] == "top" else "free"), removed=set(map(tuple, removed)), grip=g)
    return k, (None if leg is None else [np.asarray(q).tolist() for q in leg])


def place_with_approach(layer, cols, removed, prev_q, st_ok, Qh):
    """在已知上一个姿态 prev_q 的前提下，为一批箱子找一套"接近路径可行"的取法；返回 [part...]（可能被拆成小组）或 None"""
    for part in G.partitions(cols):
        got, rem, pq, ok = [], set(removed), prev_q, True
        for sub in part:
            found = None
            pref = BS.mode_for_layer(layer)
            for m in [pref] + [x for x in ("front", "top") if x != pref]:
                for cand in iter_part2(layer, sub, rem, st_ok, m):
                    leg = _approach_job((0, pq, cand, sorted(rem)))[1]
                    if leg is not None:
                        found = dict(cand); found["approach"] = leg
                        break
                if found is not None:
                    break
            if found is None:
                ok = False
                break
            got.append(found)
            rem |= set((c, layer) for c in sub)
            pq = Qh[found["mode"] + "_" + found["dir"]]
        if ok:
            return got
    return None


def with_j1(q, d):
    q = np.array(q, float)
    q[0] = math.pi if d == "L" else -math.pi
    return q


def main():
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    workers = max(1, min((os.cpu_count() or 4) - 2, 6))
    print(f"碰撞几何体 {col.m.ngeom}；并行 {workers} 进程", flush=True)
    Qp, Qh = {}, {}
    for mode in ("front", "top"):
        q, qh = G.place_poses(mode)
        if q is None or qh is None:
            print("放箱姿态无解:", mode); return
        for d in ("L", "R"):                                              # J1=+180°(左转到车尾) 和 -180°(右转) 是同一个物理姿态
            Qp[mode + "_" + d], Qh[mode + "_" + d] = with_j1(q, d), with_j1(qh, d)
        print(f"放箱姿态 {mode}: θ°={np.degrees(q).round(0).tolist()}", flush=True)
    jobs = [(k.split("_")[0], st, Qh[k].tolist(), Qp[k].tolist(), k) for k in Qp for st in STATIONS2]
    place_legs = {k: {} for k in Qp}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for (mode, st, r), job in zip(ex.map(G._place_leg_job, [j[:4] for j in jobs], chunksize=1), jobs):
            if r is not None:
                place_legs[job[4]][str(st)] = r
    st_ok = {k: sorted(v) for k, v in place_legs.items()}
    print("放箱下放/抬起路径可用的工位数:", {k: len(v) for k, v in place_legs.items()}, f"{time.time()-t0:.0f}s", flush=True)
    if not all(st_ok.values()):
        print("放箱路径规划失败"); return
    q_hi = {k: Qh[k].tolist() for k in Qh}
    cands = [(layer, cols) for layer in range(BS.N_LAYERS - 1, -1, -1) for cols in BS.groups_for_layer(layer)]
    reach, parts_final = None, None
    prev_before, prev_res = {}, {}
    for it in range(6):
        jobs, reuse = [], {}
        for idx, (layer, cols) in enumerate(cands):
            before = [b for (l2, c2) in cands[:idx] for b in [(c, l2) for c in c2] if reach is None or b in reach]
            if prev_before.get(idx) == before and idx in prev_res:
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
    global Q_HI
    Q_HI = {k: np.asarray(v) for k, v in q_hi.items()}
    ok_parts, dropped, removed_run = [], [], set()
    prev_q = Qh["front_L"]
    for k, p in enumerate(parts_final):
        leg = _approach_job((k, prev_q, p, sorted(removed_run)))[1]
        if leg is None:                                   # 接近路径走不通：同一批箱子换站位/转向/吸取方式重新规划，仍不行就拆成小组
            got = place_with_approach(p["layer"], p["cols"], removed_run, prev_q, st_ok, Qh)
            if got is None:
                dropped.append(p); continue
            for q_ in got:
                ok_parts.append(q_)
                removed_run |= set((c, q_["layer"]) for c in q_["cols"])
                prev_q = Qh[q_["mode"] + "_" + q_["dir"]]
            print(f"  组 层{p['layer']+1} 列{[c+1 for c in p['cols']]} 接近路径失败，改为 " + "；".join(f"{q_['mode']} 列{[c+1 for c in q_['cols']]} 站位{q_['station']:+.2f} {q_['dir']}转" for q_ in got), flush=True)
            continue
        p = dict(p); p["approach"] = leg
        ok_parts.append(p)
        removed_run |= set((c, p["layer"]) for c in p["cols"])
        prev_q = Qh[p["mode"] + "_" + p["dir"]]
    if dropped:
        print("接近路径规划失败而丢弃的组:", [(d["layer"], d["cols"]) for d in dropped], flush=True)
    reach_boxes = set((c, p["layer"]) for p in ok_parts for c in p["cols"])
    out = dict(version=4, simple=True, box=dict(D=BS.BOX_D, W=BS.BOX_W, H=BS.BOX_H, mass=BS.BOX_MASS),
               place={k: dict(down=Qp[k].tolist(), hi=Qh[k].tolist()) for k in Qp}, place_legs=place_legs,
               start=Qh["front_L"].tolist(), parts=ok_parts,
               unreachable=[[c, l] for l in range(BS.N_LAYERS) for c in range(BS.N_COLS) if (c, l) not in reach_boxes])
    json.dump(out, open("data/unload_plan3_raw.json", "w"))
    stations_used = sorted(set(p["station"] for p in ok_parts))
    print(f"完成：{len(ok_parts)} 组 / {len(reach_boxes)} 箱可取（共 {BS.N_COLS * BS.N_LAYERS} 箱）；用时 {time.time()-t0:.0f}s；"
          f"用到的站位 {stations_used}，左转 {sum(p['dir'] == 'L' for p in ok_parts)} 组 / 右转 {sum(p['dir'] == 'R' for p in ok_parts)} 组；未取走：{out['unreachable']}", flush=True)


if __name__ == "__main__":
    main()
