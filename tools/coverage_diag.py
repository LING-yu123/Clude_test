import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))   # 让 core/ 里的模块可被导入
"""逐箱诊断：单个箱子（其上面各层已取走）在哪一步失败：预取 IK / 贴箱 / 拔出 / 转到车尾 / 接近。python tools/coverage_diag.py [层号...]"""
import math, time
import numpy as np
import box_spec as BS
import plan_groups as G
import plan_simple as P
import plan_motion as PM


def diag(args):
    c, layer = args
    G_ = G
    P.Q_HI = {}
    for m in ("front", "top"):
        qh = np.asarray(G_.place_poses(m)[1])
        P.Q_HI[m + "_L"] = P.with_j1(qh, "L"); P.Q_HI[m + "_R"] = P.with_j1(qh, "R")
    removed = G_.above(layer)
    yc, w, widths, pw = P.group_info2(layer, [c])
    out = {}
    for mode in ("front", "top"):
        g = P.gstate(mode, w, pw)
        cm0 = "pick" if mode == "top" else "free"
        stage = dict(pre=0, contact=0, pull=0, carry=0)
        for st in P.STATIONS2:
            chk = lambda q: P.col.ok(q, station_y=st, holding=False, mode=cm0, removed=removed, grip=g)
            qpre = P.ik_at(P.center_of(mode, layer, yc, 0.0), st, mode, P.K.SEEDS, 25, chk)
            if qpre is None:
                continue
            stage["pre"] += 1
            n = max(1, int(math.ceil(P.S / P.STEP)))
            cont = P.cart_path(qpre, [(P.center_of(mode, layer, yc, -P.S * k / n), "pick") for k in range(1, n + 1)], st, mode, False, removed, g)
            if cont is None:
                continue
            stage["contact"] += 1
            rem = removed | {(c, layer)}
            pull = carry = None
            for lat in (False, True):
                pull = P.cart_path(cont[-1], P.pull_steps(mode, layer, yc, st, lat), st, mode, True, rem, g)
                if pull is None:
                    continue
                for dd in ("L", "R"):
                    carry = P.leg_robust(pull[-1], np.array(P.Q_HI[mode + "_" + dd]), station_y=st, holding=True, mode="free", removed=rem, grip=g)
                    if carry is not None:
                        break
                if carry is not None:
                    break
            if pull is not None:
                stage["pull"] += 1
            if carry is not None:
                stage["carry"] += 1
        out[mode] = stage
    return (c, layer), out


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    layers = [int(x) for x in sys.argv[1:] if x.isdigit()] or list(range(BS.N_LAYERS - 1, -1, -1))
    jobs = [(c, l) for l in layers for c in (range(BS.N_COLS) if len(sys.argv) < 3 else [0, 11])]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=6) as ex:
        for (c, l), out in ex.map(diag, jobs, chunksize=1):
            f, t = out["front"], out["top"]
            ok = f["carry"] > 0 or t["carry"] > 0
            print(f"层{l} 列{c:2d} {'可取' if ok else '**漏**'}  前吸 预取/贴箱/拔出/转后 {f['pre']}/{f['contact']}/{f['pull']}/{f['carry']}   顶吸 {t['pre']}/{t['contact']}/{t['pull']}/{t['carry']}   {time.time()-t0:.0f}s", flush=True)
