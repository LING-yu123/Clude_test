"""把 plan_groups 的原始规划（折线路点）整理成"连续运动"用的稠密无碰撞路径，写到 data/unload_plan3.json。
每个组有：chain_a（上一组抬起+接近，站位不变时合并成一条）、chain_b（拔出+搬运+下放，一条）、
站位变化时 chain_u（上一组抬起）单独先走，车体横移后再走 chain_a。最后一组之后还有 tail（抬起离开）。
运行：python core/smooth_plan.py
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import box_spec as BS
import plan_groups as G
import traj as TJ

col = G.col


def okfn(station, holding, mode, removed, grips):
    def f(q):
        return all(col.ok(q, station_y=station, holding=holding, mode=mode, removed=removed, grip=g) for g in grips)
    return f


def tagged(legs):
    """legs: [(waypoint 列表, 该段内部/端点用的 ok 函数)]，拼接去掉重复端点。返回 (路点, 每个内部拐角的 ok 函数)"""
    pts, fns = [], []
    for k, (wps, fn) in enumerate(legs):
        wps = [np.asarray(w, float) for w in wps]
        if k == 0:
            pts += wps
            fns += [fn] * (len(wps) - 2 if len(wps) > 2 else 0)
        else:
            if len(pts):
                fns.append(None)            # 衔接点，稍后统一指定
            pts += wps[1:]
            fns += [fn] * (len(wps) - 2 if len(wps) > 2 else 0)
    return pts, fns


def chain(legs, junction_fns):
    """legs: [(wps, ok_fn)]；junction_fns: 相邻两段衔接点用的 ok_fn（长度 len(legs)-1）"""
    pts, fns, ji = [], [], 0
    for k, (wps, fn) in enumerate(legs):
        wps = [np.asarray(w, float) for w in wps]
        if k == 0:
            pts += wps
            fns += [fn] * max(len(wps) - 2, 0)
        else:
            fns.append(junction_fns[ji]); ji += 1
            pts += wps[1:]
            fns += [fn] * max(len(wps) - 2, 0)
    # 衔接点出现在 fns 的位置：重新按顺序构造，保证长度 = len(pts)-2
    return pts, fns


def build_chain(legs, junction_fns):
    pts, fns = [], []
    for k, (wps, fn) in enumerate(legs):
        wps = [np.asarray(w, float) for w in wps]
        if k == 0:
            pts = list(wps)
            fns = [fn] * max(len(wps) - 2, 0)
        else:
            # 上一段末点 = 本段首点：该点是内部拐角
            fns.append(junction_fns[k - 1])
            pts += wps[1:]
            fns += [fn] * max(len(wps) - 2, 0)
            if len(wps) == 1:
                pass
    # 当前一段只有 2 个点、没有内部拐角时，上面的"首尾"计数仍然成立：拐角数 = len(pts)-2
    if len(fns) != max(len(pts) - 2, 0):
        raise RuntimeError(f"拐角数不一致 {len(fns)} vs {len(pts) - 2}")
    dense, stops = TJ.smooth_chain(pts, fns)
    return dict(q=np.round(dense, 4).tolist(), stops=[int(s) for s in stops])


def main():
    raw = json.load(open("data/unload_plan3_raw.json", encoding="utf-8"))
    parts = raw["parts"]
    t0 = time.time()
    Qh = {m: np.array(raw["place"][m]["hi"]) for m in raw["place"]}
    out_parts = []
    removed = set()
    prev = None
    for k, p in enumerate(parts):
        mode, st = p["mode"], p["station"]
        boxes = set((c, p["layer"]) for c in p["cols"])
        ge = G.grip_state(mode, p["width"], p["yoff"], pi=0.0, rho=(BS.TRAY_PLACE_RETRACT if mode == "front" else 0.0), empty=True)
        gh = G.grip_state(mode, p["width"], p["yoff"], pi=0.0, rho=0.0)
        gf = G.full_grip(mode)
        pl = raw["place_legs"][mode][str(st)]
        entry = dict(layer=p["layer"], cols=p["cols"], mode=mode, station=st, ypl=p["ypl"], yoff=p["yoff"], width=p["width"], widths=p["widths"])
        # ---- chain_a / chain_u
        amode = "pick" if mode == "top" else "free"
        appr = p["approach"]
        if prev is None:
            entry["chain_a"] = build_chain([(appr, okfn(st, False, amode, removed, [ge]))], [])
        else:
            pm, pst = prev["mode"], prev["station"]
            up = raw["place_legs"][pm][str(pst)]["up"]
            gu = G.grip_state(pm, BS.BOX_W * BS.MAX_GROUP, 0.0, pi=0.0, rho=BS.TRAY_PLACE_RETRACT, empty=True)
            f_up = okfn(pst, False, "free", removed, [gu])
            if pst == st:
                entry["chain_a"] = build_chain([(up, f_up), (appr, okfn(st, False, amode, removed, [ge]))],
                                               [okfn(st, False, "free", removed, [gu, ge])])
            else:
                entry["chain_u"] = build_chain([(up, f_up)], [])
                entry["chain_a"] = build_chain([(appr, okfn(st, False, amode, removed, [ge]))], [])
        # ---- chain_b：拔出 + 搬运 + 下放
        rem_h = removed | boxes
        pull, carry, down = p["pull"], p["carry"], pl["down"]
        f_pull = okfn(st, True, "pick", rem_h, [gh])
        f_free = okfn(st, True, "free", rem_h, [gh])
        f_down = okfn(st, True, "place", rem_h, [gf])
        entry["chain_b"] = build_chain([(pull, f_pull), (carry, f_free), (down, f_down)],
                                       [f_pull, okfn(st, True, "free", rem_h, [gh, gf])])
        out_parts.append(entry)
        removed |= boxes
        prev = entry
        print(f"  {k + 1}/{len(parts)} 层{p['layer']} 列{p['cols']} {mode} 站位{st:+.1f}  chain_b {len(entry['chain_b']['q'])} 点, 停点 {len(entry['chain_b']['stops'])}  {time.time()-t0:.0f}s", flush=True)
    # 最后一组之后的抬起
    last = out_parts[-1]
    pl = raw["place_legs"][last["mode"]][str(last["station"])]
    gu = G.grip_state(last["mode"], BS.BOX_W * BS.MAX_GROUP, 0.0, pi=0.0, rho=BS.TRAY_PLACE_RETRACT, empty=True)
    tail = build_chain([(pl["up"], okfn(last["station"], False, "free", removed, [gu]))], [])
    out = dict(version=3, box=raw["box"], place=raw["place"], start=raw["start"], parts=out_parts, tail=tail,
               unreachable=raw["unreachable"], limits=dict(front_layer_min=BS.FRONT_LAYER_MIN, stroke=BS.STROKE,
                                                          place_push=BS.PLACE_PUSH, tray_retract=BS.TRAY_RETRACT,
                                                          tray_place_retract=BS.TRAY_PLACE_RETRACT, place_x=BS.PLACE_X, bed_half_w=BS.BED_HALF_W))
    json.dump(out, open("data/unload_plan3.json", "w"))
    n = sum(len(p["cols"]) for p in out_parts)
    print(f"完成：{len(out_parts)} 组，{n} 箱，用时 {time.time()-t0:.0f}s，写入 data/unload_plan3.json")


if __name__ == "__main__":
    main()
