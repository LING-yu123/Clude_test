import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))   # 让 core/ 里的模块可被导入
"""把安川 GP110B 的 STEP（07vxb110-a00.asm_0.stp）转成 MuJoCo 能用的网格。
STEP 装配是"图上画的姿态"（下臂竖直、前臂水平向前，法兰朝前），坐标：Y 向上、-Z 向前、-X 向左。
转成机器人坐标（x 前、y 左、z 上，单位 mm，原点在 S 轴与安装面交点），再按关节拆成各连杆：
  BASE → 固定；S → J1；L1 → J2(L)；L2 → J3(E)；U → J4(U)；R → J5；B → J6；T → J7
每个连杆的网格相对"该关节在零位的原点"表示（MuJoCo 里连杆体在 θ=0 时没有旋转）。
输出：meshes/gp_*.stl、gp110b/links.json（各连杆的体积占比、体积质心、外形尺寸）。
运行一次即可：python build_gp110b_meshes.py
"""
import json, os, time
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # 工程根目录
STP = os.path.join(HERE, "data", "07vxb110-a00.asm_0.stp")
GLB = os.path.join(HERE, "gp110b", "gp110b.glb")
OUT = os.path.join(HERE, "meshes")
TARGET_FACES = 14000            # 每个零件抽稀到的面数上限（原始 20–25 万面，太重）

# 关节在零位的原点（mm，机器人系）：与尺寸图一致
J_ORIGIN = {1: (0, 0, 0), 2: (320, 0, 540), 3: (320, 0, 975), 4: (320, 0, 1410), 5: (320, 0, 1645), 6: (1340, 0, 1645), 7: (1340, 0, 1645)}
PART_LINK = {"BASE": 0, "S": 1, "L1": 2, "L2": 3, "U": 4, "R": 5, "B": 6, "T": 7}
COLOR = {0: (0.18, 0.2, 0.24), 1: (0.12, 0.32, 0.62), 2: (0.12, 0.32, 0.62), 3: (0.12, 0.32, 0.62), 4: (0.12, 0.32, 0.62),
         5: (0.12, 0.32, 0.62), 6: (0.12, 0.32, 0.62), 7: (0.7, 0.72, 0.75)}


def to_robot(v):
    """STEP(Y 上, -Z 前, -X 左)[m] → 机器人系[mm]"""
    return np.stack([-v[:, 2], -v[:, 0], v[:, 1]], 1) * 1000.0


def main():
    t0 = time.time()
    os.makedirs(os.path.dirname(GLB), exist_ok=True)
    if not os.path.exists(GLB):
        import cascadio
        cascadio.step_to_glb(STP, GLB, 0.2, 0.5)
    sc = trimesh.load(GLB, force="scene")
    parents = sc.graph.transforms.parents
    out, info = [], {}
    for node in sc.graph.nodes_geometry:
        T, gname = sc.graph[node]
        g = sc.geometry[gname].copy()
        g.apply_transform(T)
        part = node
        while part not in PART_LINK and (not part.endswith("_ASM")):          # 向上找到零件名
            key = part.split("_VIEW")[0]
            if key in PART_LINK:
                part = key; break
            part = parents.get(part)
            if part is None:
                break
        else:
            part = part.split("_VIEW")[0]
        if part is None or part not in PART_LINK:
            print("忽略节点", node); continue
        link = PART_LINK[part]
        V = to_robot(np.asarray(g.vertices))
        F = np.asarray(g.faces)
        m = trimesh.Trimesh(V, F, process=True)
        vol = abs(float(m.volume)) if m.is_volume else float(m.convex_hull.volume) * 0.5
        cen = np.array(m.center_mass if m.is_volume else m.centroid)
        if len(m.faces) > TARGET_FACES:
            import fast_simplification
            v2, f2 = fast_simplification.simplify(m.vertices.astype(np.float64), m.faces.astype(np.int64),
                                                  target_reduction=1 - TARGET_FACES / len(m.faces))
            m = trimesh.Trimesh(v2, f2, process=True)
        o = np.array(J_ORIGIN.get(link, (0, 0, 0)), float)
        m.vertices = m.vertices - o                       # 相对该关节原点
        name = f"gp_{part}_{len(out)}"
        m.export(os.path.join(OUT, name + ".stl"))
        out.append(dict(file=name + ".stl", link=link, part=part, faces=len(m.faces), vol_L=vol / 1e6, color=COLOR[link]))
        info.setdefault(str(link), dict(vol=0.0, cen=np.zeros(3)))
        info[str(link)]["vol"] += vol
        info[str(link)]["cen"] = info[str(link)]["cen"] + vol * (cen - o)
        print(f"  {name:14s} link {link}  面 {len(m.faces):6d}  体积 {vol/1e6:6.1f} L")
    for k, v in info.items():
        v["cen"] = (v["cen"] / max(v["vol"], 1e-9) / 1000.0).tolist()     # 体积质心，本关节系，m
    tot = sum(v["vol"] for k, v in info.items() if k != "0")
    for k, v in info.items():
        v["frac"] = v["vol"] / tot if k != "0" else 0.0
        v["vol_L"] = v["vol"] / 1e6
        del v["vol"]
    json.dump(dict(parts=out, links=info), open(os.path.join(HERE, "gp110b", "links.json"), "w"), ensure_ascii=False, indent=1)
    print("各连杆体积占比（0=底座，固定）：", {k: round(v["frac"], 3) for k, v in info.items()})
    print("用时 %.0f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
