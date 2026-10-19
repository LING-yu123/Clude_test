import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))   # 让 core/ 里的模块可被导入
"""机械臂连杆的碰撞网格：把每个连杆零件沿最长方向切成若干段，每段取凸包（MuJoCo 的网格碰撞用凸包，整块凸包太"胖"，
切开后更贴合真实外形）。输入 meshes/gp_*.stl（相对关节零位原点，mm），输出 meshes/col_*.stl 和 gp110b/collision.json。
运行一次即可：python build_collision_meshes.py"""
import json, os
import numpy as np
import trimesh

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # 工程根目录
MESH = os.path.join(HERE, "meshes")
PIECE_MM = 190.0          # 每段最大长度


def main():
    info = json.load(open(os.path.join(HERE, "gp110b", "links.json"), encoding="utf-8"))
    out = []
    for p in info["parts"]:
        if p["link"] == 0:
            continue                                     # 底座不需要（车体碰撞用方盒）
        m = trimesh.load(os.path.join(MESH, p["file"]))
        ext = m.bounds[1] - m.bounds[0]
        if max(ext) < 40:                                # 太小的件（螺栓、接头）不参与
            continue
        ax = int(np.argmax(ext))
        n = max(1, int(np.ceil(ext[ax] / PIECE_MM)))
        edges = np.linspace(m.bounds[0][ax], m.bounds[1][ax], n + 1)
        cen = m.triangles_center[:, ax]
        for k in range(n):
            sel = (cen >= edges[k]) & (cen <= edges[k + 1] + (1e-6 if k == n - 1 else 0))
            if sel.sum() < 4:
                continue
            pts = m.vertices[np.unique(m.faces[sel].ravel())]
            try:
                hull = trimesh.convex.convex_hull(pts)
            except Exception:
                continue
            if len(hull.faces) < 4:
                continue
            name = f"col_{p['file'][:-4]}_{k}.stl"
            hull.export(os.path.join(MESH, name))
            out.append(dict(file=name, link=p["link"], faces=len(hull.faces)))
    json.dump(out, open(os.path.join(HERE, "gp110b", "collision.json"), "w"), indent=1)
    by = {}
    for o in out:
        by[o["link"]] = by.get(o["link"], 0) + 1
    print("碰撞块数（按连杆）：", by, "共", len(out))


if __name__ == "__main__":
    main()
