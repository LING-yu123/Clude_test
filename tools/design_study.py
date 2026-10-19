import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""设计优化 第 1 步：轴距 / 轮距 / 质心 / 配重 对 卸货全过程（10° 坡）最小静态倾翻角的影响。
判据：最小倾翻角 >= TARGET（10° 工作坡 + 5° 动态余量）。结果写到 output/tables/design_study_tip.csv
运行：python tools/design_study.py
"""
import csv, itertools
import numpy as np
import chassis_sim as C
import stability_map as S

TARGET = 15.0
L0, B0 = 0.89, 0.73          # 页面原轴距、轮距


def variant(ts, ws, cx, ballast, bx=-0.40, bz=0.20):
    m0, c0 = 1000.0, np.array([cx, 0.0, 0.40])
    M, com = m0, c0
    if ballast:
        pb = np.array([bx, 0.0, bz]); M = m0 + ballast
        com = (m0 * c0 + ballast * pb) / M
    return dict(track_scale=ts, wheelbase_scale=ws, chassis_mass=M, chassis_com=tuple(com))


def main():
    plist = S.poses(); seen, uniq = set(), []
    for q, pl in plist:
        k = (tuple(np.round(q, 2)), pl)
        if k not in seen:
            seen.add(k); uniq.append((q, pl))
    rows = []
    grid = list(itertools.product((1.0, 1.15, 1.25, 1.4, 1.55), (1.0, 1.15, 1.25, 1.4), (-0.25, -0.15, -0.05), (0.0, 200.0, 400.0)))
    print("变体数", len(grid), "姿态数", len(uniq), flush=True)
    for i, (ts, ws, cx, bal) in enumerate(grid):
        ex = variant(ts, ws, cx, bal)
        w = S.evaluate(ex, uniq)
        rows.append(dict(track=round(B0 * ts, 3), wheelbase=round(L0 * ws, 3), com_x=cx, ballast=bal, chassis_mass=1000 + bal,
                         min_tip=round(float(w.min()), 1), worst_heading=S.HEADINGS[int(w.argmin())],
                         tip_0=round(float(w[0]), 1), tip_90=round(float(w[S.HEADINGS.index(90)]), 1),
                         tip_180=round(float(w[S.HEADINGS.index(180)]), 1), tip_270=round(float(w[S.HEADINGS.index(270)]), 1)))
        if i % 10 == 0:
            print("  %d/%d" % (i + 1, len(grid)), flush=True)
    os.makedirs("output/tables", exist_ok=True)
    with open("output/tables/design_study_tip.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    ok = [r for r in rows if r["min_tip"] >= TARGET]
    print("\n满足 >= %.0f° 的变体 %d/%d" % (TARGET, len(ok), len(rows)))
    # 同一整车质量下，足迹（轮距×轴距）最小的方案
    for bal in (0.0, 200.0, 400.0):
        sub = sorted([r for r in ok if r["ballast"] == bal], key=lambda r: r["track"] * r["wheelbase"])
        print("配重 +%3.0f kg（底盘 %4.0f kg）：" % (bal, 1000 + bal) + ("无满足方案" if not sub else ""))
        for r in sub[:3]:
            print("   轮距 %.2f 轴距 %.2f 质心x %+.2f -> 最小倾翻角 %.1f°（坡向 %d°）足迹 %.2f m²" % (
                r["track"], r["wheelbase"], r["com_x"], r["min_tip"], r["worst_heading"], r["track"] * r["wheelbase"]))


if __name__ == "__main__":
    main()
