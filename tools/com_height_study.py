import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""设计优化 第 4 步：底盘质心高度 z 与整车质量对最小静态倾翻角的影响（当前足迹 轮距 0.91 / 轴距 1.11，质心 x=-0.15）。
运行：python tools/com_height_study.py"""
import numpy as np
import stability_map as S
plist = S.poses(); seen, uniq = set(), []
for q, pl in plist:
    k = (tuple(np.round(q, 2)), pl)
    if k not in seen:
        seen.add(k); uniq.append((q, pl))
print("底盘质心离地高度 z（底盘 1000 kg）→ 最小静态倾翻角")
for z in (0.20, 0.30, 0.40, 0.50, 0.60):
    w = S.evaluate(dict(chassis_com=(-0.15, 0.0, z)), uniq)
    print("   z=%.2f m：%.1f°（最差坡向 %d°）" % (z, w.min(), S.HEADINGS[int(w.argmin())]), flush=True)
print("底盘质量（质心 z=0.40 不变，等于均匀加减质量）→ 最小静态倾翻角")
for M in (700.0, 850.0, 1000.0, 1200.0):
    w = S.evaluate(dict(chassis_mass=M), uniq)
    print("   底盘 %4.0f kg：%.1f°" % (M, w.min()), flush=True)
