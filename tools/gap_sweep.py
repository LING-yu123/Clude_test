import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""间隙宽度扫描：看多宽的缝过不去。python tools/gap_sweep.py"""
import mujoco
from capability_check import fresh, drive
for susp in ("on", "off"):
    row = []
    for w in (50, 70, 90, 110, 130):
        d = fresh(susp=susp, payload=50.0, speed_gain=3000.0, gap_w=w / 1000, gap_x=1.0)
        d.d.qpos[0] = -1.0; mujoco.mj_forward(d.m, d.d)
        r = drive(d, 0.1, 60.0, a=0.1, stop_x=2.2)
        row.append("%d mm:%s" % (w, "通过" if r["x"] > 2.2 else "卡住(x=%.2f)" % r["x"]))
    print("悬挂%s：%s" % ("开" if susp == "on" else "关", "  ".join(row)), flush=True)
