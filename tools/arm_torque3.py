import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))   # 让 core/ 里的模块可被导入
"""成组卸货时机械臂各关节力矩（逆动力学，平地，含 3 箱 51 kg + 夹具）。python tools/arm_torque3.py"""
import json
import numpy as np
from tip_view import TipDemo
from unload_player3 import UnloadPlayer3

plan = json.load(open("data/unload_plan3.json", encoding="utf-8"))
d = TipDemo("stow", 0, "off", 0.0, 0.0)
P = UnloadPlayer3(d, plan, True); P.start()
dt = d.m.opt.timestep; k = 0; pk = np.zeros(7); acc = np.zeros(7); n = 0; hist = []
while P.active and d.d.time < 6000:
    P.update(dt); d.step(); k += 1
    if k % 100 == 0:
        ta, _, _ = d.torques(); pk = np.maximum(pk, ta); acc += ta ** 2; n += 1; hist.append(ta)
rms = np.sqrt(acc / max(n, 1))
print("关节      S      L      E      U      R      B      T")
H = np.array(hist)
p99 = np.percentile(H, 99, axis=0)
print("99%分位" + " ".join("%6.0f" % x for x in p99) + "  N·m   （峰值含吸合/释放时质量突变和夹具动作的瞬态尖峰）")
print("峰值 " + " ".join("%6.0f" % x for x in pk) + "  N·m")
print("均方根" + " ".join("%6.0f" % x for x in rms) + "  N·m")
