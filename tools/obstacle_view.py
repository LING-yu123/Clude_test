import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # 让 core/ 里的模块可被导入
"""越障可视化：车从 x=-1 m 以 0.1 m/s（可调）直行去过台阶/间隙，看在 MuJoCo 里是怎么过（或过不去）的。
运行：python tools/obstacle_view.py [step|gap] [高度或宽度 mm=20|50] [单轮最大力矩 N·m=120] [悬挂 on|off=on] [负载 kg=50] [车速 m/s=0.1] [摩擦系数=0.6]
按键：Backspace = 重新开始
"""
import math, time
import numpy as np
import mujoco, mujoco.viewer
from tip_view import TipDemo


def build(kind, size_mm, torque, susp, payload, mu=0.6):
    from capability_check import fresh               # 与 capability_check 同一套越障接触设置（胶囊轮、金字塔摩擦锥、硬摩擦）
    kw = dict(step_h=size_mm / 1000, step_x=1.0) if kind == "step" else dict(gap_w=size_mm / 1000, gap_x=1.0)
    d = fresh(susp=susp, payload=payload, motor_torque=torque, speed_gain=3000.0, friction=mu, floor_friction=mu, **kw)
    d.d.qpos[0] = -1.0
    mujoco.mj_forward(d.m, d.d)
    d.fws.set_brake(False)
    return d


def main():
    a = sys.argv[1:]
    kind = a[0] if len(a) > 0 else "step"
    size = float(a[1]) if len(a) > 1 else (20.0 if kind == "step" else 50.0)
    torque = float(a[2]) if len(a) > 2 else 120.0
    susp = a[3] if len(a) > 3 else "on"
    payload = float(a[4]) if len(a) > 4 else 50.0
    mu = float(a[6]) if len(a) > 6 else 0.6
    speed = float(a[5]) if len(a) > 5 else 0.1          # 过障碍的车速；0.9 m/s 只是平地最高速度
    print("%s %.0f mm，单轮最大力矩 %.0f N·m，悬挂 %s，负载 %.0f kg，车速 %.2f m/s   （Backspace 重来）" % ("台阶" if kind == "step" else "间隙", size, torque, susp, payload, speed), flush=True)
    st = {"d": build(kind, size, torque, susp, payload, mu), "reset": False}

    def on_key(key):
        if key == 259:
            st["reset"] = True

    d = st["d"]
    with mujoco.viewer.launch_passive(d.m, d.d, key_callback=on_key) as v:
        v.cam.lookat[:] = [1.0, 0, 0.2]; v.cam.distance = 3.5; v.cam.azimuth = 120; v.cam.elevation = -12
        v_cmd, last = 0.0, 0.0
        while v.is_running():
            t0 = time.time()
            if st["reset"]:
                st["reset"] = False
                d.reset(); d.d.qpos[0] = -1.0; mujoco.mj_forward(d.m, d.d); d.fws.set_brake(False); v_cmd = 0.0
            dt = d.m.opt.timestep
            for _ in range(16):
                v_cmd = min(speed, v_cmd + 0.2 * dt)
                d.fws.command(v_cmd, 0.0, 0.0, dt)
                d.step()
            v.sync()
            if time.time() - last > 0.5:
                last = time.time()
                x = d.d.xpos[d.base][0]
                print("t=%5.1f s  x=%+.2f m  倾斜 %.1f°  %s" % (d.d.time, x, d.tilt_deg(), "已过" if x > 2.2 else ""), flush=True)
            time.sleep(max(0, 16 * dt - (time.time() - t0)))


if __name__ == "__main__":
    main()
