import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""爬坡可视化：车头朝上坡，从坡底开上去，终端实时打印速度、轮端力矩、倾斜。
运行：python tools/slope_view.py [坡度°=10] [车速 m/s=0.3] [悬挂 on|off=on] [负载 kg=50] [摩擦=0.6]
按键：B = 抱闸/继续行驶（看坡上停车溜不溜车）   Backspace = 重新开始
"""
import time
import numpy as np
import mujoco, mujoco.viewer
from capability_check import fresh


def main():
    a = sys.argv[1:]
    alpha = float(a[0]) if len(a) > 0 else 10.0
    speed = float(a[1]) if len(a) > 1 else 0.3
    susp = a[2] if len(a) > 2 else "on"
    payload = float(a[3]) if len(a) > 3 else 50.0
    mu = float(a[4]) if len(a) > 4 else 0.6
    print("坡度 %.0f°  车速 %.2f m/s  悬挂 %s  负载 %.0f kg  摩擦 %.2f   （B=抱闸/行驶，Backspace=重来）" % (alpha, speed, susp, payload, mu), flush=True)
    d = fresh(alpha=alpha, phi=180.0, susp=susp, payload=payload, friction=mu, floor_friction=mu)   # phi=180：车头(+x)朝上坡
    st = {"brake": False, "reset": False}
    d.fws.set_brake(False)

    def on_key(key):
        if key == ord("B"):
            st["brake"] = not st["brake"]
        elif key == 259:
            st["reset"] = True

    with mujoco.viewer.launch_passive(d.m, d.d, key_callback=on_key) as v:
        v.cam.lookat[:] = [0.5, 0, 0.5]; v.cam.distance = 5.0; v.cam.azimuth = 100; v.cam.elevation = -8
        v_cmd, last = 0.0, 0.0
        dt = d.m.opt.timestep
        p0 = d.d.xpos[d.base].copy()
        while v.is_running():
            t0 = time.time()
            if st["reset"]:
                st["reset"] = False; st["brake"] = False
                d.reset(); d.fws.set_brake(False); v_cmd = 0.0; p0 = d.d.xpos[d.base].copy()
            for _ in range(16):
                if st["brake"]:
                    d.fws.stop_and_brake(); v_cmd = 0.0
                else:
                    v_cmd = min(speed, v_cmd + 0.2 * dt)
                    d.fws.command(v_cmd, 0.0, 0.0, dt)
                d.step()
            v.sync()
            if time.time() - last > 0.5:
                last = time.time()
                fwd = d.d.xmat[d.base].reshape(3, 3)[:, 0]
                vel = float(d.d.qvel[:3] @ fwd)
                tq = max(abs(d.d.actuator_force[a_]) for a_ in d.fws.act_drive)
                print("t=%5.1f s  %s  沿坡速度 %+.3f m/s  已爬高 %.2f m  最大轮端力矩 %3.0f N·m  倾斜 %.1f°" % (
                    d.d.time, "抱闸" if st["brake"] else "行驶", vel, d.d.xpos[d.base][2] - p0[2], tq, d.tilt_deg()), flush=True)
            time.sleep(max(0, 16 * dt - (time.time() - t0)))


if __name__ == "__main__":
    main()
