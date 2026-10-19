import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core"))   # 让 core/ 里的模块可被导入
"""按星猿哲 RockyOne 官网指标校核底盘：最高速度 0.9 m/s、爬坡 10°(18%)、越障 台阶≤20 mm / 间隙≤50 mm。
轮胎-地板摩擦取 CONFIG 里的 0.6（聚氨酯-木板）。运行：python tools/capability_check.py
"""
import math
import numpy as np
import mujoco
from tip_view import TipDemo
from chassis_sim import CONFIG


def drive(d, v_target, t_total, a=0.3, x0=None, stop_x=None, log=None):
    """沿车体 x 直行，加速度 a，到 t_total 或位置 stop_x 为止。返回统计量。"""
    f = d.fws
    f.set_brake(False)
    dt = d.m.opt.timestep
    v_cmd, t = 0.0, 0.0
    vs, tq, tilt, rolls = [], 0.0, 0.0, 0.0
    fwd = lambda: d.d.xmat[d.base].reshape(3, 3)[:, 0]
    while t < t_total:
        v_cmd += float(np.clip(v_target - v_cmd, -a * dt, a * dt))
        f.command(v_cmd, 0.0, 0.0, dt)
        d.step(); t += dt
        if int(t / dt) % 20 == 0:
            vs.append((t, float(d.d.qvel[:3] @ fwd())))
            tq = max(tq, max(abs(d.d.actuator_force[a_]) for a_ in f.act_drive))
            tilt = max(tilt, d.tilt_deg())
        if stop_x is not None and d.d.xpos[d.base][0] > stop_x:
            break
    return dict(t=t, vs=vs, torque=tq, tilt=tilt, x=float(d.d.xpos[d.base][0]), z=float(d.d.xpos[d.base][2]))


def fresh(alpha=0.0, phi=0.0, susp="on", payload=0.0, pose="stow", **extra):
    ce = dict(friction=0.6)
    if "step_h" in extra or "gap_w" in extra:
        ce.setdefault("speed_gain", 3000.0)        # 轮速环增益：堵转时要能顶到力矩上限（200 时 0.1 m/s 指令下最多只有 ~270 N·m 的指令力矩）
        ce["wheel_capsule"] = True                # 圆柱-方块棱的接触法向不准，车轮在越障测试里用等半径的胶囊（x-z 轮廓是同一个圆）
    ce.update(extra)
    d = TipDemo(pose, payload, susp, phi, alpha, cfg_extra=ce)
    if "step_h" in extra or "gap_w" in extra:
        d.m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL     # 椭圆锥在车轮撞台阶棱时接触力会爆到 100 kN 以上，金字塔锥稳定
        d.m.pair_solimp[:] = [0.99, 0.99, 0.001, 0.5, 2]     # 默认 0.9 的接触阻抗让摩擦很软（滑移 0.1 m/s 才饱和），台阶棱上爬不起来；越障测试用硬摩擦
        d.m.opt.noslip_iterations = 0
    return d


def main():
    print("车体参数：整机 %.0f kg   轮径 %.0f mm   摩擦 0.6   单轮最大力矩 %.0f N·m   轮速环 kv=%.0f\n" % (
        CONFIG["chassis_mass"] + CONFIG["arm_mass"], 2 * CONFIG["wheel_radius"] * 1000, CONFIG["motor_torque"], CONFIG["speed_gain"]))

    # ---- A 最高速度 ----
    print("【A 最高行驶速度 目标 0.9 m/s】平地，加速度 0.3 m/s²")
    for pl in (0.0, 50.0):
        d = fresh(payload=pl)
        r = drive(d, 0.9, 7.0)
        v_end = np.mean([v for t, v in r["vs"] if t > 5.5])
        print("   负载 %2.0f kg：稳态速度 %.3f m/s，最大轮端力矩 %.0f / %.0f N·m，最大倾斜 %.2f°" % (pl, v_end, r["torque"], CONFIG["motor_torque"], r["tilt"]))

    # ---- B 爬坡 ----
    print("\n【B 爬坡 目标 10°(18%)】车头朝上坡，摩擦 0.6")
    for pl, v in ((0.0, 0.3), (50.0, 0.3), (50.0, 0.9)):
        d = fresh(alpha=10.0, phi=180.0, payload=pl)
        for _ in range(500): d.step()                  # 先在坡上静置
        x0 = float(d.d.xpos[d.base] @ np.array([1.0, 0, 0]))
        r = drive(d, v, 8.0 if v < 0.5 else 9.0, a=0.2)
        s = float((d.d.xpos[d.base] @ np.array([1.0, 0, 0])) - x0)
        v_end = np.mean([vv for t, vv in r["vs"] if t > r["t"] - 1.5])
        print("   负载 %2.0f kg 指令 %.1f m/s：爬升距离 %.2f m，末段速度 %.3f m/s，最大轮端力矩 %.0f N·m，最大倾斜(相对坡面) %.2f°" % (pl, v, s, v_end, r["torque"], r["tilt"]))
    d = fresh(alpha=10.0, phi=180.0, payload=50.0)
    for _ in range(300): d.step()
    r = drive(d, 0.3, 4.0, a=0.2)
    d.fws.stop_and_brake()
    for _ in range(3000): d.step()                       # 先让车停稳
    p0 = d.d.xpos[d.base].copy()
    for _ in range(3000): d.step()
    print("   坡上抱闸停稳后再保持 3 s，溜车 %.1f mm" % (np.linalg.norm(d.d.xpos[d.base] - p0) * 1000))

    # ---- C 台阶 ----
    print("\n【C 越障：台阶 20 mm】平地，车头先上台阶，车速 0.1 m/s（台阶边缘 x=1.0 m，车从 x=-1.0 出发）")
    for susp in ("on", "off"):
        for pl in (0.0, 50.0):
            d = fresh(susp=susp, payload=pl, step_h=0.02, step_x=1.0)
            d.d.qpos[0] = -1.0
            mujoco.mj_forward(d.m, d.d)
            r = drive(d, 0.1, 60.0, a=0.1, stop_x=2.2)
            ok = r["x"] > 2.2
            print("   悬挂%s 负载 %2.0f kg：%s（到达 x=%.2f m，用时 %.1f s，最大轮端力矩 %.0f N·m，最大倾斜 %.2f°）" % (
                "开" if susp == "on" else "关", pl, "通过" if ok else "未通过", r["x"], r["t"], r["torque"], r["tilt"]))

    # ---- D 间隙 ----
    print("\n【D 越障：间隙 50 mm】平地，横向缝（缝在 x=1.0~1.05 m），车速 0.1 m/s")
    for susp in ("on", "off"):
        d = fresh(susp=susp, payload=50.0, gap_w=0.05, gap_x=1.0)
        d.d.qpos[0] = -1.0
        mujoco.mj_forward(d.m, d.d)
        r = drive(d, 0.1, 60.0, a=0.1, stop_x=2.2)
        ok = r["x"] > 2.2
        print("   悬挂%s 负载 50 kg：%s（到达 x=%.2f m，用时 %.1f s，最大轮端力矩 %.0f N·m，最大倾斜 %.2f°）" % (
            "开" if susp == "on" else "关", "通过" if ok else "未通过", r["x"], r["t"], r["torque"], r["tilt"]))

    # ---- E 极限 ----
    print("\n【E 极限台阶】（悬挂开，负载 50 kg，0.1 m/s）")
    for h in (0.03, 0.04, 0.05, 0.075):
        d = fresh(susp="on", payload=50.0, step_h=h, step_x=1.0)
        d.d.qpos[0] = -1.0
        mujoco.mj_forward(d.m, d.d)
        r = drive(d, 0.1, 60.0, a=0.1, stop_x=2.2)
        print("   %2.0f mm：%s（到达 x=%.2f m，最大轮端力矩 %.0f N·m）" % (h * 1000, "通过" if r["x"] > 2.2 else "未通过", r["x"], r["torque"]))

    # ---- F 驱动力矩敏感性 ----
    print("\n【F 单轮最大力矩对越障的影响】（悬挂开，负载 50 kg，0.1 m/s，四驱）")
    for T in (120.0, 250.0, 400.0, 600.0):
        row = []
        for kind in ("step", "gap"):
            kw = dict(step_h=0.02, step_x=1.0) if kind == "step" else dict(gap_w=0.05, gap_x=1.0)
            d = fresh(susp="on", payload=50.0, motor_torque=T, speed_gain=3000.0, **kw)
            d.d.qpos[0] = -1.0
            mujoco.mj_forward(d.m, d.d)
            r = drive(d, 0.1, 60.0, a=0.1, stop_x=2.2)
            row.append("%s %s(x=%.2f，最大倾斜 %.1f°)" % ("台阶20mm" if kind == "step" else "间隙50mm", "通过" if r["x"] > 2.2 else "未通过", r["x"], r["tilt"]))
        print("   %3.0f N·m：%s" % (T, "；".join(row)))


if __name__ == "__main__":
    main()
