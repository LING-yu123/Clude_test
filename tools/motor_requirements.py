import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""设计优化 第 5 步：电机选型数据（行走电机 / 转向电机 / 机械臂关节）。
运行：python tools/motor_requirements.py      结果：output/logs/motor_requirements.txt
"""
import json, math
import numpy as np
import mujoco
from capability_check import fresh, drive
from tip_view import TipDemo
from unload_player import UnloadPlayer
import chassis_sim as C

R = C.CONFIG["wheel_radius"]


def wheel_stats(d, t_total, v_target, a, alpha_note=""):
    f = d.fws; f.set_brake(False)
    dt = d.m.opt.timestep; v = 0.0; t = 0.0; pk_t = 0.0; pk_p = 0.0; pk_w = 0.0
    while t < t_total:
        v += float(np.clip(v_target - v, -a * dt, a * dt))
        f.command(v, 0, 0, dt); d.step(); t += dt
        if int(round(t / dt)) % 20 == 0:
            tq = np.array([d.d.actuator_force[a_] for a_ in f.act_drive])
            w = np.array([d.d.qvel[x] for x in d.wheel_dofs])
            pk_t = max(pk_t, float(np.max(np.abs(tq)))); pk_p = max(pk_p, float(np.sum(np.abs(tq * w)))); pk_w = max(pk_w, float(np.max(np.abs(w))))
    return pk_t, pk_p, pk_w * 60 / (2 * math.pi)


def main():
    out = []
    P = lambda s: (print(s, flush=True), out.append(s))
    M = C.CONFIG["chassis_mass"] + C.CONFIG["arm_mass"] + 50
    P("整车（含 50 kg 箱子）%.0f kg，轮径 %.0f mm，4 轮驱动" % (M, 2000 * R))
    P("\n【行走电机（单轮，轮端）】")
    d = fresh(payload=50.0)
    t, p, n = wheel_stats(d, 6.0, 0.9, 0.3)
    P("  平地 0→0.9 m/s，加速度 0.3 m/s²：峰值力矩 %.0f N·m，四轮总机械功率 %.0f W，最高转速 %.0f rpm" % (t, p, n))
    for v in (0.3, 0.9):
        d = fresh(alpha=10.0, phi=180.0, payload=50.0)
        for _ in range(500): d.step()
        t, p, n = wheel_stats(d, 9.0 if v > 0.5 else 8.0, v, 0.2)
        P("  10° 坡匀速上坡 %.1f m/s：峰值力矩 %.0f N·m，四轮总机械功率 %.0f W，转速 %.0f rpm" % (v, t, p, n))
    hold = M * 9.81 * math.sin(math.radians(10)) * R / 4
    P("  10° 坡抱闸保持：每轮 %.0f N·m（理论 m·g·sin10°·r/4）" % hold)
    P("  20 mm 台阶（μ=0.6）：每轮需 ≥160 N·m 的短时峰值（仿真下限在 120~160 之间，推荐 ≥200）")
    P("  → 推荐：单轮持续 ≥100 N·m、峰值 ≥200 N·m（含减速器后轮端）；最高转速 ≥%.0f rpm；四轮总持续功率 ≥%.1f kW（0.9 m/s 上 10° 坡，未计效率）" % (n, p / 1000))

    P("\n【转向电机（单轮绕竖直轴）】")
    for label, alpha, phi in (("平地", 0.0, 0.0), ("10° 坡", 10.0, 0.0)):
        d = fresh(alpha=alpha, phi=phi, payload=50.0, steer_torque=800.0)
        f = d.fws; f.set_brake(False)
        dt = d.m.opt.timestep; pk = 0.0; pks = 0.0
        for k in range(int(8.0 / dt)):                         # 原地旋转 0.3 rad/s，舵轮先转到切向再滚动
            f.command(0.0, 0.0, 0.3, dt); d.step()
            if k % 20 == 0:
                tq = [abs(d.d.actuator_force[a_]) for a_ in f.act_steer]
                pk = max(pk, max(tq))
        P("  %s 原地旋转：转向峰值力矩 %.0f N·m（上限 %.0f）" % (label, pk, 800.0))
        d = fresh(alpha=alpha, phi=phi, payload=50.0, steer_torque=800.0)
        f = d.fws; f.set_brake(False); pk = 0.0
        for k in range(int(6.0 / dt)):                         # 蟹行：横移，先把舵角转 90°
            f.command(0.0, 0.3, 0.0, dt); d.step()
            if k % 20 == 0:
                pk = max(pk, max(abs(d.d.actuator_force[a_]) for a_ in f.act_steer))
        P("  %s 90° 蟹行起步：转向峰值力矩 %.0f N·m" % (label, pk))
    P("  平地原地旋转稳态约 100~185 N·m（起步瞬态 ~390 N·m）；10° 坡上原地旋转时车体会下溜、舵轮被拖到 >400~800 N·m，不建议在坡上原地旋转")
    P("  转向速率限幅 120°/s（steering.py）；→ 推荐：转向电机（含减速器）峰值 ≥450 N·m、持续 ≥200 N·m、速度 ≥120°/s（轮胎刮擦模型是简化的，需用样机校核）")

    P("\n【机械臂关节力矩（平地卸 6 个箱子，逆动力学，GP110B）】")
    plan = json.load(open("data/unload_plan2.json", encoding="utf-8")); plan["order"] = plan["order"][:6]
    d = TipDemo("stow", 0, "off", 0.0, 0.0)
    pl = UnloadPlayer(d, plan, True); pl.start()
    dt = d.m.opt.timestep; k = 0; pk = np.zeros(7); acc = np.zeros(7); n = 0
    while pl.active and d.d.time < 600:
        pl.update(dt); d.step(); k += 1
        if k % 100 == 0:
            ta, _, _ = d.torques(); pk = np.maximum(pk, ta); acc += ta ** 2; n += 1
    rms = np.sqrt(acc / max(n, 1))
    P("  关节      S      L      E      U      R      B      T")
    P("  峰值 " + " ".join("%6.0f" % x for x in pk) + "  N·m")
    P("  均方根" + " ".join("%6.0f" % x for x in rms) + "  N·m")
    open("output/logs/motor_requirements.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
