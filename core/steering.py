"""四舵轮运动学与控制：给定车体速度指令 (vx, vy, wz)，算出每个舵轮的舵角和轮速。

每个舵轮：绕竖直轴转向（关节 s_xx，位置伺服）+ 绕轮轴滚动（关节 j_xx，速度伺服）。
车体系：x 向前，y 向左，z 向上，原点在底盘中心。轮 i 位于 (x_i, y_i)：
    v_i = (vx − wz·y_i, vy + wz·x_i)         轮心处的地面速度
    舵角 δ_i = atan2(v_iy, v_ix)，轮速 = |v_i|
舵角限制在 (−90°, 90°]，超出时把舵角转 180° 并反转轮速，避免大角度反转；
舵角没转到位时按 cos(误差) 缩减轮速（先转舵再行驶），转向速率限幅。
"""
import math
import numpy as np
import mujoco

WHEELS = ["fl", "fr", "rl", "rr"]


class FourWheelSteer:
    def __init__(self, m, d, wheel_radius=0.075, steer_rate=math.radians(120)):
        self.m, self.d, self.r = m, d, wheel_radius
        mj = mujoco
        self.act_steer = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, f"st_{n}") for n in WHEELS]
        self.act_drive = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, f"m_{n}") for n in WHEELS]
        self.j_steer = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, f"s_{n}") for n in WHEELS]
        self.j_roll = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, f"j_{n}") for n in WHEELS]
        self.dof_roll = [m.jnt_dofadr[j] for j in self.j_roll]
        self.steer_rate = steer_rate
        self.cmd = np.zeros(4)                    # 当前舵角指令（经速率限幅）
        self.base = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "base")
        self.body_steer = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, f"steer_{n}") for n in WHEELS]
        self.brake_Nm = 2000.0                    # 抱闸力矩上限 [N·m]
        # 抱闸 = 轮角位置保持（PD）：速度伺服的 ctrl = (kp/kv)·(θ_hold − θ)，等效 力矩 = kp·Δθ − kv·ω。
        # 以前用 kv=1e5 的纯速度伺服，微小转速就把 2000 N·m 力矩上限顶满，成了 bang-bang，
        # 开着摆动后桥在坡上会引起摇摆溜车；位置保持没有这个问题，保持 78 N·m 的坡道力矩只需约 2 mrad 的偏移。
        self.kv0 = [float(m.actuator_gainprm[a, 0]) for a in self.act_drive]
        self.fr0 = [m.actuator_forcerange[a].copy() for a in self.act_drive]
        self.kv_brake = 3000.0                    # 抱闸阻尼 [N·m·s/rad]
        self.kp_brake = 4.0e4                     # 抱闸位置刚度 [N·m/rad]
        self.hold_theta = np.zeros(4)
        self.braked = False

    def wheel_xy(self):
        """各舵轮转向轴在车体系下的位置（取当前位姿）"""
        d = self.d
        R = d.xmat[self.base].reshape(3, 3)
        o = d.xpos[self.base]
        out = []
        for b in self.body_steer:
            p = R.T @ (d.xpos[b] - o)
            out.append((p[0], p[1]))
        return out

    def set_brake(self, on):
        if on and self.braked:
            return
        self.braked = on
        for i, (a, kv0, fr0) in enumerate(zip(self.act_drive, self.kv0, self.fr0)):
            kv = self.kv_brake if on else kv0
            self.m.actuator_gainprm[a, 0] = kv
            self.m.actuator_biasprm[a, 2] = -kv
            self.m.actuator_forcerange[a] = [-self.brake_Nm, self.brake_Nm] if on else fr0
            if on:
                self.hold_theta[i] = self.d.qpos[self.m.jnt_qposadr[self.j_roll[i]]]

    def hold_update(self):
        """抱闸时每个仿真步调用：把各驱动轮拉回抱闸瞬间的轮角"""
        for i, a in enumerate(self.act_drive):
            th = self.d.qpos[self.m.jnt_qposadr[self.j_roll[i]]]
            self.d.ctrl[a] = self.kp_brake / self.kv_brake * (self.hold_theta[i] - th)

    def command(self, vx, vy, wz, dt):
        """设置执行器：返回各轮 (舵角 rad, 轮速 m/s)"""
        d = self.d
        moving = abs(vx) + abs(vy) + abs(wz) > 1e-6
        if moving and self.braked:
            self.set_brake(False)
        res = []
        for i, (x, y) in enumerate(self.wheel_xy()):
            vix, viy = vx - wz * y, vy + wz * x
            sp = math.hypot(vix, viy)
            ang = math.atan2(viy, vix) if sp > 1e-4 else self.cmd[i]
            cur = d.qpos[self.m.jnt_qposadr[self.j_steer[i]]]
            # 选 ±180° 等价角中离当前舵角最近的一个（轮速反号）
            sign = 1.0
            da = (ang - cur + math.pi) % (2 * math.pi) - math.pi
            if abs(da) > math.pi / 2:
                ang += math.pi; sign = -1.0
            target = cur + ((ang - cur + math.pi) % (2 * math.pi) - math.pi)
            step = self.steer_rate * dt
            self.cmd[i] = self.cmd[i] + float(np.clip(target - self.cmd[i], -step, step))
            err = abs(target - cur)
            speed = sign * sp * max(0.0, math.cos(min(err, math.pi / 2)))      # 没转到位就先慢下来
            d.ctrl[self.act_steer[i]] = self.cmd[i]
            d.ctrl[self.act_drive[i]] = speed / self.r
            res.append((cur, speed))
        if not moving:
            for a in self.act_drive:
                d.ctrl[a] = 0.0
        return res

    def stop_and_brake(self):
        for a in self.act_drive:
            self.d.ctrl[a] = 0.0
        self.set_brake(True)
