"""MuJoCo 倾覆可视化：带真实关节的七轴机械臂 + 倾斜台，实时看臂姿态和负载对倾覆的影响。

运行：双击 tip_view.bat，或
      python tip_view.py [姿态] [负载kg] [悬挂 on|off] [坡向°] [起始坡度°]
  例：python tip_view.py ready 0 off 0

窗口里的按键（先点一下窗口）：
  .  /  ,     机械臂切到 下一个 / 上一个 预设姿态（臂会真的转过去，质心随之移动）
  =  /  -     末端负载 +10 / -10 kg（棕色箱子出现，质量真实加到末端）
  ]  /  [     坡度 +1° / -1°            '  /  ;     坡向 +15° / -15°
  N           自动抬升地面（0.5°/秒），再按一次停止
  U           卸货演示：前方箱墙（烟箱）逐个取下，转到后面放到理料机构上（用 unload_opt.py 优化的动作）；再按停止
  F           播放速度 ×1 / ×4 / ×10 切换
  Y           自动演示装卸货循环：准备→前伸→吸取(吸起50kg)→举高→侧向→准备(放下)→收纳，再按一次停止
  M           开/关 后桥悬挂（摆动桥自由 / 锁死）
  Backspace   重置
  坡向：0=车头朝下坡  90=左侧朝下坡  180=车头朝上坡  270=右侧朝下坡

画面里：黄色线框 = 支承多边形（悬挂开启是三角形，关闭是四边形）；
        黄球 = 整机质心沿重力方向在地面的投影，红球 = 整机质心；
        投影越出线框就会翻。  车顶小球：绿 = 稳定，橙 = 有轮子离地，红 = 已倾翻
命令行每 0.5 秒打印 姿态 / 负载 / 坡度 / 倾斜角 / 四轮载荷。

离屏出图（不开窗口）：python tip_view.py --render
"""
import json, math, sys, time
import numpy as np
from arm_abb import LIM_DEG
import mujoco
from chassis_sim import CONFIG, cfg_live, compile_model, WHEELS, G
from a_model import preset_q
from steering import FourWheelSteer
from unload_player import UnloadPlayer as UnloadPlayer2

ORDER = ["stow", "ready", "reach", "pick", "lift", "side", "rear", "zero"]
CN = {"ready": "准备", "stow": "收纳", "reach": "前伸", "pick": "吸取",
      "lift": "举高", "side": "侧向", "rear": "朝后扫描", "zero": "零位"}
ARM_RATE = math.radians(35.0)          # 机械臂关节最大转速 [rad/s]
CYCLE = ["ready", "reach", "pick", "lift", "side", "ready", "stow"]    # 取自页面"自动演示"
CYCLE_DWELL = 4.0                      # 每个姿态停留 [s]
CYCLE_PAYLOAD = 50.0                   # 吸起的箱子 [kg]
ARM_ACC = math.radians(70.0)           # 关节最大角加速度 [rad/s²]


class TipDemo:
    def __init__(self, pose="reach", payload=50.0, susp="off", phi=0.0, alpha=0.0, cfg_extra=None):
        base_cfg = dict(CONFIG, suspension=susp, friction=CONFIG["friction"], tilt_floor=True, scene=True)     # 聚氨酯-木板 0.6（10° 坡验证过 0.6 与 3.0 结果一致）
        base_cfg.update(cfg_extra or {})                    # 例如 friction=0.6, step_h, gap_w, scene=False
        self.cfg = cfg_live(base_cfg, pose, mesh=True)
        self.m = compile_model(self.cfg, self.cfg.get("step_h"))
        self.d = mujoco.MjData(self.m)
        self.r = self.cfg["wheel_radius"]
        mj = mujoco
        self.base = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, "base")
        self.pay = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, "payload")
        self.pay_geom = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_GEOM, "payload_box")
        self.wb = [mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, f"wheel_{n}") for n in WHEELS]
        self.arm_act = [mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_ACTUATOR, f"a_j{i}") for i in range(1, 8)]
        self.arm_qadr = [self.m.jnt_qposadr[self.m.actuator_trnid[a][0]] for a in self.arm_act]
        self.arm_dofs = [self.m.jnt_dofadr[self.m.actuator_trnid[a][0]] for a in self.arm_act]
        self.arm_damp = self.m.dof_damping[self.arm_dofs].copy()      # 我为数值稳定加的人为阻尼/转子惯量，算力矩时要扣掉
        self.arm_arm = self.m.dof_armature[self.arm_dofs].copy()
        self.wheel_dofs = [self.m.jnt_dofadr[self.m.joint(f"j_{n}").id] for n in WHEELS]
        self.tilt_mocap = self.m.body_mocapid[mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, "tilt")]
        self.scene_mocap = self.m.body_mocapid[mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_BODY, "scene")]
        self.wall_geom = {(c, l): mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_GEOM, f"wb_{c}_{l}") for c in range(8) for l in range(8)}
        self.bed_geom = mj.mj_name2id(self.m, mj.mjtObj.mjOBJ_GEOM, "bed_box")
        self.station = 0.0                                # 机器人在车厢里的横向位置 y [m]（用移动场景来等效）
        self.susp = susp == "on"
        self.fws = FourWheelSteer(self.m, self.d, self.r)   # 四舵轮运动学；停车时轮子抱闸（干摩擦）
        self.fws.set_brake(True)
        self.external_arm = False                           # True：臂由规划路径直接驱动（不走限速轨迹）
        self.pose_i = ORDER.index(pose)
        self.q_cmd = np.array(preset_q(pose))
        self.q_tgt = self.q_cmd.copy()
        self.phi, self.alpha = phi, alpha
        self.payload = 0.0
        self.set_payload(payload)
        self.reset()

    # ---- 机械臂 / 负载 ----
    @property
    def pose(self):
        return ORDER[self.pose_i]

    def set_pose(self, i):
        self.pose_i = i % len(ORDER)
        self.q_tgt = np.array(preset_q(self.pose))

    def set_arm(self, q):
        """直接把臂放到关节角 q（弧度）并作为目标"""
        self.q_cmd = np.array(q, float); self.q_tgt = self.q_cmd.copy(); self.v_cmd = np.zeros(7)
        for adr, qq in zip(self.arm_qadr, self.q_cmd):
            self.d.qpos[adr] = qq
        for a, qq in zip(self.arm_act, self.q_cmd):
            self.d.ctrl[a] = qq
        mujoco.mj_forward(self.m, self.d)

    def set_arm_target(self, q):
        """外部（规划器）给定关节角：直接作为伺服目标"""
        self.q_cmd = np.array(q, float)
        for a, qq in zip(self.arm_act, self.q_cmd):
            self.d.ctrl[a] = qq

    def arm_settled(self, tol=0.004):
        return bool(np.max(np.abs(self.q_tgt - self.q_cmd)) < tol and np.max(np.abs(getattr(self, "v_cmd", np.zeros(7)))) < 0.02)

    def set_payload(self, kg):
        self.payload = float(max(0.0, min(100.0, kg)))
        mass = max(self.payload, 0.01)
        self.m.body_mass[self.pay] = mass
        self.m.body_inertia[self.pay] = 0.4 * mass * 0.2 ** 2
        self.m.geom_rgba[self.pay_geom][3] = 0.85 if self.payload > 0 else 0.0
        # mj_setConst 会把 qpos 复位到 qpos0（车会瞬间回到平地姿态），所以先存后取
        saved = (self.d.qpos.copy(), self.d.qvel.copy(), self.d.ctrl.copy(), self.d.mocap_quat.copy(), self.d.mocap_pos.copy())
        mujoco.mj_setConst(self.m, self.d)
        self.d.qpos[:], self.d.qvel[:], self.d.ctrl[:], self.d.mocap_quat[:], self.d.mocap_pos[:] = saved
        mujoco.mj_forward(self.m, self.d)

    def _drive_arm(self, dt):
        """每个关节：限速 ARM_RATE、限加速度 ARM_ACC 的梯形速度曲线（真实机械臂不会瞬间起停）"""
        if not hasattr(self, "v_cmd"):
            self.v_cmd = np.zeros(7)
        rate = getattr(self, "rate", ARM_RATE)           # 可以是标量，也可以是 7 个关节各自的数组
        acc = getattr(self, "acc", ARM_ACC)
        e = self.q_tgt - self.q_cmd
        v_des = np.sign(e) * np.minimum(rate, np.sqrt(2 * acc * np.abs(e)))
        self.v_cmd += np.clip(v_des - self.v_cmd, -acc * dt, acc * dt)
        self.q_cmd += self.v_cmd * dt
        for a, q in zip(self.arm_act, self.q_cmd):
            self.d.ctrl[a] = q

    # ---- 倾斜台 ----
    def normal(self):
        a, p = math.radians(self.alpha), math.radians(self.phi)
        return np.array([math.sin(a) * math.cos(p), math.sin(a) * math.sin(p), math.cos(a)])

    def quat(self):
        n = self.normal()
        ax = np.cross([0, 0, 1.0], n)
        s = np.linalg.norm(ax)
        q = np.zeros(4)
        if s < 1e-9:
            q[0] = 1
        else:
            ang = math.atan2(s, n[2])
            q[0], q[1:] = math.cos(ang / 2), ax / s * math.sin(ang / 2)
        return q

    def set_tilt(self):
        q = self.quat()
        self.d.mocap_quat[self.tilt_mocap] = q            # 倾斜台是 mocap 体
        self.d.mocap_quat[self.scene_mocap] = q           # 车厢和箱墙跟着倾斜台一起倾斜
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        self.d.mocap_pos[self.scene_mocap] = R.reshape(3, 3) @ np.array([0.0, -self.station, 0.0])   # 机器人横移 = 场景反向平移

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.set_tilt()
        self.d.qpos[:3] = self.normal() * self.r
        self.d.qpos[3:7] = self.quat()
        self.q_cmd = np.array(preset_q(self.pose))
        self.q_tgt = self.q_cmd.copy()
        for adr, q in zip(self.arm_qadr, self.q_cmd):
            self.d.qpos[adr] = q
        for a, q in zip(self.arm_act, self.q_cmd):
            self.d.ctrl[a] = q
        self.d.eq_active[:] = 0 if self.susp else 1
        mujoco.mj_forward(self.m, self.d)

    def set_susp(self, on):
        self.susp = on

    def step(self):
        if self.m.neq:
            self.d.eq_active[0] = 0 if self.susp else 1
        if not self.external_arm:
            self._drive_arm(self.m.opt.timestep)
        self.set_tilt()
        if self.fws.braked:
            self.fws.hold_update()
        mujoco.mj_step(self.m, self.d)

    # ---- 状态 ----
    def tilt_deg(self):
        zb = self.d.xmat[self.base].reshape(3, 3)[:, 2]
        return math.degrees(math.acos(max(-1, min(1, float(zb @ self.normal())))))

    def loads(self):
        f6 = np.zeros(6); out = {n: 0.0 for n in WHEELS}
        gid = {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f"g_{n}"): n for n in WHEELS}
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            for g in (c.geom1, c.geom2):
                if g in gid:
                    mujoco.mj_contactForce(self.m, self.d, i, f6)
                    out[gid[g]] += f6[0]
        w = float(self.m.body_subtreemass[self.base]) * G
        return {n: v / w * 4 for n, v in out.items()}      # 相对静载份额（1.0 = 平均分担）

    # ---- 力矩 ----
    def torques(self):
        """返回 (臂 7 个关节力矩 N·m, 4 个车轮轮端保持扭矩 N·m, 4 个舵轮转向扭矩估算 N·m)，都取绝对值。
        臂：刚体逆动力学（实际运动所需力矩，扣掉人为阻尼和转子惯量）。
        轮：轮子被抱死，扭矩 = 抱死阻尼的反力矩（= 坡上保持车不溜所需的轮端扭矩）。
        转向：原地转舵时轮胎与地面的刮磨扭矩 ≈ μ·N·r̄（接触印痕按径向刚度 1500 N/mm 估算），是估算值。"""
        mujoco.mj_inverse(self.m, self.d)
        tau = self.d.qfrc_inverse[self.arm_dofs] - self.arm_damp * self.d.qvel[self.arm_dofs] - self.arm_arm * self.d.qacc[self.arm_dofs]
        hold = self._roll_torque()
        steer = np.array([self.d.actuator_force[a] for a in self.fws.act_steer])       # 转向伺服保持舵角所需的力矩
        return np.abs(tau), np.abs(hold), steer

    def _roll_torque(self):
        """各轮沿滚动方向的地面切向力 × 半径 = 轮端扭矩（抱闸时就是坡上保持车不溜所需的扭矩）"""
        out = np.zeros(4)
        gid = {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f"g_{n}"): i for i, n in enumerate(WHEELS)}
        f6 = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            for g in (c.geom1, c.geom2):
                if g in gid:
                    mujoco.mj_contactForce(self.m, self.d, i, f6)
                    fw = c.frame.reshape(3, 3).T @ f6[:3]                      # 接触力（世界系）
                    k = gid[g]
                    body = self.wb[k]
                    fwd = self.d.xmat[body].reshape(3, 3)[:, 0]               # 轮子的前进方向
                    out[k] += float(fw @ fwd) * self.r
        return out

    def status(self):
        t = self.tilt_deg()
        L = self.loads()
        front_off = min(L["fl"], L["fr"]) < 0.08
        any_off = min(L.values()) < 0.08
        if t > 3.0:
            return "tipped", t, L
        return ("liftoff" if (any_off if not self.susp else front_off) else "stable"), t, L

    # ---- 绘制辅助几何（追加在 scn 已有几何之后）----
    def draw(self, scn):
        n = self.normal()
        pts = {w: self.d.xpos[b] - n * self.r + n * 0.012 for w, b in zip(WHEELS, self.wb)}
        poly = ([pts["fl"], pts["fr"], (pts["rl"] + pts["rr"]) / 2] if self.susp
                else [pts["fl"], pts["fr"], pts["rr"], pts["rl"]])
        k = scn.ngeom
        yellow = np.array([1.0, 0.85, 0.1, 1.0], dtype=np.float32)
        I = np.eye(3).flatten()
        for i in range(len(poly)):
            g = scn.geoms[k]; k += 1
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), I, yellow)
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012, poly[i], poly[(i + 1) % len(poly)])
        com = self.d.subtree_com[self.base].copy()
        t = float(n @ com) / n[2]
        proj = com - np.array([0, 0, t]) + n * 0.012
        g = scn.geoms[k]; k += 1
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), I,
                            np.array([1, 0.3, 0.3, 0.6], dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006, com, proj)
        for pos, size, col in ((proj, 0.045, yellow), (com, 0.04, np.array([1, 0.2, 0.2, 1], dtype=np.float32))):
            g = scn.geoms[k]; k += 1
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([size, 0, 0]), pos, I, col)
        st, _, _ = self.status()
        colr = {"stable": [0.2, 0.8, 0.3, 1], "liftoff": [1.0, 0.6, 0.1, 1], "tipped": [0.9, 0.15, 0.15, 1]}[st]
        g = scn.geoms[k]; k += 1
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([0.07, 0, 0]),
                            com + np.array([0, 0, 1.4]), I, np.array(colr, dtype=np.float32))
        k = self._bars(scn, k)
        scn.ngeom = k

    def _bars(self, scn, k):
        """车体左侧：7 根橙色柱 = 臂 J1–J7 力矩（1 m = 6000 N·m）；右侧：每个轮子 2 根柱，蓝=轮端扭矩、紫=转向扭矩（1 m = 200 N·m）"""
        tau, hold, steer = self.torques()
        R = self.d.xmat[self.base].reshape(3, 3)
        o = self.d.xpos[self.base]
        up = R[:, 2]

        def bar(x, y, h, col, label):
            nonlocal k
            if k >= scn.maxgeom - 1:
                return
            p0 = o + R @ np.array([x, y, -self.r])
            g = scn.geoms[k]; k += 1
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(), np.array(col, dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.035, p0, p0 + up * max(h, 0.02))
            g.label = label
        for i in range(7):
            bar(-0.6 + 0.2 * i, 1.25, tau[i] / 6000.0, (1.0, 0.55, 0.1, 1.0), f"J{i+1} {tau[i]:.0f}")
        wx = {"fl": 0.45, "fr": 0.45, "rl": -0.45, "rr": -0.45}
        for i, n in enumerate(WHEELS):
            y0 = -1.25 if n in ("fr", "rr") else -1.9
            bar(wx[n] - 0.08, y0, hold[i] / 200.0, (0.25, 0.5, 1.0, 1.0), f"{n} drive {hold[i]:.0f}")
            bar(wx[n] + 0.08, y0, steer[i] / 200.0, (0.7, 0.35, 0.9, 1.0), f"{n} steer {steer[i]:.0f}")
        return k


_PATCH_CACHE = {}


def scrub_torque(N, mu=0.8, width=0.07, r=0.075, k=1.5e6):
    """舵轮原地转向的刮磨扭矩估算：矩形接触印痕（宽 70 mm，长 = 2√(2rδ)，δ = N/k），摩擦力矩 = μ·N·印痕平均半径"""
    if N <= 1.0:
        return 0.0
    delta = N / k
    l = 2 * math.sqrt(2 * r * delta)
    key = round(l, 4)
    if key not in _PATCH_CACHE:
        xs = (np.arange(24) + 0.5) / 24
        X, Y = np.meshgrid((xs - 0.5) * width, (xs - 0.5) * l)
        _PATCH_CACHE[key] = float(np.mean(np.hypot(X, Y)))
    return mu * N * _PATCH_CACHE[key]


class UnloadPlayer:
    """回放 unload_result.json：按列优先顺序取前排烟箱，放到后部理料机构上"""
    def __init__(self, demo, result):
        import unload_plan as U
        self.demo, self.U = demo, U
        self.order = [tuple(b) for b in result["order"]]
        if abs(result.get("j2_shift", 0.0) - demo.cfg.get("j2_shift", 0.0)) > 1e-6:
            print("警告：unload_result.json 是按 J2 前移 %.2f m 规划的，当前配置是 %.2f m，请重新运行 python unload_opt.py" % (result.get("j2_shift", 0.0), demo.cfg.get("j2_shift", 0.0)), flush=True)
        self.picks = result["picks"]
        self.k, self.state, self.i, self.wps, self.t_dwell, self.act = -1, "idle", 0, [], 0.0, None
        self.active = False
        self.done = 0

    def start(self):
        d = self.demo
        for g in d.wall_geom.values():
            d.m.geom_rgba[g][3] = 1.0
        d.m.geom_rgba[d.bed_geom][3] = 0.0
        d.set_payload(0.0)
        d.station = 0.0
        first = self.picks["%d,%d" % self.order[0]]
        d.set_arm(np.array(first["place_hi"]))
        self.k, self.state, self.active, self.done = -1, "next", True, 0
        print("开始卸货：%d 个箱子，列优先，从上到下" % len(self.order), flush=True)

    def stop(self):
        self.active = False
        print("停止卸货（已卸 %d 个）" % self.done, flush=True)

    def _begin_box(self):
        self.k += 1
        if self.k >= len(self.order):
            self.active = False
            print("全部卸完：%d 个" % self.done, flush=True)
            return
        b = self.order[self.k]
        g = self.picks["%d,%d" % b]
        self.box, self.g = b, g
        qpre, qpick, qhi, qpl = (np.array(g[x]) for x in ("pre", "pick", "place_hi", "place"))
        self.wps = [(qpre, 0.0, None), (qpick, self.U.SUCTION_S, "grab"), (qpre, 0.0, None),
                    (qhi, 0.0, None), (qpl, self.U.RELEASE_S, "release"), (qhi, 0.0, None)]
        self.i = -1
        self.state = "station"

    def update(self, dt):
        d, U = self.demo, self.U
        if not self.active:
            return
        if self.state == "next":
            self._begin_box()
        elif self.state == "station":
            tgt = self.g["station"]
            e = tgt - d.station
            if abs(e) < 1e-4:
                d.station = tgt; self.state = "move"; self.i = -1; self._advance()
            else:
                d.station += np.clip(e, -U.STATION_SPEED * dt, U.STATION_SPEED * dt)
        elif self.state == "move":
            if d.arm_settled():
                self.state = "dwell"; self.t_dwell = self.dwell
        elif self.state == "dwell":
            self.t_dwell -= dt
            if self.t_dwell <= 0:
                if self.act == "grab":
                    d.set_payload(50.0)
                    d.m.geom_rgba[d.wall_geom[self.box]][3] = 0.0
                    d.m.geom_rgba[d.bed_geom][3] = 0.0
                elif self.act == "release":
                    d.set_payload(0.0)
                    d.m.geom_rgba[d.bed_geom][3] = 1.0
                    self.done += 1
                    print("  已卸 %d/%d：列%d 层%d（工位 y=%+.1f m）" % (self.done, len(self.order), self.box[0] + 1, self.box[1] + 1, self.g["station"]), flush=True)
                self._advance()

    def _advance(self):
        d, U = self.demo, self.U
        self.i += 1
        if self.i >= len(self.wps):
            self.state = "next"
            return
        q, self.dwell, self.act = self.wps[self.i]
        d.q_tgt = np.array(q, float)
        d.rate = U.VMAX
        d.acc = d.rate / 0.5
        self.state = "move"


def _order_from(plan):
    """列优先，每列从上到下，只含规划成功的箱子"""
    return [[c, l] for c in range(8) for l in range(7, -1, -1) if "%d,%d" % (c, l) in plan["picks"]]


JN = ["S 回转", "L 下臂", "E 上臂", "U 肘部旋转", "R 前臂旋转", "B 腕摆", "T 法兰旋转"]


def interactive(args):
    import mujoco.viewer
    pose = args[0] if len(args) > 0 else "reach"
    payload = float(args[1]) if len(args) > 1 else 50.0
    susp = args[2] if len(args) > 2 else "off"
    phi = float(args[3]) if len(args) > 3 else 0.0
    alpha = float(args[4]) if len(args) > 4 else 0.0
    print("正在载入模型（约几秒）...", flush=True)
    demo = TipDemo(pose, payload, susp, phi, alpha)
    print(f"姿态 {CN[pose]}  负载 {payload:.0f} kg  悬挂 {'开启' if demo.susp else '关闭'}  "
          f"总质量 {demo.m.body_subtreemass[demo.base]:.0f} kg")
    print("手动控制：1-7=选关节(J1..J7)  左右/上下方向键=点动该关节  J=切换步长(1/3/10°)  R=回到预设姿态（自动卸货时先按 U 停止）", flush=True)
    print("按键：  . ,=换姿态   = -=负载   ] [=坡度   ' ;=坡向   N=自动抬升   M=悬挂   Backspace=重置", flush=True)
    state = {"joint": 0, "jog": 3.0, "auto": False, "reset": False, "cycle": False, "t": 0.0, "i": -1, "speed": 1}
    import os
    player = None
    if os.path.exists("data/unload_plan2.json"):
        player = UnloadPlayer2(demo, dict(json.load(open("data/unload_plan2.json", encoding="utf-8")),
                                          order=json.load(open("data/unload_plan2.json", encoding="utf-8"))["order"]))
    elif os.path.exists("archive/unload_result.json"):
        player = UnloadPlayer(demo, json.load(open("archive/unload_result.json", encoding="utf-8")))

    def on_key(key):
        if key == ord("]"):
            demo.alpha = min(60, demo.alpha + 1)
        elif key == ord("["):
            demo.alpha = max(0, demo.alpha - 1)
        elif key == ord("'"):
            demo.phi = (demo.phi + 15) % 360
        elif key == ord(";"):
            demo.phi = (demo.phi - 15) % 360
        elif key == ord("."):
            demo.set_pose(demo.pose_i + 1); print("姿态 →", CN[demo.pose], flush=True)
        elif key == ord(","):
            demo.set_pose(demo.pose_i - 1); print("姿态 →", CN[demo.pose], flush=True)
        elif key == ord("="):
            demo.set_payload(demo.payload + 10); print("负载 →", demo.payload, "kg", flush=True)
        elif key == ord("-"):
            demo.set_payload(demo.payload - 10); print("负载 →", demo.payload, "kg", flush=True)
        elif key == ord("N"):
            state["auto"] = not state["auto"]
        elif key == ord("Y"):
            state["cycle"] = not state["cycle"]; state["t"] = 1e9; state["i"] = -1
            print("装卸货循环", "开始" if state["cycle"] else "停止", flush=True)
        elif key == ord("U") and player is not None:
            if player.active:
                player.stop()
            else:
                state["cycle"] = False; player.start()
        elif key == ord("F"):
            state["speed"] = {1: 4, 4: 10, 10: 1}[state["speed"]]
            print("播放速度 x%d" % state["speed"], flush=True)
        elif key == ord("M"):
            demo.set_susp(not demo.susp)
            print("悬挂", "开启（后桥自由摆动）" if demo.susp else "关闭（后桥锁死）", flush=True)
        elif ord("1") <= key <= ord("7"):
            state["joint"] = key - ord("1"); print("选中关节 J%d（%s） 当前 %.1f°" % (state["joint"] + 1, JN[state["joint"]], math.degrees(demo.q_tgt[state["joint"]])), flush=True)
        elif key in (262, 263, 265, 264, ord("R")):         # 右/上 = 正向，左/下 = 反向；R = 回到当前预设姿态
            if key == ord("R"):
                demo.set_pose(demo.pose_i); print("关节回到预设姿态", flush=True); return
            if player is not None and player.active:
                print("自动卸货进行中，按 U 停止后再手动控制", flush=True); return
            j = state["joint"]
            step = math.radians(state["jog"]) * (1 if key in (262, 265) else -1)
            lo, hi = (math.radians(x) for x in LIM_DEG[j])
            demo.q_tgt[j] = float(np.clip(demo.q_tgt[j] + step, lo, hi))
            print("J%d → %.1f°  (范围 %d~%d°)" % (j + 1, math.degrees(demo.q_tgt[j]), *LIM_DEG[j]), flush=True)
        elif key == ord("J"):
            state["jog"] = {1.0: 3.0, 3.0: 10.0, 10.0: 1.0}[state["jog"]]; print("点动步长 %.0f°" % state["jog"], flush=True)
        elif key == 259:
            state["reset"] = True

    with mujoco.viewer.launch_passive(demo.m, demo.d, key_callback=on_key) as v:
        v.cam.lookat[:] = [0.3, 0, 0.9]; v.cam.distance = 6.0; v.cam.azimuth = 150; v.cam.elevation = -15
        last = 0.0
        while v.is_running():
            t0 = time.time()
            if state["reset"]:
                demo.alpha = 0.0; demo.reset(); state["reset"] = False
            if state["auto"]:
                demo.alpha = min(60.0, demo.alpha + 0.5 * 16 * state["speed"] * demo.m.opt.timestep)
            if state["cycle"]:                              # 装卸货循环：每个姿态停 CYCLE_DWELL 秒
                state["t"] += 16 * demo.m.opt.timestep
                if state["t"] >= CYCLE_DWELL:
                    state["t"] = 0.0; state["i"] += 1
                    k = state["i"] % len(CYCLE)
                    demo.set_pose(ORDER.index(CYCLE[k]))
                    if CYCLE[k] == "lift":
                        demo.set_payload(CYCLE_PAYLOAD)      # 吸取完成，吸起箱子
                    elif k == 5:
                        demo.set_payload(0.0)                # 回到准备姿态，放下箱子
                    print(f"  循环：{CN[CYCLE[k]]}  负载 {demo.payload:.0f} kg", flush=True)
            n_sub = 16 * state["speed"]
            for _ in range(n_sub):
                if player is not None and player.active:
                    player.update(demo.m.opt.timestep)
                demo.step()
            v.user_scn.ngeom = 0
            demo.draw(v.user_scn)
            v.sync()
            if time.time() - last > 0.5:
                last = time.time()
                st, tl, L = demo.status()
                print(f"{CN[demo.pose]} {demo.payload:3.0f}kg  坡度 {demo.alpha:5.1f}° 坡向 {demo.phi:3.0f}°  倾斜 {tl:4.1f}°  "
                      f"载荷 fl/fr/rl/rr = {L['fl']:.2f}/{L['fr']:.2f}/{L['rl']:.2f}/{L['rr']:.2f}  [{st}]", flush=True)
                ta, th, ts = demo.torques()
                print("    臂关节力矩 J1..J7 N·m: " + " ".join(f"{x:5.0f}" for x in ta)
                      + "   轮端扭矩 fl/fr/rl/rr: " + "/".join(f"{x:.0f}" for x in th)
                      + "   转向扭矩(估) fl/fr/rl/rr: " + "/".join(f"{x:.0f}" for x in ts), flush=True)
            time.sleep(max(0, 16 * demo.m.opt.timestep - (time.time() - t0)))


def render_cases():
    """离屏出图：每个工况画 临界坡度前 / 后 两张"""
    from PIL import Image
    from tip_analysis import analytic_tip
    from chassis_sim import cfg_for_pose
    cases = [("reach", 50, "off", 0), ("lift", 50, "off", 0), ("side", 50, "on", 90), ("ready", 0, "on", 90)]
    for pose, pl, susp, phi in cases:
        crit = analytic_tip(cfg_for_pose(dict(CONFIG, suspension=susp), pose, pl), susp, phi)
        for dalpha, tag in ((-2.5, "before"), (+3.0, "after")):
            a = max(0.5, crit + dalpha)
            demo = TipDemo(pose, pl, susp, phi, a)
            for _ in range(3000):
                demo.step()
            r = mujoco.Renderer(demo.m, 720, 1280)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = demo.d.subtree_com[demo.base]; cam.distance = 6.0
            cam.azimuth = 90 if phi == 0 else 180
            cam.elevation = -8
            r.update_scene(demo.d, cam)
            demo.draw(r.scene)
            Image.fromarray(r.render()).save(f"output/images/tip_{pose}_{pl}_{susp}_{phi}_{tag}.png")
            st, tl, _ = demo.status()
            print(f"{pose} {pl}kg 悬挂{susp} 坡向{phi}°  坡度 {a:.1f}°（解析临界 {crit:.1f}°）: {st}，倾斜 {tl:.1f}°")
            r.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--render":
        render_cases()
    else:
        interactive(sys.argv[1:])
