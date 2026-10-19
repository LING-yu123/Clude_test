"""按 unload_plan2.json 回放 / 执行卸货：机器人先用四舵轮开到工位，臂沿规划好的无碰撞路径取箱、转到车尾放箱。

关节运动：每段（路点到路点）用最小冲击(min-jerk)曲线、所有关节同步，段时长由 GP110B 各轴最大速度和加速度限制决定。
车体：舵轮蟹行/斜行到工位 y，到位后抱闸。
"""
import math
import numpy as np
import mujoco
import arm_abb

VMAX = np.radians(arm_abb.MAX_SPEED_DEG)           # 各轴最大速度 [rad/s]（安川样本）
ACC_TIME = 0.8                                    # 加速到最大速度的时间 [s]（tools/cycle_opt.py 扫描：0.8 s + 90% 速度，最差坡向倾斜 <2°、裕度 >=7 cm；再快侧坡会倾斜 8°）
AMAX = VMAX / ACC_TIME
DRIVE_V = 0.30                                     # 车体横移速度 [m/s]
DRIVE_A = 0.30                                     # 车体加速度 [m/s²]
SUCTION_S, RELEASE_S = 0.6, 0.5
SERVO_LAG = (6e4 + 3000.0) / 1e7                   # 臂关节位置伺服的滞后时间常数 [s]


def seg_duration(q0, q1, speed_scale=1.0):
    d = np.abs(np.asarray(q1) - np.asarray(q0))
    t = 0.0
    for di, vi, ai in zip(d, VMAX * speed_scale, AMAX * speed_scale ** 2):
        if di < 1e-6:
            continue
        t = max(t, 1.875 * di / vi, math.sqrt(5.7735 * di / ai))
    return max(t, 0.05) if d.max() > 1e-6 else 0.0


def minjerk(tau):
    return 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5


class UnloadPlayer:
    def __init__(self, demo, plan, drive=True):
        self.demo, self.plan, self.drive_enabled = demo, plan, drive
        self.order = [tuple(b) for b in plan["order"]]
        self.picks = plan["picks"]
        self.active = False
        self.done = 0
        self.speed_scale = 0.9                      # 整体速度比例（相对 GP110B 最大速度）
        self.tasks, self.cur = [], None
        self.grabbed, self.paths_since_grab = False, 0     # 吸着箱子后的第 1 段路径 = 沿箱面拔出，允许贴着箱墙（与规划的 pick 模式一致）
        self.fws = demo.fws

    # ---- 任务构造 ----
    def _tasks_for(self, b):
        g = self.picks["%d,%d" % b]
        L = g["legs"]
        q_hi, q_pl = np.array(self.plan["place_hi"]), np.array(self.plan["place"])
        T = [("drive", g["station"])]
        T.append(("path", [np.array(q) for q in L["approach"]]))
        T.append(("path", [np.array(q) for q in L["in"]]))
        T.append(("wait", SUCTION_S, "grab"))
        T.append(("path", [np.array(q) for q in L["out"]]))
        T.append(("path", [np.array(q) for q in L["carry"]]))
        st_key = str(g["station"])
        down = self.plan["place_down"][st_key] if isinstance(self.plan.get("place_down"), dict) else self.plan.get("place_down", [q_hi, q_pl])
        up = self.plan["place_up"][st_key] if isinstance(self.plan.get("place_up"), dict) else self.plan.get("place_up", [q_pl, q_hi])
        T.append(("path", [np.array(q) for q in down]))
        T.append(("wait", RELEASE_S, "release"))
        T.append(("path", [np.array(q) for q in up]))
        return T

    def start(self):
        d = self.demo
        for g in d.wall_geom.values():
            d.m.geom_rgba[g][3] = 1.0
        d.m.geom_rgba[d.bed_geom][3] = 0.0
        d.set_payload(0.0)
        d.external_arm = True
        q0 = np.array(self.plan["place_hi"])
        d.set_arm(q0)
        self.q_now = q0.copy()
        self.k, self.active, self.done, self.cur, self.tasks = -1, True, 0, None, []
        self.t_wait = 0.0
        print("开始卸货：%d 个箱子（列优先，从上到下）；车体 %s" % (len(self.order), "用四舵轮开到各工位" if self.drive_enabled else "固定不动"), flush=True)

    def stop(self):
        self.active = False
        self.demo.external_arm = False
        self.fws.stop_and_brake()
        print("停止卸货（已卸 %d 个）" % self.done, flush=True)

    # ---- 执行 ----
    def _next_task(self):
        if not self.tasks:
            self.k += 1
            if self.k >= len(self.order):
                self.active = False
                self.demo.external_arm = False
                print("全部卸完：%d 个" % self.done, flush=True)
                return False
            self.box = self.order[self.k]
            self.tasks = self._tasks_for(self.box)
        self.cur = self.tasks.pop(0)
        kind = self.cur[0]
        if kind == "path":
            if self.grabbed:
                self.paths_since_grab += 1
            self.wps = list(self.cur[1])
            self.seg_i = 0
            self._start_seg()
        elif kind == "wait":
            self.t_wait = self.cur[1]
        return True

    def _start_seg(self):
        while self.seg_i < len(self.wps) - 1:
            a, b = self.wps[self.seg_i], self.wps[self.seg_i + 1]
            self.T = seg_duration(a, b, self.speed_scale)
            if self.T > 0:
                self.t_seg, self.qa, self.qb = 0.0, a, b
                return True
            self.seg_i += 1
        self.cur = None
        return False

    def update(self, dt):
        d = self.demo
        if not self.active:
            return
        if self.cur is None and not self._next_task():
            return
        kind = self.cur[0]
        if kind == "drive":
            tgt = self.cur[1]
            if not self.drive_enabled:
                self.cur = None; return
            done = self._drive(tgt, dt)
            if done:
                self.cur = None
        elif kind == "path":
            self.t_seg += dt
            tau = min(1.0, self.t_seg / self.T)
            q = self.qa + (self.qb - self.qa) * minjerk(tau)
            qdot = (self.qb - self.qa) * (30 * tau ** 2 - 60 * tau ** 3 + 30 * tau ** 4) / self.T
            d.set_arm_target(q + SERVO_LAG * qdot)          # 速度前馈：抵消伺服的跟踪滞后（(kv+关节阻尼)/kp）
            self.q_now = q
            if tau >= 1.0:
                self.seg_i += 1
                if not self._start_seg():
                    self.cur = None
        elif kind == "wait":
            self.t_wait -= dt
            if self.t_wait <= 0:
                act = self.cur[2]
                if act == "grab":
                    self.grabbed, self.paths_since_grab = True, 0
                    d.set_payload(50.0)
                    d.m.geom_rgba[d.wall_geom[self.box]][3] = 0.0
                    d.m.geom_rgba[d.bed_geom][3] = 0.0
                elif act == "release":
                    self.grabbed = False
                    d.set_payload(0.0)
                    d.m.geom_rgba[d.bed_geom][3] = 1.0
                    self.done += 1
                    print("  已卸 %d/%d：列%d 层%d（工位 y=%+.1f m）" % (self.done, len(self.order), self.box[0] + 1, self.box[1] + 1,
                                                                  self.picks["%d,%d" % self.box]["station"]), flush=True)
                self.cur = None

    # ---- 车体横移：四舵轮 ----
    def _drive(self, y_tgt, dt):
        d, f = self.demo, self.fws
        R = d.d.xmat[d.base].reshape(3, 3)
        pos = d.d.xpos[d.base]
        y_err = y_tgt - self._plane_y(pos)
        vel = self._plane_vy()
        if abs(y_err) < 0.004 and abs(vel) < 0.01:
            f.stop_and_brake()
            return True
        # 梯形速度：到位前按 sqrt(2 a e) 减速
        v_des = math.copysign(min(DRIVE_V, math.sqrt(2 * DRIVE_A * abs(y_err))), y_err) if abs(y_err) > 0.002 else 0.0
        self._vy_cmd = getattr(self, "_vy_cmd", 0.0)
        self._vy_cmd += float(np.clip(v_des - self._vy_cmd, -DRIVE_A * dt, DRIVE_A * dt))
        f.command(0.0, self._vy_cmd, 0.0, dt)
        return False

    def _plane_y(self, pos):
        n = self.demo.normal()
        # 沿坡面的横向坐标：把世界位置投到坡面，取与下坡方向(车头x)正交的分量
        e_y = np.cross(n, [1.0, 0.0, 0.0]); e_y /= max(np.linalg.norm(e_y), 1e-9)
        return float(pos @ e_y)

    def _plane_vy(self):
        n = self.demo.normal()
        e_y = np.cross(n, [1.0, 0.0, 0.0]); e_y /= max(np.linalg.norm(e_y), 1e-9)
        return float(self.demo.d.qvel[:3] @ e_y)
