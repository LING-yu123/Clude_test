"""成组卸货播放器（unload_plan3.json）：
  前吸（上半部各层）：吸盘板伸出贴箱 → 吸合 → 吸盘板缩回、底托同步前伸托住箱底 → 拔出 → 转到车尾（夹具不翻转）→ 下放到辊床
  → 吸盘板前推、底托同步回缩，箱子留在辊床上 → 释放 → 抬起；
  顶吸（下半部各层）：吸盘板下伸贴箱顶 → 吸合 → 缩回提起 → 转到车尾 → 下放 → 释放（没有底托）。
臂的运动是连续轨迹（路点圆角过渡 + 速度/加速度限幅的时间参数化），拐角不停；只有夹具动作、吸合/释放和车体横移时臂才停。
一次最多 3 箱（同层相邻），质量 17 kg x 箱数。
"""
import math
import numpy as np
import arm_abb
import box_spec as BS
import traj as TJ
from unload_player import UnloadPlayer as _Old

VMAX = np.radians(arm_abb.MAX_SPEED_DEG)
SERVO_LAG = (6e4 + 3000.0) / 1e7
GRIP_SPEED = 0.40                       # 夹具滑轴速度 [m/s]
SUCTION_S, RELEASE_S = 0.3, 0.3         # 吸合/释放时间 [s]
ACC_TIME = 0.8                          # 加速到最大速度的时间 [s]（0.8 s + 90% 速度：10° 各坡向倾斜 <=1.8°、裕度 >=9.7 cm；0.6 s + 100% 可到 8 s/箱，但 270° 侧坡倾斜 6°、吸盘与侧壁最深贴进 86 mm，不采用）
SPEED_SCALE = 0.9


class UnloadPlayer3(_Old):
    def __init__(self, demo, plan, drive=True, acc_time=None, speed_scale=None, max_parts=None):
        self.demo, self.plan, self.drive_enabled = demo, plan, drive
        self.parts = plan["parts"][:max_parts] if max_parts else plan["parts"]
        self.fws = demo.fws
        self.acc_time = acc_time or ACC_TIME
        self.speed_scale = speed_scale or SPEED_SCALE
        self.vmax = VMAX * self.speed_scale
        self.amax = VMAX / self.acc_time * self.speed_scale ** 2
        self.active = False
        self.done = 0
        self.items = []
        self.cur = None
        self.log = []
        self.station_now = 0.0
        self.simple = bool(plan.get("simple", False))
        self.order = [(c, p["layer"]) for p in self.parts for c in p["cols"]]

    # ---------------------------------------------------------------- 构造任务序列
    def _traj(self, chain, slow=False):
        k = 0.5 if slow else 1.0                                        # 贴箱/拔出的最后一段放慢
        return TJ.Trajectory(np.array(chain["q"]), self.vmax * k, self.amax * k * k, chain["stops"])

    def _part_items(self, k, p):
        items = []
        S, mode = BS.STROKE, p["mode"]
        front = mode == "front"
        rho_empty = BS.TRAY_RETRACT if front else 0.0
        tag = (k, p)
        if "chain_u" in p:
            items.append(("traj", self._traj(p["chain_u"]), "up"))
            items.append(("drive", p["station"]))
        elif k == 0 or abs(p["station"] - self.station_now) > 1e-6:
            items.append(("drive", p["station"]))
        items.append(("traj", self._traj(p["chain_a"]), "approach"))
        if self.simple:                                                   # 纯吸盘：臂慢速直线推进贴箱，没有吸盘/底托的相对运动
            if "chain_in" in p:
                items.append(("traj", self._traj(p["chain_in"], slow=True), "in"))
        else:
            items.append(("grip", S, rho_empty, S / GRIP_SPEED))              # 吸盘板伸出贴箱
        items.append(("wait", SUCTION_S, "grab", p))
        if not self.simple:
            items.append(("grip", 0.0, 0.0 if front else 0.0, (BS.TRAY_RETRACT if front else S) / GRIP_SPEED))   # 吸盘板缩回，底托同步前伸
        items.append(("traj", self._traj(p["chain_b"]), "carry"))
        if self.simple:
            items.append(("wait", RELEASE_S, "release", p))
        elif front:
            items.append(("grip", BS.PLACE_PUSH, BS.TRAY_PLACE_RETRACT, BS.PLACE_PUSH / GRIP_SPEED))   # 吸盘板前推、底托同步回缩
            items.append(("wait", RELEASE_S, "release", p))
            items.append(("grip", 0.0, BS.TRAY_PLACE_RETRACT, BS.PLACE_PUSH / GRIP_SPEED))
        else:
            items.append(("wait", RELEASE_S, "release", p))
        return items

    def start(self):
        d = self.demo
        for g in d.wall_geom.values():
            d.m.geom_rgba[g][3] = 1.0
        for g in d.bed_geoms:
            d.m.geom_rgba[g][3] = 0.0
        d.set_load([])
        d.external_arm = True
        q0 = np.array(self.plan["start"])
        d.set_arm(q0)
        d.d.qpos[d.grip_qadr[0]] = 0.0
        d.d.qpos[d.grip_qadr[1]] = -BS.TRAY_PLACE_RETRACT
        d.set_grip(0.0, BS.TRAY_PLACE_RETRACT)
        self.station_now = 0.0
        self.k, self.done, self.items, self.cur = -1, 0, [], None
        self.active = True
        self.n_traj = 0
        print("开始卸货（成组）：%d 组 / %d 箱；车体 %s" % (len(self.parts), len(self.order),
                                                      "用四舵轮开到各工位" if self.drive_enabled else "固定不动"), flush=True)

    def stop(self):
        self.active = False
        self.demo.external_arm = False
        self.fws.stop_and_brake()
        print("停止卸货（已卸 %d 箱）" % self.done, flush=True)

    # ---------------------------------------------------------------- 执行
    def _next(self):
        if not self.items:
            self.k += 1
            if self.k >= len(self.parts):
                if self.k == len(self.parts) and self.plan.get("tail") and not getattr(self, "_tail_done", False):
                    self._tail_done = True
                    self.items = [("traj", self._traj(self.plan["tail"]), "tail")]
                else:
                    self.active = False
                    self.demo.external_arm = False
                    print("全部卸完：%d 箱" % self.done, flush=True)
                    return False
            else:
                self.items = self._part_items(self.k, self.parts[self.k])
        self.cur = self.items.pop(0)
        kind = self.cur[0]
        self.t_item = 0.0
        if kind == "grip":
            self.g0 = tuple(self.demo.grip_state())
        return True

    def update(self, dt):
        d = self.demo
        if not self.active:
            return
        if self.cur is None and not self._next():
            return
        kind = self.cur[0]
        if kind == "traj":
            tr = self.cur[1]
            self.t_item += dt
            q, qd = tr.sample(self.t_item)
            if q is not None:
                d.set_arm_target(q + SERVO_LAG * qd)
            if self.t_item >= tr.T:
                self.cur = None
        elif kind == "drive":
            if not self.drive_enabled:
                self.station_now = self.cur[1]; self.cur = None; return
            if self._drive(self.cur[1], dt):
                self.station_now = self.cur[1]
                self.cur = None
        elif kind == "grip_set":
            d.set_grip(d.grip_state()[0], self.cur[2])
            self.cur = None
        elif kind == "grip":
            _, pi, rho, T = self.cur
            self.t_item += dt
            a = min(1.0, self.t_item / max(T, 1e-6))
            a = a * a * (3 - 2 * a)
            d.set_grip(self.g0[0] + (pi - self.g0[0]) * a, self.g0[1] + (rho - self.g0[1]) * a)
            if self.t_item >= T:
                self.cur = None
        elif kind == "wait":
            self.t_item += dt
            if self.t_item >= self.cur[1]:
                act, p = self.cur[2], self.cur[3]
                if act == "grab":
                    d.set_load(p["widths"], top=(p["mode"] == "top"), yoff=p["yoff"])
                    for c in p["cols"]:
                        d.m.geom_rgba[d.wall_geom[(c, p["layer"])]][3] = 0.0
                elif act == "release":
                    d.set_load([])
                    self._show_on_bed(p)
                    self.done += len(p["cols"])
                    print("  已卸 %d/%d：层%d 列%s（%s 吸，站位 y=%+.1f m）" % (self.done, len(self.order), p["layer"] + 1,
                                                                    [c + 1 for c in p["cols"]], "前" if p["mode"] == "front" else "顶", p["station"]), flush=True)
                self.cur = None

    def _show_on_bed(self, p):
        d = self.demo
        n = len(p["widths"])
        ys = []
        y = -sum(p["widths"]) / 2 + p["yoff"]
        for w in p["widths"]:
            ys.append(y + w / 2); y += w
        sgn = -1.0 if p["mode"] == "front" else 1.0                    # 朝后(绕 z 转 180°)时 ee y 与世界 y 反向
        r = d.r
        for j, g in enumerate(d.bed_geoms):
            if j < n:
                d.m.geom_pos[g] = [BS.PLACE_X, sgn * ys[j], BS.BED_TOP + BS.BOX_H / 2 - r]
                d.m.geom_size[g] = [BS.BOX_D / 2, p["widths"][j] / 2 - 0.003, BS.BOX_H / 2]
                d.m.geom_rgba[g][3] = 1.0
            else:
                d.m.geom_rgba[g][3] = 0.0
