"""机械臂 / 夹具 / 箱子 与 车体、理料机构、箱墙、车厢 的碰撞检测（用 MuJoCo 的碰撞引擎，真实外形）。

  臂的 7 个连杆：GP110B 的 STEP 网格切成若干段，每段取凸包（build_collision_meshes.py 生成）
  夹具：吸盘板、底托板；负载：烟箱 0.30×0.40×0.25
  车体：顶面板以下的车体盒子、转盘外罩、四个角柱、雷达；理料机构：页面里每个零件的包围盒（46 个）
  相机立柱：挂在转盘上随 J1 转，所以只和 E/U/R/B/T 连杆比
  场景：车厢（侧壁、顶、地板）、前排 64 个箱子、后面的整块箱墙

用法：
    from collision import Collider
    col = Collider()
    pairs = col.check(q, station_y=0.0, holding=False, mode="free")   # [] 表示没有碰撞
    mode："free" 全部检测；"pick" 夹具允许贴着箱墙；"place" 负载允许压在辊床上
"""
import json, math, os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # 工程根目录
MARGIN = 0.008         # 臂与环境保持的最小距离 [m]（接触检测提前量）
ARM_Z0 = 0.79          # 臂座高度（世界系）

# 碰撞位：连杆 i 用 1<<i（i=1..7），夹具 G，负载 P
G_BIT, P_BIT, T_BIT = 1 << 8, 1 << 9, 1 << 10          # 吸盘板 / 负载箱子 / 底托
GRIP_BITS = G_BIT | P_BIT | T_BIT
ALL_ARM = sum(1 << i for i in range(1, 8)) | GRIP_BITS
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import box_spec as BS


def _box(name, lo, hi, ct, ca, extra=""):
    c = (np.array(lo) + np.array(hi)) / 2
    h = (np.array(hi) - np.array(lo)) / 2
    return (f'<geom name="{name}" type="box" pos="{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}" size="{max(h[0],1e-3):.4f} {max(h[1],1e-3):.4f} {max(h[2],1e-3):.4f}" '
            f'contype="{ct}" conaffinity="{ca}" {extra}/>')


class Collider:
    def __init__(self):
        import a_model as A
        import arm_abb as AB
        parts, _ = A._load()
        self.AB = AB
        col = json.load(open(os.path.join(HERE, "gp110b", "collision.json"), encoding="utf-8"))
        env = []
        # ---- 车体 ----
        veh_ca = sum(1 << i for i in range(2, 8)) | GRIP_BITS        # 除 J1 连杆外都要躲车体
        env.append(_box("veh_body", (-0.621, -0.514, 0.02), (0.599, 0.514, 0.70), 0, veh_ca))
        for sx in (-1, 1):
            for sy in (-1, 1):
                env.append(_box(f"veh_post_{sx}_{sy}", (sx * 0.56 - 0.05, sy * 0.465 - 0.05, 0.70), (sx * 0.56 + 0.05, sy * 0.465 + 0.05, 0.79), 0, veh_ca))
        env.append(f'<geom name="veh_cover" type="cylinder" pos="0.04 0 0.75" size="0.318 0.05" contype="0" conaffinity="{veh_ca}"/>')
        # ---- 理料机构（页面 xfer 层每个零件的包围盒）----
        self.bed_names = []
        for i, p in enumerate(parts):
            if p["l"] != "xfer" or "示意箱" in p["n"] or any(k in p["n"] for k in ("推板", "后挡板")):      # 原页面 3 工位的推板/挡板：多箱放置要改成平辊床，不参与碰撞
                continue
            import a_model
            v, _ = a_model._arrays(p)
            lo, hi = v.min(0) / 1000, v.max(0) / 1000
            if "快拆" not in p["n"]:                       # 理料机构整体后移（快拆接口留在车上）
                lo = lo + np.array([BS.BED_SHIFT, 0, 0]); hi = hi + np.array([BS.BED_SHIFT, 0, 0])
            name = f"xf_{i}"
            if any(k in p["n"] for k in ("辊床", "护条", "支座", "承载块")):
                self.bed_names.append(name)
            env.append(_box(name, lo, hi, 0, veh_ca))
        env.append(_box("xf_wide", (-1.05 + BS.BED_SHIFT - 0.10, -BS.BED_HALF_W, 0.67), (-0.62, BS.BED_HALF_W, BS.BED_TOP), 0, veh_ca))      # 加宽到可放 3 箱的辊床
        self.bed_names.append("xf_wide")
        for i, p in enumerate(parts):
            if p["l"] == "sensor" and "安装板" not in p["n"]:
                import a_model
                v, _ = a_model._arrays(p)
                env.append(_box(f"sn_{i}", v.min(0) / 1000, v.max(0) / 1000, 0, veh_ca))
        # ---- 场景（随机器人横移）----
        scene = ['<body name="scene" mocap="true">']
        w = ALL_ARM
        scene.append(_box("c_left", (-2.5, 1.5, 0), (3.0, 1.51, 3.0), 0, w))
        scene.append(_box("c_right", (-2.5, -1.51, 0), (3.0, -1.5, 3.0), 0, w))
        scene.append(_box("c_roof", (-2.5, -1.51, 3.0), (3.0, 1.51, 3.01), 0, w))
        scene.append(_box("c_front", (3.0, -1.51, 0), (3.01, 1.51, 3.0), 0, w))
        scene.append(_box("c_back", (-2.51, -1.51, 0), (-2.5, 1.51, 3.0), 0, w))
        edges = BS.COL_EDGES
        self.wall_names = {}
        for c in range(BS.N_COLS):
            for l in range(BS.N_LAYERS):
                nm = f"wb_{c}_{l}"
                self.wall_names[(c, l)] = nm
                scene.append(_box(nm, (BS.WALL_FRONT, edges[c], BS.BOX_H * l), (BS.WALL_FRONT + BS.BOX_D, edges[c + 1], BS.BOX_H * (l + 1)), 0, w))
        scene.append(_box("wall_back", (BS.WALL_FRONT + BS.BOX_D, -1.5, 0), (2.998, 1.5, 2.85), 0, w))
        scene.append("</body>")
        # ---- 臂 ----
        mast = (f'<geom name="mast_pole" type="capsule" fromto="{BS.MAST_XY[0]} {BS.MAST_XY[1]} 0.10 {BS.MAST_XY[0]} {BS.MAST_XY[1]} {BS.MAST_TOP}" size="0.09" '
                f'contype="0" conaffinity="0"/>')            # 相机立柱：单独设置位，见下
        bodies, close = "", 0
        for i, j in enumerate(AB.JOINTS, start=1):
            o = np.array(j["o"]) / 1000.0
            ax = {"x": "1 0 0", "y": "0 1 0", "z": "0 0 1"}[j["a"]]
            bodies += f'<body name="L{i}" pos="{o[0]:.4f} {o[1]:.4f} {o[2]:.4f}"><joint name="J{i}" type="hinge" axis="{ax}" limited="false"/>\n'
            close += 1
            ca = sum(1 << k for k in range(1, 8) if abs(k - i) >= 2)
            for c in col:
                if c["link"] == i:
                    bodies += (f'<geom type="mesh" mesh="{c["file"][:-4]}" contype="{1 << i}" conaffinity="{ca}" margin="{MARGIN}"/>\n')
            if i == 1:                                        # 相机立柱：只和 E、U、R、B、T 比较
                bodies += mast.replace('contype="0" conaffinity="0"', f'contype="{1 << 1}" conaffinity="0"') + "\n"
            if i == 7:                                        # 夹具 + 负载
                gca = sum(1 << k for k in range(1, 6))
                pw, pa = BS.PLATE_HALF_W, BS.PLATE_HALF_A
                gmax = max(BS.BOX_D, BS.BOX_H) / 2
                bodies += ('<body name="ee" pos="0.2 0 0" xyaxes="0 0 -1 0 1 0">\n'
                           f'<geom name="g_plate" type="box" pos="0 0 0.196" size="{pa} {pw} 0.026" contype="{G_BIT}" conaffinity="{gca}" margin="{MARGIN}"/>\n'
                           f'<geom name="g_tray" type="box" pos="{BS.TRAY_X} 0 0.222" size="{BS.TRAY_T / 2} {pw} {BS.TRAY_HALF_L}" contype="{T_BIT}" conaffinity="{gca}" margin="{MARGIN}"/>\n'
                           f'<geom name="g_box" type="box" pos="0 0 0.372" size="{gmax} {BS.BOX_W * BS.MAX_GROUP / 2} {gmax}" contype="{P_BIT}" conaffinity="{gca}" margin="{MARGIN}"/>\n</body>\n')
        bodies += "</body>" * close
        assets = "".join(f'<mesh name="{c["file"][:-4]}" file="{c["file"]}" scale="0.001 0.001 0.001"/>' for c in col)
        xml = f"""<mujoco><option gravity="0 0 0"/><asset>{assets}</asset><worldbody>
  <geom name="floor" type="plane" size="30 30 0.1" contype="0" conaffinity="{ALL_ARM & ~G_BIT}"/>
  {chr(10).join(env)}
  {chr(10).join(scene)}
  <body name="arm" pos="0 0 {ARM_Z0}">{bodies}</body>
</worldbody></mujoco>"""
        files = {c["file"]: open(os.path.join(HERE, "meshes", c["file"]), "rb").read() for c in col}
        self.m = mujoco.MjModel.from_xml_string(xml, files)
        self.d = mujoco.MjData(self.m)
        g = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n)
        self.g_wall = [g(n) for n in self.wall_names.values()] + [g("wall_back")]
        self.g_bed = [g(n) for n in self.bed_names]
        self.g_box = g("g_box")
        self.g_plate, self.g_tray = g("g_plate"), g("g_tray")
        self.g_floor = g("floor")
        self.tray_ct = int(self.m.geom_contype[self.g_tray])
        self.tray_ca = int(self.m.geom_conaffinity[self.g_tray])
        self.qadr = [self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, f"J{i}")] for i in range(1, 8)]
        self.scene_body = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "scene")
        self.mocap = self.m.body_mocapid[self.scene_body]
        self.ca0 = self.m.geom_conaffinity.copy()
        self.ct0 = self.m.geom_contype.copy()
        self.wall_gid = {k: g(v) for k, v in self.wall_names.items()}

    def set_scene(self, removed=()):
        """已取走的箱子不再参与碰撞"""
        self.m.geom_conaffinity[:] = self.ca0
        for k, gid in self.wall_gid.items():
            if k in removed:
                self.m.geom_conaffinity[gid] = 0

    def check(self, q, station_y=0.0, holding=False, mode="free", removed=(), grip=None):
        """grip：夹具状态 dict(n, width=总宽, top=是否顶吸, pi=吸盘板伸出量, rho=底托回缩量)；None = 旧行为（单箱，空手时底托缩回 12 cm）"""
        m, d = self.m, self.d
        if grip is None:
            grip = dict(width=BS.BOX_W, top=False, pi=0.0, rho=0.0 if holding else BS.TRAY_RETRACT)
        top, pi, rho = grip.get("top", False), grip.get("pi", 0.0), grip.get("rho", 0.0)
        self.m.geom_conaffinity[:] = self.ca0
        for k in removed:
            self.m.geom_conaffinity[self.wall_gid[k]] = 0
        if mode == "pick":                                   # 夹具/负载可以贴着箱墙
            for gid in self.g_wall:
                self.m.geom_conaffinity[gid] &= ~GRIP_BITS
            self.m.geom_conaffinity[self.g_floor] &= ~P_BIT              # 底层箱子的箱底贴着车厢地板拔出（滑动接触）
        if mode == "place":                                  # 负载和底托可以压在辊床上
            for gid in self.g_bed:
                self.m.geom_conaffinity[gid] &= ~(P_BIT | T_BIT)
        self.m.geom_contype[self.g_box] = P_BIT if holding else 0
        tray_on = bool(grip.get("tray", False)) and not top                # 已取消底托（纯吸盘）；关掉时 contype 和 conaffinity 都要清零（碰撞是对称的）
        self.m.geom_contype[self.g_tray] = self.tray_ct if tray_on else 0
        self.m.geom_conaffinity[self.g_tray] = self.tray_ca if tray_on else 0
        self.m.geom_pos[self.g_plate, 2] = 0.196 + pi
        self.m.geom_size[self.g_plate] = [BS.PLATE_TOP_HALF if top else BS.PLATE_FRONT_HALF, grip.get("pw", BS.PLATE_HALF_W), 0.026]
        self.m.geom_pos[self.g_tray] = [BS.TRAY_X, 0.0, BS.TRAY_Z_HOLD - rho]
        hw = grip.get("width", BS.BOX_W) / 2
        yo = grip.get("yoff", 0.0)                           # 这组箱子的中心相对吸盘板中心的横向偏移（ee y）
        if top:
            self.m.geom_pos[self.g_box] = [0.0, yo, 0.222 + BS.BOX_H / 2 + pi]
            self.m.geom_size[self.g_box] = [BS.BOX_D / 2, hw, BS.BOX_H / 2]
        else:
            self.m.geom_pos[self.g_box] = [0.0, yo, 0.222 + BS.BOX_D / 2 + pi]
            self.m.geom_size[self.g_box] = [BS.BOX_H / 2, hw, BS.BOX_D / 2]
        d.mocap_pos[self.mocap] = [0, -station_y, 0]
        for adr, v in zip(self.qadr, q):
            d.qpos[adr] = v
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        pairs = []
        for i in range(d.ncon):
            c = d.contact[i]
            pairs.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or f"mesh_body{m.geom_bodyid[c.geom1]}",
                          mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or f"mesh_body{m.geom_bodyid[c.geom2]}",
                          float(c.dist)))
        return pairs

    def ok(self, q, **kw):
        return len(self.check(q, **kw)) == 0


if __name__ == "__main__":
    import time
    col = Collider()
    import arm_kin as K
    print("几何体数", col.m.ngeom)
    for name in ("zero", "pick", "rear", "stow", "ready"):
        q = K.preset_theta(name)
        t = time.time()
        pr = col.check(q, mode="pick" if name == "pick" else "free", holding=(name == "rear"))
        print(f"{name:6s}", "无碰撞" if not pr else f"碰撞 {len(pr)} 处：{pr[:4]}", "(%.1f ms)" % ((time.time() - t) * 1000))
