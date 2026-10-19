"""从 chassis_A_3d.html 里提取真实零件网格，生成 MuJoCo 外观（纯显示，不参与碰撞和质量）。

用法（由 chassis_sim.build_xml 调用）：
    from a_model import get_visuals
    v = get_visuals("ready")     # 姿态名见 POSES
首次调用会把网格导出到 meshes/*.stl（毫米->米靠 mesh scale），之后直接复用。
"""
import base64, json, math, os, re
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # 工程根目录
HTML = os.path.join(HERE, "data", "chassis_A_3d.html")                       # 工程内的副本（云端/移动工程时也能用）
if not os.path.exists(HTML):
    HTML = os.path.join(HERE, "..", "3Dmodel", "chassis_A_3d.html")
MESH_DIR = os.path.join(HERE, "meshes")
Z0 = 75.0                      # base 坐标原点 = 轮轴高度 75 mm（页面里地面 z=0）
SKIP_LAYERS = {"scene", "fov", "air", "cable", "conv"}   # 车厢场景、视场锥、气流、线缆、外部输送线

# 页面里的预设姿态（J1..J7 角度, 吸盘标志, 吸盘/底托行程%）
POSES = {"zero": "零位", "stow": "收纳", "ready": "准备", "reach": "前伸",
         "lift": "举高", "side": "侧向", "pick": "吸取", "rear": "朝后扫描"}

_cache = {}


def _load():
    if "d" in _cache:
        return _cache["d"]
    s = open(HTML, encoding="utf-8").read()
    dec = json.JSONDecoder()
    i = s.index("window.__PARTS__=") + len("window.__PARTS__=")
    parts, end = dec.raw_decode(s[i:])
    r = s[i + end:]
    k = r.index("window.__ROBOT__=") + len("window.__ROBOT__=")
    rob, _ = dec.raw_decode(r[k:])
    _cache["d"] = (parts, rob)
    return parts, rob


def _arrays(p):
    v = np.frombuffer(base64.b64decode(p["v"]), np.float32).reshape(-1, 3)
    f = np.frombuffer(base64.b64decode(p["f"]), np.uint32).reshape(-1, 3)
    return v, f


def _write_stl(path, v, f):
    tri = v[f]                                           # (M,3,3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = np.divide(n, ln, out=np.zeros_like(n), where=ln > 0)
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as fh:
        fh.write(b"\0" * 80)
        fh.write(np.uint32(len(f)).tobytes())
        fh.write(rec.tobytes())


def _area_centroid(v, f):
    tri = v[f]
    a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    c = tri.mean(1)
    return a.sum(), (a[:, None] * c).sum(0) / max(a.sum(), 1e-9)


def _rgba(c, a=1.0):
    return f"{c[0]/255:.3f} {c[1]/255:.3f} {c[2]/255:.3f} {a:.2f}"


def _rot(axis, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


PIN_Z = 262.0                  # 后桥摆动销轴离地高度 [mm]（页面里 pivot 位置）


# ---- 精简规则（detail="lite"）：只留对外形、受力、运动有用的零件 ----
LITE_DROP_LAYERS = {"ctrl", "power", "lock"}   # 电气舱内件、泵/电池、锁止小件：都被外壳挡住，不影响质心之外的任何东西
LITE_DROP_WORDS = ("螺栓", "平键", "油嘴", "隔套", "压条", "拉杆", "油封", "压盖", "联轴器", "编码器",
                   "读数头", "码带", "光栅", "丝杠", "限位", "螺母", "导向套", "谐波", "小齿轮", "线缆护管",
                   "伺服电机", "驱动座", "驱动小齿轮", "交叉滚子轴承", "电机安装立板", "电机支座",
                   "舵轮法兰", "回转支承与减速器", "立式驱动电机", "示意箱")
LITE_MIN_DIAG = 0.08                           # 对角线小于 80 mm 的零件（紧固件、挡片等）直接去掉


def _keep(p, v, detail):
    if detail == "full":
        return True
    if p.get("l") in LITE_DROP_LAYERS:
        return False
    if any(w in p.get("n", "") for w in LITE_DROP_WORDS):
        return False
    return float(np.linalg.norm(v.max(0) - v.min(0))) / 1000 >= LITE_MIN_DIAG


_J2_X0 = None


def set_j2_shift(shift_m):
    """把 J2（肩部俯仰）轴在 J1 转盘上向前移 shift_m（米）。直接改关节链，FK/IK/MuJoCo 模型都会跟着变。"""
    global _J2_X0
    _, rob = _load()
    j = rob["joints"][1]
    if _J2_X0 is None:
        _J2_X0 = float(j["o"][0])
    j["o"][0] = _J2_X0 + float(shift_m) * 1000.0
    import arm_abb
    arm_abb.JOINTS[1]["o"][0] = float(shift_m) * 1000.0       # ABB 臂：J2 本来在 J1 轴上，前移量直接加在 x 上


ARM_TYPE = "abb"      # "abb" = 按 ABB 尺寸图的 6 轴臂；"page" = chassis_A_3d.html 里的七轴臂


def get_visuals(pose="ready", swing=False, detail="lite", pin_x=-0.30, live=False, arm_mass=800.0, j2_shift=0.0, arm_type=None):
    parts, rob = _load()
    set_j2_shift(j2_shift)
    os.makedirs(MESH_DIR, exist_ok=True)
    assets, names = [], {}

    def mesh(key, p):
        """登记一个 mesh 资源，返回名字（首次写 STL）"""
        if key in names:
            return names[key]
        fn = os.path.join(MESH_DIR, f"{key}.stl")
        v, f = _arrays(p)
        if not os.path.exists(fn):
            _write_stl(fn, v, f)
        nm = f"m_{key}"
        assets.append(f'<mesh name="{nm}" file="{key}.stl" scale="0.001 0.001 0.001"/>')
        names[key] = nm
        return nm

    GEOM = 'contype="0" conaffinity="0" mass="0" group="1"'
    # ---- 车轮：识别 4 个轮的中心 ----
    wheel_c = {}
    for p in parts:
        if p["l"] == "wheel" and p["n"].startswith("车轮"):
            v, _ = _arrays(p)
            c = (v.min(0) + v.max(0)) / 2
            wheel_c[("f" if c[0] > 0 else "r") + ("l" if c[1] > 0 else "r")] = c
    wheel_xml = {n: [] for n in wheel_c}
    steer_xml = {n: [] for n in wheel_c}      # 随转向转动的件（车轮叉架）
    base_xml, rear_xml = [], []
    stats = {"total": 0, "kept": 0}
    for i, p in enumerate(parts):
        if p["l"] in SKIP_LAYERS:
            continue
        v, f = _arrays(p)
        if len(f) < 4:
            continue
        a = p.get("a", 1.0)
        is_wheel_part = p["l"] == "wheel" and (p["n"].startswith(("车轮", "轮毂", "轮轴")))
        stats["total"] += 1
        if not is_wheel_part and not _keep(p, v, detail):
            continue
        stats["kept"] += 1
        if is_wheel_part:
            c = (v.min(0) + v.max(0)) / 2
            wn = ("f" if c[0] > 0 else "r") + ("l" if c[1] > 0 else "r")
            wc = wheel_c[wn]
            nm = mesh(f"p{i}", p)
            wheel_xml[wn].append(
                f'<geom {GEOM} type="mesh" mesh="{nm}" pos="{-wc[0]/1000:.4f} {-wc[1]/1000:.4f} {-Z0/1000:.4f}" '
                f'rgba="{_rgba(p["c"], a)}"/>')
        elif p["l"] == "wheel" and p["n"].startswith("车轮叉架"):
            c = (v.min(0) + v.max(0)) / 2
            wn = ("f" if c[0] > 0 else "r") + ("l" if c[1] > 0 else "r")
            wc = wheel_c[wn]
            nm = mesh(f"p{i}", p)
            steer_xml[wn].append(f'<geom {GEOM} type="mesh" mesh="{nm}" pos="{-wc[0]/1000:.4f} {-wc[1]/1000:.4f} {-Z0/1000:.4f}" '
                                 f'rgba="{_rgba(p["c"], a)}"/>')
        elif swing and p.get("g") == "rear":      # 后桥（摆动桥）上的零件：随摆动桥一起转
            nm = mesh(f"p{i}", p)
            rear_xml.append(f'<geom {GEOM} type="mesh" mesh="{nm}" pos="{-pin_x:.4f} 0 {-PIN_Z/1000:.4f}" rgba="{_rgba(p["c"], a)}"/>')
        else:
            nm = mesh(f"p{i}", p)
            base_xml.append(f'<geom {GEOM} type="mesh" mesh="{nm}" pos="0 0 {-Z0/1000:.4f}" rgba="{_rgba(p["c"], a)}"/>')

    # ---- 机械臂（按预设姿态摆成固定形状）----
    pre = rob["pre"][POSES.get(pose, pose)]
    q, qs = pre[:7], pre[8]
    off = rob["off"]

    def link_geoms(lst, tag):
        out, cen = [], []
        for j, p in enumerate(lst):
            v, f = _arrays(p)
            if len(f) < 4:
                continue
            area, cen_ = _area_centroid(v, f)
            cen.append((area, cen_))                 # 质心计算用全部零件，不受精简影响
            stats["total"] += 1
            if not _keep(p, v, detail):
                continue
            stats["kept"] += 1
            nm = mesh(f"{tag}_{j}", p)
            out.append(f'<geom {GEOM} type="mesh" mesh="{nm}" rgba="{_rgba(p["c"])}"/>')
        return "\n".join(out), cen

    if (arm_type or ARM_TYPE) == "abb":
        from a_model_abb import abb_visuals
        return abb_visuals(locals())

    # 关节旋转累乘，用来得到臂质心 / 末端在 base 系的位置（供 chassis_sim 调整质心）
    T = np.eye(4)
    T[:3, 3] = np.array(rob["base"]) - [0, 0, Z0]
    base_arm_T = T.copy()
    body_stack = []
    arm_cen = []                                     # (面积权重, 世界系质心[mm])
    # live 模式：机械臂做成真实的 7 个转动关节 + 位置执行器，质量按各连杆网格面积分摊
    link_mass, link_cen = {}, {}
    if live:
        def stat_of(lst, shift=None):
            A, C = 0.0, np.zeros(3)
            for p in lst:
                v, f = _arrays(p)
                if len(f) < 4:
                    continue
                a_, c_ = _area_centroid(v, f)
                A += a_; C += a_ * (c_ + (shift if shift is not None else 0))
            return A, C / max(A, 1e-9)
        MAST_KG, GRIP_KG = 20.0, 30.0                    # 相机立柱、夹具（吸盘板+底托）的估算质量
        st = {i: stat_of(rob["links"].get(str(i), [])) for i in range(1, 8)}
        tot_a = sum(a for a, _ in st.values())
        body_kg = max(arm_mass - MAST_KG - GRIP_KG, 0.0)
        extra = {1: [], 7: []}
        mA, mC = stat_of(list(rob["mastFixed"]))
        rA, rC = stat_of(list(rob["mastRot"]), np.array(rob["mastO"]))
        extra[1].append((MAST_KG, (mA * mC + rA * rC) / max(mA + rA, 1e-9)))
        gA, gC = stat_of(list(rob["slide"]) + list(rob["tray"]))
        extra[7].append((GRIP_KG, gC))
        for i, (a, c) in st.items():
            ms = [(body_kg * a / tot_a, c)] + extra.get(i, [])
            m_tot = sum(m for m, _ in ms)
            link_mass[i] = m_tot
            link_cen[i] = sum(m * cc for m, cc in ms) / max(m_tot, 1e-9) / 1000.0
    act_xml, arm_q0, arm_names = [], [], []
    xml = f'<body name="arm_base" pos="0 0 {(rob["base"][2]-Z0)/1000:.4f}">\n'
    g, cen = link_geoms(rob["links"]["0"], "L0")
    xml += g + "\n"
    close = 1
    Tcur = T.copy()
    ee_T = None
    mast_T = None
    for ji, j in enumerate(rob["joints"]):
        ang = j["s"] * (q[ji] + off[ji])
        pos = np.array(j["o"], float)
        R = _rot(j["a"], ang)
        Tl = np.eye(4); Tl[:3, :3] = R; Tl[:3, 3] = pos
        Tcur = Tcur @ Tl
        ax = {"x": "1 0 0", "y": "0 1 0", "z": "0 0 1"}[j["a"]]
        if live:
            th = q[ji] + off[ji]
            lo, hi = rob["jn"][ji][1] + off[ji], rob["jn"][ji][2] + off[ji]
            m_i, c_i = link_mass[ji + 1], link_cen[ji + 1]
            jaxis = " ".join(str(j["s"] * float(t)) for t in ax.split())
            xml += (f'<body name="arm_j{ji+1}" pos="{pos[0]/1000:.4f} {pos[1]/1000:.4f} {pos[2]/1000:.4f}">\n'
                    f'<joint name="arm_j{ji+1}" type="hinge" axis="{jaxis}" range="{lo-5:.1f} {hi+5:.1f}" limited="true" '
                    f'damping="3000" armature="20"/>\n'
                    f'<inertial pos="{c_i[0]:.4f} {c_i[1]:.4f} {c_i[2]:.4f}" mass="{m_i:.2f}" '
                    f'diaginertia="{m_i*0.12:.2f} {m_i*0.12:.2f} {m_i*0.12:.2f}"/>\n')
            act_xml.append(f'<position name="a_j{ji+1}" joint="arm_j{ji+1}" kp="1e7" kv="6e4"/>')
            arm_q0.append(math.radians(th)); arm_names.append(f"a_j{ji+1}")
        else:
            xml += f'<body name="arm_j{ji+1}" pos="{pos[0]/1000:.4f} {pos[1]/1000:.4f} {pos[2]/1000:.4f}" axisangle="{ax} {ang:.3f}">\n'
        close += 1
        g, cen = link_geoms(rob["links"].get(str(ji + 1), []), f"L{ji+1}")
        xml += g + "\n"
        for area, c in cen:
            arm_cen.append((area, (Tcur @ np.append(c, 1))[:3]))
        if ji == 0 and j2_shift > 1e-6:              # J2 前移后，转盘上加一个肩座，把 J2 轴座接到 J1 转盘上
            xml += (f'<geom {GEOM} type="box" pos="{j2_shift/2:.4f} 0 0.120" '
                    f'size="{j2_shift/2 + 0.09:.4f} 0.13 0.11" rgba="0.80 0.82 0.85 1"/>\n')
        if ji == 0:                                  # 相机立柱：固定部分 + 自转部分（随 J1 转）
            g, _ = link_geoms(rob["mastFixed"], "MF")
            xml += g + "\n"
            mo = rob["mastO"]
            xml += (f'<body name="mast" pos="{mo[0]/1000:.4f} {mo[1]/1000:.4f} {mo[2]/1000:.4f}" '
                    f'axisangle="0 0 1 180">\n')
            g, _ = link_geoms(rob["mastRot"], "MR")
            xml += g + "\n</body>\n"
        if ji == 6:                                  # 末端：吸盘板 / 底托板沿 z 反向滑动
            ee_T = Tcur.copy()
            d = rob["stroke"] * qs / 100.0 / 1000
            xml += f'<body name="ee_slide" pos="0 0 {d:.4f}">\n'
            g, cen = link_geoms(rob["slide"], "SL")
            tool_loc = sum(a * c for a, c in cen) / max(sum(a for a, _ in cen), 1e-9)   # 吸盘板面积质心（J7 坐标系）
            xml += g + "\n</body>\n"
            xml += f'<body name="ee_tray" pos="0 0 {-d:.4f}">\n'
            g, cen = link_geoms(rob["tray"], "TR")
            xml += g + "\n</body>\n"
            if live:                                 # 末端负载：质量点 + 半透明箱子（质量运行时可改）
                # 烟箱 0.30(深,沿吸盘方向)×0.40(宽)×0.25(高)，坐在底托板上、贴着吸嘴面：中心在 J7 系 (0.069, 0, 0.372)
                xml += ('<body name="payload" pos="0.069 0 0.372">\n'
                        '<inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>\n'
                        f'<geom name="payload_box" {GEOM} type="box" size="0.125 0.2 0.15" rgba="0.80 0.64 0.42 0"/>\n</body>\n')
    xml += "</body>\n" * close

    tot = sum(a for a, _ in arm_cen) or 1.0
    arm_com = sum(a * c for a, c in arm_cen) / tot / 1000.0       # base 系 [m]
    ee_pos = ee_T[:3, 3] / 1000.0
    tool_pos = (ee_T @ np.append(tool_loc, 1))[:3] / 1000.0        # 末端负载质心（吸盘板中心）
    base_xml.append(xml)

    return dict(
        assets="\n    ".join(assets),
        base="\n      ".join(base_xml),
        rear="\n        ".join(rear_xml),
        pin_z=(PIN_Z - Z0) / 1000,
        steer={n: "\n        ".join(v) for n, v in steer_xml.items()},
        wheels={n: "\n        ".join(v) for n, v in wheel_xml.items()},
        wheel_xy={n: (c[0] / 1000, c[1] / 1000) for n, c in wheel_c.items()},
        arm_com=arm_com, ee_pos=ee_pos, tool_pos=tool_pos,
        holding=bool(pre[7]),
        stats=stats,
        arm_act="\n    ".join(act_xml), arm_q0=arm_q0, arm_act_names=arm_names,
    )


if __name__ == "__main__":
    v = get_visuals("reach")
    print("wheel xy:", v["wheel_xy"])
    print("arm com:", v["arm_com"], "ee:", v["ee_pos"], "holding:", v["holding"])
    print("meshes:", v["assets"].count("<mesh "))


def preset_q(pose):
    """预设姿态的 7 个关节角（弧度，已加零位偏置，直接作为 MuJoCo 关节目标）"""
    if ARM_TYPE == "abb":
        import arm_abb
        return list(arm_abb.preset_theta(pose))
    _, rob = _load()
    pre = rob["pre"][POSES.get(pose, pose)]
    return [math.radians(pre[i] + rob["off"][i]) for i in range(7)]
