"""安川 MOTOMAN-GP110B（YR-1-07VXB110-A00）7 轴机械臂：S、L、E（中臂）、U、R、B、T。

尺寸（单位 mm，来自你给的图）：
  L 轴（J2）：在 S 轴（J1）前方 320、底座平面上方 540
  下臂 L→U（J2→J3）：竖直向上 435+435=870
  前臂：U 轴（J3）到腕中心 1020，前臂轴线比 U 轴高 235
  腕中心 → 法兰中心 200
  校核：870 + √(1020²+235²) + 320 = 2237 ≈ 图上的最大水平伸出 2236 ✓

θ=0 的姿态 = 图里画的姿态：下臂竖直、前臂水平向前、法兰朝前。
关节：J1(S) 绕 z；J2(L)、J3(E)、J4(U) 绕 y（正角度：向前倾 / 前臂向下压）；J5(R) 绕前臂轴 x；
      J6(B) 绕 y（腕俯仰）；J7(T) 绕 x（法兰扭转）。
夹具坐标系（EE）：原点在法兰中心，z = 吸盘朝向，x = 托板方向（θ=0 时 z 朝前、x 朝下），沿用你原来的夹具网格。
"""
import json, math, os
import numpy as np

JOINTS = [   # 安川 GP110B（7 轴）：S、L、E、U、R、B、T。原点相对上一个关节坐标系，单位 mm
    {"n": "S", "a": "z", "o": [0.0, 0.0, 0.0], "s": 1},
    {"n": "L", "a": "y", "o": [320.0, 0.0, 540.0], "s": 1},       # 下臂：在 S 轴前方 320、高 540
    {"n": "E", "a": "y", "o": [0.0, 0.0, 435.0], "s": 1},         # 中臂：L 轴上方 435（尺寸图 435+435）
    {"n": "U", "a": "y", "o": [0.0, 0.0, 435.0], "s": 1},         # 上臂：E 轴上方 435
    {"n": "R", "a": "x", "o": [0.0, 0.0, 235.0], "s": 1},         # 腕滚转：沿前臂轴线，比 U 轴高 235
    {"n": "B", "a": "y", "o": [1020.0, 0.0, 0.0], "s": 1},        # 腕俯仰：前臂 1020
    {"n": "T", "a": "x", "o": [0.0, 0.0, 0.0], "s": 1},           # 法兰扭转
]
# 运动范围 [°]：安川 GP 系列样本（GP110B, YR-1-07VXB110-A00）
LIM_DEG = [(-180, 180), (-45, 155), (-45, 120), (-70, 90), (-360, 360), (-125, 125), (-360, 360)]
# 最大速度 [°/s]：同上样本
MAX_SPEED_DEG = [140, 110, 110, 130, 175, 175, 255]
OFF_DEG = [0.0] * 7
EE_POS = np.array([0.200, 0.0, 0.0])                                   # 法兰中心（J7 系）
EE_R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)             # 夹具系的轴在 J7 系里：x→(0,0,-1)，y→(0,1,0)，z→(1,0,0)
# 质量分摊（臂总质量扣掉相机立柱 20 kg、夹具 30 kg 后，按比例分给各连杆）
MAST_KG, GRIP_KG = 20.0, 10.0
FRAC = {1: 0.24, 2: 0.17, 3: 0.17, 4: 0.17, 5: 0.12, 6: 0.09, 7: 0.048}
# 各连杆质心（本连杆坐标系，m）
LINK_COM = {1: (0.0, 0.0, 0.30), 2: (0.0, 0.0, 0.435), 3: (0.0, 0.0, 0.12), 4: (0.0, 0.0, 0.0),
            5: (0.51, 0.0, 0.0), 6: (0.05, 0.0, 0.0), 7: (0.10, 0.0, 0.0)}
ORANGE = "0.93 0.46 0.10 1"
DARK = "0.22 0.23 0.26 1"

# ---- 若已用 build_gp110b_meshes.py 从 STEP 生成了真实网格，就用 STEP 的几何、体积占比和体积质心 ----
_LINKS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gp110b", "links.json")
STEP_PARTS = []
if os.path.exists(_LINKS_FILE):
    _li = json.load(open(_LINKS_FILE, encoding="utf-8"))
    STEP_PARTS = _li["parts"]
    for _k, _v in _li["links"].items():
        if int(_k) >= 1:
            FRAC[int(_k)] = _v["frac"]
            LINK_COM[int(_k)] = tuple(_v["cen"])

# 固定底座（BASE）：它是机器人本体质量的一部分，但固定在车上，质量算进底盘；转动连杆只分到剩下的
BASE_FRAC = 0.0                                   # 底座占本体质量的比例（按 STEP 体积）
BASE_COM = np.zeros(3)                            # 底座质心（机器人系：原点在 S 轴与安装面交点，m）
if STEP_PARTS:
    _vols = {int(k): v["vol_L"] for k, v in _li["links"].items()}
    BASE_FRAC = _vols.get(0, 0.0) / max(sum(_vols.values()), 1e-9)
    BASE_COM = np.array(_li["links"]["0"]["cen"], float)


def base_mass(arm_mass):
    """固定底座的质量 [kg]（本体 = 总质量 - 立柱 - 夹具）"""
    return max(arm_mass - MAST_KG - GRIP_KG, 0.0) * BASE_FRAC


_PRESET_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "abb_presets.json")
PRESETS_DEG = json.load(open(_PRESET_FILE)) if os.path.exists(_PRESET_FILE) else {}
PRESET_NAMES = {"zero", "stow", "ready", "reach", "pick", "lift", "side", "rear"}


def rot(axis, ang):
    c, s = math.cos(ang), math.sin(ang)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def preset_theta(name):
    d = PRESETS_DEG.get(name)
    return np.radians(d) if d is not None else np.zeros(7)


def fk_chain(theta, base_m):
    """返回 7 个关节坐标系（4×4，m）和夹具系 T_ee"""
    T = np.eye(4); T[:3, 3] = base_m
    out = []
    for j, th in zip(JOINTS, theta):
        Tl = np.eye(4); Tl[:3, :3] = rot(j["a"], j["s"] * th); Tl[:3, 3] = np.array(j["o"]) / 1000.0
        T = T @ Tl; out.append(T.copy())
    Te = np.eye(4); Te[:3, :3] = EE_R; Te[:3, 3] = EE_POS
    return out, out[-1] @ Te


def cap(a, b, r, rgba=ORANGE):
    return (f'<geom contype="0" conaffinity="0" mass="0" group="1" type="capsule" '
            f'fromto="{a[0]:.4f} {a[1]:.4f} {a[2]:.4f} {b[0]:.4f} {b[1]:.4f} {b[2]:.4f}" size="{r}" rgba="{rgba}"/>')


def cyl_y(c, r, half, rgba=DARK):
    return (f'<geom contype="0" conaffinity="0" mass="0" group="1" type="cylinder" pos="{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}" '
            f'euler="90 0 0" size="{r} {half}" rgba="{rgba}"/>')


def cyl_x(c, r, half, rgba=DARK):
    return (f'<geom contype="0" conaffinity="0" mass="0" group="1" type="cylinder" pos="{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}" '
            f'euler="0 90 0" size="{r} {half}" rgba="{rgba}"/>')


def box(c, h, rgba=ORANGE):
    return (f'<geom contype="0" conaffinity="0" mass="0" group="1" type="box" pos="{c[0]:.4f} {c[1]:.4f} {c[2]:.4f}" '
            f'size="{h[0]} {h[1]} {h[2]}" rgba="{rgba}"/>')


def link_geoms(i):
    """各连杆的外观（用基本体按尺寸图搭出来）"""
    if i == 1:     # 转盘 + 肩座
        return [f'<geom contype="0" conaffinity="0" mass="0" group="1" type="cylinder" pos="0 0 0.11" size="0.30 0.11" rgba="{DARK}"/>',
                box((0.0, 0.0, 0.40), (0.20, 0.26, 0.30))]
    if i == 2:     # J2 轴座 + 下臂（J2 → J3：竖直向上 0.87）
        return [cyl_y((0, 0, 0), 0.20, 0.26), cap((0, 0, 0), (0, 0, 0.87), 0.13)]
    if i == 3:     # 肘部 + 到前臂轴线的 0.235 立柱
        return [cyl_y((0, 0, 0), 0.17, 0.20), cap((0, 0, 0), (0, 0, 0.235), 0.14)]
    if i == 4:
        return [cyl_y((0, 0, 0), 0.15, 0.18), cap((0, 0, 0), (0, 0, 0.235), 0.12)]
    if i == 5:     # 前臂（沿 x，1.02）
        return [cap((0, 0, 0), (1.02, 0, 0), 0.11), cyl_x((0.05, 0, 0), 0.15, 0.08)]
    if i == 6:     # 腕
        return [cyl_y((0, 0, 0), 0.10, 0.13), cap((0, 0, 0), (0.12, 0, 0), 0.09)]
    if i == 7:     # 法兰（到夹具 0.2）
        return [cyl_x((0.10, 0, 0), 0.085, 0.10, DARK)]
    return []


def link_masses(arm_mass, grip_loc_ee, mast_loc_j1):
    """各连杆（含并入的夹具、相机立柱）的质量 [kg] 和质心 [m]（本关节系）。
    grip_loc_ee：夹具质心在夹具系里的位置 (m)；mast_loc_j1：相机立柱质心在 J1 系里的位置 (m)"""
    body = max(arm_mass - MAST_KG - GRIP_KG, 0.0) * (1.0 - BASE_FRAC)     # 转动连杆分到的质量（扣掉固定底座）
    out = {}
    tot = sum(FRAC.values())
    for i in range(1, 8):
        m = body * FRAC[i] / tot
        c = np.array(LINK_COM[i], float)
        if i == 1:
            c = (m * c + MAST_KG * np.array(mast_loc_j1)) / (m + MAST_KG); m += MAST_KG
        if i == 7:
            g = EE_POS + EE_R @ np.array(grip_loc_ee)
            c = (m * c + GRIP_KG * g) / (m + GRIP_KG); m += GRIP_KG
        out[i] = (m, c)
    return out
