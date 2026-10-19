"""
机器人底盘动力学校核模板 (MuJoCo)
=====================================
四轮底盘（差速 / 四驱），车体用 "总质量 + 质心 + 转动惯量" 描述，
这些数值可直接从 Solid Edge 装配体的「物理属性」里读出来填入 CONFIG。

包含的校核：
  1. 侧向 / 纵向静态倾翻角（仿真搜索 + 解析公式对照）
  2. 爬坡：所需轮端力矩、能否达到目标速度、前轮载荷
  3. 平地急停：最大减速度、后轮最小载荷、俯仰角、制动距离
  4. 越台阶：给定台阶高度能否通过
  5. 参数扫描：轮距 × 质心高度 → 侧倾临界角（输出 CSV）

运行：
  pip install mujoco numpy
  python chassis_sim.py            # 跑全部校核
  python -m mujoco.viewer --mjcf chassis.xml   # 打开生成的模型可视化查看

坐标约定：x 前进方向，y 向左，z 向上；原点在底盘几何中心正下方的地面。
"""

import csv
import math
import numpy as np
import mujoco

# ============================================================
# 参数区：改这里
# ============================================================
CONFIG = dict(
    # ---- 几何（参考 chassis_A_3d.html 方案 A v11，数值为估算）----
    wheelbase=1.1125,     # 轴距 L [m]（前后轮中心距）= 原页面 0.89 x 1.25（10° 坡上卸货全过程静态倾翻角 >= 15°）
    wheelbase_scale=1.25, # 模型里实际放大轴距/轮距的倍数（页面车轮位置 x 倍数）
    track=0.9125,         # 轮距 B [m]（左右轮中心距）= 原页面 0.73 x 1.25
    track_scale=1.25,
    wheel_radius=0.075,   # 轮半径 r [m]（Ø150 聚氨酯舵轮）
    wheel_width=0.07,     # 轮宽 [m]
    body_size=(1.20, 1.00, 0.35),   # 车体外形 长宽高 [m]，仅用于显示/碰撞
    body_bottom=0.228,    # 车体底面离地高度 [m]（下沉舱底面离地 228 mm）

    # ---- 质量属性 ----
    chassis_mass=1000.0,  # 底盘总质量（含 4 个轮子，不含机械臂）[kg]（用户给定）
    chassis_com=(-0.15, 0.0, 0.40),   # 底盘质心 (x, y, 离地高度 z) [m]（估算：车架+电池+配重，偏后偏低；需用 CAD 实测值替换）
    chassis_inertia=(104.0, 141.0, 203.0),   # 底盘绕自身质心的惯量 [kg·m²]（按 1.2×1.0×0.5 m 均质箱体估算）
    arm_mass=820.0,       # 机械臂总质量 [kg] = GP110B 本体 790（安川样本）+ 夹具 10（用户给定）+ 相机立柱 20（估算）
    arm_inertia=72.0,     # 机械臂自身惯量（各向同性近似）[kg·m²]
    payload=0.0,          # 末端负载 0–50 kg，质心取吸盘板中心
    j2_shift=0.32,        # 安川 GP110B：L 轴（J2）在 S 轴（J1）前方 320 mm（尺寸图）
    wheel_mass=8.0,       # 单个舵轮模组质量 [kg]（已包含在底盘质量内）

    # ---- 驱动（质量大幅增加后需要重新选型，这里是占位值）----
    motor_torque=200.0,   # 单轮轮端最大力矩 [N·m]（20 mm 台阶、μ=0.6 在 160 N·m 以上稳定通过，取 200 留余量）
    motor_armature=0.5,   # 电机转子惯量折算到轮端 [kg·m²]
    speed_gain=200.0,     # 轮速环增益 [N·m/(rad/s)]
    drive_wheels="all",   # "all" = 四驱, "rear" = 后驱, "front" = 前驱

    # ---- 后桥摆动悬挂（页面方案：前轴刚性，后桥绕纵向销轴摆动 ±4°，无弹簧）----
    suspension="on",      # "on" = 摆动桥自由摆动, "off" = 锁死（等同刚性车架）, None = 不建后桥刚体
    swing_limit=4.0,      # 摆角限位 [°]
    swing_damping=300.0,  # 销轴阻尼 [N·m·s/rad]（轴套/阻尼器的等效；30 太小，坡上急停后后桥会撞限位反复摇晃）
    rear_axle_mass=100.0, # 后桥总成（横梁+两个舵轮模组，不含轮）质量 [kg]（估算，已包含在底盘质量内）
    pin_x=-0.30,          # 摆动销轴支承点的 x 位置 [m]（页面：销轴 x 从 -442 到 -166 mm，取中点附近）
    bump=0.0,             # 左侧单边斜坡台高度 [m]，用于演示悬挂（0 = 无）

    # ---- 地面 & 工况 ----
    friction=0.6,         # 聚氨酯轮 - 木板地面 摩擦系数（干燥木板约 0.5~0.7）
    floor_friction=0.6,   # 车厢木地板摩擦系数
    tire_k=250.0,         # 轮胎/地板接触刚度（MuJoCo 参考加速度刚度，配合 solimp 放大约 13 倍）→ 静态压缩约 3 mm，竖向固有频率约 9 Hz，对应页面的轮胎径向刚度 1500 N/mm
    tire_c=5.0,           # 接触阻尼（阻尼比约 0.5）
    steer_kp=6000.0,      # 舵轮转向位置伺服刚度 [N·m/rad]
    steer_kv=400.0,       # 转向伺服阻尼 [N·m·s/rad]
    steer_torque=800.0,   # 转向电机最大扭矩 [N·m]（占位值：单轮约 600 kg 载荷时原地转舵约需 350 N·m）
    steer_damping=5.0, steer_armature=0.5, steer_mass=6.0,   # 转向关节阻尼 / 折算惯量 / 转向模组质量(含在底盘质量内)
    slope_deg=10.0,       # 爬坡校核坡度 [°]
    v_max=1.0,            # 最高速度 [m/s]
    step_height=0.02,     # 越障台阶高度 [m]
)

G = 9.81
TIMESTEP = 0.001


# ============================================================
# 模型生成
# ============================================================
WHEELS = ["fl", "fr", "rl", "rr"]  # 前左 前右 后左 后右



# ============================================================
# 外观（纯显示，不参与碰撞和质量）：车架、桥、电气舱、机械臂
# ============================================================
VIS = 'contype="0" conaffinity="0" mass="0"'
# 机械臂姿态：(各段 (长度m, 在竖直面内与水平夹角°) 列表, 绕竖直轴转角°, 是否抓箱)
ARM_VIS = {
    "ready": ([(0.8, 75), (0.6, 170), (0.5, -90)], 0, False),   # 准备：折叠在车体上方
    "reach": ([(0.8, 35), (0.6, 5), (0.7, -5)], 0, True),       # 前伸抓箱
    "side":  ([(0.8, 35), (0.6, 5), (0.7, -5)], -90, True),     # 侧向
}


def vis_xml(cfg, pose):
    L, B, r = cfg["wheelbase"], cfg["track"], cfg["wheel_radius"]
    bx, by, bz = cfg["body_size"]
    zb = cfg["body_bottom"] - r            # 车体底面（base 系，轴高为 0）
    zt = zb + bz                           # 车体顶面
    o = []
    # 纵梁 + 前后桥
    for sy in (1, -1):
        o.append(f'<geom {VIS} type="box" pos="0 {sy*by*0.36} {zb+0.05}" size="{bx*0.46} 0.04 0.05" rgba="0.35 0.38 0.42 1"/>')
    for x in (L/2, -L/2):
        o.append(f'<geom {VIS} type="cylinder" fromto="{x} {-B/2} 0 {x} {B/2} 0" size="0.03" rgba="0.55 0.58 0.6 1"/>')
    # 舵轮立柱
    for x in (L/2, -L/2):
        for sy in (1, -1):
            o.append(f'<geom {VIS} type="box" pos="{x} {sy*B/2} {r+0.05}" size="0.05 0.04 0.1" rgba="0.3 0.3 0.33 1"/>')
    # 电池舱（中部）、电气舱（车尾，橙色）、配重（尾部低位）
    o.append(f'<geom {VIS} type="box" pos="0.05 0 {zb+0.12}" size="0.25 0.3 0.1" rgba="0.2 0.6 0.35 1"/>')
    o.append(f'<geom {VIS} type="box" pos="{-bx*0.3} 0 {zb+0.2}" size="0.2 {by*0.4} 0.15" rgba="0.85 0.5 0.15 1"/>')
    o.append(f'<geom {VIS} type="box" pos="{-bx*0.5+0.1} 0 {zb+0.08}" size="0.08 {by*0.35} 0.07" rgba="0.25 0.25 0.25 1"/>')
    # 顶盖 + 四角雷达
    o.append(f'<geom {VIS} type="box" pos="0 0 {zt}" size="{bx/2} {by/2} 0.01" rgba="0.92 0.93 0.95 1"/>')
    for sx in (1, -1):
        for sy in (1, -1):
            o.append(f'<geom {VIS} type="cylinder" pos="{sx*(bx/2-0.06)} {sy*(by/2-0.06)} {zt+0.03}" size="0.04 0.03" rgba="0.1 0.1 0.12 1"/>')

    if pose in ARM_VIS:
        segs, yaw, hold = ARM_VIS[pose]
        ax, az = bx * 0.5 - 0.25, zt + 0.3          # 立柱位置（车头偏后）、肩高
        c, sn = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        def P(px, pz):                               # 臂面内坐标 -> base 系
            dx = px - ax
            return (ax + dx * c, dx * sn, pz)
        o.append(f'<geom {VIS} type="cylinder" fromto="{ax} 0 {zt} {ax} 0 {az}" size="0.1" rgba="0.3 0.3 0.33 1"/>')
        px, pz = ax, az
        pts = [(px, pz)]
        for ln, ang in segs:
            px += ln * math.cos(math.radians(ang)); pz += ln * math.sin(math.radians(ang))
            pts.append((px, pz))
        for (x0, z0), (x1, z1) in zip(pts[:-1], pts[1:]):
            a, b = P(x0, z0), P(x1, z1)
            o.append(f'<geom {VIS} type="capsule" fromto="{a[0]} {a[1]} {a[2]} {b[0]} {b[1]} {b[2]}" size="0.05" rgba="0.95 0.6 0.1 1"/>')
        for x0, z0 in pts:
            q = P(x0, z0)
            o.append(f'<geom {VIS} type="sphere" pos="{q[0]} {q[1]} {q[2]}" size="0.065" rgba="0.2 0.2 0.22 1"/>')
        e = P(*pts[-1])                              # 末端：吸盘板 300×580 mm
        o.append(f'<geom {VIS} type="box" pos="{e[0]} {e[1]} {e[2]-0.04}" size="0.15 0.29 0.015" euler="0 0 {yaw}" rgba="0.15 0.15 0.18 1"/>')
        if hold:
            o.append(f'<geom {VIS} type="box" pos="{e[0]} {e[1]} {e[2]-0.04-0.015-0.15}" size="0.2 0.28 0.15" euler="0 0 {yaw}" rgba="0.7 0.55 0.35 1"/>')
    return "\n      ".join(o)


def build_xml(cfg, step_height=None):
    if "body_mass" not in cfg:                     # 还没合成整机质量属性：按当前姿态/负载合成
        if cfg.get("arm_live"):
            cfg = cfg_live(cfg, cfg.get("arm_pose", "ready"), mesh=cfg.get("mesh", True))
        else:
            cfg = cfg_for_pose(cfg, cfg.get("arm_pose", "ready"), mesh=cfg.get("mesh", False))
    L, B, r = cfg["wheelbase"], cfg["track"], cfg["wheel_radius"]
    w = cfg["wheel_width"]
    bx, by, bz = cfg["body_size"]
    cx, cy, cz = cfg["body_com"]
    ixx, iyy, izz = cfg["body_inertia"]
    mu = cfg["friction"]

    # base 坐标原点放在轮轴高度
    body_center_z = cfg["body_bottom"] + bz / 2 - r
    pos = {"fl": (L / 2, B / 2), "fr": (L / 2, -B / 2),
           "rl": (-L / 2, B / 2), "rr": (-L / 2, -B / 2)}

    V = None
    susp = cfg.get("suspension")
    pin_z = 0.262 - r                         # 销轴在 base 系的高度（页面：离地 262 mm）
    if cfg.get("mesh"):                       # 用页面里的真实零件网格做外观
        from a_model import get_visuals, MESH_DIR
        V = get_visuals(cfg.get("arm_pose", "ready"), swing=bool(susp), detail=cfg.get("detail", "lite"), pin_x=cfg["pin_x"],
                        live=cfg.get("arm_live", False), arm_mass=cfg["arm_mass"], j2_shift=cfg.get("j2_shift", 0.0))
        sx, sy = cfg.get("wheelbase_scale", 1.0), cfg.get("track_scale", 1.0)      # 研究用：整体放大轴距/轮距
        pos = {n: (float(xy[0]) * sx, float(xy[1]) * sy) for n, xy in V["wheel_xy"].items()}
        pin_z = V["pin_z"]
    wheel_rgba = "0 0 0 0" if V else "0.15 0.15 0.15 1"
    wheel_parts = {}
    for n in WHEELS:
        x, y = pos[n]
        wvis = V["wheels"].get(n, "") if V else ""
        px = cfg["pin_x"]
        if susp and n in ("rl", "rr"):
            x, dz = x - px, -pin_z                                # 后轮挂在摆动桥坐标系下
        else:
            dz = 0.0
        svis = V["steer"].get(n, "") if (V and "steer" in V) else ""
        # 聚氨酯轮胎：径向刚度取页面的 1500 N/mm（solref 取负值 = 直接给刚度/阻尼）
        tire = (f'solref="{-cfg["tire_k"]} {-cfg["tire_c"]}" solimp="0.9 0.95 0.02" '
                f'friction="{cfg["friction"]} 0.01 0.002"')
        wheel_parts[n] = f"""
      <body name="steer_{n}" pos="{x} {y} {dz}">
        <joint name="s_{n}" type="hinge" axis="0 0 1" damping="{cfg['steer_damping']}" armature="{cfg['steer_armature']}"/>
        <inertial pos="0 0 0.1" mass="{cfg['steer_mass']}" diaginertia="0.3 0.3 0.2"/>
        {svis}
        <body name="wheel_{n}" pos="0 0 0">
          <joint name="j_{n}" type="hinge" axis="0 1 0" damping="0.01" armature="{cfg['motor_armature']}"/>
          <geom name="g_{n}" type="{"capsule" if cfg.get("wheel_capsule") else "cylinder"}" size="{r} {w/2}" zaxis="0 1 0"
                mass="{cfg['wheel_mass']}" rgba="{wheel_rgba}" condim="4" contype="0" conaffinity="0" {tire}/>
          {wvis}
        </body>
      </body>"""
    wheels_xml = wheel_parts["fl"] + wheel_parts["fr"]
    mass_base, com_base = cfg["body_mass"], (cx, cy, cz)
    eq_xml = ""
    if susp:
        mr = cfg["rear_axle_mass"]
        rcx, rcz = pos["rl"][0], r + 0.06                   # 后桥质心（地面系），约在轮轴上方 6 cm
        M = cfg["body_mass"]
        mass_base = M - mr
        com_base = ((M * cx - mr * rcx) / mass_base, M * cy / mass_base, (M * cz - mr * rcz) / mass_base)
        lim = cfg["swing_limit"]
        wheels_xml += f"""
      <body name="rear_axle" pos="{cfg['pin_x']} 0 {pin_z}">
        <joint name="swing" type="hinge" axis="1 0 0" range="{-lim} {lim}" limited="true"
               damping="{cfg['swing_damping']}" armature="2.0"/>
        <inertial pos="{rcx - cfg['pin_x']} 0 {rcz - r - pin_z}" mass="{mr}" diaginertia="{mr*0.07:.2f} {mr*0.03:.2f} {mr*0.1:.2f}"/>
        {V["rear"] if V else ""}{wheel_parts["rl"]}{wheel_parts["rr"]}
      </body>"""
        eq_xml = (f'<equality><joint name="swing_lock" joint1="swing" polycoef="0 0 0 0 0" '
                  f'active="{"true" if susp == "off" else "false"}"/></equality>')
    else:
        wheels_xml += wheel_parts["rl"] + wheel_parts["rr"]

    drive = {"all": WHEELS, "rear": ["rl", "rr"], "front": ["fl", "fr"]}[cfg["drive_wheels"]]
    T = cfg["motor_torque"]
    kv = cfg["speed_gain"]
    act_xml = "".join(
        f'\n    <velocity name="m_{n}" joint="j_{n}" kv="{kv}" forcelimited="true" '
        f'forcerange="{-T} {T}"/>' for n in drive)
    act_xml += "".join(                                    # 舵轮转向：位置伺服（ctrl = 舵角 rad）
        f'\n    <position name="st_{n}" joint="s_{n}" kp="{cfg["steer_kp"]}" kv="{cfg["steer_kv"]}" forcelimited="true" '
        f'forcerange="{-cfg["steer_torque"]} {cfg["steer_torque"]}"/>' for n in WHEELS)
    if cfg.get("arm_live") and V:
        act_xml += "\n    " + V["arm_act"]

    floor_geom = (f'<geom name="floor" type="plane" size="30 30 0.1" material="wood" '
                  f'friction="{cfg["floor_friction"]} 0.01 0.002" solref="{-cfg["tire_k"]} {-cfg["tire_c"]}" solimp="0.9 0.95 0.02"/>')
    step_xml = ""
    extra_floor = []                               # 需要和车轮建接触对的其他地面几何
    if step_height:
        x_step = cfg.get("step_x", 1.0)
        step_xml = (f'<geom name="step" type="box" pos="{x_step + 1.0} 0 {step_height/2}" '
                    f'size="1.0 2 {step_height/2}" rgba="0.6 0.5 0.4 1" material="wood" '
                    f'friction="{cfg["floor_friction"]} 0.01 0.002"/>')
        extra_floor.append("step")
    if cfg.get("gap_w"):                           # 地板上一条横向缝（宽 gap_w），把 plane 换成两块厚板
        xg, gw = cfg.get("gap_x", 1.0), cfg["gap_w"]
        fr = f'material="wood" friction="{cfg["floor_friction"]} 0.01 0.002" solref="{-cfg["tire_k"]} {-cfg["tire_c"]}" solimp="0.9 0.95 0.02"'
        floor_geom = (f'<geom name="floor" type="box" pos="{(xg - 30) / 2} 0 -0.5" size="{(xg + 30) / 2} 30 0.5" {fr}/>'
                      f'<geom name="floor_b" type="box" pos="{(xg + gw + 30) / 2} 0 -0.5" size="{(30 - xg - gw) / 2} 30 0.5" {fr}/>')
        extra_floor.append("floor_b")

    if cfg.get("bump"):                       # 左侧（y>0）单边斜坡台：一侧车轮抬高，演示摆动桥均衡载荷
        h = cfg["bump"]; th = math.radians(8); hl = h / (2 * math.sin(th)); t = 0.01; x0 = 1.5
        step_xml += (f'<geom name="ramp" type="box" euler="0 {-8} 0" '
                     f'pos="{x0 + hl*math.cos(th) + t*math.sin(th)} 0.8 {hl*math.sin(th) - t*math.cos(th)}" '
                     f'size="{hl} 0.8 {t}" rgba="0.75 0.6 0.35 1"/>'
                     f'<geom name="plateau" type="box" pos="{x0 + 2*hl*math.cos(th) + 0.8} 0.8 {h/2}" '
                     f'size="0.8 0.8 {h/2}" rgba="0.75 0.6 0.35 1"/>')
    # 轮胎-地板接触对：法向软（聚氨酯刚度），摩擦约束硬（solreffriction 很短），这样坡上不蠕动、转向也不会被卡
    contact_xml = "<contact>" + "".join(
        f'<pair geom1="floor" geom2="g_{n}" condim="4" friction="{cfg["friction"]} {cfg["friction"]} 0.01 0.002 0.002" '
        f'solref="{-cfg["tire_k"]} {-cfg["tire_c"]}" solimp="0.9 0.95 0.02" solreffriction="0.004 1"/>'
        for n in WHEELS) + "".join(
        f'<pair geom1="{og}" geom2="g_{n}" condim="4" friction="{cfg["friction"]} {cfg["friction"]} 0.01 0.002 0.002" '
        f'solref="{-cfg["tire_k"]} {-cfg["tire_c"]}" solimp="0.9 0.95 0.02" solreffriction="0.004 1"/>'
        for og in extra_floor for n in WHEELS) + "</contact>"
    return f"""
<mujoco model="chassis">
  <option timestep="{TIMESTEP}" gravity="0 0 {-G}" integrator="implicitfast" cone="elliptic"/>
  <default><geom friction="{mu} 0.005 0.0001"/></default>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.75 0.82 0.9" rgb2="0.95 0.96 0.98" width="512" height="512"/>
    <texture type="2d" name="grid" builtin="checker" rgb1="0.86 0.88 0.9" rgb2="0.78 0.81 0.85" width="300" height="300"/>
    <material name="grid" texture="grid" texrepeat="12 12" reflectance="0.03"/>
    <texture type="2d" name="wood" builtin="checker" rgb1="0.66 0.50 0.32" rgb2="0.60 0.45 0.29" width="256" height="256" mark="edge" markrgb="0.35 0.25 0.15"/>
    <material name="wood" texture="wood" texrepeat="40 8" reflectance="0.02"/>
    {V["assets"] if V else ""}
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1"/>
    <light pos="2 -2 3" dir="-0.5 0.5 -1" diffuse="0.5 0.5 0.5"/>
    {__import__("scene").scene_xml() if cfg.get("scene") else ""}
    {('<body name="tilt" mocap="true">' + floor_geom + '</body>') if cfg.get("tilt_floor") else floor_geom}
    {step_xml}
    <body name="base" pos="0 0 {r}">
      <freejoint/>
      <inertial pos="{com_base[0]} {com_base[1]} {com_base[2] - r}" mass="{mass_base}"
                diaginertia="{ixx} {iyy} {izz}"/>
      <geom name="body" type="box" pos="0 0 {body_center_z}" size="{bx/2} {by/2} {bz/2}"
            mass="0" rgba="0.3 0.5 0.8 {0 if V else 0.25}"/>
      {V["base"] if V else vis_xml(cfg, cfg.get("arm_pose", "ready"))}
      {__import__("scene").bed_box_xml(r) if cfg.get("scene") else ""}
      <site name="com" pos="{cx} {cy} {cz - r}" size="0.02" rgba="1 0 0 1"/>
      {wheels_xml}
    </body>
  </worldbody>
  {eq_xml}
  {contact_xml}
  <actuator>{act_xml}
  </actuator>
</mujoco>"""


def cfg_for_pose(cfg, pose="ready", payload=None, mesh=False):
    """由 底盘 + 机械臂(按页面姿态的真实正运动学求质心) + 末端负载 合成整机质量、质心、惯量。
    返回的 cfg 带 body_mass / body_com / body_inertia（不含 4 个轮子，轮子单独建模）。"""
    from a_model import get_visuals
    v = get_visuals(pose, detail="full", j2_shift=cfg.get("j2_shift", 0.0))
    r, mw = cfg["wheel_radius"], cfg["wheel_mass"]
    pl = cfg.get("payload", 0.0) if payload is None else payload
    up = np.array([0.0, 0.0, r])                     # a_model 的坐标原点在轮轴高度，这里换成地面系
    try:
        import arm_abb as AB
        mb = AB.base_mass(cfg["arm_mass"]) if cfg.get("arm_type", "abb") == "abb" else 0.0
        pbase = np.array([AB.BASE_COM[0], AB.BASE_COM[1], 0.79 + AB.BASE_COM[2]]) if mb else np.zeros(3)
    except Exception:
        mb, pbase = 0.0, np.zeros(3)
    items = [(cfg["chassis_mass"] - 4 * (mw + cfg["steer_mass"]), np.array(cfg["chassis_com"], float), np.array(cfg["chassis_inertia"], float)),
             (mb, pbase, np.zeros(3)),
             (cfg["arm_mass"] - mb, v["arm_com"] + up, np.full(3, cfg["arm_inertia"])),
             (pl, v["tool_pos"] + up, np.zeros(3))]
    M = sum(m for m, _, _ in items)
    com = sum(m * c for m, c, _ in items) / M
    I = np.zeros(3)
    for m, c, i0 in items:                           # 平行轴定理（忽略惯性积）
        d = c - com
        I += i0 + m * (d @ d - d * d)
    return dict(cfg, body_mass=float(M), body_com=tuple(float(x) for x in com),
                body_inertia=tuple(float(x) for x in I), arm_pose=pose, mesh=mesh, payload=pl)


def cfg_live(cfg, pose="ready", mesh=True):
    """机械臂做成真实关节（质量按连杆分摊），底盘本体只含底盘自己的质量；臂和负载的质心由动力学自然产生"""
    import arm_abb as AB
    mw, r = cfg["wheel_mass"], cfg["wheel_radius"]
    m0 = cfg["chassis_mass"] - 4 * (mw + cfg["steer_mass"])         # 转向模组质量也含在底盘质量内
    mb = AB.base_mass(cfg["arm_mass"])                       # 机器人固定底座，焊在车上
    pb = np.array([AB.BASE_COM[0], AB.BASE_COM[1], 0.79 + AB.BASE_COM[2]])
    com0 = np.array(cfg["chassis_com"], float)
    M = m0 + mb
    com = (m0 * com0 + mb * pb) / M
    I = np.array(cfg["chassis_inertia"], float)
    for m_, p_ in ((m0, com0), (mb, pb)):                   # 平行轴定理（底座自身惯量忽略；底盘惯量已在其质心处给出）
        d_ = p_ - com
        I = I + m_ * (d_ @ d_ - d_ * d_)
    return dict(cfg, body_mass=float(M), body_com=tuple(float(x) for x in com),
                body_inertia=tuple(float(x) for x in I), arm_pose=pose, mesh=mesh, arm_live=True)


def compile_model(cfg, step_height=None):
    """编译 MJCF；网格文件以内存字典传入（避免 Windows 下中文路径打不开）"""
    xml = build_xml(cfg, step_height)
    assets = None
    if cfg.get("mesh"):
        import os
        from a_model import MESH_DIR
        assets = {fn: open(os.path.join(MESH_DIR, fn), "rb").read()
                  for fn in os.listdir(MESH_DIR) if fn.endswith(".stl")}
    return mujoco.MjModel.from_xml_string(xml, assets)


class Sim:
    def __init__(self, cfg, step_height=None):
        self.cfg = cfg
        self.m = compile_model(cfg, step_height)
        self.d = mujoco.MjData(self.m)
        self.wheel_geoms = {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, f"g_{n}"): n
                            for n in WHEELS}
        self.base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.total_mass = float(self.m.body_subtreemass[self.base])
        self.weight = self.total_mass * G
        self.r = cfg["wheel_radius"]

    def reset(self, gravity_dir=(0, 0, -1)):
        mujoco.mj_resetData(self.m, self.d)
        self.m.opt.gravity[:] = np.array(gravity_dir) * G
        self.d.ctrl[:] = 0
        mujoco.mj_forward(self.m, self.d)

    def set_speed(self, v):
        self.d.ctrl[:] = v / self.r

    def step(self, n=1):
        mujoco.mj_step(self.m, self.d, n)

    def wheel_loads(self):
        """各轮法向力 [N]"""
        loads = {n: 0.0 for n in WHEELS}
        f6 = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            for g in (c.geom1, c.geom2):
                if g in self.wheel_geoms:
                    mujoco.mj_contactForce(self.m, self.d, i, f6)
                    loads[self.wheel_geoms[g]] += f6[0]
        return loads

    def base_vel(self):
        return self.d.qvel[:3].copy()

    def roll_pitch(self):
        q = self.d.qpos[3:7]
        w, x, y, z = q
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = math.asin(max(-1, min(1, 2 * (w * y - z * x))))
        return math.degrees(roll), math.degrees(pitch)

    def act_torques(self):
        return self.d.actuator_force.copy()


# ============================================================
# 解析参考值
# ============================================================
def composite_com(cfg):
    mb, mw, r = cfg["body_mass"], cfg["wheel_mass"], cfg["wheel_radius"]
    cx, cy, cz = cfg["body_com"]
    M = mb + 4 * mw
    return M, (mb * cx) / M, (mb * cy) / M, (mb * cz + 4 * mw * r) / M


def analytic_tip(cfg):
    M, x, y, h = composite_com(cfg)
    L, B = cfg["wheelbase"], cfg["track"]
    return dict(
        lateral=math.degrees(math.atan((B / 2 - abs(y)) / h)),
        forward=math.degrees(math.atan((L / 2 - x) / h)),    # 车头朝下坡/急停前翻
        backward=math.degrees(math.atan((L / 2 + x) / h)),   # 车头朝上坡/急加速后翻
        h=h, M=M,
    )


# ============================================================
# 校核 1：静态倾翻角
# ============================================================
def tipped_at(cfg, angle_deg, direction):
    """在给定坡度下静置，判断是否倾翻 / 侧滑"""
    sim = Sim(cfg)
    a = math.radians(angle_deg)
    # 用倾斜重力等效坡面
    if direction == "lateral":     # 坡面向右下（-y），左轮为上坡侧
        g = (0, -math.sin(a), -math.cos(a)); uphill = ["fl", "rl"]
    elif direction == "forward":   # 车头朝下坡，后轮为上坡侧
        g = (math.sin(a), 0, -math.cos(a)); uphill = ["rl", "rr"]
    else:                          # 车头朝上坡，前轮为上坡侧
        g = (-math.sin(a), 0, -math.cos(a)); uphill = ["fl", "fr"]
    sim.reset(g)
    # 静置时轮子刹死：用很大的关节阻尼抱死（隐式积分，数值稳定）
    for n in WHEELS:
        sim.m.dof_damping[sim.m.jnt_dofadr[sim.m.joint(f"j_{n}").id]] = 1e5
    sim.step(1500)
    loads, slid = [], False
    for _ in range(300):
        sim.step()
        L = sim.wheel_loads()
        loads.append(sum(L[n] for n in uphill) / sim.weight)
        if np.linalg.norm(sim.base_vel()[:2]) > 0.05:
            slid = True
    roll, pitch = sim.roll_pitch()
    tipped = np.mean(loads) < 0.005 or abs(roll) > 2 or abs(pitch) > 2
    return tipped, slid


def search_tip_angle(cfg, direction, hi=70.0):
    """纯几何倾翻角：搜索时把摩擦调到足够大，排除侧滑干扰"""
    c = dict(cfg, friction=3.0)
    lo = 0.0
    for _ in range(9):          # 二分法，精度约 0.15°
        mid = (lo + hi) / 2
        tipped, _ = tipped_at(c, mid, direction)
        if tipped:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def slide_angle(cfg):
    """刹死状态下开始整体下滑的坡度（约等于 atan(μ)）"""
    lo, hi = 0.0, 70.0
    for _ in range(9):
        mid = (lo + hi) / 2
        _, slid = tipped_at(cfg, mid, "forward")
        if slid:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


# ============================================================
# 校核 2：爬坡
# ============================================================
def climb_test(cfg, slope_deg=None, v=None):
    slope_deg = cfg["slope_deg"] if slope_deg is None else slope_deg
    v = cfg["v_max"] * 0.5 if v is None else v
    sim = Sim(cfg)
    a = math.radians(slope_deg)
    sim.reset((-math.sin(a), 0, -math.cos(a)))   # 车头朝上坡
    sim.set_speed(v)
    torques, front, speeds = [], [], []
    for k in range(4000):
        sim.step()
        if k > 2500:
            torques.append(np.abs(sim.act_torques()))
            L = sim.wheel_loads()
            front.append((L["fl"] + L["fr"]) / sim.weight)
            speeds.append(sim.base_vel()[0])
    torques = np.array(torques)
    # 解析：所需总驱动力 = m g (sinα + f cosα)，f 为滚阻(此处模型近似 0)
    n_drive = sim.m.nu
    T_need = sim.total_mass * G * math.sin(a) * sim.r / n_drive
    return dict(
        slope=slope_deg,
        v_target=v,
        v_achieved=float(np.mean(speeds)),
        torque_mean=float(torques.mean()),
        torque_peak=float(torques.max()),
        torque_analytic=T_need,
        torque_limit=cfg["motor_torque"],
        front_load_min=float(np.min(front)),
        ok=float(np.mean(speeds)) > 0.9 * v,
    )


# ============================================================
# 校核 3：平地急停
# ============================================================
def brake_test(cfg, v=None, decel=None):
    """decel=None：速度指令直接置零（满力矩硬刹，最恶劣）
       decel=数值：按给定减速度斜坡减速（实际控制器的做法）"""
    v = cfg["v_max"] if v is None else v
    sim = Sim(cfg)
    sim.reset()
    sim.set_speed(v)
    sim.step(3000)
    v0 = sim.base_vel()[0]
    x0 = sim.d.qpos[0]
    win = 20                      # 载荷做 20 ms 滑动平均，滤掉接触抖动
    buf_r, buf_f = [], []
    rear_min, front_max, pitch_max = 1.0, 0.0, 0.0
    t = 0.0
    for k in range(5000):
        t = (k + 1) * TIMESTEP
        cmd = 0.0 if decel is None else max(0.0, v0 - decel * t)
        sim.set_speed(cmd)
        sim.step()
        L = sim.wheel_loads()
        buf_r.append((L["rl"] + L["rr"]) / sim.weight)
        buf_f.append((L["fl"] + L["fr"]) / sim.weight)
        if len(buf_r) >= win:
            rear_min = min(rear_min, float(np.mean(buf_r[-win:])))
            front_max = max(front_max, float(np.mean(buf_f[-win:])))
        pitch_max = max(pitch_max, abs(sim.roll_pitch()[1]))
        if cmd == 0 and abs(sim.base_vel()[0]) < 0.005:
            break
    return dict(
        v0=float(v0),
        stop_distance=float(sim.d.qpos[0] - x0),
        stop_time=t,
        decel_avg=float(v0 / t),
        rear_load_min=rear_min,
        front_load_max=front_max,
        pitch_max=pitch_max,
    )


# ============================================================
# 校核 4：越台阶
# ============================================================
def step_test(cfg, height=None, v=0.3):
    height = cfg["step_height"] if height is None else height
    sim = Sim(cfg, step_height=height)
    sim.reset()
    sim.set_speed(v)
    torque_peak = 0.0
    for k in range(int(10 / TIMESTEP)):
        sim.step()
        if k > 500:   # 跳过起步加速段
            torque_peak = max(torque_peak, float(np.abs(sim.act_torques()).max()))
        if sim.d.qpos[0] > 1.0 + cfg["wheelbase"] + 0.3:
            return dict(height=height, passed=True, torque_peak=torque_peak)
    return dict(height=height, passed=False, torque_peak=torque_peak)


def max_step_height(cfg):
    """二分搜索可通过的最大台阶高度"""
    lo, hi = 0.0, cfg["wheel_radius"]
    for _ in range(7):
        mid = (lo + hi) / 2
        if step_test(cfg, mid)["passed"]:
            lo = mid
        else:
            hi = mid
    return lo


# ============================================================
# 校核 5：参数扫描
# ============================================================
def sweep(cfg, tracks, com_heights, out="output/tables/sweep_tip_angle.csv"):
    rows = []
    for B in tracks:
        for h in com_heights:
            c = dict(cfg, track=B, body_com=(cfg["body_com"][0], cfg["body_com"][1], h))
            ang = search_tip_angle(c, "lateral")
            rows.append((B, h, ang, analytic_tip(c)["lateral"]))
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["轮距B[m]", "车体质心高[m]", "侧倾临界角_仿真[°]", "侧倾临界角_解析[°]"])
        wr.writerows([(f"{a:.3f}", f"{b:.3f}", f"{c:.1f}", f"{d:.1f}") for a, b, c, d in rows])
    return rows, out


# ============================================================
# 校核 6：后桥摆动悬挂 开 / 关 对比
# ============================================================
def suspension_test(cfg, bump=0.04, v=0.5):
    """左侧单边斜坡台 bump [m]，比较摆动桥 开(on)/关(off) 时的四轮载荷均衡"""
    out = {}
    for mode in ("on", "off"):
        sim = Sim(dict(cfg, suspension=mode, bump=bump))
        sim.reset(); sim.set_speed(v)
        adr = sim.m.joint("swing").qposadr[0]
        xs, S, sw = [], [], []
        for k in range(9000):
            sim.step()
            if k % 5 == 0:
                L = sim.wheel_loads()
                xs.append(sim.d.qpos[0]); sw.append(math.degrees(sim.d.qpos[adr]))
                S.append([L[n] / (sim.weight / 4) for n in WHEELS])
        xs, S, sw = np.array(xs), np.array(S), np.array(sw)
        Ss = np.array([S[max(0, i - 20):i + 1].mean(0) for i in range(len(S))])   # 100 ms 滑动平均
        mn = Ss[(xs > 1.2) & (xs < 3.6)].min(1)
        out[mode] = dict(unloaded=float((mn < 0.1).mean()), p10=float(np.percentile(mn, 10)),
                         swing_max=float(np.abs(sw).max()))
    return out


def main():
    cfg = cfg_for_pose(CONFIG, "ready", 0.0)       # 常规校核：准备姿态、空载
    with open("output/chassis.xml", "w", encoding="utf-8") as f:
        f.write(build_xml(cfg))

    ref = analytic_tip(cfg)
    print("=" * 60)
    print(f"整机质量 {ref['M']:.1f} kg，整机质心高度 {ref['h']*1000:.0f} mm")
    print("=" * 60)

    print("\n[1] 静态倾翻角（纯几何，不考虑打滑）")
    for d, name in [("lateral", "侧向"), ("forward", "车头朝下坡"), ("backward", "车头朝上坡")]:
        ang = search_tip_angle(cfg, d)
        print(f"  {name:6s}: 仿真 {ang:5.1f}°   解析 {ref[d]:5.1f}°")
    sa = slide_angle(cfg)
    print(f"  刹车状态下整车开始下滑的坡度：{sa:.1f}°（摩擦系数 {cfg['friction']}，"
          f"解析 atan(μ) = {math.degrees(math.atan(cfg['friction'])):.1f}°）")

    print(f"\n[2] 爬坡 {cfg['slope_deg']}°")
    c = climb_test(cfg)
    print(f"  目标速度 {c['v_target']:.2f} m/s，实际 {c['v_achieved']:.2f} m/s  -> "
          f"{'通过' if c['ok'] else '不通过（力矩不足或打滑）'}")
    print(f"  单轮力矩 平均 {c['torque_mean']:.1f} / 峰值 {c['torque_peak']:.1f} N·m"
          f"（解析 {c['torque_analytic']:.1f}，上限 {c['torque_limit']:.1f}）")
    print(f"  前轮载荷最低占比 {c['front_load_min']*100:.0f}%（<15% 时转向和抓地变差）")

    print(f"\n[3] 平地 {cfg['v_max']} m/s 急停")
    a_tip = G * (cfg["wheelbase"] / 2 - composite_com(cfg)[1]) / ref["h"]
    print(f"  解析：后轮离地的临界减速度 g·(L/2)/h = {a_tip:.2f} m/s²")
    for label, dec in [("满力矩硬刹", None), ("斜坡 3 m/s²", 3.0), ("斜坡 5 m/s²", 5.0)]:
        b = brake_test(cfg, decel=dec)
        warn = "  ⚠ 后轮离地，有前翻风险" if b["rear_load_min"] < 0.03 else ""
        print(f"  {label:9s}: 距离 {b['stop_distance']*100:5.1f} cm，平均减速 {b['decel_avg']:.2f} m/s²，"
              f"后轮最低载荷 {b['rear_load_min']*100:3.0f}%，俯仰 {b['pitch_max']:.2f}°{warn}")

    print(f"\n[4] 越台阶")
    s = step_test(cfg)
    print(f"  {cfg['step_height']*1000:.0f} mm 台阶：{'通过' if s['passed'] else '未通过'}，"
          f"峰值单轮力矩 {s['torque_peak']:.1f} N·m")
    print(f"  可通过的最大台阶高度约 {max_step_height(cfg)*1000:.0f} mm"
          f"（刚性轮主要受摩擦系数限制，增大轮径比加力矩更有效）")

    print("\n（机械臂各姿态 × 负载的倾覆分析见 tip_analysis.py）")

    print("\n[4c] 后桥摆动悬挂 开/关 对比（过 40 mm 单边斜坡台 + 侧倾临界角）")
    sus = suspension_test(cfg)
    for mode, name in (("on", "悬挂开启(摆动桥自由)"), ("off", "悬挂关闭(锁死)    ")):
        r_ = sus[mode]
        print(f"  {name}: 有轮子卸载(<10%载荷)的时间占 {r_['unloaded']*100:3.0f}%，四轮载荷最低 10 分位 {r_['p10']*100:3.0f}%，最大摆角 {r_['swing_max']:.1f}°")
    for mode, name in (("on", "开启"), ("off", "关闭")):
        print(f"  悬挂{name}：侧向倾翻角 {search_tip_angle(dict(cfg, suspension=mode), 'lateral'):.1f}°")

    print("\n[5] 参数扫描：轮距 × 车体质心高度 → 侧倾临界角")
    tracks = [0.70, 0.80, 0.85, 0.90, 1.00]
    heights = [0.35, 0.45, 0.55, 0.70]
    rows, out = sweep(cfg, tracks, heights)
    print("  B\\h  " + "".join(f"{h:>8.2f}" for h in heights))
    for B in tracks:
        print(f"  {B:.2f}" + "".join(f"{r[2]:>7.1f}°" for r in rows if r[0] == B))
    print(f"  已保存 {out}")
    print("\n模型已保存 chassis.xml，可用 python -m mujoco.viewer --mjcf chassis.xml 查看")


if __name__ == "__main__":
    main()