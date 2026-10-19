"""安川 GP110B 臂的 MuJoCo 外观 / 质量 / 关节（由 a_model.get_visuals 在 arm_type="abb" 时调用）。
夹具（吸盘板 + 底托板）、相机立柱沿用 chassis_A_3d.html 里的网格；臂的各连杆按尺寸图用基本体搭建。"""
import math
import numpy as np
import arm_abb as AB


def step_geoms(i, L):
    """STEP 里属于第 i 个关节的零件：登记 mesh 资源并返回 geom 字符串（网格文件在 meshes/ 下，相对关节零位原点）"""
    out = []
    for p in AB.STEP_PARTS:
        if p["link"] != i:
            continue
        stem = p["file"][:-4]
        tag = f'name="m_{stem}"'
        if not any(tag in a for a in L["assets"]):
            L["assets"].append(f'<mesh name="m_{stem}" file="{p["file"]}" scale="0.001 0.001 0.001"/>')
        c = p["color"]
        out.append(f'<geom {L["GEOM"]} type="mesh" mesh="m_{stem}" rgba="{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} 1"/>')
    return out


def abb_visuals(L):
    import a_model as A
    rob, mesh, link_geoms = L["rob"], L["mesh"], L["link_geoms"]
    live, pose, arm_mass, GEOM = L["live"], L["pose"], L["arm_mass"], L["GEOM"]
    AB.JOINTS[1]["o"][0] = float(L["j2_shift"]) * 1000.0               # J2（L 轴）在 S 轴前方的距离
    base_m = np.array(rob["base"], float) / 1000.0 - np.array([0, 0, A.Z0 / 1000.0])
    q = AB.preset_theta(pose)
    Ts, Tee = AB.fk_chain(q, base_m)

    # ---- 夹具网格（夹具系，沿用页面）----
    g_slide, gcen = link_geoms(rob["slide"], "SL")
    g_tray, _ = link_geoms(rob["tray"], "TR")
    tool_loc = sum(a * c for a, c in gcen) / max(sum(a for a, _ in gcen), 1e-9) / 1000.0
    # ---- 相机立柱：页面里它挂在 J1 转盘上（J1 节点比转盘面高 36 mm）----
    g_mf, _ = link_geoms(rob["mastFixed"], "MF")
    g_mr, _ = link_geoms(rob["mastRot"], "MR")
    A_, C_ = 0.0, np.zeros(3)
    for lst, sh in ((rob["mastFixed"], np.zeros(3)), (rob["mastRot"], np.array(rob["mastO"], float))):
        for p in lst:
            v, f = A._arrays(p)
            if len(f) < 4:
                continue
            a_, c_ = A._area_centroid(v, f)
            A_ += a_; C_ += a_ * (c_ + sh)
    mast_loc = C_ / max(A_, 1e-9) / 1000.0 + np.array([0, 0, 0.036])
    masses = AB.link_masses(arm_mass, tool_loc, mast_loc)

    act, q0, names = [], [], []
    xml = f'<body name="arm_base" pos="{base_m[0]:.4f} {base_m[1]:.4f} {base_m[2]:.4f}">\n'
    if AB.STEP_PARTS:
        xml += "\n".join(step_geoms(0, L)) + "\n"
    for i, (j, th) in enumerate(zip(AB.JOINTS, q), start=1):
        o = np.array(j["o"]) / 1000.0
        ax = {"x": "1 0 0", "y": "0 1 0", "z": "0 0 1"}[j["a"]]
        m_i, c_i = masses[i]
        if live:
            lo, hi = AB.LIM_DEG[i - 1]
            if lo == hi:
                lo, hi = lo - 0.05, hi + 0.05
            xml += (f'<body name="arm_j{i}" pos="{o[0]:.4f} {o[1]:.4f} {o[2]:.4f}">\n'
                    f'<joint name="arm_j{i}" type="hinge" axis="{ax}" range="{lo:.2f} {hi:.2f}" limited="true" damping="3000" armature="20"/>\n'
                    f'<inertial pos="{c_i[0]:.4f} {c_i[1]:.4f} {c_i[2]:.4f}" mass="{m_i:.2f}" '
                    f'diaginertia="{m_i*0.12:.2f} {m_i*0.12:.2f} {m_i*0.12:.2f}"/>\n')
            act.append(f'<position name="a_j{i}" joint="arm_j{i}" kp="1e7" kv="6e4"/>')
            q0.append(float(th)); names.append(f"a_j{i}")
        else:
            xml += f'<body name="arm_j{i}" pos="{o[0]:.4f} {o[1]:.4f} {o[2]:.4f}" axisangle="{ax} {math.degrees(th):.3f}">\n'
        xml += "\n".join(step_geoms(i, L) if AB.STEP_PARTS else AB.link_geoms(i)) + "\n"
        if i == 1:
            xml += '<body name="mastroot" pos="0 0 0.036">\n' + g_mf + "\n"
            mo = rob["mastO"]
            xml += f'<body name="mast" pos="{mo[0]/1000:.4f} {mo[1]/1000:.4f} {mo[2]/1000:.4f}" axisangle="0 0 1 180">\n{g_mr}\n</body>\n</body>\n'
        if i == 7:                                    # 法兰 + 夹具
            xml += ('<body name="ee" pos="0.2 0 0" xyaxes="0 0 -1 0 1 0">\n'
                    f'<body name="ee_slide" pos="0 0 0">\n{g_slide}\n</body>\n<body name="ee_tray" pos="0 0 0">\n{g_tray}\n</body>\n')
            if live:
                xml += ('<body name="payload" pos="0.069 0 0.372">\n<inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>\n'
                        f'<geom name="payload_box" {GEOM} type="box" size="0.125 0.2 0.15" rgba="0.80 0.64 0.42 0"/>\n</body>\n')
            xml += "</body>\n"
    xml += "</body>\n" * 8                           # 7 个关节体 + arm_base
    L["base_xml"].append(xml)

    M, com = 0.0, np.zeros(3)
    for i, T in enumerate(Ts, start=1):
        m_i, c_i = masses[i]
        M += m_i; com += m_i * (T @ np.append(c_i, 1))[:3]
    return dict(
        assets="\n    ".join(L["assets"]),
        base="\n      ".join(L["base_xml"]),
        rear="\n        ".join(L["rear_xml"]),
        pin_z=(A.PIN_Z - A.Z0) / 1000,
        wheels={n: "\n        ".join(v) for n, v in L["wheel_xml"].items()},
        steer={n: "\n        ".join(v) for n, v in L["steer_xml"].items()},
        wheel_xy={n: (c[0] / 1000, c[1] / 1000) for n, c in L["wheel_c"].items()},
        arm_com=com / M, ee_pos=Tee[:3, 3].copy(), tool_pos=(Tee @ np.append(tool_loc, 1))[:3],
        holding=False, stats=L["stats"],
        arm_act="\n    ".join(act), arm_q0=q0, arm_act_names=names,
    )


def grip_mast_locs():
    """夹具质心（夹具系，m）和相机立柱质心（J1 系，m）——给 arm_kin 的质量模型用"""
    import a_model as A
    parts, rob = A._load()
    def cen(lst, sh=None):
        Ar, C = 0.0, np.zeros(3)
        for p in lst:
            v, f = A._arrays(p)
            if len(f) < 4:
                continue
            a_, c_ = A._area_centroid(v, f); Ar += a_; C += a_ * (c_ + (0 if sh is None else sh))
        return Ar, C
    ga, gc = cen(rob["slide"])
    ma, mc = cen(rob["mastFixed"])
    ra, rc = cen(rob["mastRot"], np.array(rob["mastO"], float))
    return gc / max(ga, 1e-9) / 1000.0, (mc + rc) / max(ma + ra, 1e-9) / 1000.0 + np.array([0, 0, 0.036])
