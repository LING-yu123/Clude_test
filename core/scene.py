"""卸货场景（纯显示）：车厢（截面 3×3 m，长 5.5 m）+ 前方箱墙（烟箱）+ 后部理料机构上的"已放置箱子"。
尺寸取自 chassis_A_3d.html。世界系：地面 z=0，车厢 x∈[-2.5, 3.0]，y∈[-1.5, 1.5]，z∈[0, 3.0]。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from box_spec import WALL_FRONT, BOX_D, BOX_W, BOX_H, COL_EDGES, COLS, LAYERS, PLACE_X, BED_TOP
OFF = 'contype="0" conaffinity="0" mass="0"'


def scene_xml():
    """放在 worldbody 里的 mocap 体 'scene'（随倾斜台和机器人横移一起动）"""
    o = ['<body name="scene" mocap="true">']
    glass = 'rgba="0.55 0.65 0.78 0.10"'
    o.append(f'<geom {OFF} type="box" pos="0.25 1.505 1.5" size="2.75 0.005 1.5" {glass}/>')       # 左壁
    o.append(f'<geom {OFF} type="box" pos="0.25 -1.505 1.5" size="2.75 0.005 1.5" {glass}/>')      # 右壁
    o.append(f'<geom {OFF} type="box" pos="0.25 0 3.005" size="2.75 1.5 0.005" {glass}/>')         # 顶
    o.append(f'<geom {OFF} type="box" pos="3.005 0 1.5" size="0.005 1.5 1.5" rgba="0.55 0.65 0.78 0.25"/>')  # 前壁
    # 前排：8 列 × 8 层
    for c in range(len(COLS)):
        w = COL_EDGES[c + 1] - COL_EDGES[c]
        for l in range(len(LAYERS)):
            shade = 0.80 - 0.04 * ((c + l) % 2)
            o.append(f'<geom name="wb_{c}_{l}" {OFF} type="box" pos="{WALL_FRONT + BOX_D/2 + 0.0005} {COLS[c]} {LAYERS[l]}" '
                     f'size="{BOX_D/2 - 0.002} {w/2 - 0.004} {BOX_H/2 - 0.002}" rgba="{shade} {shade*0.8:.2f} {shade*0.52:.2f} 1"/>')
    # 后面几排（整块表示，高 2.85）
    xb = WALL_FRONT + BOX_D
    o.append(f'<geom {OFF} type="box" pos="{(xb + 3.0) / 2} 0 1.425" size="{(3.0 - xb) / 2} 1.5 1.425" rgba="0.72 0.57 0.38 1"/>')
    o.append('</body>')
    return "\n    ".join(o)


def bed_box_xml(r=0.075):
    """后部理料机构（辊床）上"已放置"的箱子：3 个位置（base 系，初始隐藏），运行时由播放器摆位"""
    out = []
    for j in range(3):
        name = "bed_box" if j == 0 else f"bed_box_{j}"
        out.append(f'<geom name="{name}" {OFF} type="box" pos="{PLACE_X} {(j - 1) * BOX_W} {BED_TOP + BOX_H/2 - r:.4f}" '
                   f'size="{BOX_D/2} {BOX_W/2} {BOX_H/2}" rgba="0.80 0.64 0.42 0"/>')
    return "\n      ".join(out)
