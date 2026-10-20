"""烟箱与夹具参数（整套卸货规划、碰撞、仿真共用；改这里即可）。

烟箱：泰山(红将军) 整件烟箱 长 451 x 宽 243 x 高 568（用户给定，按 mm），单箱 17 kg。
装车方式（假设）：长 451 沿车厢纵向（垂直箱墙，即“深”），宽 243 沿车厢横向，高 568 竖直。
整车厢截面 3 x 3 m：横向 12 列 x 243 mm = 2.916 m，竖向 5 层 x 568 mm = 2.84 m。
"""
BOX_D, BOX_W, BOX_H = 0.451, 0.243, 0.568   # 深（垂直箱墙方向）x 宽（沿车厢横向）x 高 [m]
BOX_MASS = 17.0                              # 单箱质量 [kg]
MAX_GROUP = 3                                # 一次最多同时抓取的箱数（同层相邻的几列）

# 箱墙：N_COLS 列 x N_LAYERS 层，前表面 x=1.402
WALL_FRONT = 1.402
N_COLS = int(3.0 // BOX_W)
N_LAYERS = int(3.0 // BOX_H)
_y0 = -N_COLS * BOX_W / 2
COL_EDGES = [_y0 + BOX_W * k for k in range(N_COLS + 1)]
COLS = [(COL_EDGES[i] + COL_EDGES[i + 1]) / 2 for i in range(N_COLS)]
COL_W = [BOX_W] * N_COLS
LAYERS = [BOX_H / 2 + BOX_H * k for k in range(N_LAYERS)]
FRONT_LAYER_MIN = 3                          # 层号 >= 3（上面两层）优先用正面吸+底托；更低的层优先顶吸（两种方式都会试）

# 夹具（J7/ee 系：z 向前，x 向下）
PLATE_FRONT_HALF = BOX_H / 2 - 0.005         # 前吸时吸盘板半高（不超过箱子高度，免得放箱时比箱底还低撞到辊床）
PLATE_TOP_HALF = BOX_D / 2 - 0.005           # 顶吸时吸盘板沿 x 的半深（不超过箱深）
PLATE_HALF_A = 0.30                          # 吸盘板在 ee x 方向的半尺寸（前吸时=竖直半高，顶吸时=沿 x 的半深）
PLATE_HALF_W = 0.42                          # 吸盘板半宽，可覆盖 3 个箱子（3*0.243=0.73 m）
STROKE = 0.20                                # 吸盘板伸缩行程（也是取箱预取点离箱面/箱顶的距离）[m]
TRAY_T = 0.012                               # 底托厚度
TRAY_HALF_L = BOX_D / 2                      # 底托半长（沿 ee z），托住箱底前半段
TRAY_X = BOX_H / 2 + TRAY_T / 2              # 底托中心在 ee x 的位置（箱子中心与吸盘板中心对齐，底托托在箱底下面）
TRAY_Z_HOLD = 0.222                          # 托着箱子时底托中心在 ee 系的 z（与箱子重叠一半深度）
TRAY_RETRACT = BOX_D / 2 + 0.17              # 空手时底托回缩量：尖端缩到吸盘面后面
PULL_OUT = BOX_D + 0.04                      # 取出箱子后总共要离开箱墙的距离（吸盘缩回 STROKE + 臂再后退）
TOP_LIFT = BOX_H + 0.05                      # 顶吸取出箱子要向上提起的高度
PLACE_PUSH = 0.0                             # 已取消吸盘板/底托的相对运动
TRAY_PLACE_RETRACT = BOX_D / 2 + 0.03 - PLACE_PUSH   # 放箱时底托回缩量：与前推量之和 >= 底托压着箱底的长度，箱子就脱离底托

# 理料机构（辊床）：整体后移，朝后放箱时 J7 离 S 轴需要足够远；改成可放 3 箱的平辊床
BED_SHIFT = -0.50
PLACE_X, PLACE_Y = -0.85 + BED_SHIFT, 0.0    # 出箱位中心，base 系
MAST_XY = (-0.32, 0.0)                       # 相机立柱在转盘上的位置（base 系）：原页面在右后方 (-0.26,-0.34)，会和肘部连杆打架，挪到后中间
MAST_TOP = 0.80                              # 相机立柱顶高（base 系，原 1.12）：再高会和够顶层靠边箱子的前臂打架
BED_TOP = 0.73                               # 辊床上表面高度（世界系）
BED_HALF_W = 0.55


def groups_for_layer(l):
    """一层的取箱分组：先中间后两边，每组 <=3 列"""
    mid = N_COLS // 2
    cols = list(range(N_COLS))
    first = [mid - 2, mid - 1, mid]                     # 中间三列
    groups = [first]
    right = [c for c in cols if c > mid]
    left = [c for c in cols if c < mid - 2]
    while right:
        groups.append(right[:3]); right = right[3:]
    while left:
        groups.append(left[-3:][::1]); left = left[:-3]
    return [g for g in groups if g]


def mode_for_layer(l):
    return "front" if l >= FRONT_LAYER_MIN else "top"


def group_geometry(cols):
    """一组箱子的横向中心 y 和总宽"""
    lo, hi = COL_EDGES[min(cols)], COL_EDGES[max(cols) + 1]
    return (lo + hi) / 2, hi - lo
