# 重载移动卸货机器人 MuJoCo 仿真

**设计优化结论见 [设计优化报告.md](设计优化报告.md)**(最佳底盘质量/轮距/轴距/轮径、电机选型数据、卸货动作优化)。

四舵轮底盘(1000 kg) + 安川 GP110B 七轴臂(820 kg 含立柱、夹具) + 烟箱(50 kg)。所有命令都在本目录运行(双击 .bat 即可)。

## 直接运行(根目录的 .bat)
| 文件 | 作用 |
|---|---|
| `tip_view.bat` | **主程序**:MuJoCo 可视化。U=自动卸货,1-7+方向键=手动点动关节,J=点动步长,R=回预设姿态,`[ ]`=坡度,M=悬挂,`= -`=负载 |
| `validate.bat` | 无界面动力学验证:`validate.bat 坡度 坡向 箱数 是否开车`,如 `validate.bat 0 0 8 1` |
| `replan.bat` | 重新规划卸货动作(很慢,多进程),结果写入 `data/unload_plan2.json` |
| `view_susp.bat` | 悬挂摆动单独演示 |
| `view_menu.bat` | 菜单:选 1-8 分别看 卸货(平地/侧坡/上坡)、爬坡、最高速度、台阶、间隙 |
| `slope_view.bat` | 爬坡可视化:`slope_view.bat 10 0.3 on 50 0.6`(坡度, 车速, 悬挂, 负载, 摩擦);B=抱闸/行驶 |
| `obstacle_view.bat` | 越障可视化:`obstacle_view.bat step 20 120 on 50 0.1`(台阶/间隙, mm, 单轮力矩, 悬挂, 负载, 车速;默认 0.1 m/s) |

## 目录
| 目录 | 内容 |
|---|---|
| `core/` | 核心代码(被其他脚本导入) |
| `tools/` | 验证 / 分析 / 出图的独立脚本 |
| `build/` | 一次性生成模型数据的脚本(STEP→网格、预设姿态),一般不用再跑 |
| `data/` | 输入与规划结果:STEP 文件、`chassis_A_3d.html`、`unload_plan2.json`(卸货动作)、`abb_presets.json`、`place_cache.json` |
| `gp110b/`, `meshes/` | 由 STEP 生成的连杆网格、质量占比(`links.json`)、碰撞凸包(`collision.json`) |
| `output/` | 运行产生的图片、表格、日志 |
| `archive/` | 旧版/不再使用的文件(旧优化结果、旧看板等) |

## core/ 各文件
| 文件 | 作用 |
|---|---|
| `chassis_sim.py` | **底盘与整机模型**:CONFIG 参数(质量、质心、轮距、轮胎刚度…)、生成 MJCF、悬挂、四舵轮、`cfg_live` |
| `arm_abb.py` | GP110B 参数:关节链、限位、各轴速度、质量分配(含固定底座 BASE) |
| `arm_kin.py` | 正/逆运动学、质心计算 |
| `a_model.py`, `a_model_abb.py` | 机械臂/车体的外观与关节(MuJoCo 几何) |
| `scene.py` | 车厢、箱墙、理料机构 |
| `collision.py` | 真实外形碰撞检测(臂、夹具、箱子 vs 车厢/箱墙/车体) |
| `plan_motion.py` | 无碰撞卸货动作规划(RRT-Connect/RRT*,多进程) |
| `unload_plan.py` | 箱子取放顺序、工位、携带姿态 |
| `unload_player.py` | 执行规划好的动作(最小冲击曲线,**速度/加速度在文件开头 `ACC_TIME`、`speed_scale`**) |
| `steering.py` | 四舵轮运动学与控制 |
| `tip_view.py` | 可视化主程序与按键 |
| `tip_analysis.py` | 倾覆解析计算(支承多边形 + 质心高度) |
| `cycle_test.py` | 倾覆裕度计算等公用函数 |

## tools/
- 设计研究:`design_study.py`(轴距/轮距/质心/配重→倾翻角) · `com_height_study.py`(质心高度、底盘质量) · `wheel_study.py`(轮径) · `traction_study.py`(质心/摩擦→台阶) · `gap_sweep.py`(间隙宽度) · `motor_requirements.py`(电机选型数据) · `cycle_opt.py`(卸货节拍) · `stability_map.py`(卸货全过程静态倾翻角)
- 校核/验证:`capability_check.py`(RockyOne 指标:速度/爬坡/台阶/间隙) · `validate_plan.py`(卸货动力学验证) · `step_robust.py`(台阶稳健性) · `chassis_tune.py`
- 可视化:`slope_view.py` · `obstacle_view.py` · `view_sim.py` · `render.py`
- 旧脚本在 `archive/old_tools/`(用旧臂数据,不再维护)

## 已知
- 越障(台阶/间隙)测试用了专门的接触设置:车轮胶囊代替圆柱(圆柱撞方块棱时接触法向不准)、金字塔摩擦锥、更硬的接触阻抗;结论:20 mm 台阶受摩擦限制,μ=0.6 时临界,μ≥0.9 且单轮力矩≥300 N·m 稳定通过;50 mm 间隙可过。
- 脚本都要在本目录(工程根)下运行;生成的图/表默认写到 `output/`。
- `design_check.py`、`torque_report.py`、`j2_shift_study.py`、`unload_opt.py` 还没按新的臂质量(含底座)更新。
- 底盘已改:轴距 1.11 m、轮距 0.91 m(页面原值 x1.25)、底盘质心 x=-0.15、单轮力矩 200 N·m、后桥摆动阻尼 300;10° 坡上 8 个坡向卸货均不倾覆(`tools/stability_map.py` 静态最小倾翻角 15.8°)。
