import os, sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
"""成组卸货的节拍优化：扫 (加速时间, 速度比例)，在最差坡向 10° 坡上看 节拍 / 倾覆裕度 / 倾斜 / 跟踪误差。
运行：python tools/cycle_opt3.py [组数=4] [坡向...]"""
import validate_plan3 as V

NP = int(sys.argv[1]) if len(sys.argv) > 1 else 4
HEADINGS = [float(x) for x in sys.argv[2:]] or [90.0, 180.0]

if __name__ == "__main__":
    print("加速时间 速度比例 | 坡向 | 节拍 s/箱 | 卸出 | 最小裕度 cm | 最大倾斜° | 跟踪误差° | 碰撞复查", flush=True)
    for acc, sc in ((1.5, 0.6), (1.0, 0.8), (0.8, 0.9), (0.6, 1.0), (0.5, 1.0)):
        for phi in HEADINGS:
            r = V.run(10.0, phi, NP, True, acc, sc, "off", quiet=True)
            print("  %.1f s   %.1f   | %3.0f° | %6.1f | %d/%d | %6.1f | %5.2f | %5.2f | %d/%d" % (
                acc, sc, phi, r["per_box"], r["done"], r["boxes"], r["margin"], r["tilt"], r["err"], r["ncol"], r["ncheck"]), flush=True)
