"""连续轨迹：把规划出的关节空间折线（路点）变成"不停顿"的平滑运动。

1. smooth_chain：在路点拐角处用二次 Bézier 圆角倒圆（半径逐级减小直到碰撞检测通过，不通过的拐角保留为停点），
   输出稠密的无碰撞路径（在规划阶段做，需要 Collider）；
2. Trajectory：对稠密路径做时间参数化（数值 TOPP：各关节速度 <= vmax、加速度 <= amax，沿路径前向+后向积分），
   再用滑动平均把加速度的"急变"磨平（降低冲击、减小底盘晃动）。运行时只用 numpy，很快。
"""
import math
import numpy as np


def _dedupe(P, eps=1e-6):
    out = [np.asarray(P[0], float)]
    for q in P[1:]:
        q = np.asarray(q, float)
        if np.max(np.abs(q - out[-1])) > eps:
            out.append(q)
    return out


def densify(a, b, step=0.01):
    n = max(1, int(math.ceil(np.max(np.abs(b - a)) / step)))
    return [a + (b - a) * (k / n) for k in range(1, n + 1)]


def smooth_chain(waypoints, ok_fns, ratios=(0.45, 0.25, 0.12, 0.05), step=0.01):
    """waypoints：关节空间折线（含起点终点）；ok_fns：每个内部拐角一个 q->bool 的碰撞检测（长度 = len(waypoints)-2，
    可以是单个函数，表示全部拐角共用）。返回 (稠密路径 N x 7, 停点下标列表)。起点和终点总是停点。"""
    P = _dedupe(waypoints)
    n = len(P)
    if n == 1:
        return np.array(P), [0]
    if n == 2:
        dense = [P[0]] + densify(P[0], P[1], step)
        return np.array(dense), [0, len(dense) - 1]
    if callable(ok_fns):
        ok_fns = [ok_fns] * (n - 2)
    seg = [float(np.linalg.norm(P[i + 1] - P[i])) for i in range(n - 1)]
    cuts = []                                   # 每个内部拐角：(A, 控制点, B, 是否倒圆成功)
    for i in range(1, n - 1):
        a, b, c = P[i - 1], P[i], P[i + 1]
        ua, uc = (a - b) / seg[i - 1], (c - b) / seg[i]
        res = None
        for r in ratios:
            d = r * min(seg[i - 1], seg[i])
            A, B = b + ua * d, b + uc * d
            arc = [(1 - t) ** 2 * A + 2 * t * (1 - t) * b + t ** 2 * B for t in np.linspace(0, 1, 15)]
            if all(ok_fns[i - 1](q) for q in arc):
                res = (A, b, B)
                break
        cuts.append(res)
    dense, stops = [P[0]], [0]
    cur = P[0]
    for i in range(1, n - 1):
        res = cuts[i - 1]
        if res is None:                         # 倒圆失败：保留尖角，必须在此停一下
            dense += densify(cur, P[i], step); cur = P[i]
            stops.append(len(dense) - 1)
        else:
            A, ctrl, B = res
            dense += densify(cur, A, step)
            # 沿 Bézier 采样（密度按弧长）
            L = np.linalg.norm(A - ctrl) + np.linalg.norm(B - ctrl)
            m = max(6, int(math.ceil(L / step * 0.5)))
            for t in np.linspace(0, 1, m)[1:]:
                dense.append((1 - t) ** 2 * A + 2 * t * (1 - t) * ctrl + t ** 2 * B)
            cur = B
    dense += densify(cur, P[-1], step)
    stops.append(len(dense) - 1)
    return np.array(dense), stops


class Trajectory:
    """稠密路径 + 速度/加速度限制 → q(t)、q̇(t)。"""

    def __init__(self, dense, vmax, amax, stops=None, smooth_s=0.20, dt=0.002):
        Q = np.asarray(dense, float)
        self.dt = dt
        if len(Q) < 2 or np.max(np.abs(Q[-1] - Q[0])) < 1e-6 and len(Q) < 3:
            self.T = 0.0
            self.q_end = Q[-1] if len(Q) else None
            self.qs = Q[:1]
            self.qd = np.zeros_like(Q[:1])
            return
        vmax, amax = np.asarray(vmax, float), np.asarray(amax, float)
        dq = np.diff(Q, axis=0)
        ds = np.max(np.abs(dq) / vmax, axis=1)                     # 时间度量：全速下的最短时间
        keep = np.concatenate([[True], ds > 1e-9])
        Q = Q[keep]
        if stops is not None:                                       # 去重后重新映射停点（取最近的保留点）
            idx_map = np.cumsum(keep) - 1
            stops = sorted(set(int(idx_map[min(s, len(idx_map) - 1)]) for s in stops))
        dq = np.diff(Q, axis=0)
        ds = np.max(np.abs(dq) / vmax, axis=1)
        s = np.concatenate([[0.0], np.cumsum(ds)])
        N = len(Q)
        # 导数（对 s）：一阶 q'，二阶 q''
        d1 = np.zeros_like(Q); d2 = np.zeros_like(Q)
        d1[1:-1] = (Q[2:] - Q[:-2]) / (s[2:] - s[:-2])[:, None]
        d1[0] = dq[0] / ds[0]; d1[-1] = dq[-1] / ds[-1]
        v_half = dq / ds[:, None]                                    # 每段的 q'
        d2[1:-1] = (v_half[1:] - v_half[:-1]) / (0.5 * (ds[1:] + ds[:-1]))[:, None]
        # 速度上限：|q'|·ṡ <= vmax 由度量保证 ṡ<=1；曲率限制 |q''|·ṡ² <= amax
        vlim = np.ones(N)
        with np.errstate(divide="ignore", invalid="ignore"):
            cur = np.where(np.abs(d2) > 1e-9, amax / np.abs(d2), np.inf)
        vlim = np.minimum(vlim, np.sqrt(np.min(cur, axis=1)) * 0.9)
        if stops is None:
            stops = [0, N - 1]
        for k in stops:
            vlim[k] = 0.0
        vlim[0] = vlim[-1] = 0.0

        def sddot_range(k, v2):
            lo, hi = -np.inf, np.inf
            for i in range(Q.shape[1]):
                a1 = d1[k, i]
                if abs(a1) < 1e-9:
                    continue
                u = (amax[i] - d2[k, i] * v2) / a1
                l = (-amax[i] - d2[k, i] * v2) / a1
                if a1 > 0:
                    hi = min(hi, u); lo = max(lo, l)
                else:
                    hi = min(hi, l); lo = max(lo, u)
            return lo, hi

        v2 = np.minimum(vlim ** 2, 1.0)
        for k in range(N - 1):                                       # 前向（加速）
            lo, hi = sddot_range(k, v2[k])
            hi = max(hi, 0.0)
            v2[k + 1] = min(v2[k + 1], v2[k] + 2 * hi * ds[k])
        for k in range(N - 2, -1, -1):                               # 后向（减速）
            lo, hi = sddot_range(k + 1, v2[k + 1])
            lo = min(lo, 0.0)
            v2[k] = min(v2[k], v2[k + 1] - 2 * lo * ds[k])
        v = np.sqrt(np.maximum(v2, 0.0))
        t = np.zeros(N)
        for k in range(N - 1):
            vm = max(0.5 * (v[k] + v[k + 1]), 1e-3)
            # 起/终点速度为 0 的段用匀加速公式避免除零：dt = 2 ds / (v0+v1)，v0+v1 很小时取 sqrt(2 ds / a)
            if v[k] + v[k + 1] < 1e-3:
                t[k + 1] = t[k] + math.sqrt(2 * ds[k] / max(float(np.min(amax / np.max(np.abs(d1[k]) + 1e-9))), 1e-3))
            else:
                t[k + 1] = t[k] + ds[k] / vm
        T = float(t[-1])
        tt = np.arange(0.0, T + dt, dt)
        qt = np.empty((len(tt), Q.shape[1]))
        for i in range(Q.shape[1]):
            qt[:, i] = np.interp(tt, t, Q[:, i])
        if smooth_s > 0:                                             # 滑动平均：把加速度的突变磨成斜坡
            w = max(1, int(round(smooth_s / dt)))
            ker = np.ones(w) / w
            pad = w // 2
            qs = np.empty_like(qt)
            for i in range(Q.shape[1]):
                x = np.concatenate([np.full(pad, qt[0, i]), qt[:, i], np.full(w - pad, qt[-1, i])])
                qs[:, i] = np.convolve(x, ker, mode="valid")[: len(qt)]
            # 追加尾部，让位置收敛到终点
            tail = int(round(smooth_s / dt))
            ext = np.empty((len(qt) + tail, Q.shape[1]))
            for i in range(Q.shape[1]):
                x = np.concatenate([np.full(pad, qt[0, i]), qt[:, i], np.full(w + tail, qt[-1, i])])
                ext[:, i] = np.convolve(x, ker, mode="valid")[: len(qt) + tail]
            qt = ext
            qt[-1] = Q[-1]
        acc = np.gradient(np.gradient(qt, dt, axis=0), dt, axis=0)
        r = float(np.max(np.abs(acc) / amax))
        if r > 1.0:                                                  # 数值积分/平滑的残差：整体拉长时间，让加速度不超限
            lam = math.sqrt(r) * 1.01
            tt0 = np.arange(len(qt)) * dt
            tt2 = np.arange(0.0, tt0[-1] * lam + dt, dt)
            qt = np.stack([np.interp(tt2 / lam, tt0, qt[:, i]) for i in range(qt.shape[1])], axis=1)
        self.qs = qt
        self.qd = np.gradient(qt, dt, axis=0)
        self.T = (len(qt) - 1) * dt
        self.q_end = Q[-1]

    def sample(self, t):
        if self.T <= 0:
            return self.qs[0] if len(self.qs) else None, np.zeros(7)
        k = min(int(t / self.dt), len(self.qs) - 1)
        return self.qs[k], self.qd[k]


if __name__ == "__main__":
    # 小自测：3 个路点
    vmax = np.radians([140, 110, 110, 130, 175, 175, 255]) * 0.9
    amax = vmax / 0.8
    P = [np.zeros(7), np.radians([40, 30, 10, 0, 20, 0, 0]), np.radians([60, 10, 40, 10, 0, 30, 0])]
    dense, stops = smooth_chain(P, lambda q: True)
    tr = Trajectory(dense, vmax, amax, stops)
    print("稠密点", len(dense), "停点", stops, "时长 %.2f s" % tr.T, "最大速度比", float(np.max(np.abs(tr.qd) / vmax)),
          "最大加速度比", float(np.max(np.abs(np.gradient(tr.qd, tr.dt, axis=0)) / amax)))
