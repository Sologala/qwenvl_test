
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# ===============================
# 基础工具
# ===============================


def project_point(Xw, R, t, K):
    """世界点投影到像素"""
    Xc = R @ Xw + t
    x, y, z = Xc
    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]
    return np.array([u, v]), z


def make_camera_pose(t):
    """简单相机运动：沿 x 轴移动"""
    R = np.eye(3)
    T = np.array([0.1 * t, 0.0, 0.0])
    return R, T


# ===============================
# 1. 生成 GT 轨迹（匀速）
# ===============================

def generate_gt(T, dt):
    X0 = np.array([0.0, 0.0, 20.0])
    V = np.array([0.2, 0.1, -0.1])
    Xs = np.array([X0 + V * t * dt for t in range(T)])
    return Xs, V


# ===============================
# 2. 生成观测 box
# ===============================

def generate_observations(Xs, K, H, noise_std=1.0):
    obs = []
    Rs, Ts = [], []

    for t, X in enumerate(Xs):
        R, T = make_camera_pose(t)
        (u, v), z = project_point(X, R, T, K)

        h = K[1, 1] * H / z
        w = 0.5 * h

        # 加噪声
        u += np.random.randn() * noise_std
        v += np.random.randn() * noise_std
        h += np.random.randn() * noise_std * 1.0

        obs.append([u, v, w, h])
        Rs.append(R)
        Ts.append(T)

    return np.array(obs), Rs, Ts


# ===============================
# 3. BA 残差函数
# ===============================

def residual(params, obs, K, Rs, Ts, dt, H):
    Tn = len(obs)
    Xs = params[:3 * Tn].reshape(Tn, 3)
    V = params[3 * Tn:3 * Tn + 3]

    res = []

    for t in range(Tn):
        R, T = Rs[t], Ts[t]
        Xc = R @ Xs[t] + T
        x, y, z = Xc

        # 位置投影
        u = K[0, 0] * x / z + K[0, 2]
        v = K[1, 1] * y / z + K[1, 2]

        res.append(u - obs[t, 0])
        res.append(v - obs[t, 1])

        # 尺寸约束（高度）
        h_pred = K[1, 1] * H / z
        res.append(h_pred - obs[t, 3])

        # 匀速运动约束
        if t < Tn - 1:
            res.extend(Xs[t + 1] - (Xs[t] + V * dt))

    return np.array(res)


# ===============================
# 4. 主函数
# ===============================

def main():
    np.random.seed(0)

    # 参数
    T = 100
    dt = 0.2
    H = 2.0  # 目标真实高度

    K = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ])

    # GT
    X_gt, V_gt = generate_gt(T, dt)

    # 观测
    obs, Rs, Ts = generate_observations(X_gt, K, H)

    # 初值（故意给得很烂）
    X_init = X_gt + np.random.randn(*X_gt.shape) * 10.0
    V_init = np.array([0.0, 0.0, 0.0])

    x0 = np.hstack([X_init.flatten(), V_init])

    # 优化
    result = least_squares(
        residual,
        x0,
        args=(obs, K, Rs, Ts, dt, H),
        verbose=2,
        loss="huber"
    )

    X_opt = result.x[:3 * T].reshape(T, 3)
    V_opt = result.x[3 * T:3 * T + 3]

    print("\nGT velocity :", V_gt)
    print("EST velocity:", V_opt)

    # ===============================
    # 5. 可视化
    # ===============================

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(X_gt[:, 0], X_gt[:, 2], 'g-', label="GT")
    plt.plot(X_opt[:, 0], X_opt[:, 2], 'r--', label="Optimized")
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.legend()
    plt.title("Top View (XZ)")

    plt.subplot(1, 2, 2)
    plt.plot(X_gt[:, 2], 'g-', label="GT Z")
    plt.plot(X_opt[:, 2], 'r--', label="Optimized Z")
    plt.title("Depth over Time")
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig("trajectory_result.png", dpi=150)
    print("Saved to trajectory_result.png")


if __name__ == "__main__":
    main()
