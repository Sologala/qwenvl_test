
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
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


def make_camera_pose(t, dt, radius=50.0, height=30.0, angular_velocity=0.05):
    """飞机（相机）在更高平面上绕圈飞行
    Args:
        t: 帧序号
        dt: 时间步长
        radius: 绕圈半径
        height: 飞行高度（z坐标）
        angular_velocity: 角速度
    """
    # 相机在xy平面绕圈，z固定在高处
    time = t * dt
    angle = angular_velocity * time
    
    # 相机位置（绕圈）
    cam_x = radius * np.cos(angle)
    cam_y = radius * np.sin(angle)
    cam_z = height
    
    # 相机朝向目标（船）的方向
    # 简化：相机始终朝向原点（或可以朝向船的位置）
    # 这里使用简单的旋转矩阵，让相机朝向下方和前方
    R = np.eye(3)  # 简化：保持相机坐标系与世界坐标系一致
    # 可以添加更复杂的旋转，让相机朝向目标
    
    T = np.array([cam_x, cam_y, cam_z])
    return R, T


# ===============================
# 1. 生成 GT 轨迹（船在z固定的平面上运动）
# ===============================

def generate_gt(T, dt, z_plane=0.0):
    """生成船的轨迹，船在z固定的平面上运动
    Args:
        T: 总帧数
        dt: 时间步长
        z_plane: 船的z坐标（固定平面）
    """
    X0 = np.array([0.0, 0.0, z_plane])  # 起始位置，z固定
    V0 = np.array([0.3, 0.2, 0.0])  # 初始速度（z方向为0，保持在平面上）
    A = np.array([0.01, 0.008, 0.0])  # 加速度（z方向为0）
    omega = 0.03  # 角速度（用于曲线运动）
    
    Xs = []
    for t in range(T):
        time = t * dt
        # 船在xy平面上的非线性运动（z固定）
        x = X0[0] + V0[0] * time + 0.5 * A[0] * time**2 + 3.0 * np.sin(omega * time)
        y = X0[1] + V0[1] * time + 0.5 * A[1] * time**2 + 3.0 * np.cos(omega * time)
        z = z_plane  # z固定
        Xs.append([x, y, z])
    
    return np.array(Xs), None  # 非线性运动没有固定速度


# ===============================
# 2. 生成观测 box（考虑船的长宽比）
# ===============================

def generate_observations(Xs, K, H, L, W, dt, camera_radius=50.0, camera_height=30.0, 
                          camera_angular_velocity=0.05, noise_std=1.0):
    """生成观测box，考虑船的长宽比
    Args:
        Xs: 船的位置序列
        K: 相机内参
        H: 船的高度
        L: 船的长度（长边）
        W: 船的宽度（短边，L > W）
        dt: 时间步长
        camera_radius: 飞机绕圈半径
        camera_height: 飞机飞行高度
        camera_angular_velocity: 飞机角速度
        noise_std: 噪声标准差
    """
    obs = []
    Rs, Ts = [], []

    for t, X in enumerate(Xs):
        R, T = make_camera_pose(t, dt, camera_radius, camera_height, camera_angular_velocity)
        (u, v), z = project_point(X, R, T, K)

        # 计算船在图像中的尺寸
        # 高度（垂直方向）
        h = K[1, 1] * H / z
        
        # 宽度和长度（需要考虑船的朝向）
        # 简化：假设船的长度方向与运动方向一致
        # 计算运动方向
        if t > 0:
            direction = Xs[t] - Xs[t-1]
            direction[2] = 0  # 只在xy平面
            if np.linalg.norm(direction) > 0.01:
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.array([1.0, 0.0, 0.0])  # 默认方向
        else:
            direction = np.array([1.0, 0.0, 0.0])  # 第一帧默认方向
        
        # 计算相机到船的向量
        cam_to_boat = X - T
        cam_to_boat_normalized = cam_to_boat / np.linalg.norm(cam_to_boat)
        
        # 计算船的长度和宽度在图像中的投影
        # 简化处理：假设相机从上方观察，长度和宽度在图像中的投影
        length_proj = K[0, 0] * L / z
        width_proj = K[0, 0] * W / z
        
        # 取较大的作为宽度（因为box是轴对齐的）
        w = max(length_proj, width_proj) * 0.8  # 稍微缩小以适应轴对齐box

        # 加噪声
        u += np.random.randn() * noise_std
        v += np.random.randn() * noise_std
        h += np.random.randn() * noise_std * 1.0
        w += np.random.randn() * noise_std * 0.5

        obs.append([u, v, w, h])
        Rs.append(R)
        Ts.append(T)

    return np.array(obs), Rs, Ts


# ===============================
# 3. BA 残差函数
# ===============================

def residual(params, obs, K, Rs, Ts, dt, H, boat_z_plane=None):
    Tn = len(obs)
    Xs = params[:3 * Tn].reshape(Tn, 3)

    res = []

    for t in range(Tn):
        R, T = Rs[t], Ts[t]
        Xc = R @ Xs[t] + T
        x, y, z = Xc

        # 保护：确保z为正数（避免除零或负深度）
        # 使用软约束：对过小的z添加惩罚到现有残差中
        z_safe = max(z, 0.1)  # 用于计算的安全值
        
        # 位置投影
        u = K[0, 0] * x / z_safe + K[0, 2]
        v = K[1, 1] * y / z_safe + K[1, 2]

        u_res = u - obs[t, 0]
        v_res = v - obs[t, 1]
        
        # 如果z太小，对位置残差添加惩罚
        if z <= 0.1:
            penalty = 10.0 * (0.1 - z)  # 惩罚项
            u_res += penalty
            v_res += penalty
        
        res.append(u_res)
        res.append(v_res)

        # 尺寸约束（高度）
        h_pred = K[1, 1] * H / z_safe
        h_res = h_pred - obs[t, 3]
        
        # 如果z太小，对高度残差也添加惩罚
        if z <= 0.1:
            h_res += 10.0 * (0.1 - z)
        
        res.append(h_res)

        # 约束：船应该在z固定的平面上
        if boat_z_plane is not None:
            z_constraint = Xs[t, 2] - boat_z_plane
            res.append(z_constraint * 1.0)  # 约束z坐标等于boat_z_plane

        # 非线性运动：平滑性约束（相邻帧之间的位置变化应该平滑）
        # 使用二阶差分来约束平滑性（允许加速度变化，但变化要平滑）
        if t < Tn - 2:
            # 二阶差分：X[t+2] - 2*X[t+1] + X[t] 应该较小
            second_diff = Xs[t + 2] - 2 * Xs[t + 1] + Xs[t]
            # 增加权重，使平滑性约束更强
            res.extend(second_diff * 0.5)  # 从0.1增加到0.5，加强平滑性约束

    return np.array(res)


# ===============================
# 4. 主函数
# ===============================

def main():
    np.random.seed(0)

    # 参数
    T = 40
    dt = 5.0
    H = 2.0  # 船的高度
    L = 8.0  # 船的长度（长边）
    W = 3.0  # 船的宽度（短边，L > W）
    boat_z_plane = 0.0  # 船所在的z平面
    camera_height = 30.0  # 飞机飞行高度
    camera_radius = 50.0  # 飞机绕圈半径
    camera_angular_velocity = 0.1  # 飞机角速度

    K = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ])

    # GT：船在z固定的平面上运动
    X_gt, V_gt = generate_gt(T, dt, z_plane=boat_z_plane)

    # 观测：飞机在更高平面绕圈飞行，观测船
    obs, Rs, Ts = generate_observations(X_gt, K, H, L, W, dt, 
                                         camera_radius, camera_height, camera_angular_velocity)

    # 初值（故意给得很烂，但不要太差以加快收敛）
    X_init = X_gt + np.random.randn(*X_gt.shape) * 10.0  # 减小噪声从10.0到5.0
    
    # 计算初值误差
    init_error = np.mean(np.linalg.norm(X_init - X_gt, axis=1))
    print(f"初值平均误差: {init_error:.4f}")

    x0 = X_init.flatten()

    # 优化
    print("开始优化...")
    result = least_squares(
        residual,
        x0,
        args=(obs, K, Rs, Ts, dt, H, boat_z_plane),
        verbose=2,  # 减少输出
        loss="huber",  # huber损失对异常值更鲁棒
        method='trf',  # Trust Region Reflective算法，支持huber损失且通常收敛较快
        ftol=1e-6,   # 函数值收敛容差（默认1e-8，可能太严格）
        xtol=1e-6,   # 参数值收敛容差（默认1e-8，可能太严格）
        gtol=1e-6,   # 梯度收敛容差（默认1e-8，可能太严格）
        max_nfev=1000,  # 最大函数评估次数，防止无限迭代
    )
    print(f"优化完成: 迭代 {result.nfev} 次, 最终残差 {result.cost:.6f}")

    X_opt = result.x[:3 * T].reshape(T, 3)

    print("\n优化完成！")
    print(f"优化状态: {result.status}")
    print(f"残差: {result.cost}")
    print(f"迭代次数: {result.nfev}")

    # ===============================
    # 保存结果到文本文件
    # ===============================
    output_file = "trajectory_result.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("轨迹优化结果\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"总帧数: {T}\n")
        f.write(f"时间步长: {dt}\n")
        f.write(f"优化状态: {result.status}\n")
        f.write(f"最终残差: {result.cost:.6f}\n")
        f.write(f"函数评估次数: {result.nfev}\n\n")
        
        # 提取相机位置
        camera_positions = np.array(Ts)
        
        f.write("-" * 80 + "\n")
        f.write("每一帧的位置对比 (GT vs Optimized) 和相机位置\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Frame':<8} {'GT_X':<12} {'GT_Y':<12} {'GT_Z':<12} "
                f"{'OPT_X':<12} {'OPT_Y':<12} {'OPT_Z':<12} "
                f"{'CAM_X':<12} {'CAM_Y':<12} {'CAM_Z':<12} "
                f"{'Error_X':<12} {'Error_Y':<12} {'Error_Z':<12}\n")
        f.write("-" * 80 + "\n")
        
        errors = []
        for t in range(T):
            error = X_opt[t] - X_gt[t]
            errors.append(error)
            f.write(f"{t:<8} {X_gt[t, 0]:<12.6f} {X_gt[t, 1]:<12.6f} {X_gt[t, 2]:<12.6f} "
                    f"{X_opt[t, 0]:<12.6f} {X_opt[t, 1]:<12.6f} {X_opt[t, 2]:<12.6f} "
                    f"{camera_positions[t, 0]:<12.6f} {camera_positions[t, 1]:<12.6f} {camera_positions[t, 2]:<12.6f} "
                    f"{error[0]:<12.6f} {error[1]:<12.6f} {error[2]:<12.6f}\n")
        
        errors = np.array(errors)
        f.write("\n" + "-" * 80 + "\n")
        f.write("误差统计\n")
        f.write("-" * 80 + "\n")
        f.write(f"平均误差 (X, Y, Z): ({np.mean(np.abs(errors[:, 0])):.6f}, "
                f"{np.mean(np.abs(errors[:, 1])):.6f}, {np.mean(np.abs(errors[:, 2])):.6f})\n")
        f.write(f"最大误差 (X, Y, Z): ({np.max(np.abs(errors[:, 0])):.6f}, "
                f"{np.max(np.abs(errors[:, 1])):.6f}, {np.max(np.abs(errors[:, 2])):.6f})\n")
        f.write(f"标准差 (X, Y, Z): ({np.std(errors[:, 0]):.6f}, "
                f"{np.std(errors[:, 1]):.6f}, {np.std(errors[:, 2]):.6f})\n")
        
        # 找出误差最大的帧
        max_error_frames = np.argmax(np.abs(errors), axis=0)
        f.write("\n误差最大的帧:\n")
        f.write(f"  X方向: 帧 {max_error_frames[0]}, 误差 = {errors[max_error_frames[0], 0]:.6f}\n")
        f.write(f"  Y方向: 帧 {max_error_frames[1]}, 误差 = {errors[max_error_frames[1], 1]:.6f}\n")
        f.write(f"  Z方向: 帧 {max_error_frames[2]}, 误差 = {errors[max_error_frames[2], 2]:.6f}\n")
        
        # 检查异常值（误差超过3个标准差）
        f.write("\n异常值检测 (误差 > 3*标准差):\n")
        thresholds = 3 * np.std(errors, axis=0)
        for t in range(T):
            if (np.abs(errors[t, 0]) > thresholds[0] or 
                np.abs(errors[t, 1]) > thresholds[1] or 
                np.abs(errors[t, 2]) > thresholds[2]):
                f.write(f"  帧 {t}: 误差 = ({errors[t, 0]:.6f}, {errors[t, 1]:.6f}, {errors[t, 2]:.6f})\n")
        
        # 检查深度是否合理（Z应该为正且不会太小）
        f.write("\n深度检查 (Z值):\n")
        invalid_z_frames = []
        for t in range(T):
            if X_opt[t, 2] <= 0:
                invalid_z_frames.append(t)
                f.write(f"  帧 {t}: Z = {X_opt[t, 2]:.6f} (无效，应该>0)\n")
            elif X_opt[t, 2] < 1.0:
                f.write(f"  帧 {t}: Z = {X_opt[t, 2]:.6f} (过小，可能有问题)\n")
        
        if not invalid_z_frames:
            f.write("  所有帧的深度值都有效\n")
        
        # 相机到目标的距离
        f.write("\n" + "-" * 80 + "\n")
        f.write("相机到目标的距离\n")
        f.write("-" * 80 + "\n")
        distances_gt = []
        distances_opt = []
        for t in range(T):
            dist_gt = np.linalg.norm(X_gt[t] - camera_positions[t])
            dist_opt = np.linalg.norm(X_opt[t] - camera_positions[t])
            distances_gt.append(dist_gt)
            distances_opt.append(dist_opt)
            if t % 10 == 0 or t == T - 1:
                f.write(f"  帧 {t}: GT距离 = {dist_gt:.6f}, 优化距离 = {dist_opt:.6f}, "
                        f"距离误差 = {abs(dist_opt - dist_gt):.6f}\n")
        
        distances_gt = np.array(distances_gt)
        distances_opt = np.array(distances_opt)
        f.write(f"\n平均距离: GT = {np.mean(distances_gt):.6f}, 优化 = {np.mean(distances_opt):.6f}\n")
        f.write(f"距离误差: 平均 = {np.mean(np.abs(distances_opt - distances_gt)):.6f}, "
                f"最大 = {np.max(np.abs(distances_opt - distances_gt)):.6f}\n")
    
    print(f"\n结果已保存到: {output_file}")

    # ===============================
    # 5. 可视化
    # ===============================

    # 计算每一帧相对于第一帧的位置
    X_gt_relative = X_gt - X_gt[0]  # GT相对位置
    X_opt_relative = X_opt - X_opt[0]  # 优化后的相对位置
    
    # 提取相机位置（Ts是平移向量，就是相机在世界坐标系中的位置）
    camera_positions = np.array(Ts)  # shape: (T, 3)
    camera_relative = camera_positions - camera_positions[0]  # 相机相对位置

    print(f"\n场景信息:")
    print(f"  船在 z = {boat_z_plane:.1f} 平面上运动")
    print(f"  飞机在 z = {camera_height:.1f} 高度绕圈飞行 (半径 = {camera_radius:.1f})")
    print(f"  船尺寸: 长={L:.1f}m, 宽={W:.1f}m, 高={H:.1f}m")

    fig = plt.figure(figsize=(16, 10))

    # 1. 3D轨迹图
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(X_gt[:, 0], X_gt[:, 1], X_gt[:, 2], 'g-', label="GT 船", linewidth=2)
    ax1.plot(X_opt[:, 0], X_opt[:, 1], X_opt[:, 2], 'r--', label="优化 船", linewidth=2)
    ax1.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
             'b-', label="飞机", linewidth=2, alpha=0.7)
    ax1.scatter(X_gt[0, 0], X_gt[0, 1], X_gt[0, 2], c='green', s=100, marker='o', label='船起点')
    ax1.scatter(X_gt[-1, 0], X_gt[-1, 1], X_gt[-1, 2], c='darkgreen', s=100, marker='s', label='船终点')
    ax1.scatter(camera_positions[0, 0], camera_positions[0, 1], camera_positions[0, 2], 
                c='blue', s=100, marker='^', label='飞机起点')
    ax1.scatter(camera_positions[-1, 0], camera_positions[-1, 1], camera_positions[-1, 2], 
                c='darkblue', s=100, marker='v', label='飞机终点')
    # 绘制一些相机到目标的连线（每10帧）
    for t in range(0, T, 10):
        ax1.plot([camera_positions[t, 0], X_gt[t, 0]], 
                 [camera_positions[t, 1], X_gt[t, 1]], 
                 [camera_positions[t, 2], X_gt[t, 2]], 
                 'k--', alpha=0.2, linewidth=0.5)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("3D轨迹 (船在z=0平面, 飞机在z=30高度绕圈)")
    ax1.legend(fontsize=8)
    # 添加z平面参考线
    ax1.plot([-100, 100], [0, 0], [boat_z_plane, boat_z_plane], 'k--', alpha=0.3, linewidth=1, label='船平面')
    ax1.plot([0, 0], [-100, 100], [boat_z_plane, boat_z_plane], 'k--', alpha=0.3, linewidth=1)

    # 2. XZ平面视图（俯视图）
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(X_gt[:, 0], X_gt[:, 2], 'g-', label="GT Target", linewidth=2)
    ax2.plot(X_opt[:, 0], X_opt[:, 2], 'r--', label="Optimized Target", linewidth=2)
    ax2.plot(camera_positions[:, 0], camera_positions[:, 2], 'b-', label="Camera", linewidth=2, alpha=0.7)
    # 绘制每一帧的点
    ax2.scatter(X_gt[::5, 0], X_gt[::5, 2], c='green', s=30, alpha=0.6, marker='o')
    ax2.scatter(X_opt[::5, 0], X_opt[::5, 2], c='red', s=30, alpha=0.6, marker='x')
    ax2.scatter(camera_positions[::5, 0], camera_positions[::5, 2], c='blue', s=30, alpha=0.6, marker='^')
    # 绘制相机到目标的连线（每10帧）
    for t in range(0, T, 10):
        ax2.plot([camera_positions[t, 0], X_gt[t, 0]], 
                 [camera_positions[t, 2], X_gt[t, 2]], 
                 'k--', alpha=0.2, linewidth=0.5)
    ax2.set_xlabel("X")
    ax2.set_ylabel("Z")
    ax2.set_title("Top View (XZ) - Target & Camera")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. XY平面视图（前视图）
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(X_gt[:, 0], X_gt[:, 1], 'g-', label="GT Target", linewidth=2)
    ax3.plot(X_opt[:, 0], X_opt[:, 1], 'r--', label="Optimized Target", linewidth=2)
    ax3.plot(camera_positions[:, 0], camera_positions[:, 1], 'b-', label="Camera", linewidth=2, alpha=0.7)
    ax3.scatter(X_gt[::5, 0], X_gt[::5, 1], c='green', s=30, alpha=0.6, marker='o')
    ax3.scatter(X_opt[::5, 0], X_opt[::5, 1], c='red', s=30, alpha=0.6, marker='x')
    ax3.scatter(camera_positions[::5, 0], camera_positions[::5, 1], c='blue', s=30, alpha=0.6, marker='^')
    # 绘制相机到目标的连线（每10帧）
    for t in range(0, T, 10):
        ax3.plot([camera_positions[t, 0], X_gt[t, 0]], 
                 [camera_positions[t, 1], X_gt[t, 1]], 
                 'k--', alpha=0.2, linewidth=0.5)
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.set_title("Front View (XY) - Target & Camera")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 相对位置 - XZ平面
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(X_gt_relative[:, 0], X_gt_relative[:, 2], 'g-', label="GT Target Relative", linewidth=2)
    ax4.plot(X_opt_relative[:, 0], X_opt_relative[:, 2], 'r--', label="Optimized Target Relative", linewidth=2)
    ax4.plot(camera_relative[:, 0], camera_relative[:, 2], 'b-', label="Camera Relative", linewidth=2, alpha=0.7)
    # 绘制每一帧的相对位置点
    for t in range(0, T, 5):
        ax4.scatter(X_gt_relative[t, 0], X_gt_relative[t, 2], c='green', s=50, alpha=0.7, marker='o')
        ax4.scatter(X_opt_relative[t, 0], X_opt_relative[t, 2], c='red', s=50, alpha=0.7, marker='x')
        ax4.scatter(camera_relative[t, 0], camera_relative[t, 2], c='blue', s=50, alpha=0.7, marker='^')
        # 标注帧号
        if t % 10 == 0:
            ax4.annotate(f't={t}', (X_gt_relative[t, 0], X_gt_relative[t, 2]), 
                        fontsize=8, alpha=0.7)
    ax4.set_xlabel("Relative X")
    ax4.set_ylabel("Relative Z")
    ax4.set_title("Relative Position (XZ) - 每一帧")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. 相对位置 - XY平面
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(X_gt_relative[:, 0], X_gt_relative[:, 1], 'g-', label="GT Target Relative", linewidth=2)
    ax5.plot(X_opt_relative[:, 0], X_opt_relative[:, 1], 'r--', label="Optimized Target Relative", linewidth=2)
    ax5.plot(camera_relative[:, 0], camera_relative[:, 1], 'b-', label="Camera Relative", linewidth=2, alpha=0.7)
    # 绘制每一帧的相对位置点
    for t in range(0, T, 5):
        ax5.scatter(X_gt_relative[t, 0], X_gt_relative[t, 1], c='green', s=50, alpha=0.7, marker='o')
        ax5.scatter(X_opt_relative[t, 0], X_opt_relative[t, 1], c='red', s=50, alpha=0.7, marker='x')
        ax5.scatter(camera_relative[t, 0], camera_relative[t, 1], c='blue', s=50, alpha=0.7, marker='^')
        # 标注帧号
        if t % 10 == 0:
            ax5.annotate(f't={t}', (X_gt_relative[t, 0], X_gt_relative[t, 1]), 
                        fontsize=8, alpha=0.7)
    ax5.set_xlabel("Relative X")
    ax5.set_ylabel("Relative Y")
    ax5.set_title("Relative Position (XY) - 每一帧")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. 深度随时间变化
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(X_gt[:, 2], 'g-', label="GT Target Z", linewidth=2)
    ax6.plot(X_opt[:, 2], 'r--', label="Optimized Target Z", linewidth=2)
    ax6.plot(camera_positions[:, 2], 'b-', label="Camera Z", linewidth=2, alpha=0.7)
    ax6.scatter(range(0, T, 5), X_gt[::5, 2], c='green', s=30, alpha=0.6, marker='o')
    ax6.scatter(range(0, T, 5), X_opt[::5, 2], c='red', s=30, alpha=0.6, marker='x')
    ax6.scatter(range(0, T, 5), camera_positions[::5, 2], c='blue', s=30, alpha=0.6, marker='^')
    ax6.set_xlabel("Frame")
    ax6.set_ylabel("Depth (Z)")
    ax6.set_title("Depth over Time (Target & Camera)")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("trajectory_result.png", dpi=150)
    print("Saved to trajectory_result.png")
    plt.close()  # 关闭图形以释放内存


if __name__ == "__main__":
    main()
