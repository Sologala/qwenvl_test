
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


def unproject_point(u, v, R, t, K, z_world):
    """从像素坐标反投影到世界坐标
    给定像素坐标(u,v)和估计的世界z坐标，计算世界坐标
    注意：这里假设z_world是估计值，可能是错误的
    
    Args:
        u, v: 像素坐标
        R, t: 相机外参（旋转和平移）
        K: 相机内参
        z_world: 估计的世界z坐标（可能是错误的）
    
    Returns:
        Xw: 世界坐标系中的3D点
    """
    # 从像素坐标转换到归一化相机坐标
    x_norm = (u - K[0, 2]) / K[0, 0]
    y_norm = (v - K[1, 2]) / K[1, 1]
    
    # 归一化方向向量（在相机坐标系中）
    ray_cam = np.array([x_norm, y_norm, 1.0])
    ray_cam = ray_cam / np.linalg.norm(ray_cam)
    
    # 转换到世界坐标系
    R_inv = R.T  # 旋转矩阵的逆等于转置
    ray_world = R_inv @ ray_cam
    
    # 计算射线与z=z_world平面的交点
    # 射线方程: X = t + lambda * ray_world
    # 其中t是相机位置（世界坐标系）
    # 我们需要找到lambda使得 X[2] = z_world
    # t[2] + lambda * ray_world[2] = z_world
    # lambda = (z_world - t[2]) / ray_world[2]
    
    if abs(ray_world[2]) < 1e-6:
        # 如果射线几乎平行于xy平面，使用一个默认距离
        lambda_val = 50.0
    else:
        lambda_val = (z_world - t[2]) / ray_world[2]
    
    # 计算世界坐标
    Xw = t + lambda_val * ray_world
    
    return Xw


def make_camera_pose(t, dt, boat_position, radius=50.0, height=30.0, angular_velocity=0.05):
    """飞机（相机）跟随船移动并围绕船旋转
    Args:
        t: 帧序号
        dt: 时间步长
        boat_position: 船的位置 [x, y, z]
        radius: 围绕船的旋转半径
        height: 飞行高度（相对于船的z坐标）
        angular_velocity: 角速度
    """
    # 飞机围绕船旋转，同时跟随船移动
    time = t * dt
    angle = angular_velocity * time
    
    # 在船的xy平面坐标系中，飞机围绕船旋转
    # 旋转半径在xy平面上的投影
    cam_offset_x = radius * np.cos(angle)
    cam_offset_y = radius * np.sin(angle)
    
    # 飞机位置 = 船的位置 + 旋转偏移 + 高度偏移
    cam_x = boat_position[0] + cam_offset_x
    cam_y = boat_position[1] + cam_offset_y
    cam_z = boat_position[2] + height  # 飞机在船的上方
    
    # 相机朝向船的方向
    # 计算从相机到船的向量
    cam_to_boat = boat_position - np.array([cam_x, cam_y, cam_z])
    cam_to_boat_normalized = cam_to_boat / np.linalg.norm(cam_to_boat)
    
    # 构建相机坐标系
    # z轴：指向船的方向（相机前方）
    z_axis = -cam_to_boat_normalized  # 相机z轴指向船
    
    # x轴：在xy平面上的投影方向（右方）
    if abs(z_axis[2]) > 0.9:
        # 如果相机几乎垂直向下，使用世界坐标系的x轴
        x_axis = np.array([1.0, 0.0, 0.0])
    else:
        # 否则使用z轴与[0,0,1]的叉积
        world_up = np.array([0.0, 0.0, 1.0])
        x_axis = np.cross(world_up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
    
    # y轴：z轴与x轴的叉积（上方）
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # 重新正交化x轴
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # 构建旋转矩阵（从世界坐标系到相机坐标系）
    # 相机坐标系的三个轴在世界坐标系中的表示
    R = np.array([x_axis, y_axis, z_axis]).T  # 转置，因为列向量是坐标轴
    
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
        R, T = make_camera_pose(t, dt, X, camera_radius, camera_height, camera_angular_velocity)
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
# 3. 从射线和框大小反推位置
# ===============================

def ray_from_observation(u, v, R, T, K):
    """从观测(u,v)计算射线方向（世界坐标系）"""
    # 从像素坐标转换到归一化相机坐标
    x_norm = (u - K[0, 2]) / K[0, 0]
    y_norm = (v - K[1, 2]) / K[1, 1]
    
    # 归一化方向向量（在相机坐标系中）
    ray_cam = np.array([x_norm, y_norm, 1.0])
    ray_cam = ray_cam / np.linalg.norm(ray_cam)
    
    # 转换到世界坐标系
    R_inv = R.T
    ray_world = R_inv @ ray_cam
    
    return ray_world


def position_from_ray_and_height(ray_world, camera_pos, R, H, h_obs, K, boat_z_plane):
    """从射线、框高度和船高度反推位置
    Args:
        ray_world: 射线方向（世界坐标系）
        camera_pos: 相机位置（世界坐标系）
        R: 相机旋转矩阵
        H: 船的高度（待优化）
        h_obs: 观测到的框高度（像素）
        K: 相机内参
        boat_z_plane: 船所在的z平面
    Returns:
        X: 世界坐标系中的位置
    """
    if h_obs < 1.0:
        h_obs = 1.0  # 避免除零
    
    # 方法1：从框高度和H计算深度（相机坐标系）
    # h_obs = K[1, 1] * H / z_cam
    # z_cam = K[1, 1] * H / h_obs
    z_cam = K[1, 1] * H / h_obs  # 相机坐标系中的深度
    
    # 将射线方向转换回相机坐标系
    ray_cam = R @ ray_world  # ray_world = R^T @ ray_cam, 所以 ray_cam = R @ ray_world
    
    # 在相机坐标系中，位置是 z_cam * ray_cam / ray_cam[2]
    # 因为 ray_cam 是归一化的，ray_cam[2] 应该是接近1的
    if abs(ray_cam[2]) < 1e-6:
        # 如果射线几乎垂直于z轴，使用深度直接计算
        X_cam = z_cam * ray_cam
    else:
        # 归一化，使得z坐标等于z_cam
        X_cam = z_cam * ray_cam / ray_cam[2]
    
    # 转换到世界坐标系
    # X_world = R^T @ X_cam + camera_pos
    # 但更简单：X_world = camera_pos + lambda * ray_world
    # 其中 lambda 使得距离等于 z_cam
    # 实际上，我们需要找到射线上的点，使得它在相机坐标系中的深度是z_cam
    
    # 更直接的方法：找到射线与z=boat_z_plane平面的交点
    # 然后验证深度
    if abs(ray_world[2]) < 1e-6:
        # 如果射线几乎平行于xy平面，使用深度信息
        lambda_val = z_cam / np.linalg.norm(ray_world)
    else:
        # 找到射线与z=boat_z_plane平面的交点
        lambda_val = (boat_z_plane - camera_pos[2]) / ray_world[2]
    
    X = camera_pos + lambda_val * ray_world
    
    # 验证：计算实际深度，如果差异太大，使用深度约束的位置
    X_rel = X - camera_pos
    actual_depth = np.linalg.norm(X_rel)
    if abs(actual_depth - z_cam) > z_cam * 0.5:  # 如果差异超过50%，使用深度约束
        # 使用深度约束重新计算
        lambda_val = z_cam / np.linalg.norm(ray_world)
        X = camera_pos + lambda_val * ray_world
        X[2] = boat_z_plane  # 强制z坐标
    
    return X


# ===============================
# 4. BA 残差函数（优化H和速度）
# ===============================

def residual_optimize_H_V(params, obs, K, Rs, Ts, dt, boat_z_plane, X0_init):
    """优化船的高度H和速度V
    Args:
        params: [H, Vx, Vy] - 船的高度和速度（2D，因为z=0）
        obs: 观测数据 [u, v, w, h]
        K: 相机内参
        Rs, Ts: 相机位姿
        dt: 时间步长
        boat_z_plane: 船所在的z平面
        X0_init: 第一帧的初始位置估计（用于确定起始点）
    """
    Tn = len(obs)
    H = params[0]
    V = np.array([params[1], params[2], 0.0])  # 速度向量（z=0）
    
    # 从第一帧的观测和H反推第一帧位置
    u0, v0, w0, h0 = obs[0]
    ray0 = ray_from_observation(u0, v0, Rs[0], Ts[0], K)
    X0 = position_from_ray_and_height(ray0, Ts[0], Rs[0], H, h0, K, boat_z_plane)
    
    # 如果H太小导致位置不合理，使用初始估计
    if np.linalg.norm(X0 - X0_init) > 100.0:
        X0 = X0_init.copy()
        X0[2] = boat_z_plane  # 确保z在正确平面上
    
    res = []
    
    # 构建所有帧的位置（基于速度模型）
    Xs = [X0]
    for t in range(1, Tn):
        X_t = Xs[t-1] + V * dt
        X_t[2] = boat_z_plane  # 确保z在正确平面上
        Xs.append(X_t)
    
    Xs = np.array(Xs)
    
    # 计算残差
    for t in range(Tn):
        R, T = Rs[t], Ts[t]
        X = Xs[t]
        
        # 投影到相机坐标系
        Xc = R @ X + T
        x, y, z = Xc
        
        if z <= 0.1:
            z = 0.1  # 保护
        
        # 位置投影残差
        u_pred = K[0, 0] * x / z + K[0, 2]
        v_pred = K[1, 1] * y / z + K[1, 2]
        
        res.append(u_pred - obs[t, 0])
        res.append(v_pred - obs[t, 1])
        
        # 框高度残差
        h_pred = K[1, 1] * H / z
        res.append(h_pred - obs[t, 3])
        
        # 约束：位置应该在射线上（从观测反推的位置应该与速度模型一致）
        # 从观测反推位置
        u_obs, v_obs, w_obs, h_obs = obs[t]
        ray = ray_from_observation(u_obs, v_obs, R, T, K)
        X_from_ray = position_from_ray_and_height(ray, T, R, H, h_obs, K, boat_z_plane)
        
        # 位置应该接近（在xy平面上）
        xy_diff = (X[:2] - X_from_ray[:2])
        res.extend(xy_diff * 0.5)  # 权重0.5
    
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
    L = 30.0  # 船的长度（长边）
    W = 3.0  # 船的宽度（短边，L > W）
    boat_z_plane = 0.0  # 船所在的z平面
    camera_height = 30.0  # 飞机飞行高度
    camera_radius = 100.0  # 飞机绕圈半径
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

    # 初值：从观测反推第一帧位置（使用错误的高度估计）
    # 然后估计H和速度
    wrong_z_offset = 10.0  # 错误的高度偏移
    
    print(f"使用错误的初始高度估计: z = {boat_z_plane} ± {wrong_z_offset}")
    
    # 从第一帧观测反推初始位置（使用错误的高度）
    u0, v0 = obs[0, 0], obs[0, 1]
    R0, T0 = Rs[0], Ts[0]
    
    # 估计一个错误的高度
    if np.random.rand() > 0.5:
        estimated_z = boat_z_plane + wrong_z_offset * (1.0 + np.random.rand() * 0.5)
    else:
        estimated_z = boat_z_plane - wrong_z_offset * (1.0 + np.random.rand() * 0.5)
    
    X0_init = unproject_point(u0, v0, R0, T0, K, estimated_z)
    X0_init[2] = boat_z_plane  # 确保z在正确平面上
    
    # 估计初始H和速度
    # H的初值：从第一帧的框高度和估计的深度反推
    z_cam_est = np.linalg.norm(X0_init - T0)  # 估计的相机到目标的距离
    h_obs_0 = obs[0, 3]  # 第一帧的框高度
    H_init = h_obs_0 * z_cam_est / K[1, 1]  # H = h * z / K[1,1]
    H_init = -120.0
    # 速度的初值：从前几帧的位置差估计
    if T >= 3:
        # 从观测反推前3帧的位置（使用错误的高度）
        X_init_list = [X0_init]
        for t in range(1, min(3, T)):
            u, v = obs[t, 0], obs[t, 1]
            R, T_cam = Rs[t], Ts[t]
            estimated_z_t = estimated_z + (np.random.rand() - 0.5) * 0.5  # 稍微变化
            X_t = unproject_point(u, v, R, T_cam, K, estimated_z_t)
            X_t[2] = boat_z_plane
            X_init_list.append(X_t)
        
        # 估计速度
        if len(X_init_list) >= 2:
            V_init_xy = (X_init_list[1][:2] - X_init_list[0][:2]) / dt
        else:
            V_init_xy = np.array([0.0, 0.0])
    else:
        V_init_xy = np.array([0.0, 0.0])
    
    V_init_xy = np.array([0.0, 0.0])
    print(f"初始估计:")
    print(f"  H = {H_init:.4f} m (真实值 = {H:.4f} m)")
    print(f"  速度 = [{V_init_xy[0]:.4f}, {V_init_xy[1]:.4f}] m/s")
    print(f"  第一帧位置 = [{X0_init[0]:.4f}, {X0_init[1]:.4f}, {X0_init[2]:.4f}]")
    
    # 优化参数：[H, Vx, Vy]
    x0 = np.array([H_init, V_init_xy[0], V_init_xy[1]])

    # 优化
    print("\n开始优化 H 和速度...")
    result = least_squares(
        residual_optimize_H_V,
        x0,
        args=(obs, K, Rs, Ts, dt, boat_z_plane, X0_init),
        verbose=0,
        loss="huber",
        method='trf',
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        max_nfev=1000,
    )
    print(f"优化完成: 迭代 {result.nfev} 次, 最终残差 {result.cost:.6f}")

    # 提取优化结果
    H_opt = result.x[0]
    V_opt = np.array([result.x[1], result.x[2], 0.0])
    
    print(f"\n优化结果:")
    print(f"  H: {H_opt:.4f} m (真实值 = {H:.4f} m, 误差 = {abs(H_opt - H):.4f} m)")
    print(f"  速度: [{V_opt[0]:.4f}, {V_opt[1]:.4f}] m/s")
    
    # 从优化结果重建轨迹
    X_opt = [X0_init]
    for t in range(1, T):
        X_t = X_opt[t-1] + V_opt * dt
        X_t[2] = boat_z_plane
        X_opt.append(X_t)
    X_opt = np.array(X_opt)

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

    # 提取相机位置（Ts是平移向量，就是相机在世界坐标系中的位置）
    camera_positions = np.array(Ts)  # shape: (T, 3)

    print(f"\nScene Information:")
    print(f"  Boat on z = {boat_z_plane:.1f} plane")
    print(f"  Aircraft at z = {camera_height:.1f} height, orbiting radius = {camera_radius:.1f}")
    print(f"  Boat dimensions: L={L:.1f}m, W={W:.1f}m, H={H:.1f}m")

    fig = plt.figure(figsize=(18, 6))

    # 1. 3D轨迹图
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.plot(X_gt[:, 0], X_gt[:, 1], X_gt[:, 2], 'g-', label="GT Boat", linewidth=2)
    ax1.plot(X_opt[:, 0], X_opt[:, 1], X_opt[:, 2], 'r--', label="Optimized Boat", linewidth=2)
    ax1.plot(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
             'b-', label="Aircraft", linewidth=2, alpha=0.7)
    ax1.scatter(X_gt[0, 0], X_gt[0, 1], X_gt[0, 2], c='green', s=100, marker='o', label='Boat Start')
    ax1.scatter(X_gt[-1, 0], X_gt[-1, 1], X_gt[-1, 2], c='darkgreen', s=100, marker='s', label='Boat End')
    ax1.scatter(camera_positions[0, 0], camera_positions[0, 1], camera_positions[0, 2], 
                c='blue', s=100, marker='^', label='Aircraft Start')
    ax1.scatter(camera_positions[-1, 0], camera_positions[-1, 1], camera_positions[-1, 2], 
                c='darkblue', s=100, marker='v', label='Aircraft End')
    # 绘制一些相机到目标的连线（每10帧）
    for t in range(0, T, 10):
        ax1.plot([camera_positions[t, 0], X_gt[t, 0]], 
                 [camera_positions[t, 1], X_gt[t, 1]], 
                 [camera_positions[t, 2], X_gt[t, 2]], 
                 'k--', alpha=0.2, linewidth=0.5)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("3D Trajectory")
    ax1.legend(fontsize=8)
    # 添加z平面参考线
    ax1.plot([-200, 200], [0, 0], [boat_z_plane, boat_z_plane], 'k--', alpha=0.3, linewidth=1)
    ax1.plot([0, 0], [-200, 200], [boat_z_plane, boat_z_plane], 'k--', alpha=0.3, linewidth=1)

    # 2. XZ平面视图（俯视图）
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(X_gt[:, 0], X_gt[:, 2], 'g-', label="GT Boat", linewidth=2)
    ax2.plot(X_opt[:, 0], X_opt[:, 2], 'r--', label="Optimized Boat", linewidth=2)
    ax2.plot(camera_positions[:, 0], camera_positions[:, 2], 'b-', label="Aircraft", linewidth=2, alpha=0.7)
    # 绘制每一帧的点
    ax2.scatter(X_gt[::5, 0], X_gt[::5, 2], c='green', s=30, alpha=0.6, marker='o')
    ax2.scatter(X_opt[::5, 0], X_opt[::5, 2], c='red', s=30, alpha=0.6, marker='x')
    ax2.scatter(camera_positions[::5, 0], camera_positions[::5, 2], c='blue', s=30, alpha=0.6, marker='^')
    # 绘制相机到目标的连线（每10帧）
    for t in range(0, T, 10):
        ax2.plot([camera_positions[t, 0], X_gt[t, 0]], 
                 [camera_positions[t, 2], X_gt[t, 2]], 
                 'k--', alpha=0.2, linewidth=0.5)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Z (m)")
    ax2.set_title("Top View (XZ)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. XY平面视图（前视图）
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(X_gt[:, 0], X_gt[:, 1], 'g-', label="GT Boat", linewidth=2)
    ax3.plot(X_opt[:, 0], X_opt[:, 1], 'r--', label="Optimized Boat", linewidth=2)
    ax3.plot(camera_positions[:, 0], camera_positions[:, 1], 'b-', label="Aircraft", linewidth=2, alpha=0.7)
    ax3.scatter(X_gt[::5, 0], X_gt[::5, 1], c='green', s=30, alpha=0.6, marker='o')
    ax3.scatter(X_opt[::5, 0], X_opt[::5, 1], c='red', s=30, alpha=0.6, marker='x')
    ax3.scatter(camera_positions[::5, 0], camera_positions[::5, 1], c='blue', s=30, alpha=0.6, marker='^')
    # 绘制相机到目标的连线（每10帧）
    for t in range(0, T, 10):
        ax3.plot([camera_positions[t, 0], X_gt[t, 0]], 
                 [camera_positions[t, 1], X_gt[t, 1]], 
                 'k--', alpha=0.2, linewidth=0.5)
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_title("Front View (XY)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("trajectory_result.png", dpi=150)
    print("Saved to trajectory_result.png")
    plt.close()  # 关闭图形以释放内存


if __name__ == "__main__":
    main()
