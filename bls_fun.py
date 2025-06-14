import numpy as np

from data import Vel_ver


# ---------------------
# 批量最小二乘法（BLS）函数
# ---------------------
def batch_least_squares(data, omega_input):
    """
    批量最小二乘法估计 Bias 和 Factor，并估计测量噪声方差。

    参数:
        data (np.ndarray): shape (500, 7)，实测的角速度数据
        omega_input (np.ndarray): shape (7,)，输入的真实角速度值

    返回:
        bias_est (float): 估计的 Bias
        factor_est (float): 估计的 Factor
        var_noise_est (float): 估计的测量噪声方差
    """
    num_samples = data.shape[0]  # 应该是500
    num_axes = data.shape[1]     # 应该是7

    # 构造设计矩阵 A 和观测向量 y
    A = np.zeros((num_samples * num_axes, 2))
    y = np.zeros((num_samples * num_axes,))

    for i in range(num_samples):
        for j in range(num_axes):
            index = i * num_axes + j
            A[index, 0] = 1.0  # 对应 Bias
            A[index, 1] = omega_input[j] + Vel_ver  # 对应 Factor * (omega + Vel_ver)
            y[index] = data[i, j]  # 观测值 ω_G

    # 最小二乘解: θ = (A^T A)^{-1} A^T y
    ATA = A.T @ A
    theta = np.linalg.inv(ATA) @ A.T @ y

    bias_est = theta[0]
    factor_est = theta[1]

    # 计算残差用于估计噪声方差
    residuals = y - A @ theta
    var_noise_est = np.var(residuals)

    return bias_est, factor_est, var_noise_est

# ---------------------
# 递推最小二乘法（RLS）函数
# ---------------------
def recursive_least_squares(data_matrix, omega_input, init_guess=None, lambda_=1.0):
    """
    使用递推最小二乘法 (RLS) 来联合估计所有7个通道的 Bias 和 Factor。

    参数:
        data_matrix (np.ndarray): shape (500, 7)，每列是一个通道的测量数据
        omega_input (np.ndarray): shape (7,)，输入的真实角速度值
        Vel_ver (float): 垂直方向地球自转角速度
        init_guess (np.ndarray): 初始估计值 [bias_init, factor_init]
        lambda_ (float): 遗忘因子，默认为1.0

    返回:
        bias_estimates (list): 每一步估计的 Bias 值
        factor_estimates (list): 每一步估计的 Factor 值
    """
    num_samples, num_axes = data_matrix.shape

    # 添加地球自转影响
    omega_with_vel = omega_input + Vel_ver  # shape: (7,)

    # 初始化参数
    if init_guess is None:
        theta = np.zeros(2)
    else:
        theta = np.array(init_guess)

    # 初始化协方差矩阵
    P = np.eye(2) * 1000  # 初始协方差较大，表示不确定

    # 存储每一步的估计值
    bias_estimates = []
    factor_estimates = []

    for i in range(num_samples):
        for j in range(num_axes):
            phi = np.array([1.0, omega_with_vel[j]])  # 对应第 j 个轴
            y_k = data_matrix[i, j]  # 第 i 次观测、第 j 个轴的测量值

            # 计算误差
            error = y_k - phi @ theta

            # 计算增益
            numerator = P @ phi
            denominator = lambda_ + phi @ numerator
            K = numerator / denominator

            # 更新参数估计
            theta += K * error

            # 更新协方差矩阵
            outer_product = np.outer(K, phi)
            P = (np.eye(2) - outer_product) @ P / lambda_

            # 保存每一步的估计值
            bias_estimates.append(theta[0])
            factor_estimates.append(theta[1])

        # 计算残差用于估计噪声方差


    return bias_estimates, factor_estimates