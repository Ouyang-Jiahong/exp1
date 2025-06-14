import numpy as np

from data import Vel_ver


# ---------------------
# 批量最小二乘法（BLS）函数
# ---------------------
def batch_least_squares(data, omega_input):
    X = np.ones((data.shape[0], 2))
    X[:, 1] = omega_input + Vel_ver
    y = data.mean(axis=1).reshape(-1, 1)  # 对每行取均值作为观测值
    theta = np.linalg.pinv(X.T @ X) @ X.T @ y
    bias_est = theta[0, 0]
    factor_est = theta[1, 0]
    residuals = y - X @ theta
    sigma2 = np.var(residuals)  # 测量噪声方差估计
    return bias_est, factor_est, sigma2

# ---------------------
# 递推最小二乘法（RLS）函数
# ---------------------
def recursive_least_squares(data, omega_input, P0=1e6, init_guess=None):
    N = data.shape[0]
    if init_guess is None:
        theta = np.zeros((2, 1))
    else:
        theta = init_guess.reshape(2, 1)
    P = P0 * np.eye(2)
    I = np.eye(2)

    bias_history = []
    factor_history = []

    for k in range(N):
        phi = np.array([[1, omega_input[k] + Vel_ver]])
        y = data[k]

        # RLS 更新公式
        e = y - phi @ theta
        K = P @ phi.T / (phi @ P @ phi.T + 1e-8)
        theta += K * e
        P = (I - K @ phi) @ P

        bias_history.append(theta[0, 0])
        factor_history.append(theta[1, 0])

    return np.array(bias_history), np.array(factor_history)