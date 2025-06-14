import numpy as np
import matplotlib.pyplot as plt

from bls_fun import batch_least_squares, recursive_least_squares
from data import load_matlab_v73, num_experiments, omega

# ---------------------
# 加载真值数据
# ---------------------
bias_data = load_matlab_v73('GyroMeasData 50/GyroBias.mat')["GyroBias"]
factor_data = load_matlab_v73('GyroMeasData 50/GyroFactor.mat')["GyroFactor"]

# 提取 bias 和 factor 的真值
true_bias = np.array([bias_data[i][0] for i in range(num_experiments)])
true_factor = np.array([factor_data[i][0] for i in range(num_experiments)])

# ---------------------
# 存储估计结果
# ---------------------
bls_bias_estimates = []
bls_factor_estimates = []
noise_var_estimates = []

rls_bias_estimates_random = []
rls_factor_estimates_random = []

rls_bias_estimates_bls = []
rls_factor_estimates_bls = []

# ---------------------
# 主循环：对每组实验进行估计
# ---------------------
for x in range(num_experiments):
    mat_file = f'GyroMeasData 50/GyroMeasData_{x + 1}.mat'
    data = load_matlab_v73(mat_file)["GyroMeasData"]  # shape: 500×7

    # 使用第0列作为示例轴，每个轴可单独估计
    omega_input = np.tile(omega[x % 7], data.shape[0])  # 每次实验使用不同的输入角速度
    measurements = data[:, 0]  # 取第一列进行估计

    # Batch LS
    bias_bls, factor_bls, var_noise = batch_least_squares(data, omega_input)
    bls_bias_estimates.append(bias_bls)
    bls_factor_estimates.append(factor_bls)
    noise_var_estimates.append(var_noise)

    # RLS with random initial guess
    bias_rls_rand, factor_rls_rand = recursive_least_squares(measurements, omega_input, init_guess=np.random.rand(2))
    rls_bias_estimates_random.append(bias_rls_rand[-1])
    rls_factor_estimates_random.append(factor_rls_rand[-1])

    # RLS with BLS initial guess
    bias_rls_bls, factor_rls_bls = recursive_least_squares(measurements, omega_input, init_guess=np.array([bias_bls, factor_bls]))
    rls_bias_estimates_bls.append(bias_rls_bls[-1])
    rls_factor_estimates_bls.append(factor_rls_bls[-1])

    # 显示第一次仿真的收敛过程
    if x == 0:
        plt.figure(figsize=(12, 5))
        plt.plot(bias_rls_rand, label='RLS Bias (Random Init)')
        plt.plot(factor_rls_rand, label='RLS Factor (Random Init)')
        plt.plot(bias_rls_bls, label='RLS Bias (BLS Init)')
        plt.plot(factor_rls_bls, label='RLS Factor (BLS Init)')
        plt.axhline(y=true_bias[x], color='r', linestyle='--', label='True Bias')
        plt.axhline(y=true_factor[x], color='g', linestyle='--', label='True Factor')
        plt.title("RLS Convergence Behavior (First Experiment)")
        plt.xlabel("Iteration")
        plt.legend()
        plt.grid(True)
        plt.show()

# ---------------------
# 结果对比与分析
# ---------------------
plt.figure(figsize=(12, 5))

# Bias Estimation Comparison
plt.subplot(1, 2, 1)
plt.plot(true_bias[:], 'k--', label='True Bias')
plt.plot(bls_bias_estimates[:], 'b-o', label='BLS Estimate')
plt.plot(rls_bias_estimates_random[:], 'r-o', label='RLS Random Init')
plt.plot(rls_bias_estimates_bls[:], 'g-o', label='RLS BLS Init')
plt.title("Bias Estimation Comparison")
plt.legend()
plt.grid(True)

# Factor Estimation Comparison
plt.subplot(1, 2, 2)
plt.plot(true_factor[:], 'k--', label='True Factor')
plt.plot(bls_factor_estimates[:], 'b-o', label='BLS Estimate')
plt.plot(rls_factor_estimates_random[:], 'r-o', label='RLS Random Init')
plt.plot(rls_factor_estimates_bls[:], 'g-o', label='RLS BLS Init')
plt.title("Factor Estimation Comparison")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()