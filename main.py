import numpy as np
import matplotlib.pyplot as plt

from bls_fun import batch_least_squares, recursive_least_squares
from data import load_matlab_v73, num_experiments, omega, random_factor

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

# 初始化存储 RLS 迭代路径
rls_bias_paths_random = []  # shape: (num_experiments, num_steps)
rls_factor_paths_random = []

rls_bias_paths_bls = []
rls_factor_paths_bls = []

# ---------------------
# 主循环：对每组实验进行估计
# ---------------------
for x in range(num_experiments):
    mat_file = f'GyroMeasData 50/GyroMeasData_{x + 1}.mat'
    data = load_matlab_v73(mat_file)["GyroMeasData"].T  # shape: 500×7
    omega_input = np.array(omega)

    # Batch LS
    bias_bls, factor_bls, var_noise_bls = batch_least_squares(data, omega_input)
    bls_bias_estimates.append(bias_bls)
    bls_factor_estimates.append(factor_bls)
    noise_var_estimates.append(var_noise_bls)

    # RLS with random initial guess
    bias_rls_rand, factor_rls_rand = recursive_least_squares(data, omega_input, init_guess=np.array(
        [true_bias[x], true_factor[x]]) + random_factor)
    rls_bias_estimates_random.append(bias_rls_rand[-1])
    rls_factor_estimates_random.append(factor_rls_rand[-1])
    rls_bias_paths_random.append(bias_rls_rand)  # 保存完整路径
    rls_factor_paths_random.append(factor_rls_rand)

    # RLS with BLS initial guess
    bias_rls_bls, factor_rls_bls = recursive_least_squares(data, omega_input,
                                                           init_guess=np.array([bias_bls, factor_bls]))
    rls_bias_estimates_bls.append(bias_rls_bls[-1])
    rls_factor_estimates_bls.append(factor_rls_bls[-1])
    rls_bias_paths_bls.append(bias_rls_bls)  # 保存完整路径
    rls_factor_paths_bls.append(factor_rls_bls)

# ---------------------
# 结果对比与分析
# ---------------------
# 三种求解思路的最终结果对比
bls_bias_estimates_residuals = np.array(bls_bias_estimates) - true_bias
rls_bias_estimates_random_residuals = np.array(rls_bias_estimates_random) - true_bias
rls_bias_estimates_bls_residulas = np.array(rls_bias_estimates_bls) - true_bias

plt.figure(figsize=(12, 4))
plt.plot(bls_bias_estimates_residuals, 'b', label='BLS Residuals')
plt.plot(rls_bias_estimates_random_residuals, 'r--', label='RLS (random initial) Residuals')
plt.plot(rls_bias_estimates_bls_residulas, 'g--', label='RLS (BLS initial) Residuals')
plt.axhline(0, color='black', linestyle='--')
plt.title("Residuals of Bias Estimation")
plt.xlabel("Experiment Number")
plt.ylabel("Estimate - True")
plt.legend()
plt.grid(True)
plt.show()

# 使用BLS结果作为初值的RLS迭代收敛情况

# 使用随机初值的RLS迭代收敛情况