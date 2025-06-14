import h5py
import numpy as np

# ---------------------
# 参数设置
# ---------------------
PhiN = 28.209167  # 地理纬度（角度）
r2d = 180 / np.pi
d2r = np.pi / 180
Vel_ver = 7.292e-5 * r2d * np.sin(PhiN * d2r)  # 垂直方向地球自转角速度
omega = np.array([60, 40, 50, 80, 33, 24, 17])  # 输入角速度 (7 维)
num_experiments = 50  # 实验组数从 0 到 99
random_factor = np.random.rand(2) * 0.5 # 用于给RLS的初值加随机变量

def load_matlab_v73(file_path):
    data = {}
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                # 简单数据集直接读取
                data[key] = np.array(obj)
            elif isinstance(obj, h5py.Group):
                # 如果是嵌套结构，可以递归处理
                data[key] = {k: np.array(v) for k, v in obj.items()}
    return data