# 飞行器试验统计学实验一报告项目

## 项目概述
本项目是2025春飞行器试验统计学的实验一，主要使用批量最小二乘法（Batch LS）和递推最小二乘法（RLS）来估计陀螺仪的偏差（bias）和标度因子（scale factor），并比较这两种估计方法的结果。同时，还会估计批量最小二乘法下测量噪声的方差，以及通过仿真探究递推最小二乘法中初始参数对估计误差曲线的影响。

## 项目结构
```plaintext
├── .gitignore
├── GyroMeasData 50/
│   ├── GyroBias.mat
│   ├── GyroFactor.mat
│   ├── GyroMeasData_X.mat
├── fun.py
├── data.py
├── main.py
└── report/
    ├── package.tex
    └── report.tex
```

## 主要文件说明
- `fun.py`：包含批量最小二乘法和递推最小二乘法的实现函数。
- `data.py`：负责加载Matlab v7.3格式的数据文件，并定义了一些实验参数。
- `main.py`：主程序，调用`bls_fun.py`中的函数进行参数估计，并绘制结果对比图。
- `report/`：包含实验报告的LaTeX文件。

## 安装依赖
本项目使用Python编写，需要安装以下依赖库：
```plaintext
numpy
matplotlib
h5py
```
你可以使用以下命令安装这些依赖：
```bash
pip install numpy matplotlib h5py
```

## 使用方法
1. 确保已经安装了所需的依赖库。
2. 将`GyroMeasData 50`文件夹放在项目根目录下，确保其中包含`GyroBias.mat`、`GyroFactor.mat`和`GyroMeasData_X.mat`文件。
3. 运行`main.py`文件：
```bash
python main.py
```
4. 运行程序后，会自动绘制各种结果对比图，分析批量最小二乘法和递推最小二乘法的性能。

## 注意事项
- 本项目运行在Windows操作系统上。
- 确保`GyroMeasData 50`文件夹中的数据文件格式正确，否则可能会导致程序出错。
- 本项目使用`Python 3.13.2`编写，如果发生运行错误，请尝试调整`Python`版本。