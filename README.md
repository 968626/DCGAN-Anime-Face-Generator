# DCGAN 动漫人脸生成器
# 源码获取：https://mbd.pub/o/bread/YZaTl5dpZg==

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-success.svg)]()

## 项目简介

本项目是基于 **PyTorch** 实现的深度卷积生成对抗网络（DCGAN），用于生成高质量的动漫人脸图像。生成器通过转置卷积层将随机噪声映射为 **64×64 RGB 图像**，判别器则通过卷积层评估图像的真实性。项目采用了 **批量归一化（Batch Normalization）**、**标签平滑（Label Smoothing）** 和 **噪声注入（Noise Injection）** 等技术来提升训练稳定性和收敛速度。

## 功能特性

- 基于 DCGAN 卷积网络架构的动漫人脸生成
- 自动生成损失曲线可视化（生成器与判别器）
- 每 10 个训练轮次保存中间生成结果
- 支持 **GPU 加速训练**
- 支持预训练模型加载（`generator.pth` / `discriminator.pth`）
- 标签平滑和噪声注入技术提升训练稳定性
- 基于 PyQt5 的图形界面，支持交互式图像生成

## 模型架构

### 生成器（Generator）
- **输入**：100 维随机噪声向量
- **结构**：多层 `ConvTranspose2d` 上采样，配合批量归一化
- **输出**：64×64 RGB 图像
- **激活函数**：ReLU（隐藏层），Tanh（输出层）

### 判别器（Discriminator）
- **输入**：64×64 RGB 图像
- **结构**：多层 `Conv2d` 下采样，配合 LeakyReLU
- **输出**：图像为真实/伪造的概率
- **激活函数**：LeakyReLU

## 训练参数

| 参数 | 数值 |
|-----------|-------|
| 训练轮次 (Epochs) | 100 |
| 批次大小 (Batch Size) | 64 |
| 学习率 (Learning Rate) | 0.0002 |
| 优化器 | Adam (betas=(0.5, 0.999)) |
| 噪声维度 (z_dim) | 100 |
| 图像尺寸 | 64×64 |
| 损失函数 | BCEWithLogitsLoss |

## 数据集

**动漫人脸数据集** 来自 Kaggle：
- 图像尺寸调整为 **64×64**
- 像素值归一化到 **[-1, 1]** 范围
- [数据集链接](https://www.kaggle.com/datasets/soumikrakshit/anime-faces)

## 环境配置

### 系统要求
- Python 3.8 或更高版本
- PyTorch 1.7 或更高版本
- CUDA 兼容的 GPU（推荐使用）

### 安装步骤
```bash
# 克隆仓库
git clone https://github.com/yourusername/anime-DCGAN.git
cd anime-DCGAN

# 安装依赖
pip install torch torchvision matplotlib numpy

# 可选：安装 GUI 依赖
pip install PyQt5
```

## 使用方法

### 训练模型
```bash
python train.py
```

训练脚本将执行以下操作：
1. 加载并预处理动漫人脸数据集
2. 初始化生成器和判别器网络
3. 使用 Adam 优化器训练 100 个轮次
4. 每 10 个轮次保存生成图像到 `outputs/images/`
5. 保存训练好的模型到 `outputs/models/`
6. 显示损失曲线

### 生成图像
```bash
python generate.py
```

### 启动图形界面
```bash
python gui/app.py
```

## 项目结构
```
anime-DCGAN/
├── data/                    # 训练数据集（动漫人脸）
├── gui/                     # 图形界面应用
│   ├── app.py              # GUI 主入口
│   └── main_window.py      # 主窗口实现
├── models/                  # 神经网络模型
│   ├── generator.py        # 生成器网络
│   └── discriminator.py    # 判别器网络
├── dataset/                 # 数据集处理
│   └── dataset.py          # 自定义数据集类
├── train.py                 # 训练脚本
├── generate.py             # 图像生成脚本
├── utils.py                # 工具函数（保存图像、绘制损失曲线）
├── outputs/                 # 输出目录
│   ├── images/             # 各轮次生成的图像
│   └── models/             # 保存的模型权重
├── README.md               # 项目文档
└── requirements.txt        # Python 依赖
```

## 生成效果展示

训练过程中的生成效果示例：

| 轮次 | 描述 | 预览 |
|-------|-------------|---------|
| 10 | 初始模糊输出 | ![Epoch 10](https://github.com/user-attachments/assets/8f01b143-f6f1-48e5-8a1b-2afcadce35e6) |
| 50 | 面部结构逐渐形成 | ![Epoch 50](https://github.com/user-attachments/assets/62f62214-51e1-4976-ac0d-ea5b7597c8fde690) |
| 100 | 清晰的动漫人脸 | ![Epoch 100](https://github.com/user-attachments/assets/3df9fb40-5e23-4404-a37b-32e8d66df351) |

## 损失曲线

![Loss Curves](https://github.com/user-attachments/assets/8bea1b7d-cfc4-49a3-8d82-7597c8fde690)

## 技术细节

### 训练稳定性技术
1. **标签平滑**：真实标签从 1.0 平滑到 0.9
2. **噪声注入**：在判别器训练时向真实图像添加高斯噪声
3. **批量归一化**：在生成器各层中应用
4. **Adam 优化器**：使用 betas=(0.5, 0.999) 进行 GAN 训练

### 损失函数
- **二元交叉熵损失**（`BCEWithLogitsLoss`）
- 分别应用于生成器和判别器

## 性能说明
- 训练时间：在现代 GPU（NVIDIA RTX 3060+）上约需 2-4 小时
- CPU 训练：速度显著较慢（不推荐）

## 常见问题

| 问题 | 解决方案 |
|-------|----------|
| CUDA 显存不足 | 减小 train.py 中的 batch_size |
| 模式崩溃（Mode collapse） | 调整学习率或 betas 参数 |
| 输出图像模糊 | 增加训练轮次或检查数据集质量 |

