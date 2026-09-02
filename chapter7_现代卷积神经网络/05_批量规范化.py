"""
=============================================================================
批量规范化（Batch Normalization）—— 完整带注释版
=============================================================================
批量规范化是训练深层神经网络的核心技术之一，由 Ioffe & Szegedy (2015) 提出。
它使得训练 100+ 层的网络成为可能，几乎成为所有现代 CNN 的标准组件。

核心思想：
  对每个 mini-batch 的中间层输出做标准化（均值0，方差1），
  再通过可学习的 γ（拉伸）和 β（偏移）恢复表达能力。

效果：
  - 允许使用更大的学习率，加速收敛
  - 减少对参数初始化精度的敏感度
  - 自带一定的正则化效果（减少对 Dropout 的依赖）
  - 使深层网络的训练更稳定

本节包含从零实现和 PyTorch 简洁实现两种方式。
=============================================================================
"""

import torch
from torch import nn
from d2l import torch as d2l


# ============================================================================
# 从零实现 batch_norm 函数
# ============================================================================
# 批量规范化的数学公式:
#   BN(x) = γ ⊙ (x - μ̂) / √(σ̂² + ε) + β
#
# 其中：
#   μ̂ (均值): 当前小批量数据的均值
#   σ̂² (方差): 当前小批量数据的方差
#   γ (gamma): 可学习的拉伸参数，初始化为 1
#   β (beta):  可学习的偏移参数，初始化为 0
#   ε (eps):   小常数(1e-5)，防止除以零
#
# 训练模式 vs 预测模式的区别：
#   - 训练时: 使用当前 mini-batch 的统计量（有噪声，但这是一种正则化）
#   - 预测时: 使用整个训练集的移动平均统计量（稳定、确定）

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    """
    批量规范化的核心计算。

    参数:
        X: 输入张量，形状为 (batch, features) 或 (batch, channels, H, W)
        gamma: 拉伸参数，形状与特征维度一致
        beta: 偏移参数，形状与特征维度一致
        moving_mean: 移动平均均值（预测时使用）
        moving_var: 移动平均方差（预测时使用）
        eps: 防止除零的小常数 (1e-5)
        momentum: 移动平均的动量，用于更新 moving_mean/var

    返回:
        Y: 批量规范化后的输出
        moving_mean.data: 更新后的移动平均均值
        moving_var.data: 更新后的移动平均方差
    """
    # torch.is_grad_enabled(): 判断当前是否处于训练模式
    #   训练模式下梯度追踪开启 → 返回 True
    #   预测模式下梯度追踪关闭 → 返回 False
    if not torch.is_grad_enabled():
        # ---- 预测模式 ----
        # 直接使用训练时累积的全局移动平均统计量
        # torch.sqrt(): 开平方根，将方差转为标准差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # ---- 训练模式 ----
        assert len(X.shape) in (2, 4)  # 只支持全连接(2D)或卷积(4D)
        if len(X.shape) == 2:
            # 全连接层的情况: X shape = (batch_size, num_features)
            # dim=0 即沿着 batch 维度求均值和方差
            # mean(dim=0): 对每个特征列独立计算 mini-batch 均值
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 二维卷积层的情况: X shape = (batch, channels, H, W)
            # 需要对每个通道独立做标准化
            # dim=(0, 2, 3): 在 batch、H、W 三个维度上求均值/方差
            #   对每个通道，有 batch×H×W 个元素参与统计
            # keepdim=True: 保持维度以便后续广播计算
            #   例如: X=(32,6,28,28), mean 变为 (1,6,1,1) 而非 (6,)
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # 训练模式下用当前 mini-batch 的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 更新移动平均的均值和方差（用于预测时使用）
        # moving = momentum × old + (1-momentum) × new
        # momentum 通常取 0.9，给历史值更大权重，平滑更新
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    # 最后应用拉伸和偏移: Y = γ * X_hat + β
    #   gamma 和 beta 是可学习参数，让网络能恢复任意分布
    Y = gamma * X_hat + beta  # 缩放和移位
    return Y, moving_mean.data, moving_var.data


# ============================================================================
# BatchNorm 层封装
# ============================================================================
class BatchNorm(nn.Module):
    """
    自定义批量规范化层。

    参数:
        num_features: 全连接层的输出数量 或 卷积层的输出通道数
        num_dims: 输入维度，2 表示全连接层，4 表示卷积层

    内部参数:
        gamma (nn.Parameter): 可学习的拉伸参数 → 需要梯度更新
        beta  (nn.Parameter): 可学习的偏移参数 → 需要梯度更新
        moving_mean: 移动平均均值 → 不是 nn.Parameter，不参与梯度计算
        moving_var:  移动平均方差 → 不是 nn.Parameter，不参与梯度计算
    """
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            # 全连接层：gamma/beta 形状为 (1, num_features)
            shape = (1, num_features)
        else:
            # 卷积层：gamma/beta 形状为 (1, num_features, 1, 1)
            #   每个通道有一个独立的 gamma 和 beta（标量）
            shape = (1, num_features, 1, 1)
        # nn.Parameter: 将张量标记为模型参数，optimizer 会自动追踪和更新
        #   初始化为 ones: γ=1 → 不改变方差（开始时不做拉伸）
        self.gamma = nn.Parameter(torch.ones(shape))
        #   初始化为 zeros: β=0 → 不改变均值（开始时不做偏移）
        self.beta = nn.Parameter(torch.zeros(shape))
        # 非模型参数的变量，初始化为 0 和 1
        #   这些值在训练中通过移动平均更新，但不参与梯度计算
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # 如果 X 在 GPU 上而 moving_mean/var 还在 CPU 上，将它们移到同一设备
        #   .device 属性返回张量所在的设备（'cpu' 或 'cuda:0'）
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 调用核心 batch_norm 函数
        #   每次前向传播都会更新 moving_mean 和 moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y


# ============================================================================
# 使用自定义 BatchNorm 的 LeNet
# ============================================================================
# BN 层的放置位置：卷积层/全连接层之后，激活函数之前
#   顺序: Conv → BN → Activation → Pool
#   原因: 先标准化再激活，让激活函数接收稳定分布的输入
#   注意: 如果使用 BN，全连接层/卷积层通常可以省略偏置（bias）
#         因为 BN 中的 β 参数已经提供了偏移能力
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))

# 对比：没有 BN 的 LeNet 学习率通常约 0.9（SGD）就需要仔细调参
# 加了 BN 后可以用 lr=1.0 这样的大学习率！训练更快更稳定
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


# ============================================================================
# 查看 BN 层学到的参数
# ============================================================================
# net[1].gamma 访问第 1 个 BN 层的拉伸参数
# reshape((-1,)): 展平为一维，方便查看
# 打印第一个 BN 层学到的 gamma 和 beta
print("第一个BN层的 gamma:", net[1].gamma.reshape((-1,)))
print("第一个BN层的 beta:",  net[1].beta.reshape((-1,)))
# gamma 的值 ≠ 1: 说明网络学到了需要放大/缩小某些通道
# beta 的值 ≠ 0:  说明网络学到了需要偏移某些通道


# ============================================================================
# PyTorch 简洁实现
# ============================================================================
# PyTorch 内置的 BN 比自定义实现更快（C++/CUDA 优化），功能也更完善
#   nn.BatchNorm2d: 用于 4D 输入 (batch, channels, H, W) 的卷积层
#   nn.BatchNorm1d: 用于 2D 输入 (batch, features) 的全连接层
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))

# 简要实现速度更快（C++/CUDA 编译优化 vs Python 实现）
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


# ============================================================================
# 补充知识点
# ============================================================================

# 1. 为什么批量规范化能加速训练？
#    原始网络中，每层的输入分布会随着前层参数更新而变化（内部协变量偏移）
#    BN 强制每层输入保持稳定分布（均值0，方差1），优化器不需要不断适应变化
#    这使得可以使用更大的学习率，训练更快收敛

# 2. 全连接层 vs 卷积层的 BN 差异
#    全连接层: 对每个特征（列）独立标准化，统计 batch_size 个样本
#    卷积层:   对每个通道独立标准化，统计 batch×H×W 个空间位置
#    卷积层中每个通道有独立的 γ 和 β（标量），体现"各通道重要性不同"

# 3. 为什么小批量（batch_size=1）不能用 BN？
#    均值 = 样本本身，方差 = 0，每个隐藏单元标准化后都变成 0
#    BN 的有效性依赖于足够大的 mini-batch（通常 16+，推荐 32~256）

# 4. BN 的正则化效果从哪来？
#    训练时使用 mini-batch 统计量（有噪声），带来随机性
#    类似于 Dropout，这种噪声迫使网络不过度依赖特定激活值
#    因此 BN 网络常常可以减小或取消 Dropout

# ============================================================================
# 核心公式回顾
# ============================================================================
#   BN(x) = γ * (x - μ_batch) / √(σ²_batch + ε) + β
#   移动平均: μ_global = momentum × μ_global + (1-momentum) × μ_batch
#   全连接均值: μ = mean(X, dim=0)                    # 对每个特征
#   卷积均值:   μ = mean(X, dim=(0,2,3), keepdim=True) # 对每个通道

# ============================================================================
# 总结
# ============================================================================
# 1. BN 通过 mini-batch 标准化让中间层输出分布稳定，可大幅提高学习率
# 2. 训练模式用小批量统计，预测模式用全局移动平均统计
# 3. 全连接和卷积的 BN 实现略有不同（统计维度、参数形状）
# 4. γ 和 β 是可学习参数，让网络恢复表达的灵活性
# 5. BN 是训练深层网络的关键技术，与残差连接并列现代CNN的基础
