"""
=============================================================================
残差网络（ResNet）—— 完整带注释版
=============================================================================
ResNet (Residual Network) 由何恺明等人于 2015 年提出，在当年 ImageNet 竞赛
夺冠。它的核心创新——残差连接（skip connection）——使得训练 100+ 层的超深
网络成为可能，深刻影响了此后所有深度网络的设计。

核心思想：
  "学习残差比学习原始映射更容易"
  F(x) = H(x) - x  →  网络只需学习目标与输入的差异部分
  极端情况：若最优映射就是恒等映射，只需将卷积层权重置零即可

为什么需要残差连接？
  深层网络的梯度在反向传播时要连乘很多层
  如果每层的 Jacobian 都 < 1 → 梯度指数衰减 → 梯度消失
  跳跃连接提供了一条"快速通道"：梯度可以不经衰减直接传回浅层

输入: Fashion-MNIST → resize 到 96×96
输出: 10 类分类
=============================================================================
"""

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


# ============================================================================
# 残差块（Residual Block）
# ============================================================================
# 残差块的两种变体：
#   1. 不改变通道数/空间尺寸：输入直接加到输出（use_1x1conv=False）
#      结构: Conv3×3→BN→ReLU→Conv3×3→BN → +x → ReLU
#
#   2. 改变通道数或下采样：用 1×1 卷积调整输入形状使匹配（use_1x1conv=True）
#      结构: Conv3×3→BN→ReLU→Conv3×3→BN → + (1×1 Conv(x)) → ReLU
#
# BN 层的放置顺序沿用原始论文：Conv → BN → ReLU

class Residual(nn.Module):  #@save
    """
    残差块：通过跳跃连接实现残差学习。

    参数:
        input_channels: 输入通道数
        num_channels:   输出通道数（块内两个卷积层统一使用）
        use_1x1conv:    是否使用 1×1 卷积调整跳跃连接
                        当 input_channels ≠ num_channels 或 strides≠1 时需设为 True
        strides:        第一个卷积层的步幅（用于下采样）
    """
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        # 第1个卷积层: 注意 stride 参数可能 > 1（用于下采样）
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        # 第2个卷积层: stride=1 保持空间尺寸
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        # 1×1 卷积：当输入/输出通道数不同或空间尺寸不同时用来调整跳跃连接
        #   kernel_size=1, stride=strides: 使输出与 conv1 的空间尺寸一致
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None  # 不需要调整时设为 None，forward 中跳过
        # 批量规范化层
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        """
        残差块的前向传播。

        X ─┬─→ Conv1 → BN1 → ReLU → Conv2 → BN2 ─→ (+) → ReLU → Y
           │                                        ↑
           └─→ (可选: 1×1 Conv 调整形状) ──────────┘
        """
        # 主路径: Conv → BN → ReLU → Conv → BN
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        # 跳跃连接: 如果形状不匹配，用 1×1 卷积调整
        if self.conv3:
            X = self.conv3(X)
        # 相加：Y += X，即 Y = Y + X
        #   张量加法要求形状一致（广播规则也会自动扩展）
        Y += X
        # 相加后再过 ReLU: 残差块的最终输出
        return F.relu(Y)


# ============================================================================
# 测试残差块
# ============================================================================
# 测试1: 输入输出形状一致（use_1x1conv=False）
blk = Residual(3, 3)  # 输入 3 通道，输出 3 通道
X = torch.rand(4, 3, 6, 6)  # (4个样本, 3通道, 6×6)
Y = blk(X)
print("同形状残差块输出:", Y.shape)  # torch.Size([4, 3, 6, 6])
# 输入 (4,3,6,6) → 输出 (4,3,6,6)，形状完全不变

# 测试2: 增加通道数并减半高宽（use_1x1conv=True, strides=2）
blk = Residual(3, 6, use_1x1conv=True, strides=2)
print("下采样残差块输出:", blk(X).shape)  # torch.Size([4, 6, 3, 3])
# 输入 (4,3,6,6) → 输出 (4,6,3,3)，通道翻倍，空间减半


# ============================================================================
# ResNet-18 模型
# ============================================================================
# ResNet-18 的整体结构（共 18 层）:
#   起始层: 7×7 Conv + BN + ReLU + 3×3 MaxPool
#   模块1 (b2): 2 个残差块，通道=64，空间不变
#   模块2 (b3): 2 个残差块，通道=128，空间减半
#   模块3 (b4): 2 个残差块，通道=256，空间减半
#   模块4 (b5): 2 个残差块，通道=512，空间减半
#   输出: 全局平均池化 → Flatten → Linear(512, 10)
# 层数计算: 1(起始Conv) + 4×2×2(残差块各有2个Conv) + 1(FC) = 18

# 起始模块: 类似 GoogLeNet 的开头
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))


def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    """
    构建由多个残差块组成的 ResNet 模块。

    参数:
        input_channels: 输入通道数
        num_channels:   输出通道数（模块内所有残差块统一）
        num_residuals:  残差块的数量
        first_block:    是否为第一个模块（第一个模块不需要在首块做下采样）

    返回:
        list: 残差块的列表（可直接解包传给 nn.Sequential）
    """
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # 非首个模块的第 1 个残差块：需要 1×1 卷积调整通道并下采样
            #   use_1x1conv=True: 调整跳跃连接的通道数
            #   strides=2: 空间尺寸减半
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            # 其他残差块：输入输出通道相同，简单加法
            blk.append(Residual(num_channels, num_channels))
    return blk


# 构建 4 个模块，每个模块包含 2 个残差块
# b2: (batch,64,56,56) → (batch,64,56,56)  空间不变
b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
# b3: (batch,64,56,56) → (batch,128,28,28) 通道翻倍，空间减半
b3 = nn.Sequential(*resnet_block(64, 128, 2))
# b4: (batch,128,28,28) → (batch,256,14,14)
b4 = nn.Sequential(*resnet_block(128, 256, 2))
# b5: (batch,256,14,14) → (batch,512,7,7)
b5 = nn.Sequential(*resnet_block(256, 512, 2))

# 组装完整 ResNet-18
net = nn.Sequential(b1, b2, b3, b4, b5,
                    nn.AdaptiveAvgPool2d((1, 1)),  # (batch,512,7,7) → (batch,512,1,1)
                    nn.Flatten(),                   # (batch,512,1,1) → (batch,512)
                    nn.Linear(512, 10))             # (batch,512) → (batch,10)


# ============================================================================
# 查看各模块输出形状
# ============================================================================
X = torch.rand(size=(1, 1, 224, 224))
print("=== ResNet-18 各模块输出形状 ===")
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
# 输出: 64@56×56 → 64@56×56 → 128@28×28 → 256@14×14 → 512@7×7 → 512 → 10


# ============================================================================
# 训练 ResNet-18
# ============================================================================
# resize=96: 使用 96×96 输入加速训练
lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


# ============================================================================
# 补充知识点
# ============================================================================

# 1. 残差学习为什么有效？—— 数学直觉
#    目标: 学习映射 H(x)
#    残差: 学习 F(x) = H(x) - x，再计算 H(x) = F(x) + x
#    如果最优 H(x) 接近恒等映射（深层常见），F(x)≈0
#    → 网络只需将权重推近 0 即可，这比学习精确的恒等映射容易得多

# 2. Bottleneck 残差块（ResNet-50/101/152 使用）
#    标准残差块（ResNet-18/34）:  3×3 → 3×3 (两个 3×3 卷积)
#    Bottleneck 残差块:          1×1(降维) → 3×3 → 1×1(升维)
#    例如: 256→64→64→256，大幅减少计算量，使更深的网络可行
#    ResNet-50: 3+4+6+3=16 个 Bottleneck 块 × 3层 = 48 卷积层 + 起始层 + FC = 50 层

# 3. Pre-activation ResNet（ResNet v2）
#    原始顺序: Conv → BN → ReLU
#    v2 顺序:  BN → ReLU → Conv（激活函数在卷积之前）
#    v2 的梯度流动路径更"纯净"，表现略好

# 4. ResNet vs GoogLeNet
#    两者都使用全局平均池化和 7×7 起始卷积
#    但 ResNet 的设计更简洁统一，不依赖精心设计的 Inception 块
#    简洁性让 ResNet 更容易修改和迁移到其他任务

# ============================================================================
# 核心公式回顾
# ============================================================================
#   残差学习:      y = F(x, {Wi}) + x
#   梯度流动:      ∂L/∂x = ∂L/∂y × (1 + ∂F/∂x)
#                  ← 那个 "+1" 保证了梯度不会消失！
#   Bottleneck:    x → 1×1(C→C/4) → 3×3 → 1×1(C/4→C) → +x

# ============================================================================
# 总结
# ============================================================================
# 1. 残差块通过跳跃连接 (x + F(x)) 解决了深层网络的梯度消失问题
# 2. 学习残差 F(x)=H(x)-x 比直接学习 H(x) 更容易，尤其是接近恒等映射时
# 3. ResNet-18 有 18 层，ResNet-50/101/152 用 Bottleneck 设计实现更深
# 4. 配合 BN，残差连接使得训练 100+ 层网络成为可能
# 5. ResNet 的简洁统一设计使其成为计算机视觉最广泛使用的骨干网络
