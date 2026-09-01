"""
=============================================================================
稠密连接网络（DenseNet）—— 完整带注释版
=============================================================================
DenseNet (Densely Connected Network) 是 ResNet 的"逻辑扩展"，将跨层连接
推进到了极致。ResNet 用相加结合输入和输出，DenseNet 用连结（concat）。

核心思想（类比泰勒展开）:
  f(x) = [x, f₁(x), f₂([x,f₁(x)]), f₃([x,f₁(x),f₂(...)])), ...]
  每层都可以访问所有前面层的特征，实现极致的特征复用。

主要构成:
  - 稠密块 (DenseBlock): 层间密集连结，每层输入 = 前面所有层的拼接
  - 过渡层 (Transition Layer): 用 1×1 卷积 + 平均池化压缩通道和空间

输入: Fashion-MNIST → resize 到 96×96
输出: 10 类分类
=============================================================================
"""

import torch
from torch import nn
from d2l import torch as d2l


# ============================================================================
# 卷积块（DenseNet 使用 BN→ReLU→Conv 的顺序，即 ResNet v2 风格）
# ============================================================================
def conv_block(input_channels, num_channels):
    """
    DenseNet 的基础卷积块。

    与 ResNet 原始块的区别：采用 BN → ReLU → Conv 的顺序
    （也称 "pre-activation" 设计），梯度流动更顺畅。

    参数:
        input_channels: 输入通道数
        num_channels:   输出通道数（也是"增长率" growth rate）

    返回:
        nn.Sequential: BN → ReLU → 3×3 Conv(padding=1)
    """
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))


# ============================================================================
# 稠密块（DenseBlock）
# ============================================================================
class DenseBlock(nn.Module):
    """
    稠密块：每个卷积块将输出与输入在通道维上连结。

    通道增长规律（增长率 k = num_channels）:
      输入: C 通道
      第1个卷积块后: C + k 通道
      第2个卷积块后: C + 2k 通道
      ...
      第 n 个卷积块后: C + n×k 通道

    这个 k 就是 "增长率" (growth rate)，通常取较小值 (12, 24, 32, 40)
    因为每个块都能看到所有前面层的特征，不需要每层输出很多通道。

    参数:
        num_convs:      卷积块的数量
        input_channels: 初始输入通道数
        num_channels:   增长率 k（每个卷积块的输出通道数）
    """
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            # 第 i 个卷积块的输入通道 = 原始输入 + 前面 i 个块的输出
            #   输入: num_channels * i + input_channels
            #   输出: num_channels (即增长率 k)
            # 例如 k=32, 初始=64:
            #   块0: 输入 64  → 输出 32, 拼接后=96
            #   块1: 输入 96  → 输出 32, 拼接后=128
            #   块2: 输入 128 → 输出 32, 拼接后=160
            layer.append(conv_block(
                num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        """
        前向传播：逐块执行 BN→ReLU→Conv→Concat。

        X 初始形状: (batch, C, H, W)
        每个块后:    通道数增加 k，H 和 W 保持不变
        """
        for blk in self.net:
            Y = blk(X)
            # torch.cat((X, Y), dim=1): 在通道维度(dim=1)上拼接
            #   将旧特征和新特征连在一起，供后续层使用
            #   X 包含之前所有层的信息，Y 是当前层的新特征
            X = torch.cat((X, Y), dim=1)
        return X


# ============================================================================
# 测试稠密块
# ============================================================================
# 2 个卷积块，增长率=10，输入 3 通道
# 输出通道 = 3 + 2×10 = 23
blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)  # 4 个样本，3 通道，8×8
Y = blk(X)
print("稠密块输出形状:", Y.shape)  # torch.Size([4, 23, 8, 8])


# ============================================================================
# 过渡层（Transition Layer）
# ============================================================================
# 用途：在稠密块之间压缩通道数和下采样空间尺寸
# 如果不压缩，通道数会随深度线性增长，模型会变得过于庞大
def transition_block(input_channels, num_channels):
    """
    过渡层：BN → ReLU → 1×1Conv(减通道) → AvgPool2d(减半高宽)。

    参数:
        input_channels: 输入通道数（通常很大，因为经过稠密块的增长）
        num_channels:   目标输出通道数

    返回:
        nn.Sequential: 过渡层
    """
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        # 1×1 卷积压缩通道数: input → num_channels
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        # 平均池化减半高宽: 使用 Average Pooling 而非 Max Pooling
        #   原因: 平均池化保留所有位置的信息，与 DenseNet 的"特征复用"理念一致
        nn.AvgPool2d(kernel_size=2, stride=2))


# 测试过渡层
blk = transition_block(23, 10)
print("过渡层输出形状:", blk(Y).shape)  # torch.Size([4, 10, 4, 4])
# 输入 (4,23,8,8) → 输出 (4,10,4,4): 通道从 23 压到 10，高宽减半


# ============================================================================
# DenseNet 完整模型
# ============================================================================
# 起始模块: 同 ResNet 的 7×7 Conv + BN + ReLU + 3×3 MaxPool
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 构建 4 个稠密块 + 3 个过渡层
#   每个稠密块有 4 个卷积块，增长率 k=32
#   每个稠密块增加 4×32 = 128 个通道
num_channels, growth_rate = 64, 32  # 初始通道=64，增长率=32
num_convs_in_dense_blocks = [4, 4, 4, 4]  # 4 个稠密块各有 4 个卷积块
blks = []
for i, num_convs in enumerate(num_convs_in_dense_blocks):
    # 添加稠密块
    blks.append(DenseBlock(num_convs, num_channels, growth_rate))
    # 更新当前通道数（稠密块增加了 num_convs * growth_rate）
    num_channels += num_convs * growth_rate
    # 在稠密块之间添加过渡层（最后一个稠密块后不加）
    #   过渡层将通道数减半，高宽减半
    if i != len(num_convs_in_dense_blocks) - 1:
        blks.append(transition_block(num_channels, num_channels // 2))
        num_channels = num_channels // 2  # 更新通道数为减半后的值

# 通道数变化轨迹:
#   b1 输出: 64
#   稠密块1: 64 + 4×32 = 192
#   过渡层1: 192 → 96
#   稠密块2: 96 + 4×32 = 224
#   过渡层2: 224 → 112
#   稠密块3: 112 + 4×32 = 240
#   过渡层3: 240 → 120
#   稠密块4: 120 + 4×32 = 248

# 组装完整网络
net = nn.Sequential(
    b1, *blks,  # *blks 解包列表，展开为独立参数
    # 最后的 BN + ReLU + 全局平均池化
    nn.BatchNorm2d(num_channels), nn.ReLU(),
    nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化: (batch,C,H,W) → (batch,C,1,1)
    nn.Flatten(),                   # (batch,C,1,1) → (batch,C)
    nn.Linear(num_channels, 10))    # 分类层: (batch,C) → (batch,10)


# ============================================================================
# 训练 DenseNet
# ============================================================================
# resize=96: 使用 96×96 输入加速（深层网络计算量大）
lr, num_epochs, batch_size = 0.1, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


# ============================================================================
# 补充知识点
# ============================================================================

# 1. DenseNet 与 ResNet 的本质区别
#    ResNet:  相加 (add)        → 信息混合，不增加维度
#    DenseNet: 连结 (concat)     → 信息保留，维度增长
#    相加 = f(x) + x，连结 = [f(x), x]
#    连结保留了所有历史特征，后续层可以直接访问，但代价是内存消耗更大

# 2. 增长率 k (growth rate)
#    k 通常很小（12、24、32、40），因为：
#    - 每个层可以访问所有前面的特征 → 不需要输出很多来"记住"信息
#    - 全局特征被重复使用 → 每个层只需贡献少量新的信息
#    k=32 时，每层只加 32 个新通道，但能看到几百个历史通道

# 3. 为什么过渡层用平均池化？
#    最大池化只保留最显著特征（丢弃其他信息）
#    平均池化保留所有位置的信息 → 与 DenseNet 的"特征复用"哲学一致
#    DenseNet 希望每层都能访问所有历史信息

# 4. DenseNet 的优缺点
#    优点:
#      - 参数效率极高（比 ResNet 用更少参数达到同等精度）
#      - 梯度流动极好（每层都直接连接 loss）
#      - 天然的正则化效果（特征复用减少过拟合）
#    缺点:
#      - 显存消耗巨大（需要存储所有中间特征用于 concat）
#      - 实际训练/推理速度较慢（concat 和大量通道处理开销）
#      - 与其他架构比，工程优化较差

# 5. DenseNet 的显存消耗
#    假设 batch=32, 特征图 56×56, 100 通道:
#      ResNet:   只需存当前层特征 + 前一层（用于反向传播）
#      DenseNet: 需要存每一层的输出（因为每层都可能被后续 concat 使用）
#    这就是 DenseNet "内存不友好" 的根本原因

# ============================================================================
# 核心公式回顾
# ============================================================================
#   稠密块通道增长: C_out = C_in + num_convs × growth_rate
#   过渡层通道压缩: C_out = C_in // 2
#   DenseNet 完整的层连接: x_l = H_l([x_0, x_1, ..., x_{l-1}])
#     第 l 层的输入 = 前面所有层的输出在通道维上的拼接

# ============================================================================
# 总结
# ============================================================================
# 1. DenseNet 用通道维连结（concat）替代 ResNet 的相加（add）
# 2. 主要构造：稠密块（密集连结）+ 过渡层（压缩通道和空间）
# 3. 增长率 k 控制通道增长速度，通常设为较小值（如 32）
# 4. 参数效率高、梯度流动好，但显存消耗大
# 5. 是 ResNet 思想在"特征复用"方向上的极致扩展
