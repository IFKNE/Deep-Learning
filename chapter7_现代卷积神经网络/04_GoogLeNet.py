"""
=============================================================================
含并行连结的网络（GoogLeNet）—— 完整带注释版
=============================================================================
GoogLeNet (又名 Inception v1) 在 2014 年 ImageNet 挑战赛夺冠，核心贡献是
Inception 块：在同一层内部并行使用多种大小的卷积核来提取多尺度特征。

关键思想：
  - 不同大小的卷积核捕获不同范围的特征（1×1 点状、3×3 小范围、5×5 大范围）
  - 1×1 卷积做 Bottleneck 降维，大幅减少计算量
  - 全局平均池化替代全连接层（继承自 NiN）

输入: Fashion-MNIST → resize 到 96×96（简化计算）
输出: 10 类分类
=============================================================================
"""

import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l


# ============================================================================
# Inception 块
# ============================================================================
# Inception 块包含 4 条并行路径，最后在通道维度连结（concat）：
#
#   路径1: 1×1 卷积 → 直接提取点状特征
#   路径2: 1×1 卷积 → 3×3 卷积 → 先降维再提取小范围特征
#   路径3: 1×1 卷积 → 5×5 卷积 → 先降维再提取大范围特征
#   路径4: 3×3 最大池化 → 1×1 卷积 → 池化提取显著性 + 调整通道
#
# 为什么路径2/3先做 1×1 卷积？
#   假设输入 192 通道，直接做 3×3卷积(192→128): 192×3×3×128 ≈ 221K 参数
#   先 1×1 (192→96) 再做 3×3 (96→128): 192×1×1×96 + 96×3×3×128 ≈ 18K+111K ≈ 129K
#   参数减少约 42%，这就是 Bottleneck 设计的精髓！

class Inception(nn.Module):
    """
    Inception 块：4 条并行路径 + 通道维度连结。

    参数:
        in_channels: 输入通道数
        c1: 路径1（1×1卷积）的输出通道数
        c2: 路径2 的中间和最终通道数 (c2[0]=1×1输出, c2[1]=3×3输出)
        c3: 路径3 的中间和最终通道数 (c3[0]=1×1输出, c3[1]=5×5输出)
        c4: 路径4（最大池化+1×1）的输出通道数
    """
    # c1--c4 是每条路径的输出通道数，总输出通道 = c1 + c2[1] + c3[1] + c4
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # ---- 线路1: 单 1×1 卷积 ----
        # 无中间层，直接对每个像素做通道线性组合
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)

        # ---- 线路2: 1×1 卷积 → 3×3 卷积 ----
        # Bottleneck: 先 1×1 降维，再 3×3 提取空间特征
        # c2[0] 是中间通道数（通常比 in_channels 小）
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)

        # ---- 线路3: 1×1 卷积 → 5×5 卷积 ----
        # 5×5 的 padding=2: 确保输出高宽与输入一致
        #   公式: (n + 2*2 - 5)/1 + 1 = n，保持不变
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        # ---- 线路4: 3×3 最大池化 → 1×1 卷积 ----
        # 池化提取最显著特征（无参数），再 1×1 调整通道
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        """
        前向传播：4 条路径并行计算，最后在通道维度连结。

        输入: x shape = (batch, in_channels, H, W)
        输出: shape = (batch, c1+c2[1]+c3[1]+c4, H, W)
        """
        # 路径1: 1×1 卷积
        p1 = F.relu(self.p1_1(x))
        # 路径2: 1×1 → ReLU → 3×3 → ReLU
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        # 路径3: 1×1 → ReLU → 5×5 → ReLU
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        # 路径4: 3×3 MaxPool → 1×1 → ReLU
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # torch.cat((p1, p2, p3, p4), dim=1): 在通道维度(dim=1)上拼接
        #   4 个输出高宽相同，通道数相加
        return torch.cat((p1, p2, p3, p4), dim=1)


# ============================================================================
# GoogLeNet 模型 —— 逐模块构建
# ============================================================================

# 模块1 (b1)：类似 AlexNet 的开头
#   7×7 卷积(1→64, stride=2) + ReLU + 3×3 MaxPool(stride=2)
#   输入 96×96 → 输出 64×24×24
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 模块2 (b2)：1×1 卷积降维 + 3×3 卷积扩展
#   Conv 1×1(64→64) + ReLU + Conv 3×3(64→192) + ReLU + MaxPool 3×3(stride=2)
#   输出: (batch, 192, 12, 12)
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.ReLU(),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 模块3 (b3)：2 个 Inception 块
#   第1个 Inception 块: 输入 192 通道
#     路径1(1×1): 64,  路径2(1×1→3×3): 96→128
#     路径3(1×1→5×5): 16→32, 路径4(池化→1×1): 32
#     总输出 = 64+128+32+32 = 256 通道
#   第2个 Inception 块: 输入 256 通道
#     总输出 = 128+192+96+64 = 480 通道
#   最后 MaxPool → (batch, 480, 6, 6)
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 模块4 (b4)：5 个 Inception 块的堆叠
#   通道数逐步增加: 512→512→512→528→832
#   每个块的路径 2（含 3×3）输出最多通道，其次是路径 1（纯 1×1）
#   各路径间的通道分配比例是通过 ImageNet 上的大量实验确定的
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

# 模块5 (b5)：2 个 Inception 块 + 全局平均池化 + Flatten
#   总输出 = 832→832→1024
#   AdaptiveAvgPool2d((1,1)): 将每个通道的 3×3 → 1×1
#   Flatten: (batch, 1024, 1, 1) → (batch, 1024)
b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   nn.AdaptiveAvgPool2d((1, 1)),
                   nn.Flatten())

# 组装完整网络：b1→b2→b3→b4→b5→ 输出层
# b5 输出 1024 通道 → nn.Linear(1024, 10) 映射到 10 个类别
net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))


# ============================================================================
# 查看各模块输出形状（使用 96×96 输入简化计算）
# ============================================================================
X = torch.rand(size=(1, 1, 96, 96))
print("=== GoogLeNet 各模块输出形状 (input=96×96) ===")
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
# 输出规律: 通道递增 (64→192→480→832→1024)，空间递减 (24→12→6→3→1)


# ============================================================================
# 训练 GoogLeNet
# ============================================================================
# resize=96: 使用 96×96 替代 224×224 加速训练（Fashion-MNIST 足够简单）
lr, num_epochs, batch_size = 0.1, 10, 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())


# ============================================================================
# 补充知识点
# ============================================================================

# 1. Inception 块的通道分配比例
#    在 ImageNet 上的经验规律（第二个 Inception 块为例）：
#      路径1(1×1): 128 (占比 128/480≈27%)
#      路径2(1×1→3×3): 192 (占比 40%)  ← 3×3 卷积输出最多通道
#      路径3(1×1→5×5): 96  (占比 20%)
#      路径4(池化→1×1): 64  (占比 13%)
#    各 Inception 块的比例略有不同，但都是通过大量实验确定的

# 2. 为什么用不同大小的卷积核？
#    不同核大小捕获不同感受野的信息：
#    - 1×1: 点状特征（如颜色、纹理）
#    - 3×3: 局部结构（如图像中的边缘片段）
#    - 5×5: 更大的局部结构（如图像中较完整的形状）
#    并行提取并融合，让网络自己决定用哪种尺度的信息

# 3. GoogLeNet 后续版本
#    - Inception v2: 加入 Batch Normalization
#    - Inception v3: 将大卷积核分解（5×5→两个3×3），类似VGG思想
#    - Inception v4 / Inception-ResNet: 融入残差连接

# ============================================================================
# 核心公式回顾
# ============================================================================
#   Bottleneck 参数节省（输入C_in，中间C_mid，输出C_out，核大小K）:
#     直接: C_in × K² × C_out
#     Bottleneck: C_in × 1² × C_mid + C_mid × K² × C_out
#   Inception 输出通道: C_total = c1 + c2[1] + c3[1] + c4

# ============================================================================
# 总结
# ============================================================================
# 1. Inception 块的 4 条并行路径可同时提取多尺度特征
# 2. 1×1 卷积的 Bottleneck 设计大幅减少计算量和参数量
# 3. GoogLeNet 用 9 个 Inception 块 + 全局平均池化构建深层网络
# 4. 通道分配比例是 ImageNet 上大量实验调优的结果
# 5. 相比 AlexNet/VGG，计算复杂度更低，但精度相当或更好
