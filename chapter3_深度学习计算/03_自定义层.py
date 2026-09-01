"""
=============================================================================
自定义层 —— 无参数层与带参数层（完整带注释版）
=============================================================================
本文件讲解如何从零构建自定义神经网络层：
  - 无参数层：仅做数据变换，无可学习参数（如 CenteredLayer）
  - 带参数层：用 nn.Parameter 定义可学习权重（如 MyLinear）
  - 自定义层与 nn.Sequential 的无缝集成
=============================================================================
"""

# ============================================================================
# 第1步：导入库
# ============================================================================
import torch
import torch.nn.functional as F
from torch import nn


# ============================================================================
# 第1节：无参数自定义层 —— CenteredLayer
# ============================================================================
# 有些层不需要可学习参数，它们只是对数据做固定的数学变换。
# 这类层仍然需要继承 nn.Module，但 __init__ 中不需要创建任何 Parameter。

class CenteredLayer(nn.Module):
    """
    中心化层：将输入数据减去其均值，使输出均值为 0。

    数学模型：CenteredLayer(X) = X - mean(X)

    这是一个"无参数层"的最简示例。
    虽然没有任何可学习参数，但：
      - 继承 nn.Module 意味着可以放进 Sequential 中
      - forward 中的操作会被 autograd 追踪（梯度可以通过它传播）
      - 可以被 .to(device) 等方法统一管理

    类比 C++ 的面向对象：
      nn.Module  = 抽象基类
      __init__() = 构造函数（这里不需要创建成员变量）
      forward()  = 必须重写的虚函数
    """
    def __init__(self):
        # 调用 nn.Module 的构造函数（必须！）
        # 即使没有参数，这一步也不能省——它初始化了内部的 _modules 等字典
        super().__init__()

    def forward(self, X):
        # X.mean() 默认对所有元素取平均，返回标量（0 维张量）
        # X - X.mean()：利用广播机制 —— 标量会自动扩展为与 X 相同形状
        # 例如 [1, 2, 3, 4, 5] → 均值 3.0 → 输出 [-2, -1, 0, 1, 2]
        return X - X.mean()

# 测试 CenteredLayer
layer = CenteredLayer()
# 均值 = (1+2+3+4+5)/5 = 3.0
# 输出 = [1-3, 2-3, 3-3, 4-3, 5-3] = [-2, -1, 0, 1, 2]
layer(torch.FloatTensor([1, 2, 3, 4, 5]))


# ============================================================================
# 第2节：将自定义层与标准层组合
# ============================================================================
# 自定义层可以无缝放入 nn.Sequential 中，与标准层混合使用。
# 这是 nn.Module 统一接口的威力 —— 只要继承 nn.Module，任何层都是"一等公民"。

# 构建：Linear(8→128) → CenteredLayer（均值归零）
# CenteredLayer 对 Linear 的输出做中心化，相当于一种数据标准化操作
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())

# 4 个样本，每个 8 维 → Linear 变换 → 128 维 → 中心化
Y = net(torch.rand(4, 8))

# 经过 CenteredLayer 后，Y 的均值应该接近 0
# （由于浮点精度限制，通常是一个极小值如 1e-8 而非精确的 0）
Y.mean()


# ============================================================================
# 第3节：带参数的层 —— MyLinear
# ============================================================================
# 大多数层需要可学习参数。本节手动实现一个简化版的全连接层，
# 帮助你理解 nn.Linear 的底层原理。

class MyLinear(nn.Module):
    """
    自己实现的全连接层（简化版），展示 nn.Parameter 的正确用法。

    公式（简化版，不使用 W^T，直接用 matmul）：
      out = relu(X @ W + b)

    其中：
      X: (batch_size, in_units)   — 输入
      W: (in_units, units)        — 权重
      b: (units,)                 — 偏置
      out: (batch_size, units)    — 输出

    与 nn.Linear 的关键区别：
      - nn.Linear 存的是 (out_features, in_features) 形状的权重，用 X @ W^T + b
      - MyLinear 存的是 (in_features, out_features) 形状的权重，用 X @ W + b
      - 数学上等价，只是权重的存储方向和乘法方式不同
    """
    def __init__(self, in_units, units):
        """
        参数:
            in_units (int): 输入特征数
            units (int):    输出特征数
        """
        super().__init__()

        # nn.Parameter(data)：将普通张量包装为"可学习参数"
        #
        # nn.Parameter 是 Tensor 的子类，但有一个关键特性：
        # 当它被赋值给 nn.Module 的属性（如 self.weight）时，
        # nn.Module.__setattr__ 会自动将该 Parameter 注册到 self._parameters 字典中。
        # → parameters() 能遍历到它
        # → optimizer 能找到并更新它
        # → state_dict() 会包含它
        # → .to(device) 会移动它
        #
        # torch.randn(shape)：从标准正态分布 N(0, 1) 中采样
        self.weight = nn.Parameter(torch.randn(in_units, units))  # 形状 (in_units, units)
        self.bias = nn.Parameter(torch.randn(units,))            # 形状 (units,)

    def forward(self, X):
        # torch.matmul(A, B) = 矩阵乘法
        # 形状：(batch, in_units) @ (in_units, units) → (batch, units)
        #
        # 这里用 self.weight.data 和 self.bias.data 而不是直接用 Parameter：
        #   .data 返回底层 Tensor，绕过 autograd 追踪
        #   为什么这样做？在这个简化实现中，如果直接用 self.weight，
        #   matmul 结果的 grad_fn 会指向这个 Parameter，
        #   而在某些 autograd 实现中，Parameter 被直接用于计算可能导致
        #   不符合预期的梯度累积路径。使用 .data 可以更精细地控制。
        #   注意：nn.Linear 的官方实现对此有更完善的处理方式。
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)  # 经过 ReLU 激活


# ============================================================================
# 第4节：测试自定义带参数层
# ============================================================================
# 创建 MyLinear 实例：5 个输入特征 → 3 个输出特征
linear = MyLinear(5, 3)

# 查看 weight 参数
# Parameter 对象：含 requires_grad=True 和一些元信息
# 形状 (5, 3)：5 个输入 → 3 个输出的权重矩阵
linear.weight

# 前向传播：2 个样本，每个 5 维 → 输出 (2, 3)
# 由于使用 ReLU 激活，所有负值都会被置零
# 如果权重和输入导致 matmul 结果为全负，输出可能全为 0
linear(torch.rand(2, 5))


# ============================================================================
# 第5节：用自定义层构建完整模型
# ============================================================================
# 自定义的 MyLinear 可以像 nn.Linear 一样放在 Sequential 中。
# 这再次体现了 nn.Module 接口设计的优雅——任何层都是可互换的组件。

# 两个 MyLinear 串联：64 → 8 → 1
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
# 前向传播：(2, 64) → MyLinear(64→8) → ReLU → MyLinear(8→1) → ReLU → (2, 1)
net(torch.rand(2, 64))


# ============================================================================
# 补充知识点
# ============================================================================

# ============================================================================
# 补充 1：自定义层的完整实现（对标 nn.Linear 的正确写法）
# ============================================================================
class MyLinearFull(nn.Module):
    """
    nn.Linear 的更完整实现，展示在 forward 中正确使用 Parameter。

    与上面 MyLinear 的关键区别：
      在 forward 中直接使用 self.weight（不取 .data），
      让 autograd 引擎正确追踪操作并构建计算图。
      这是生产代码中推荐的做法。

    公式：out = input @ weight + bias
      其中 weight 形状为 (in_features, out_features)
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        # 缩放初始值：乘以 0.01 防止初始输出过大
        # 对于深层网络，过大的初始值会导致梯度爆炸
        self.weight = nn.Parameter(
            torch.randn(in_features, out_features) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # 直接用 self.weight（不取 .data）
        # autograd 会正确记录 @ 和 + 操作，构建计算图
        # @ 运算符：Python 3.5+ 引入的矩阵乘法语法糖
        return x @ self.weight + self.bias


# ============================================================================
# 补充 2：nn.Parameter vs 普通 Tensor —— 深入理解
# ============================================================================
"""
在 __init__ 中，以下三种写法的效果完全不同：

  # 写法 1：赋值为 nn.Parameter（✅ 推荐）
  self.weight = nn.Parameter(torch.randn(4, 8))
  → self.weight 被自动注册到 self._parameters
  → parameters() 能找到它
  → optimizer 会更新它
  → state_dict() 会保存它

  # 写法 2：赋值为普通 Tensor（❌ 不参与训练！）
  self.weight = torch.randn(4, 8)
  → self.weight 是一个普通属性，不是 Parameter
  → parameters() 找不到它！
  → optimizer 不会更新它！
  → state_dict() 不会保存它！
  → 即使你写了训练循环，这个"参数"永远不会被更新

  # 写法 3：用 register_parameter() 手动注册
  weight = torch.randn(4, 8)
  self.register_parameter('weight', nn.Parameter(weight))
  → 等价于写法 1，但更繁琐

  关键教训：
    任何需要被训练的权重，必须用 nn.Parameter 包裹！
    普通 Tensor 不会被 PyTorch 当作模型参数。
"""


# ============================================================================
# 补充 3：常见问题 / 练习思考
# ============================================================================
"""
Q1: 无参数层有必要继承 nn.Module 吗？
A1: 是的。虽然它没有可学习参数，但继承 nn.Module 赋予了它：
    - 能被放入 Sequential 的能力
    - 参与 autograd 计算图（梯度可以通过它传播到之前的层）
    - 统一的设备管理（.to(device)）
    - 统一的模式切换（.train()/.eval()）
    如果不继承，它只是一个普通的 Python 函数，无法集成到 PyTorch 生态中。

Q2: MyLinear 的 forward 中用 .data 访问参数，有什么风险？
A2: .data 完全绕过了 autograd，如果在 .data 上做了会被追踪的操作，
    可能导致梯度计算不正确。在生产代码中应直接用 self.weight（不取 .data）。
    上面补充 1 中的 MyLinearFull 展示了正确的写法。

Q3: 如何自定义一个 Dropout 层？
A3: 思路：
    class MyDropout(nn.Module):
        def __init__(self, p=0.5):
            super().__init__()
            self.p = p  # p 不是 Tensor，不需要 nn.Parameter

        def forward(self, X):
            if self.training:  # self.training 是 nn.Module 的属性
                mask = (torch.rand_like(X) > self.p).float()
                return X * mask / (1 - self.p)
            return X  # 评估模式：不做任何事

Q4: 自定义层中如何实现多个可学习参数？
A4: 每个可学习参数单独声明为 nn.Parameter：
    self.alpha = nn.Parameter(torch.tensor(1.0))
    self.beta = nn.Parameter(torch.tensor(0.0))
    在 forward 中使用它们即可。
"""


# ============================================================================
# 总结
# ============================================================================
"""
1. 自定义层 = 继承 nn.Module + 重写 __init__ 和 forward

2. 无参数层（如 CenteredLayer）：
   - __init__ 中不需要创建 Parameter
   - forward 中做固定数学变换
   - 仍然可以无缝集成到 Sequential 中

3. 带参数层（如 MyLinear）：
   - 用 nn.Parameter 包裹可学习张量
   - 不能用普通 Tensor！（不会被 optimizer 找到）
   - 在 forward 中推荐直接用 self.weight（不取 .data）

4. 自定义层 = 一等公民：
   - 可以放进 Sequential
   - 可以与其他标准层任意组合
   - 这就是 nn.Module 统一接口设计的价值
"""
