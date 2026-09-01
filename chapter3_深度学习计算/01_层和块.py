"""
=============================================================================
层和块 —— 模型构造（完整带注释版）
=============================================================================
本文件讲解 PyTorch 中神经网络模型构造的核心概念：
  - nn.Module 基类与 nn.Sequential 容器
  - 自定义块：继承 nn.Module 并重写 forward
  - 手动实现 MySequential 理解层容器内部原理
  - 在 forward 中编写任意 Python 控制流（动态计算图）
  - 块的嵌套与组合
=============================================================================
"""

# ============================================================================
# 第1步：导入库
# ============================================================================
import torch
from torch import nn
from torch.nn import functional as F


# ============================================================================
# 第1节：回顾多层感知机 —— nn.Sequential
# ============================================================================
# nn.Sequential 是一个"层容器"（也叫"块"，Block），
# 将多个层按顺序串联起来，数据依次流过每一层。
#
# nn.Sequential 的本质：
#   它继承自 nn.Module，内部用一个 OrderedDict（self._modules）
#   按顺序存储子模块。调用 forward() 时，数据依次流过每个子模块。
#
# 类比 C++ 中的概念：
#   nn.Module      ≈ 抽象基类（定义接口）
#   nn.Sequential  ≈ std::vector<Layer> + 顺序执行的 pipeline
#   forward()      ≈ 虚函数，子类必须重写

# nn.Sequential(A, B, C): 将 A、B、C 三个模块串联
#   数据流向：输入 → Linear(20→256) → ReLU → Linear(256→10) → 输出
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

# torch.rand(shape): 从 [0, 1) 均匀分布中采样，创建指定形状的张量
#   X 形状：(2, 20) = 2 个样本，每个样本 20 个特征
X = torch.rand(2, 20)

# 前向传播：net(X) 等价于 net.__call__(X)
#   __call__ 内部会自动做预处理（如注册 hook），然后调用 net.forward(X)
# 形状变化追踪：
#   (2, 20) → Linear(20→256) → (2, 256)
#           → ReLU()          → (2, 256)  # ReLU 不改变形状
#           → Linear(256→10)  → (2, 10)   # 10 个类别的得分
net(X)


# ============================================================================
# 第2节：自定义块 —— 继承 nn.Module
# ============================================================================
# nn.Module 是所有神经网络模块的基类。任何自定义网络都必须：
#   (1) 继承 nn.Module
#   (2) 在 __init__() 中声明网络层（类似 C++ 声明成员变量）
#   (3) 重写 forward() 定义前向传播逻辑（类似 C++ 的虚函数实现）
#
# PyTorch 的自动微分机制：
#   你只需要定义 forward()，backward()（反向传播）由 autograd 引擎自动生成。
#   这是因为 forward 中的每一步都是可微的 PyTorch 操作，
#   autograd 会在运行时构建计算图，并自动通过链式法则计算梯度。

class MLP(nn.Module):
    """
    自定义多层感知机（MLP）。

    结构：输入(20) → 隐藏层(256) + ReLU → 输出层(10)
    """
    def __init__(self):
        # super().__init__() 调用父类 nn.Module 的构造函数
        # 这是 Python 类的标准写法（super() 返回父类代理对象）
        # 必须调用！PyTorch 在 __init__ 中初始化 _modules、_parameters 等内部字典
        super().__init__()

        # nn.Linear(in_features, out_features, bias=True): 全连接层（仿射变换）
        #   公式：y = x @ W^T + b
        #   W 形状：(out_features, in_features) — 注意！是转置关系
        #   b 形状：(out_features,)
        #   requires_grad 默认为 True，参数会被自动追踪和更新
        self.hidden = nn.Linear(20, 256)   # 隐藏层：20 个输入 → 256 个隐藏单元
        self.out = nn.Linear(256, 10)      # 输出层：256 → 10（10 个类别）

    def forward(self, X):
        """
        前向传播：定义输入 X 如何流经网络变成输出。

        只定义 forward —— PyTorch 自动推导 backward。
        这是因为 autograd 引擎在 forward 执行时记录了所有操作的计算图，
        调用 .backward() 时沿计算图反向传播计算梯度。
        """
        # F.relu(x) 是函数式 API，功能与 nn.ReLU() 相同但不需实例化
        # ReLU(x) = max(0, x)：将负值置零，正值不变，引入非线性
        # self.out(F.relu(self.hidden(X))) 是链式函数调用：
        #   X → self.hidden(X) → F.relu(...) → self.out(...)
        # 形状：(2,20) → (2,256) → (2,256) → (2,10)
        return self.out(F.relu(self.hidden(X)))

# 实例化模型
net = MLP()
# 前向传播：直接调用实例（Python 自动调用 __call__ → 内部调用 forward()）
net(X)


# ============================================================================
# 第3节：顺序块 —— 手写一个 MySequential
# ============================================================================
# 这一节手动实现一个类 Sequential 容器，目的不是替代 nn.Sequential，
# 而是帮助你理解 Sequential 的底层实现原理。

class MySequential(nn.Module):
    """
    自己实现的顺序块容器，功能类似 nn.Sequential。

    关键知识点：
      - *args（可变位置参数）：将所有传入的位置参数收集到一个元组中
        MySequential(A, B, C) → args = (A, B, C)
      - self._modules：nn.Module 内部的 OrderedDict
        必须以 _modules 命名 — PyTorch 通过 __setattr__ 魔法方法自动检测
        赋值给 self 的对象是否为 nn.Module，是则加入 _modules。
        用 _modules 而非普通 dict，PyTorch 才能追踪子模块的参数！
      - enumerate(iterable, start=0)：Python 内置函数，
        返回 (索引, 元素) 的迭代器
    """
    def __init__(self, *args):
        # *args 将 MySequential(A, B, C) 中的 A, B, C 打包为元组 args
        super().__init__()
        # 将每个子模块按顺序注册到 self._modules 字典
        # 键为 "0", "1", "2"...（str 类型），值为对应的模块
        for idx, module in enumerate(args):
            # self._modules 是 OrderedDict（有序字典）
            # 确保遍历顺序 = 插入顺序（这是 Sequential 正确性的保证）
            self._modules[str(idx)] = module

    def forward(self, X):
        # self._modules.values() 返回所有子模块（按插入顺序迭代）
        # 对每个 block（子模块）依次调用，前一个 block 的输出 = 后一个 block 的输入
        # 这就是 Sequential 的核心逻辑：链式调用
        for block in self._modules.values():
            X = block(X)
        return X

# 测试自己实现的 MySequential：行为应与 nn.Sequential 完全一致
net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
net(X)


# ============================================================================
# 第4节：在前向传播中执行任意代码 —— 动态计算图
# ============================================================================
# PyTorch 的核心特性之一：动态计算图（Define-by-Run）。
# 不需要像 TensorFlow 1.x 那样预先定义完整的静态计算图，
# 每次调用 forward 时，计算图即时构建，因此可以使用任意 Python 控制流。

class FixedHiddenMLP(nn.Module):
    """
    在前向传播中混合使用 Python 控制流的自定义网络。

    与 C++ 对比：
      PyTorch 的 forward 就像普通的 C++ 成员函数 ——
      你可以写 if/else、while 循环、定义局部变量……没有任何限制。
      框架在运行时自动追踪这些操作并构建计算图。

    动态计算图的优势：
      每次 forward 调用都重新构建计算图，这意味着：
      - 图结构可以随输入数据变化（不同输入走不同分支）
      - 可以使用 while 循环（迭代次数不确定）
      - 调试方便（可以用 pdb、print 等标准 Python 工具）
    """
    def __init__(self):
        super().__init__()
        # torch.rand() 生成 [0,1) 均匀分布的随机张量
        # requires_grad=False：这个张量不是可学习参数，不参与梯度计算
        #   训练过程中它会保持不变（相当于"固定权重"或"常量"）
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)  # 可学习的全连接层

    def forward(self, X):
        X = self.linear(X)                          # (2, 20) → (2, 20)
        # torch.mm(A, B) = 矩阵乘法（Matrix Multiplication）
        # 等价于 A @ B（@ 运算符是 Python 3.5+ 引入的矩阵乘法语法糖）
        # + 1 是广播加法：标量 1 会自动扩展为与矩阵相同形状（全 1）
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)                          # 再次经过同一个 Linear（权重共享）

        # 动态控制流！
        # X.abs().sum()：计算 X 所有元素的绝对值之和（标量）
        # while 循环：只要绝对值之和 > 1，就将 X 所有元素折半
        # 这种用 while 做迭代直到满足条件的能力是静态图框架难以实现的
        while X.abs().sum() > 1:
            X /= 2  # Python 的 /= 运算符，等价于 X = X / 2

        # .sum() 返回所有元素之和（0 维标量张量）
        # 将矩阵输出压缩为单一标量，作为网络的最终输出
        return X.sum()

# 实例化并前向传播
net = FixedHiddenMLP()
net(X)


# ============================================================================
# 第5节：混合搭配各种组合块 —— 嵌套与组合
# ============================================================================
# 除了基本的 Sequential 和继承，块还可以嵌套和任意组合。
# 这展示了 nn.Module 设计的灵活性 —— 类似于乐高积木，任意拼接。

class NestMLP(nn.Module):
    """
    嵌套 MLP：在 __init__ 中组合子模块，形成更复杂的层次化结构。

    设计模式：
      self.net = nn.Sequential(...)  将多个层打包为一个子块
      self.linear = nn.Linear(...)   顶部单独一层
      forward 中 self.net(X) 的输出作为 self.linear 的输入

    这种模式在大型网络中非常常见：
      ResNet 中的 ResidualBlock、Transformer 中的 EncoderLayer，
      都是先定义子块然后在更大的网络中组合使用。
    """
    def __init__(self):
        super().__init__()
        # 内部 Sequential 块：20 → 64 → 32（两次变换 + 激活）
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)  # 顶部层：32 → 16

    def forward(self, X):
        # 先过内部 Sequential 块，输出形状 (2, 32)
        # 再经过顶部 Linear 层，输出形状 (2, 16)
        return self.linear(self.net(X))

# 多层嵌套：将不同类的实例组合在一起
# chimera（奇美拉）= 希腊神话中的混合怪兽，暗示这个网络由"异质"模块拼接而成
chimera = nn.Sequential(
    NestMLP(),           # 第 0 层：(2,20) → (2,16)
    nn.Linear(16, 20),   # 第 1 层：(2,16) → (2,20)
    FixedHiddenMLP()     # 第 2 层：(2,20) → 标量
)
chimera(X)


# ============================================================================
# 补充知识点
# ============================================================================

# ============================================================================
# 补充 1：nn.Module 的核心机制图解
# ============================================================================
"""
nn.Module 是 PyTorch 所有神经网络模块的基类。它的核心职责：

  ┌─────────────────────────────────────────────┐
  │              nn.Module                       │
  │─────────────────────────────────────────────│
  │  _modules   : OrderedDict → 子模块字典       │
  │  _parameters: OrderedDict → 参数字典         │
  │  _buffers   : OrderedDict → 缓冲区字典       │
  │  training   : bool → 训练/评估模式标志        │
  │─────────────────────────────────────────────│
  │  forward()     → 前向传播（用户重写）         │
  │  __call__()    → 调用 forward 前做预处理      │
  │  __setattr__() → 拦截 self.xxx = ... 赋值     │
  │  parameters()  → 返回所有可学习参数           │
  │  modules()     → 返回所有子模块（含自身）      │
  │  to(device)    → 移动到指定设备               │
  │  train()       → 设置训练模式                 │
  │  eval()        → 设置评估模式                 │
  │  apply(fn)     → 递归调用 fn 到所有子模块     │
  └─────────────────────────────────────────────┘

当你写 self.fc = nn.Linear(4, 8) 时发生了什么？
  (1) Python 触发 nn.Module.__setattr__('fc', Linear(4,8))
  (2) __setattr__ 检测到赋值的对象是 nn.Module 实例
  (3) 自动将 Linear 实例注册到 self._modules['fc']
  (4) → parameters() 能遍历到 fc 的所有参数
  (5) → to(device) 能递归将 fc 搬到目标设备

当你调用 net(X) 时发生了什么？
  (1) Python 触发 net.__call__(X)
  (2) __call__ 内部做预处理（注册前向/反向 hook 等）
  (3) 调用 net.forward(X)  ← 你写的代码在这里
  (4) __call__ 做后处理，返回结果

这就是为什么继承 nn.Module 后不需要（也不应该）重写 __call__，
只重写 forward() 就够了——框架帮你处理了所有通用逻辑。
"""


# ============================================================================
# 补充 2：Sequential vs ModuleList vs ModuleDict
# ============================================================================
"""
PyTorch 提供三种子模块容器，各有不同用途：

┌──────────────┬──────────────────────────────────┬──────────────────────┐
│ 容器          │ 特点                              │ 何时使用              │
├──────────────┼──────────────────────────────────┼──────────────────────┤
│ Sequential   │ 按顺序自动连接（输出→下个输入）     │ 简单的前馈结构        │
│ ModuleList   │ 列表式容器，不自带 forward          │ 需要索引访问子模块    │
│ ModuleDict   │ 字典式容器，按名称访问              │ 需要命名访问子模块    │
└──────────────┴──────────────────────────────────┴──────────────────────┘

ModuleList 和 ModuleDict 与普通 Python list/dict 的关键区别：
  - 普通 list 中的 Module 不会被 parameters() 识别（不会注册到 _modules）
  - ModuleList/ModuleDict 中的 Module 会被正确注册和追踪
  - 所以永远用 nn.ModuleList 而不是 Python list 存放子模块！

示例：
  # ❌ 错误：parameters() 找不到 self.layers 中的参数
  self.layers = [nn.Linear(10, 10) for _ in range(5)]

  # ✅ 正确：parameters() 能遍历到所有 Linear 的参数
  self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(5)])
"""


# ============================================================================
# 补充 3：常见问题 / 练习思考
# ============================================================================
"""
Q1: 如果不调用 super().__init__() 会发生什么？
A1: nn.Module 的 __init__ 负责初始化 _modules、_parameters、_buffers 等内部字典。
    不调用会导致这些属性未初始化，后续的模块注册、参数遍历等功能全部失效。
    表现为：parameters() 返回空、to(device) 不移动参数等。

Q2: 自定义块时 __init__ 和 forward 各承担什么职责？
A2: __init__ 负责"声明"结构（定义有哪些层），类似声明成员变量。
    forward 负责"定义"计算（数据如何流过这些层），类似实现成员函数。
    这种分离使得同一个网络结构可以多次前向传播（权重共享）。

Q3: MySequential 中为什么用 self._modules 而不是普通的 dict？
A3: nn.Module 的 __setattr__ 自动检测赋值对象。如果赋给 self.xxx 的是 nn.Module，
    则自动注册到 self._modules['xxx']。但如果直接用 self._modules[str(idx)] = module，
    跳过了 __setattr__，所以需要手动写入 _modules。
    普通 dict 不会被 PyTorch 识别——parameters() 遍历不到其中的参数。

Q4: FixedHiddenMLP 中的 while 循环在反向传播时怎么处理？
A4: PyTorch 的动态计算图在每次 forward 时记录所有执行过的操作。
    while 循环展开多少次就记录多少步。反向传播时沿记录的操作序列反推即可。
    如果 while 循环次数随输入变化，每次的计算图结构也会不同。
"""


# ============================================================================
# 总结
# ============================================================================
"""
1. nn.Module 是所有网络模块的基类：
   - 必须重写 __init__（声明层）和 forward（定义计算）
   - 只需定义 forward，backward 由 autograd 自动生成

2. nn.Sequential 是有序层容器：
   - 内部用 OrderedDict 存储子模块
   - 数据按插入顺序依次流过各层

3. 自定义块本质上就是继承 + 组合：
   - 可以嵌套、组合任意 nn.Module 子类
   - 在 forward 中可以写任意 Python 控制流（动态计算图）

4. 关键 Python/PyTorch 概念：
   - super().__init__()：调用父类构造（必须！）
   - self._modules：PyTorch 内部子模块字典
   - *args：可变位置参数
   - enumerate()：(索引, 值) 迭代
   - requires_grad=False：不需要梯度追踪
"""
