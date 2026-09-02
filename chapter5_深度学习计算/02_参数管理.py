"""
=============================================================================
参数管理 —— 访问、初始化与绑定（完整带注释版）
=============================================================================
本文件深入 PyTorch 模型参数管理的全部核心操作：
  - 参数访问：state_dict / named_parameters / 直接索引
  - 参数初始化：内置方法 + 自定义策略
  - 参数绑定：多个层共享同一组权重
=============================================================================
"""

# ============================================================================
# 第1步：导入库并构建示例网络
# ============================================================================
import torch
from torch import nn

# 构建一个简单的三层感知机：
#   输入 4 → 隐藏 8 → ReLU → 输出 1
# 这个网络包含两个 Linear 层，每个 Linear 层内部有两个参数张量：
#   (1) weight：权重矩阵
#   (2) bias：偏置向量
# ReLU 层没有可学习参数，它是纯函数变换。
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))  # 批量=2，每个样本 4 个输入特征
net(X)


# ============================================================================
# 第2节：参数访问 —— state_dict()
# ============================================================================
# net[2] 用索引访问 Sequential 中的第 3 个模块（索引从 0 开始）
#   索引 0: nn.Linear(4, 8)
#   索引 1: nn.ReLU()       ← 无参数
#   索引 2: nn.Linear(8, 1)

# .state_dict() 返回 OrderedDict：键=参数名，值=参数张量
#   这是保存/加载模型的推荐方式（见第 4 节）
#   OrderedDict 是有序字典，保证遍历顺序与插入顺序一致
print(net[2].state_dict())


# ============================================================================
# 第3节：目标参数 —— Parameter 对象
# ============================================================================
# net[2].bias 是什么类型？
#   torch.nn.parameter.Parameter — 这是 Tensor 的子类
#   Parameter 与普通 Tensor 的区别：
#     Parameter 自动被注册到模块的 _parameters 字典中
#     → parameters() 能遍历到
#     → optimizer 能找到并更新它
#     → 保存/加载 state_dict 时会包含它
#     普通 Tensor 即使赋给 self.xxx，也不会被自动追踪

print(type(net[2].bias))      # <class 'torch.nn.parameter.Parameter'>
print(net[2].bias)             # Parameter 对象（含 requires_grad 等元信息）
print(net[2].bias.data)       # .data = 底层纯 Tensor 数据（绕过 autograd 追踪）

# .grad 属性存放梯度值
# 因为还没有调用 .backward()，所以梯度为 None
# .grad 只有在反向传播之后才被赋值
net[2].weight.grad == None  # True


# ============================================================================
# 第4节：一次性访问所有参数 —— named_parameters()
# ============================================================================
# .named_parameters() 返回 (参数名, 参数张量) 的迭代器
#
# 参数名的命名规则：
#   net[0].named_parameters()  →  ('weight', ...), ('bias', ...)
#                                   ↑ 相对于 net[0] 这个子模块自身
#   net.named_parameters()     →  ('0.weight', ...), ('0.bias', ...),
#                                  ('2.weight', ...), ('2.bias', ...)
#                                   ↑ 前缀 "0." / "2." 是子模块在 Sequential 中的索引
# 注意：ReLU 层没有参数，所以参数名中跳过了索引 1

# * 解包：print(*[(...), (...)]) 将列表中的每个元素解包为独立参数传给 print
# 相当于 print(('weight', ...), ('bias', ...), ...)
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
print(*[(name, param.shape) for name, param in net.named_parameters()])

# 通过 state_dict 的字符串键直接访问任意参数
# .data 访问底层数据，返回普通 Tensor（不是 Parameter）
net.state_dict()['2.bias'].data


# ============================================================================
# 第5节：从嵌套块收集参数
# ============================================================================
# 这一节展示如何在深度嵌套的网络结构中访问特定参数

def block1():
    """基础块：4→8→ReLU→4→ReLU"""
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    """
    复合块：由 4 个 block1 串联而成。

    add_module(name, module)：
      向 Sequential 中动态加入一个具名子模块。
      等价于在 __init__ 中写 self.name = module。
      用于在循环中或条件性地构建网络结构。
    """
    net = nn.Sequential()
    for i in range(4):
        # f'block {i}' 是 f-string（格式化字符串）
        # 等价于 'block ' + str(i)，但更简洁直观
        net.add_module(f'block {i}', block1())
    return net

# 深度嵌套网络：block2（含 4 个 block1）→ Linear(4, 1)
rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
rgnet(X)

# print(net) 递归打印整个网络结构
# 输出像一棵树，方便查看层次化组织
print(rgnet)

# 嵌套索引访问：用连续的 [idx] 层层深入
#   rgnet[0]     = block2 整体
#   rgnet[0][1]  = block2 中的第 2 个 block1
#   rgnet[0][1][0] = 该 block1 中的第 1 层（nn.Linear）
#   .bias.data   = 该层的偏置参数数据
rgnet[0][1][0].bias.data


# ============================================================================
# 第6节：内置初始化 —— net.apply(fn)
# ============================================================================
# net.apply(fn) 是最常用的参数初始化方式。
# 它会递归遍历网络中所有子模块（包括子模块的子模块，深度优先），
# 对每个子模块调用 fn(m)，其中 m 是子模块对象。
#
# 类比：设计模式中的"访问者模式"(Visitor Pattern)
#   apply 负责遍历整棵模块树，fn 负责在每个节点上执行操作。

def init_normal(m):
    """正态分布初始化：权重 ~ N(0, 0.01²)，偏置 = 0"""
    # type(m) == nn.Linear：精确类型匹配
    # 为什么用 type() 而非 isinstance()？
    #   type()  严格匹配类型（不匹配子类）
    #   isinstance() 匹配该类型及其所有子类
    # 这里用 type() 是为了只初始化 Linear 层，不影响可能的子类变体
    if type(m) == nn.Linear:
        # nn.init.normal_(tensor, mean=0, std=0.01)：
        #   用正态分布 N(mean, std²) 的值原地填充 tensor
        #   _ 后缀 = 原地操作（in-place），直接修改原张量，不创建副本
        nn.init.normal_(m.weight, mean=0, std=0.01)
        # nn.init.zeros_(tensor)：将所有元素填为 0
        nn.init.zeros_(m.bias)

net.apply(init_normal)  # 递归初始化所有子模块
net[0].weight.data[0], net[0].bias.data[0]


def init_constant(m):
    """常数初始化：权重 = 1，偏置 = 0"""
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)  # 所有权重填充为常数 1
        nn.init.zeros_(m.bias)

net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]


# ============================================================================
# 第7节：对不同块应用不同的初始化方法
# ============================================================================
# net.apply(fn) 统一处理整棵树。
# net[idx].apply(fn) 只处理某个子树 —— 实现精细化的初始化控制。

def init_xavier(m):
    """
    Xavier/Glorot 初始化：
    适用于 Sigmoid、Tanh 等 S 型激活函数。
    设计目标：让每层输出的方差尽量等于输入的方差，维持信号在前向传播中不爆炸不消失。
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)  # 均匀分布版本（也有 normal 版本）

def init_42(m):
    """常数 42 初始化（仅用于测试/演示）"""
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

net[0].apply(init_xavier)  # 只对第 0 个模块（输入层）用 Xavier
net[2].apply(init_42)      # 只对第 2 个模块（输出层）设常数
print(net[0].weight.data[0])
print(net[2].weight.data)


# ============================================================================
# 第8节：自定义初始化
# ============================================================================
# 除了使用内置初始化方法，你可以编写完全自定义的初始化逻辑。

def my_init(m):
    """
    自定义稀疏初始化策略：

    步骤：
      (1) 从 [-10, 10] 均匀分布中随机初始化所有权重
      (2) 将绝对值 < 5 的权重全部置零（只保留"强连接"）

    效果：
      产生稀疏的权重矩阵 —— 大部分连接权重为 0，少数非零。
      这可以看作一种手工的稀疏正则化。
    """
    if type(m) == nn.Linear:
        # 打印正在初始化的参数名和形状（方便调试）
        print("Init", *[(name, param.shape)
                        for name, param in m.named_parameters()][0])
        # nn.init.uniform_(tensor, a, b)：从 [a, b) 均匀分布采样
        nn.init.uniform_(m.weight, -10, 10)
        # 稀疏化：只保留 |weight| >= 5 的权重
        #   m.weight.data.abs() >= 5   → 布尔张量（满足条件的为 True）
        #   m.weight.data *= 布尔掩码   → True 位置保留原值，False 位置变 0
        # 因为 True=1, False=0 在乘法中自动转换
        m.weight.data *= m.weight.data.abs() >= 5

net.apply(my_init)
net[0].weight[:2]  # 查看前两行的稀疏权重


# ============================================================================
# 第9节：直接修改参数
# ============================================================================
# 参数初始化后，你仍然可以像操作普通张量一样直接修改参数数据。
# 这在某些手动调参或实验场景下很有用。

# 注意：以下操作直接修改 .data（底层数据），不会触发 autograd
net[0].weight.data[:] += 1      # 所有权重 +1（原地加法）
net[0].weight.data[0, 0] = 42   # 修改单个元素
net[0].weight.data[0]           # 查看修改后的第一行


# ============================================================================
# 第10节：参数绑定 —— 层间权重共享
# ============================================================================
# 参数绑定（Weight Sharing / Weight Tying）：
#   多个层引用同一个 Parameter 对象（同一块内存），而不是各自持有独立的副本。
#
# 典型应用场景：
#   - Siamese Network（孪生网络）：两个分支共享权重以提取一致的特征
#   - Transformer：Encoder 和 Decoder 的 embedding 层有时共享权重
#   - 等价性约束：强制网络的不同部分使用相同的变换

# 定义一个共享层
shared = nn.Linear(8, 8)  # 这个层将在网络中出现两次

# 构建网络：同一个 shared 对象出现在索引 2 和 4
# 注意：这里传入的是同一个 Python 对象，不是两个副本！
#   shared is shared → True
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),   # net[2] ← shared
                    shared, nn.ReLU(),   # net[4] ← 同一个 shared！
                    nn.Linear(8, 1))
net(X)

# 验证参数绑定：检查 weight.data[0] 是否逐元素相等
# 如果输出全是 True，说明 net[2] 和 net[4] 确实是同一组参数
print(net[2].weight.data[0] == net[4].weight.data[0])

# 修改 net[2] 的权重 —— net[4] 会同步变化
# 因为它们共享同一块内存（同一个 Tensor 对象），不是值拷贝
net[2].weight.data[0, 0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])
# 输出仍全是 True，确认共享关系


# ============================================================================
# 补充知识点
# ============================================================================

# ============================================================================
# 补充 1：常用初始化方法速查表
# ============================================================================
"""
┌───────────────────────────┬────────────────────────────────┬─────────────────────┐
│ 方法                       │ 适用场景                        │ 说明                 │
├───────────────────────────┼────────────────────────────────┼─────────────────────┤
│ nn.init.normal_()          │ 通用                            │ N(mean, std²)        │
│ nn.init.uniform_()         │ 通用/自定义                     │ [a, b) 均匀分布       │
│ nn.init.xavier_uniform_()  │ Sigmoid/Tanh 激活函数           │ 维持前向/反向方差     │
│ nn.init.kaiming_uniform_() │ ReLU/LeakyReLU (He 初始化)     │ 考虑 ReLU 单侧抑制    │
│ nn.init.constant_()        │ 调试/偏置项                     │ 常数填充              │
│ nn.init.zeros_()           │ 偏置/LSTM forget gate            │ 全零                  │
│ nn.init.ones_()            │ 特殊场景                        │ 全一                  │
│ nn.init.orthogonal_()      │ RNN/LSTM 权重                   │ 正交矩阵（保范数）    │
└───────────────────────────┴────────────────────────────────┴─────────────────────┘

初始化选择经验法则：
  - ReLU / LeakyReLU / PReLU → kaiming (He) 初始化
  - Sigmoid / Tanh → xavier (Glorot) 初始化
  - 偏置项 (bias) → 通常初始化为 0
  - BatchNorm 的 weight → 1，bias → 0
  - LSTM 的 forget gate bias → 可以初始化为 1（鼓励初始时记忆更多信息）
"""


# ============================================================================
# 补充 2：为什么不能把所有参数初始化为相同值？
# ============================================================================
"""
对称性破坏 (Symmetry Breaking)：

如果把同一层的所有神经元初始化为相同的值：
  - 前向传播：所有神经元输出相同 → 与只有一个神经元无异
  - 反向传播：所有神经元收到相同的梯度
  - 参数更新：所有神经元被更新为相同的新值
  - 结果：同一层的神经元永远同步更新，永远等价于单个神经元！

随机初始化的目的就是打破这种对称性，
使不同的神经元能够学习不同的特征。

这就是为什么神经网络参数必须随机初始化（不能全 0 或全 1）。
偏置项初始化为 0 是安全的，因为权重的随机性已经打破了对称。
"""


# ============================================================================
# 补充 3：.data vs .detach() —— 何时用哪个？
# ============================================================================
"""
.data 和 .detach() 都能获取一个与计算图分离的张量，但有重要区别：

  .data:
    - 返回一个共享底层存储的 Tensor（不是 Parameter）
    - 完全脱离 autograd 追踪
    - 危险：修改 .data 不会触发 autograd 的任何通知
    - 可能导致梯度计算错误（已被污染的梯度）

  .detach():
    - 返回一个共享底层存储的 Tensor，与原张量有相同 requires_grad 状态
    - 显式标记为 detach（脱离计算图但保留历史信息）
    - 更安全：autograd 知道这个分离操作

  推荐：
    修改参数值 → 用 .data（这是它的设计用途）
    获取用于计算的值 → 用 .detach()（更安全）
"""


# ============================================================================
# 补充 4：常见问题 / 练习思考
# ============================================================================
"""
Q1: 使用嵌套块时，parameters() 的遍历顺序是什么？
A1: 深度优先、按 _modules 的插入顺序（即 __init__ 中声明的顺序）。
    以 rgnet 为例：先遍历 block2 的所有参数，再遍历最后的 Linear 的参数。

Q2: apply(fn) 和 parameters() 遍历子模块的方式有什么不同？
A2: apply(fn) 遍历的是模块树（modules），fn 接收的是 Module 对象。
    parameters() 遍历的是参数列表，返回的是 Parameter 对象。
    apply 通常用于初始化（在参数上操作），parameters 用于传给优化器。

Q3: 参数绑定时，共享参数的梯度怎么计算？
A3: 因为两个层引用的是同一个 Parameter 对象，反向传播时，
    两个位置对该参数的梯度会累加。
    即：param.grad = grad_from_position_1 + grad_from_position_2
    每次 optimizer.step() 只用累加后的总梯度更新一次。

Q4: state_dict 中的 key 命名规则是什么？
A4: 用 . 分隔层级。例如：
    '0.weight' = 第 0 个子模块的 weight 参数
    '0.block 1.2.weight' = 第0个子模块→block 1→第2个子模块的weight
    这种命名自然反映了模块的嵌套结构。
"""


# ============================================================================
# 总结
# ============================================================================
"""
1. 参数访问：
   - state_dict()：返回参数字典（保存/加载的推荐方式）
   - named_parameters()：(参数名, 参数) 迭代器
   - net[idx].weight / .bias：直接访问特定参数

2. 参数是 nn.Parameter（Tensor 子类）：
   - 自动注册到模型的参数列表
   - .data 获取底层 Tensor（不含 autograd 元信息）
   - .grad 存放梯度（.backward() 后才有值）

3. 参数初始化：
   - net.apply(init_fn)：递归初始化所有子模块
   - 内置方法：normal_、xavier_uniform_、kaiming_uniform_ 等
   - 自定义初始化：完全灵活
   - 必须随机初始化以打破对称性

4. 参数绑定：
   - 让多个层引用同一个 Parameter 对象
   - 梯度自动累加
   - 用于孪生网络、权重共享等场景
"""
