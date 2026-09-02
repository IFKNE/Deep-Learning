"""
=============================================================================
softmax回归的简洁实现 —— 完整带注释版
=============================================================================
源文件：pytorch/chapter_linear-networks线性神经网络/softmax-regression-concise.ipynb
任务：用 PyTorch 高级 API 简洁地实现 softmax 回归，在 Fashion-MNIST 上做 10 分类。
     输入：28×28 的灰度图像（展平为 784 维向量）
     输出：10 个类别的概率分布

整体流程：加载数据 → 初始化模型参数 → 重新审视Softmax实现 → 优化算法 → 训练
=============================================================================
"""

# ============================================================================
# （原 notebook 开头：导入库 + 加载数据集）
# ============================================================================

import torch                          # PyTorch 深度学习框架
from torch import nn                  # nn = neural network，神经网络模块
# -- Python 语法：from X import Y 只导入 X 中的 Y，之后直接用 Y 而非 X.Y --
from d2l import torch as d2l          # D2L 配套工具库，as d2l = 起别名

batch_size = 256                      # 每次迭代处理 256 张图像
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# -- Python 语法：函数返回两个值时，可以用逗号同时接收多个返回值 --
# train_iter：训练集迭代器（60000 张 28×28 灰度服装图）
# test_iter：测试集迭代器（10000 张）
# 标签 0-9 对应：T恤/裤子/套头衫/连衣裙/外套/凉鞋/衬衫/运动鞋/包/短靴


# ============================================================================
# ## 初始化模型参数
# ============================================================================
# 原 notebook：softmax 回归的输出层是一个全连接层。在 Sequential 中添加一个
# 带 10 个输出的全连接层。以均值 0 和标准差 0.01 随机初始化权重。

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
# -- PyTorch 概念：nn.Sequential 是层容器，按顺序串联各层，数据依次流过 --
#    输入 (batch, 28, 28) → Flatten → (batch, 784) → Linear → (batch, 10)
# nn.Flatten()：展平层，保留 batch 维度（第0维），其余全部拉成一维
#   28×28 → 784，等价于 X.reshape(X.shape[0], -1)
# nn.Linear(784, 10)：全连接层（线性变换层）
#   参数1=784 输入特征数，参数2=10 输出类别数
#   内部有 weight (10×784) 和 bias (10,) 两个可学习参数
#   计算：output = input @ weight.T + bias

def init_weights(m):
    """
    初始化函数：对全连接层用正态分布随机初始化权重。
    参数 m：模型中的某个子层（由 net.apply 逐个传入）
    """
    if type(m) == nn.Linear:                        # 只对 Linear 层做初始化
        nn.init.normal_(m.weight, std=0.01)         # 权重 ~ N(0, 0.01²)
        # -- PyTorch 概念：_ 后缀 = 原地操作（in-place），直接修改原张量 --
        # 不带 _ 的版本（如 nn.init.normal）会返回新张量，原张量不变

# -- PyTorch 概念：net.apply(fn) 递归遍历模型的每个子层，对每个子层调用 fn(子层) --
#   遍历顺序是深度优先：Flatten → Linear
#   Flatten 没有参数 → type(m) == nn.Linear 为 False → 跳过
#   Linear  有 weight → type(m) == nn.Linear 为 True  → 执行 normal_() 初始化
net.apply(init_weights)


# ============================================================================
# ## 重新审视Softmax的实现
# ============================================================================
# 原 notebook 数学推导（LogSumExp 技巧，解决数值稳定性问题）：
#
# 回顾 softmax：ŷ_j = exp(o_j) / Σ_k exp(o_k)
#
# 问题1 — 上溢（overflow）：
#   o_k 中有很大的值（如 50），exp(50) ≈ 5.18e21 → 超出 float32 范围 → inf
#   分母或分子变成 inf → 最终 ŷ 出现 0、inf 或 nan
#
# 技巧1：先减去最大值，数学上等价（分子分母同除 exp(max)）：
#   ŷ_j = exp(o_j - max) / Σ_k exp(o_k - max)
#   所有 exp 的自变量 ≤ 0，最大为 exp(0) = 1，不会上溢
#
# 问题2 — 下溢（underflow）：
#   减法后有些值是非常负的数，exp(负大数) → 0（超出 float 最小精度）
#   ŷ_j = 0 → log(0) = -inf → 反向传播出 nan
#
# 最终方案 — LogSumExp 技巧（softmax + log 合并计算）：
#   log(ŷ_j) = log[exp(o_j - max) / Σ_k exp(o_k - max)]
#            = (o_j - max) - log(Σ_k exp(o_k - max))
#   直接计算 log(ŷ_j)，跳过了 exp(负大数) → 0 的中间步骤
#
# 损失（取负）：
#   loss_j = -(o_j - max) + log(Σ_k exp(o_k - max))
#   整个计算过程中不会上溢也不会下溢
#
# 结论：不要先将 softmax 概率传给损失函数，而是将"原始未规范化的输出"直接
# 传入损失函数，让损失函数内部用 LogSumExp 技巧完成 softmax + log + 负号。

loss = nn.CrossEntropyLoss(reduction='none')
# -- PyTorch 概念：nn.CrossEntropyLoss 内部实现了 LogSumExp 技巧 --
# 接收的是 net 的"原始输出"（raw logits），不是 softmax 后的概率！
# reduction 参数控制损失如何聚合：
#   'none' — 不聚合，返回每个样本的独立损失值，形状 (batch_size,)
#   'mean' — 取平均（默认值），返回一个标量
#   'sum'  — 求和，返回一个标量
# 这里用 'none' 是因为 d2l.train_ch3 内部会自行 .sum() 来聚合


# ============================================================================
# ## 优化算法
# ============================================================================
# 原 notebook：使用学习率为 0.1 的小批量随机梯度下降。与线性回归中相同，
# 说明了优化器的普适性。

trainer = torch.optim.SGD(net.parameters(), lr=0.1)
# -- PyTorch 概念 --
# torch.optim.SGD：内置的小批量随机梯度下降优化器
#   参数1 net.parameters()：返回模型所有可学习参数的迭代器
#     Flatten 层→无参数，Linear 层→weight(10×784) + bias(10,)
#   参数2 lr=0.1：学习率 learning rate
# trainer 的两个核心方法：
#   trainer.zero_grad() — 将所有参数的 .grad 清零
#     -- PyTorch 概念：梯度默认累加，每次 backward() 前必须手动清零 --
#   trainer.step()      — 按 param -= lr * param.grad 更新所有参数


# ============================================================================
# ## 训练
# ============================================================================
# 原 notebook：调用 d2l.train_ch3() 训练函数（与从零实现共用同一个函数）
# 内部流程：每个 epoch 扫一遍训练集 → 在测试集上评估 → 绘图

num_epochs = 10                                          # 训练 10 个 epoch
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
# 参数依次为：模型、训练数据、测试数据、损失函数、训练轮数、优化器
# 10 个 epoch 后训练精度和测试精度通常都在 80%-85% 左右


# ============================================================================
# 补充知识点
# ============================================================================

# 1. 从零实现 vs 简洁实现 对比（Softmax 回归）：
#
#   | 组件       | 从零实现                              | 简洁实现                          |
#   |-----------|--------------------------------------|---------------------------------|
#   | 模型定义   | 手写 net()：reshape + matmul + softmax | Sequential(Flatten, Linear)     |
#   | 参数初始化 | torch.normal() + torch.zeros()        | net.apply(init_weights)         |
#   | softmax   | 手写 softmax()：exp / Σexp            | 集成在 CrossEntropyLoss 中       |
#   | 损失函数   | 手写 cross_entropy()：-log(ŷ[y])      | nn.CrossEntropyLoss()           |
#   | 优化器    | 手写 sgd() + d2l.sgd()                | torch.optim.SGD                 |
#   | 训练循环   | 手写 train_epoch_ch3 + train_ch3       | 复用 d2l.train_ch3()            |

# 2. 本文件中出现的 PyTorch `_` 后缀方法（原地操作）：
#    nn.init.normal_(tensor, std=0.01) — 用正态分布填充原张量，不创建新张量
#    对比：nn.init.normal(tensor, std=0.01) 不存在（带 _ 是 PyTorch 初始化方法的默认命名）

# 3. nn.CrossEntropyLoss 使用要点：
#    - 输入必须是"原始 logits"，不是 softmax 之后的概率
#    - 标签 y 是整数（类别索引），不是 one-hot 向量
#    - 内部自动做了 softmax → log → 取负，外加 LogSumExp 数值稳定

# 4. 过拟合预警（对应原 notebook 练习）：
#    如果测试精度在某个 epoch 后开始下降而训练精度继续上升，
#    说明模型开始过拟合（overfitting）——记住了训练集的噪声而非学到了一般规律
#    解决方案：早停（early stopping）、权重衰减（weight decay）、增加数据量


# ============================================================================
# 总结
# ============================================================================
# 1. Softmax 回归 = 线性变换 + softmax + 交叉熵，是多分类任务的基础
# 2. 简洁实现用到的 4 个核心 API：
#    nn.Flatten()          — 展平输入，把 28×28 图像拉成 784 维向量
#    nn.Linear(784, 10)    — 全连接层，做矩阵乘法 X @ W.T + b
#    nn.CrossEntropyLoss() — 交叉熵损失，内部含 LogSumExp 保证数值稳定
#    net.apply(fn)         — 递归遍历子层，用 fn 初始化参数
# 3. CrossEntropyLoss 接收原始 logits（非 softmax 概率），内部自动处理
# 4. 训练流程的标准 4 步：zero_grad → backward → step（反复循环）
# 5. 从零实现帮你理解原理，简洁实现帮你提高效率——两者缺一不可
