"""
=============================================================================
线性回归的简洁实现 —— 完整带注释版
=============================================================================
本文件将 linear-regression-concise 简洁实现.ipynb 中的所有代码汇总在一起，
每一行都配有详细的中文注释，方便学习和理解。

任务：用 PyTorch 高级 API 简洁地实现线性回归，对比从零实现的版本。
     输入：2 个特征（从标准正态分布采样）
     输出：1 个连续值（标量预测）
     真实参数：w = [2, -3.4]ᵀ, b = 4.2

使用的高级 API：
   - torch.utils.data.TensorDataset + DataLoader：数据迭代器
   - torch.nn.Sequential + nn.Linear：神经网络层
   - torch.nn.MSELoss：均方损失函数
   - torch.optim.SGD：小批量随机梯度下降优化器

整体流程：生成数据集 → 读取数据 → 定义模型 → 初始化参数 → 定义损失函数 → 定义优化器 → 训练 → 评估
=============================================================================
"""

# ============================================================================
# 第1步：导入必要的库
# ============================================================================
import numpy as np                         # NumPy：科学计算库，此处仅用于类型兼容
import torch                               # PyTorch 深度学习框架
from torch.utils import data               # PyTorch 数据处理模块：提供 Dataset 和 DataLoader
from d2l import torch as d2l               # 《动手学深度学习》(D2L) 配套工具库


# ============================================================================
# 第2步：生成合成数据集
# ============================================================================
# 使用 d2l 封装好的 synthetic_data 函数（与从零实现中使用的是同一个函数）
# 真实模型：y = Xw + b + ε
# w = [2, -3.4]ᵀ, b = 4.2, ε ~ N(0, 0.01²)

true_w = torch.tensor([2, -3.4])                               # 真实权重：两个特征的权重
true_b = 4.2                                                   # 真实偏置
features, labels = d2l.synthetic_data(true_w, true_b, 1000)    # 生成 1000 个样本
# features 形状：(1000, 2)，labels 形状：(1000, 1)


# ============================================================================
# 第3步：使用 PyTorch 高级 API 读取数据
# ============================================================================
# 对比从零实现：不再需要手动写 data_iter 函数来打乱索引和切分批量
# PyTorch 提供了两个核心组件：
#   TensorDataset: 将特征张量和标签张量打包成一个数据集对象
#   DataLoader:   负责打乱、分批、多线程加载等功能

def load_array(data_arrays, batch_size, is_train=True):
    """
    构造一个 PyTorch 数据迭代器。

    参数:
        data_arrays: 元组 (features, labels)，即特征和标签张量
        batch_size:  每个小批量的样本数
        is_train:    是否在训练模式（True=打乱数据, False=不打乱）

    返回:
        DataLoader 对象，每次迭代返回一个 (batch_X, batch_y) 元组

    实现说明：
        data.TensorDataset(*data_arrays):
            * 是解包操作符，将 (features, labels) 展开为两个独立参数
            TensorDataset 将多个张量按第一维（样本维度）对齐打包
            访问 dataset[i] 会返回 (features[i], labels[i])

        data.DataLoader(dataset, batch_size, shuffle=is_train):
            shuffle=True: 每个 epoch 开始时打乱数据（训练时需要，防止模型记忆顺序）
            shuffle=False: 保持原始顺序（测试时常用）
            DataLoader 内部还支持多线程预加载（num_workers 参数），但此处省略
    """
    dataset = data.TensorDataset(*data_arrays)                 # 将特征和标签打包为数据集
    return data.DataLoader(dataset, batch_size, shuffle=is_train)  # 创建数据加载器

batch_size = 10                                                # 批量大小设为 10
data_iter = load_array((features, labels), batch_size)         # 创建数据迭代器

# 验证数据迭代器：取第一个小批量并查看
# iter(data_iter) 将 DataLoader 转为 Python 迭代器
# next() 从迭代器中取下一个元素，这里是一个包含 [特征, 标签] 的列表
print('第一个批量的数据:', next(iter(data_iter)))


# ============================================================================
# 第4步：定义模型（使用 nn.Sequential + nn.Linear）
# ============================================================================
# 对比从零实现：不再需要手动定义 linreg 函数和初始化 w, b 张量
# PyTorch 的 nn 模块提供了预定义的神经网络层：

# nn 是 torch.nn 的缩写，包含了构建神经网络的所有组件
from torch import nn

# nn.Sequential 是一个"层容器"，按顺序将多个层串联起来
#   输入 → Layer1 → Layer2 → ... → 输出
#   这里只有一个层，但使用 Sequential 可以保持良好的代码习惯

# nn.Linear(2, 1) 创建一个全连接层（也叫线性层或 Dense 层）
#   第一个参数 in_features=2：输入特征数量（我们的数据有 2 个特征）
#   第二个参数 out_features=1：输出特征数量（预测单个值）
#   这个层内部包含：
#     - weight：形状为 (out_features, in_features) = (1, 2) 的权重矩阵
#     - bias：形状为 (out_features,) = (1,) 的偏置向量
#   前向传播计算：output = input @ weight.T + bias

net = nn.Sequential(nn.Linear(2, 1))                           # 单层线性网络：2 个输入 → 1 个输出


# ============================================================================
# 第5步：初始化模型参数
# ============================================================================
# 对比从零实现：不再需要 torch.normal() 和 torch.zeros()
# PyTorch 的每个层都有 weight.data 和 bias.data 属性，可以直接原地修改

# net[0] 表示取 Sequential 中的第 0 个（第一个）层，即 nn.Linear(2, 1)
# weight.data 直接访问权重张量的底层数据（不经过 autograd 追踪）

# .normal_(均值, 标准差): 用正态分布随机初始化，下划线 _ 表示原地操作（in-place）
net[0].weight.data.normal_(0, 0.01)                            # 权重从 N(0, 0.01²) 随机采样
# .fill_(值): 用指定的值填充整个张量
net[0].bias.data.fill_(0)                                      # 偏置初始化为 0
# 这些初始化值与从零实现中的 torch.normal() 和 torch.zeros() 完全对应


# ============================================================================
# 第6步：定义损失函数（使用 nn.MSELoss）
# ============================================================================
# 对比从零实现：不再需要手动写 squared_loss 函数
# nn.MSELoss() 计算均方误差：MSE = (1/n) * Σ(ŷ - y)²
#   注意：框架的 MSELoss 默认求平均（除以样本数），而非除以 2
#   这与从零实现中的 (ŷ - y)² / 2 略有不同，但功能等价
#   reduction='mean'（默认）：返回所有样本损失的平均值
#   reduction='sum'：返回所有样本损失的总和

loss = nn.MSELoss()                                            # 均方误差损失函数


# ============================================================================
# 第7步：定义优化算法（使用 torch.optim.SGD）
# ============================================================================
# 对比从零实现：不再需要手动写 sgd 函数
# torch.optim.SGD 是 PyTorch 内置的 SGD 优化器

# net.parameters() 返回模型中所有可学习参数的迭代器
#   对于 nn.Linear(2, 1)，返回 [weight, bias] 两个参数
#   每个参数的 .grad 属性会被优化器自动读取和更新

# lr=0.03 是学习率，与从零实现中保持一致
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
# trainer 的内部工作原理：
#   {
#       'params': [weight, bias],        # net 中所有可学习参数
#       'lr': 0.03,
#       'momentum': 0 (默认),
#       'weight_decay': 0 (默认),
#   }
#   trainer.zero_grad()  # 遍历所有参数，清空 .grad 属性
#   trainer.step()       # 遍历所有参数，param = param - lr * param.grad
# 注意：SGD 的 step() 不会自动除以 batch_size，因为我们用的是 loss.mean()


# ============================================================================
# 第8步：训练模型
# ============================================================================
# 训练循环与从零实现非常相似，但更简洁：
#   不再需要手动 net(X, w, b)，改为 net(X)
#   不再需要手动 l.sum().backward()，改为 l.backward()
#   不再需要手动 sgd([w,b], lr, batch_size)，改为 trainer.step()
#   梯度清零从手动 .zero_() 改为 trainer.zero_grad()

num_epochs = 3                                                 # 训练 3 个 epoch（与从零实现的 5 不同，但效果已足够）

for epoch in range(num_epochs):                                # 外层循环：遍历整个数据集 num_epochs 次
    for X, y in data_iter:                                     # 从 DataLoader 逐批取出数据
        l = loss(net(X), y)                                    # (1)(2) 前向传播 net(X) + 计算损失
        trainer.zero_grad()                                    # (3前) 清零所有参数的梯度
        l.backward()                                           # (3) 反向传播：计算所有参数的梯度
        trainer.step()                                         # (4) 更新所有参数：θ = θ - lr * ∇θ

    # 每个 epoch 结束后计算整个训练集上的损失（用于监控）
    l = loss(net(features), labels)                            # 在整个训练集上做一次前向传播
    print(f'epoch {epoch + 1}, loss {l:f}')                    # 打印损失值


# ============================================================================
# 第9步：评估模型
# ============================================================================
# 访问训练好的参数：net[0] 取出第一个层，.weight.data 和 .bias.data 读取参数值
w = net[0].weight.data                                         # 获取学到的权重，形状 (1, 2)
print('w的估计误差：', true_w - w.reshape(true_w.shape))        # 与真实权重的差值
b = net[0].bias.data                                           # 获取学到的偏置，形状 (1,)
print('b的估计误差：', true_b - b)                               # 与真实偏置的差值
# 误差非常小，说明简洁实现的模型同样成功学到了数据背后的线性关系


# ============================================================================
# 补充知识点
# ============================================================================

# 1. 从零实现 vs 简洁实现 对比表：
#
#   | 组件       | 从零实现              | 简洁实现                        |
#   |-----------|----------------------|-------------------------------|
#   | 数据迭代    | 手写 data_iter()      | DataLoader + TensorDataset     |
#   | 模型定义    | 手写 linreg()         | nn.Sequential(nn.Linear(2,1)) |
#   | 参数初始化  | torch.normal/zeros    | .normal_() / .fill_()         |
#   | 损失函数    | 手写 squared_loss()   | nn.MSELoss()                  |
#   | 优化器     | 手写 sgd()            | torch.optim.SGD               |
#   | 梯度清零    | param.grad.zero_()    | trainer.zero_grad()           |
#   | 参数更新    | 手动更新每个参数        | trainer.step()                |

# 2. nn.Linear 内部参数形状：
#    weight: (out_features, in_features) = (1, 2)  注意与从零实现中的 (2, 1) 互为转置
#    bias:   (out_features,) = (1,)
#    前向计算：y = X @ weight.T + bias
#               (batch, 2) @ (2, 1) + (1,) → (batch, 1)

# 3. MSELoss 的 reduction 参数：
#    reduction='mean' (默认): loss = (1/n) * Σ(ŷ - y)²
#    reduction='sum':        loss = Σ(ŷ - y)²
#    reduction='none':       loss 保持原始形状，不做聚合
#    从零实现中用 (ŷ - y)² / 2，然后 .sum()，所以等价于 MSELoss(reduction='sum') / 2

# 4. 常用优化器变体：
#    - SGD:     基础随机梯度下降
#    - SGD + Momentum: 加入动量项，加速收敛
#    - Adam:    自适应学习率，最常用的优化器之一
#    - RMSprop: 对每个参数自适应调整学习率
#    用法类似：trainer = torch.optim.XXX(net.parameters(), lr=...)

# 5. Sequential 的灵活用法：
#    net = nn.Sequential(
#        nn.Linear(2, 64),     # 第一层：2 → 64
#        nn.ReLU(),            # 激活函数
#        nn.Linear(64, 32),    # 第二层：64 → 32
#        nn.ReLU(),            # 激活函数
#        nn.Linear(32, 1),     # 第三层：32 → 1
#    )
#    这就是一个简单的 3 层 MLP（多层感知机）！


# ============================================================================
# 总结
# ============================================================================
# 1. PyTorch 的高级 API 大大简化了深度学习模型的实现
# 2. 核心模块分工：
#    - torch.utils.data：数据处理（TensorDataset, DataLoader）
#    - torch.nn：神经网络层和损失函数（Linear, MSELoss, Sequential）
#    - torch.optim：优化算法（SGD, Adam 等）
# 3. 简洁实现的代码量约为从零实现的 1/3，但底层原理完全相同
# 4. 建议学习路线：先理解从零实现，再使用高级 API——知道底层原理才能更好地调试和定制
# 5. 训练循环的 4 步模板适用于几乎所有 PyTorch 模型：
#    trainer.zero_grad() → loss.backward() → trainer.step()
#    这个模板在 CNN、RNN、Transformer 等复杂模型中同样适用
