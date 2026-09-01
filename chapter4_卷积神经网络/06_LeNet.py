"""
=============================================================================
LeNet —— 第一个卷积神经网络（完整带注释版）
=============================================================================
本节将前面 5 节学到的组件（卷积层、激活函数、汇聚层、全连接层）组装成
一个完整的卷积神经网络 —— LeNet-5，并在 Fashion-MNIST 数据集上训练。

LeNet 由 Yann LeCun 于 1989 年提出，是最早成功应用反向传播训练的 CNN，
用于手写数字识别（ATM 支票读取），是深度学习史上的里程碑。
=============================================================================
"""

# ============================================================================
# 第1步：导入库
# ============================================================================
import torch
from torch import nn
from d2l import torch as d2l  # 《动手学深度学习》配套工具包


# ============================================================================
# 第1节：构建 LeNet-5 模型
# ============================================================================
# LeNet-5 架构（简化版，去掉最后一层高斯激活）：
#
#  输入 (1×28×28 灰度图)
#    │
#    ├─ Conv2d(1→6, 5×5, pad=2) ─→ (6×28×28)   ← 卷积块1
#    ├─ Sigmoid()
#    ├─ AvgPool2d(2×2, stride=2) ─→ (6×14×14)   ← 降采样，尺寸减半
#    │
#    ├─ Conv2d(6→16, 5×5)       ─→ (16×10×10)  ← 卷积块2
#    ├─ Sigmoid()
#    ├─ AvgPool2d(2×2, stride=2) ─→ (16×5×5)    ← 降采样，尺寸减半
#    │
#    ├─ Flatten()                ─→ (400,)       ← 展平为向量
#    │
#    ├─ Linear(400→120) + Sigmoid()              ← 全连接块
#    ├─ Linear(120→84)  + Sigmoid()
#    └─ Linear(84→10)                            ← 10 类输出（不做 softmax）
#
# 设计规律：
#   空间尺寸逐渐减小（28→14→5），通道数逐渐增加（1→6→16）
#   这是一个普遍规律 —— 用通道深度换取空间分辨率。

net = nn.Sequential(
    # --- 卷积块 1 ---
    # Conv2d(输入通道=1, 输出通道=6, 核=5, padding=2)
    #   为何 padding=2？28 - 5 + 2*2 + 1 = 28 → 保持尺寸不变
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    # AvgPool2d(核=2, stride=2)：将 28×28 降采样到 14×14
    nn.AvgPool2d(kernel_size=2, stride=2),

    # --- 卷积块 2 ---
    # Conv2d(6→16, 5×5, 无 padding)
    #   10 = 14 - 5 + 1 → 尺寸从 14 缩小到 10
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    # AvgPool2d(2, stride=2)：10×10 → 5×5
    nn.AvgPool2d(kernel_size=2, stride=2),

    # --- 过渡：展平 ---
    # Flatten() 将四维张量 (N, 16, 5, 5) 展平为二维 (N, 16*5*5=400)
    # 这是卷积块→全连接块的必要桥梁
    nn.Flatten(),

    # --- 全连接块 ---
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),  # 400→120
    nn.Linear(120, 84), nn.Sigmoid(),            # 120→84
    nn.Linear(84, 10)                            # 84→10（输出）
    # 注意：最后一层不加 Softmax，因为 nn.CrossEntropyLoss 内置了 LogSoftmax
)


# ============================================================================
# 第2节：检查各层输出形状
# ============================================================================
# 用随机输入走过每一层，验证维度变化是否符合预期
# 这是一个很好的调试技巧 —— 在实际训练前确认网络结构正确

X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)  # 单样本灰度图
for layer in net:
    X = layer(X)  # 逐层前向传播
    # layer.__class__.__name__：获取层类名（如 'Conv2d', 'Linear'）
    # X.shape：输出张量形状
    print(layer.__class__.__name__, 'output shape: \t', X.shape)


# ============================================================================
# 第3节：加载 Fashion-MNIST 数据
# ============================================================================
# Fashion-MNIST：28×28 灰度服装图片，10 个类别
#   (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle Boot)
# 是 MNIST 手写数字的现代替代品（手写数字太简单，Fashion-MNIST 更有挑战）

batch_size = 256  # 每批 256 张图片
# d2l.load_data_fashion_mnist()：返回 (train_iter, test_iter)
#   内部使用 torch.utils.data.DataLoader，自动完成打乱、分批
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)


# ============================================================================
# 第4节：GPU 评估函数
# ============================================================================
def evaluate_accuracy_gpu(net, data_iter, device=None):  #@save
    """
    用 GPU 计算模型在数据集上的分类准确率。

    参数:
        net: 模型
        data_iter: 数据迭代器
        device: 计算设备（GPU/CPU），默认从模型参数推断

    返回:
        float: 准确率（0~1）

    关键步骤：
        (1) net.eval()：设置为评估模式（关闭 Dropout、固定 BN 统计量）
        (2) torch.no_grad()：不追踪梯度（省显存、加速推理）
        (3) 将每个 batch 的数据和标签搬到 device
        (4) 计算预测正确的样本数 / 总样本数
    """
    # isinstance(obj, Type)：Python 运行时类型检查
    if isinstance(net, nn.Module):
        net.eval()  # 切换到评估模式！
        if not device:
            # next(iter(...))：取迭代器的第一个元素
            # net.parameters() 的第一个参数所在的设备 = 模型所在的设备
            device = next(iter(net.parameters())).device

    # d2l.Accumulator(n)：累加器，内部有 n 个变量
    #   metric[0] = 正确预测数, metric[1] = 总样本数
    metric = d2l.Accumulator(2)

    # torch.no_grad() 上下文管理器：在此范围内不构建计算图
    # → 不会记录梯度 → 节省 GPU 显存
    with torch.no_grad():
        for X, y in data_iter:
            # 处理可能的列表输入（如 BERT 的多个输入），一般 Fashion-MNIST 走 else 分支
            if isinstance(X, list):
                X = [x.to(device) for x in X]  # 列表推导式：逐元素搬到 GPU
            else:
                X = X.to(device)  # 将数据搬到 GPU
            y = y.to(device)      # 将标签搬到 GPU（必须与模型同设备！）
            # d2l.accuracy(y_hat, y)：计算预测正确的数量
            # y.numel()：y 中的元素总数（= batch_size）
            metric.add(d2l.accuracy(net(X), y), y.numel())
    # 准确率 = 正确数 / 总数
    return metric[0] / metric[1]


# ============================================================================
# 第5节：GPU 训练函数
# ============================================================================
#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """
    在 GPU 上训练卷积神经网络（第 6 章通用训练函数）。

    参数:
        net: 模型（nn.Module 实例）
        train_iter: 训练数据迭代器
        test_iter: 测试数据迭代器
        num_epochs: 训练轮数
        lr: 学习率
        device: 计算设备（如 torch.device('cuda:0')）

    训练流程：
        (1) 初始化权重（Xavier）
        (2) 模型搬到 GPU
        (3) 创建优化器（SGD）和损失函数（CrossEntropyLoss）
        (4) 每 epoch: train → evaluate
    """
    # --- 权重初始化 ---
    # net.apply(fn)：递归遍历所有子模块，对每个子模块调用 fn
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            # Xavier 初始化：维持前向/反向传播的方差稳定
            # 适合 Sigmoid/Tanh 激活函数（LeNet 使用 Sigmoid）
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)  # 递归初始化

    print('training on', device)
    net.to(device)  # 将整个模型搬到 GPU（递归搬运所有参数和缓冲区）

    # SGD 优化器：
    #   更新公式 param = param - lr * param.grad
    #   PyTorch 的 optim.SGD 自带动量(momentum)和权重衰减(weight_decay)选项
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    # CrossEntropyLoss = LogSoftmax + NLLLoss（负对数似然）
    #   内置了 Softmax，所以模型最后一层不需要加 Softmax
    #   期望输入 shape=(N, num_classes)，标签 shape=(N,) 为类别索引（非 one-hot）
    loss = nn.CrossEntropyLoss()

    # d2l.Animator：可视化训练曲线（基于 matplotlib）
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    # d2l.Timer()：计时器，记录累计时间
    timer, num_batches = d2l.Timer(), len(train_iter)

    for epoch in range(num_epochs):
        # Accumulator(3)：累加 3 个指标
        #   [0] = 训练 loss 累计, [1] = 正确预测数, [2] = 样本总数
        metric = d2l.Accumulator(3)

        net.train()  # 设置为训练模式！(启用 Dropout、更新 BN 统计量)

        for i, (X, y) in enumerate(train_iter):  # 元组解包：数据+标签
            timer.start()  # 开始计时
            optimizer.zero_grad()  # 清零梯度（PyTorch 默认累加，必须清零！）

            X, y = X.to(device), y.to(device)  # 数据搬到 GPU
            y_hat = net(X)                     # 前向传播
            l = loss(y_hat, y)                 # 计算损失
            l.backward()                       # 反向传播（计算梯度）
            optimizer.step()                   # 更新参数（w = w - lr * grad）

            # 记录指标（不追踪梯度）
            with torch.no_grad():
                # l * X.shape[0]：将平均 loss 还原为总 loss
                #   CrossEntropyLoss 默认 reduction='mean'，每批返回平均值
                #   乘以 batch_size 得该批的总损失，方便后续计算全局平均
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()  # 停止计时

            # 计算当前 epoch 的累计指标
            train_l = metric[0] / metric[2]    # 平均训练 loss
            train_acc = metric[1] / metric[2]   # 训练准确率

            # 每 1/5 个 epoch 或最后一个 batch 时更新可视化
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))

        # 每个 epoch 结束后，在测试集上评估
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))

    # 打印最终结果
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    # metric[2] = 累计样本总数, num_epochs * metric[2] = 总处理样本数
    # timer.sum() = 总训练秒数 → 计算吞吐量(样本/秒)
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


# ============================================================================
# 第6节：训练 LeNet-5
# ============================================================================
# 参数选择：
#   lr=0.9：SGD 的初始学习率（对 LeNet+Sigmoid 较高是合适的）
#   num_epochs=10：训练 10 轮
#   device=d2l.try_gpu()：有 GPU 就用 GPU 0，没有则用 CPU

lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# 预期结果：test acc ≈ 0.78-0.80（Fashion-MNIST 上的基线）


# ============================================================================
# 补充知识点
# ============================================================================

# ============================================================================
# 补充 1：LeNet-5 的参数量和计算量
# ============================================================================
"""
LeNet-5 各层参数明细：

  ┌─────────────────┬───────────────────┬──────────────────┐
  │ 层               │ 输出形状           │ 参数量            │
  ├─────────────────┼───────────────────┼──────────────────┤
  │ Conv2d(1→6, 5)  │ (6, 28, 28)       │ 6×(1×5×5+1)=156  │
  │ AvgPool(2, s=2) │ (6, 14, 14)       │ 0                │
  │ Conv2d(6→16, 5) │ (16, 10, 10)      │ 16×(6×5×5+1)=2416│
  │ AvgPool(2, s=2) │ (16, 5, 5)        │ 0                │
  │ Flatten         │ (400,)            │ 0                │
  │ Linear(400→120) │ (120,)            │ 400×120+120=48120│
  │ Linear(120→84)  │ (84,)             │ 120×84+84=10164  │
  │ Linear(84→10)   │ (10,)             │ 84×10+10=850     │
  ├─────────────────┼───────────────────┼──────────────────┤
  │ 总计             │ —                 │ ≈ 61,706         │
  └─────────────────┴───────────────────┴──────────────────┘

仅 6 万个参数！对比全连接 MLP（784→256→10 就有 20 万参数），
LeNet 用更少的参数实现了更好的图像分类效果。
"""

# ============================================================================
# 补充 2：训练流程关键技巧总结
# ============================================================================
"""
完整的训练循环需要记住的几个关键步骤（缺一不可）：

  for epoch in range(num_epochs):
      net.train()                    # 1. 训练模式！
      for X, y in train_iter:
          optimizer.zero_grad()      # 2. 清零梯度！
          X, y = X.to(device), y.to(device)  # 3. 数据搬到 GPU
          y_hat = net(X)             # 4. 前向传播
          l = loss(y_hat, y)         # 5. 计算损失
          l.backward()               # 6. 反向传播
          optimizer.step()           # 7. 更新参数

      net.eval()                     # 8. 评估模式！
      with torch.no_grad():          # 9. 关 autograd
          test model on test set

常见遗漏 bug：
  - 忘记 zero_grad() → 梯度累积，模型不收敛
  - 忘记 net.train()/eval() → Dropout/BN 行为错误
  - 忘记 X.to(device) → 模型在 GPU 但数据在 CPU，报错
  - 忘记 torch.no_grad() → 评估时浪费 GPU 显存
"""

# ============================================================================
# 补充 3：LeNet 的现代改进方向
# ============================================================================
"""
原版 LeNet 使用 Sigmoid + AvgPool，现代改进：

  1. ReLU 替换 Sigmoid：
     → 缓解梯度消失，训练更快

  2. MaxPool 替换 AvgPool：
     → 保留最显著特征，实践中效果更好

  3. 添加 BatchNorm：
     → 稳定训练，允许更高学习率

  4. 添加 Dropout：
     → 防止全连接层过拟合

  5. 更深的架构：
     → LeNet(5层) → AlexNet(8层) → VGG(16-19层) → ResNet(152层)

  改进后的"现代版 LeNet"：
    Conv2d → BN → ReLU → MaxPool  →  Conv2d → BN → ReLU → MaxPool
    → Flatten → Dropout → FC → ReLU → Dropout → FC
"""

# ============================================================================
# 补充 4：常见问题 / 练习思考
# ============================================================================
"""
Q1: 为什么 LeNet 的最后一层不加 Softmax？
A1: nn.CrossEntropyLoss 内部已包含 LogSoftmax：
    loss = -log(softmax(logits)[target])
    如果模型输出已过 Softmax → 概率值取 log 后再 softmax → 错误！
    所以：模型输出 raw logits，让 loss 函数内部做 softmax。

Q2: 为什么 CNN 的特征图越来越"厚"（通道数增加）？
A2: 通道数 = 特征的"种类数"。浅层检测简单特征（边缘、颜色），
    种类不多（6→16）。深层检测复杂特征（形状、部件），
    需要更多种类来区分（128→256→512）。
    空间分辨率下降通过通道深度增加来补偿信息容量。

Q3: Flatten 后为何是 16*5*5=400 而不是固定值？
A3: 取决于输入图像大小。28×28 输入经过两次 2×2 pooling
    （每次减半：28→14→5，第二个卷积无 padding 再缩 4 像素），
    最终特征图 = 16 通道 × 5 × 5 = 400。
    输入尺寸改变，这个值也会变，所以全连接的输入维数需要手动计算。
"""

# ============================================================================
# 总结
# ============================================================================
"""
1. LeNet = 卷积块 ×2 + 全连接块 ×3，是最早的 CNN 之一
2. CNN 设计规律：空间尺寸 ↓，通道数 ↑
3. 训练 CNN 与训练 MLP 流程相同：forward → loss → backward → step
4. 关键细节：zero_grad()、train()/eval()、to(device)、no_grad()
5. LeNet 仅 6 万参数，却能在 Fashion-MNIST 上达到 ~79% 准确率
6. 现代改进：ReLU + MaxPool + BN + Dropout = 更强、更快、更稳定
"""
