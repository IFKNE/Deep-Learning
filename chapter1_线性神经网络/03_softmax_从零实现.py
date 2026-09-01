"""
=============================================================================
Softmax回归 —— 从零开始实现（完整带注释版）
=============================================================================
本文件将 softmax-regression-scratch 从零实现.ipynb 中的所有代码汇总在一起，
每一行都配有详细的中文注释，方便学习和理解。

任务：使用 Fashion-MNIST 数据集训练一个 softmax 回归模型，实现 10 分类。
     输入：28×28 的灰度图像（共 784 个像素特征）
     输出：10 个类别的概率分布（T恤、裤子、套头衫、连衣裙、外套、凉鞋、衬衫、运动鞋、包、短靴）

整体流程：加载数据 → 初始化参数 → 定义模型 → 定义损失函数 → 训练 → 预测
=============================================================================
"""

# ============================================================================
# 第1步：导入必要的库
# ============================================================================
import torch                          # PyTorch 深度学习框架，提供张量运算和自动求导
from IPython import display           # 用于在 Jupyter Notebook 中显示动画/图表
from d2l import torch as d2l          # 《动手学深度学习》(D2L) 配套工具库，封装了数据加载、绘图等常用函数


# ============================================================================
# 第2步：加载 Fashion-MNIST 数据集
# ============================================================================
batch_size = 256                      # 批量大小：每次迭代使用 256 个样本
                                      # 较大的 batch_size 可以更稳定地估计梯度，但占用更多内存
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# d2l.load_data_fashion_mnist() 返回两个数据迭代器：
#   - train_iter: 训练集迭代器，包含 60000 张 28×28 的灰度服装图像
#   - test_iter:  测试集迭代器，包含 10000 张图像
# 每个迭代器每次 yield 一个 (特征, 标签) 的元组


# ============================================================================
# 第3步：初始化模型参数
# ============================================================================
# 为什么是 784？因为每张图像是 28×28 = 784 个像素，我们将图像展平为 784 维向量作为输入
num_inputs = 784                      # 输入特征数：28 × 28 = 784
num_outputs = 10                      # 输出类别数：Fashion-MNIST 有 10 个类别

# 权重矩阵 W 的形状为 (784, 10)
#   784 行对应 784 个输入特征（每个像素是一个特征）
#   10  列对应 10 个输出类别（每个类别有一组权重）
# torch.normal(均值, 标准差, size=形状) 从正态分布中随机采样来初始化权重
# requires_grad=True 表示需要计算该张量的梯度，这样训练时才能通过反向传播更新它
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)

# 偏置向量 b 的形状为 (10,)
#   每个输出类别有一个偏置项
# torch.zeros() 将偏置全部初始化为 0
b = torch.zeros(num_outputs, requires_grad=True)


# ============================================================================
# 第4步：理解 sum 运算符在特定维度上的行为（预备知识）
# ============================================================================
# 创建一个 2 行 3 列的示例矩阵，帮助理解 softmax 中的求和操作
X = torch.tensor([[1.0, 2.0, 3.0],    # 第 0 个样本
                   [4.0, 5.0, 6.0]])   # 第 1 个样本

# X.sum(0, keepdim=True)：沿着轴 0（行方向/竖直方向）求和
#   即对每一列求和：1+4=5, 2+5=7, 3+6=9
#   keepdim=True 保持维度不折叠，结果形状为 (1, 3) 而非 (3,)
#   (1, 3) 形状便于后续的广播机制（broadcasting）
print("沿轴0求和（对列求和）:", X.sum(0, keepdim=True))  # tensor([[5., 7., 9.]])

# X.sum(1, keepdim=True)：沿着轴 1（列方向/水平方向）求和
#   即对每一行求和：1+2+3=6, 4+5+6=15
#   结果形状为 (2, 1)
print("沿轴1求和（对行求和）:", X.sum(1, keepdim=True))  # tensor([[ 6.], [15.]])


# ============================================================================
# 第5步：定义 softmax 函数
# ============================================================================
# softmax 的数学公式：
#   softmax(X)_{ij} = exp(X_{ij}) / Σ_k exp(X_{ik})
# 含义：对于第 i 个样本，将其第 j 个类别的原始分数 X_{ij} 通过 e 的指数放大，
#       然后除以该样本所有类别分数的指数和，从而得到归一化的概率分布。

def softmax(X):
    """
    对输入 X 的每一行执行 softmax 运算，将原始分数转换为概率分布。

    参数:
        X: 形状为 (batch_size, num_classes) 的张量，每行是一个样本的原始分数

    返回:
        形状相同的张量，每行元素为非负数且和为 1（即合法的概率分布）

    分步说明：
        (1) torch.exp(X): 对每个元素取指数，将任意实数映射到正数
        (2) X_exp.sum(1, keepdim=True): 沿轴1（行方向）求和，得到每行的归一化分母
        (3) X_exp / partition: 广播除法，每行除以自己的分母，使概率之和为 1
    """
    X_exp = torch.exp(X)                     # 对所有元素求指数 e^x，确保结果为正数
    partition = X_exp.sum(1, keepdim=True)   # 对每一行求和，得到每行的配分函数（归一化常数）
    return X_exp / partition                 # 广播机制：每一行除以自己行的和，得到概率分布


# 验证 softmax 的输出：每行的和应该为 1
X_test = torch.normal(0, 1, (2, 5))         # 从标准正态分布随机生成 2 行 5 列的测试数据
X_prob = softmax(X_test)                     # 应用 softmax
print("\nsoftmax 后的概率分布:\n", X_prob)
print("每行的概率之和:", X_prob.sum(1))       # 应该都是 1.0
# 注意：实际生产中，直接对很大的数求 exp 可能导致数值溢出（如 exp(50) 会很大），
# 解决方法：先减去每行的最大值，即 softmax(X) = softmax(X - max(X))，数学上等值但数值稳定


# ============================================================================
# 第6步：定义模型（神经网络前向传播）
# ============================================================================
def net(X):
    """
    Softmax 回归模型的前向传播。

    参数:
        X: 形状为 (batch_size, 28, 28) 的原始图像批次

    返回:
        形状为 (batch_size, 10) 的预测概率分布

    计算过程：
        (1) X.reshape((-1, W.shape[0])): 将每张 28×28 的图像展平为 784 维向量
            -1 表示"自动推断该维度大小"，这里会等于 batch_size
        (2) torch.matmul(..., W): 矩阵乘法，(batch_size, 784) × (784, 10) = (batch_size, 10)
           这是线性变换：每个输入特征乘以对应的权重并求和
        (3) + b: 加上偏置项（广播机制自动将 (10,) 扩展为 (batch_size, 10)）
        (4) softmax(...): 将线性变换的结果转换为概率分布
    """
    # 展平 + 线性变换 + softmax
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


# ============================================================================
# 第7步：定义交叉熵损失函数
# ============================================================================
# 交叉熵损失的数学公式：
#   CrossEntropy = -log(ŷ_y)
# 其中 ŷ_y 是模型对真实类别 y 的预测概率。
# 直观理解：模型对正确答案越有信心（概率越接近1），损失越小（-log(1)=0）；
#          模型越不确定（概率越接近0），损失越大（-log(0)→∞）。

# 先用一个简单示例理解索引操作：
y_demo = torch.tensor([0, 2])                           # 真实标签：第1个样本是类别0，第2个样本是类别2
y_hat_demo = torch.tensor([[0.1, 0.3, 0.6],             # 第1个样本对3个类别的预测概率
                            [0.3, 0.2, 0.5]])            # 第2个样本对3个类别的预测概率
# y_hat_demo[[0, 1], y_demo] 的含义：
#   第0行取列索引0（概率0.1），第1行取列索引2（概率0.5）
#   即取出每个样本在其真实类别上的预测概率
print("\n取出真实类别的预测概率:", y_hat_demo[[0, 1], y_demo])  # tensor([0.1, 0.5])


def cross_entropy(y_hat, y):
    """
    计算交叉熵损失。

    参数:
        y_hat: 形状为 (batch_size, num_classes) 的预测概率分布
        y:     形状为 (batch_size,) 的真实标签（整数，不是 one-hot 编码）

    返回:
        形状为 (batch_size,) 的张量，每个元素是该样本的交叉熵损失

    实现细节：
        range(len(y_hat)) 生成 [0, 1, 2, ..., batch_size-1] 的行索引
        y 提供列索引（即真实类别）
        y_hat[行索引, 列索引] 用高级索引一次性取出所有样本在真实类别上的概率
        -torch.log(...) 取负对数，得到每个样本的损失值
    """
    return -torch.log(y_hat[range(len(y_hat)), y])

# 验证：第一个样本真实类别的概率为 0.1，损失 = -ln(0.1) ≈ 2.3026
#       第二个样本真实类别的概率为 0.5，损失 = -ln(0.5) ≈ 0.6931
print("交叉熵损失:", cross_entropy(y_hat_demo, y_demo))


# ============================================================================
# 第8步：定义分类精度计算函数
# ============================================================================
def accuracy(y_hat, y):
    """
    计算预测正确的样本数量。

    参数:
        y_hat: 预测值，可以是概率分布矩阵 (batch_size, num_classes)
               也可以是已经取过 argmax 的预测类别向量
        y:     形状为 (batch_size,) 的真实标签

    返回:
        预测正确的样本数量

    步骤：
        (1) 如果 y_hat 是二维矩阵（概率分布），用 argmax 找出每行最大值的索引作为预测类别
        (2) 将预测类别转换为与真实标签相同的数据类型
        (3) 逐元素比较预测类别 == 真实标签，得到一个布尔张量
        (4) 将布尔值转换为数值（True→1, False→0）并求和
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:    # 如果 y_hat 是二维矩阵（有多列）
        y_hat = y_hat.argmax(axis=1)                    # 取每行最大值的索引作为预测类别
    cmp = y_hat.type(y.dtype) == y                      # 比较预测类别与真实标签，得到 True/False 张量
    return float(cmp.type(y.dtype).sum())               # 将 True/False 转为 1/0 再求和


# 验证精度函数
print("\n精度（正确数）/ 样本数 =", accuracy(y_hat_demo, y_demo) / len(y_demo))  # 0.5：两个中预测对了一个


# ============================================================================
# 第9步：定义累加器工具类
# ============================================================================
class Accumulator:
    """
    累加器：用于在多个变量上累计求和。

    为什么需要它？
        训练时需要统计"累计损失"、"累计正确数"、"累计样本数"等指标。
        有了累加器，在每个批量后只需调用 add() 即可自动累加。

    使用示例:
        metric = Accumulator(3)       # 创建 3 个累加变量
        metric.add(1.5, 2, 10)        # 向 3 个变量分别累加 1.5, 2, 10
        print(metric[0])              # 查看第 0 个变量的累计值
    """
    def __init__(self, n):
        """初始化 n 个累加变量，初始值均为 0.0"""
        self.data = [0.0] * n

    def add(self, *args):
        """
        向每个累加变量加上对应的值。
        例如 self.data = [1.0, 3.0]; args = (2.0, 5.0) → self.data = [3.0, 8.0]
        """
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        """将所有累加变量重置为 0.0"""
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        """支持用索引访问累加值，如 metric[0]"""
        return self.data[idx]


# ============================================================================
# 第10步：定义评估函数
# ============================================================================
def evaluate_accuracy(net, data_iter):
    """
    在指定数据集上评估模型的分类精度。

    参数:
        net:      模型函数（或 torch.nn.Module）
        data_iter: 数据迭代器（如 test_iter）

    返回:
        分类精度（正确预测数 / 总预测数），范围 [0, 1]

    流程：
        (1) 若是 PyTorch 模型，切换到评估模式（eval），关闭 Dropout/BatchNorm 的训练行为
        (2) 用 Accumulator(2) 创建两个累加变量：正确数、总数
        (3) torch.no_grad() 上下文：关闭梯度计算，节省内存，加速推理
        (4) 遍历数据，累加每个批量的正确数（accuracy 函数）和样本数（y.numel()）
        (5) 返回 正确数 / 总数
    """
    if isinstance(net, torch.nn.Module):                # 判断 net 是否是 PyTorch 的模型类
        net.eval()                                       # 设置为评估模式（而非训练模式）
    metric = Accumulator(2)                              # 创建 2 个累加变量：[0]正确预测数, [1]总预测数
    with torch.no_grad():                                # 禁用梯度计算，因为在评估时不需要反向传播
        for X, y in data_iter:                           # 逐批遍历数据集
            metric.add(accuracy(net(X), y), y.numel())   # 累加正确数和总样本数
    return metric[0] / metric[1]                         # 精度 = 正确数 / 总数


# 在训练前先评估一下：用随机初始化的权重，精度应该接近 1/10 = 0.1（随机猜测水平）
print("\n训练前的测试精度:", evaluate_accuracy(net, test_iter))


# ============================================================================
# 第11步：定义动画绘图工具类
# ============================================================================
class Animator:
    """
    动画绘图器：用于在训练过程中实时绘制损失曲线和精度曲线。

    在 Jupyter Notebook 中，它会动态更新图表，让训练进度一目了然。
    """
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """初始化绘图器，设置坐标轴标签、图例、线型等"""
        if legend is None:
            legend = []
        d2l.use_svg_display()                            # 使用 SVG 格式显示图像（更清晰）
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:                           # 如果是单图，将 axes 包装为列表以便统一处理
            self.axes = [self.axes, ]
        # lambda 函数捕获参数，在每次绘图时配置坐标轴
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts     # X: x轴数据, Y: y轴数据, fmts: 线型格式

    def add(self, x, y):
        """
        向图表中添加数据点并刷新显示。

        参数:
            x: x 轴的值（通常是 epoch 编号）
            y: y 轴的值（可以是单个值或多个值的列表）
        """
        if not hasattr(y, "__len__"):                    # 如果 y 不是列表，转换为列表
            y = [y]
        n = len(y)                                       # 要绘制的曲线数量
        if not hasattr(x, "__len__"):                    # 如果 x 不是列表，复制 n 份
            x = [x] * n
        if not self.X:                                   # 首次添加数据时，初始化 X 列表
            self.X = [[] for _ in range(n)]
        if not self.Y:                                   # 首次添加数据时，初始化 Y 列表
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):           # 逐条曲线追加数据点
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()                               # 清除当前图形
        for x_data, y_data, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x_data, y_data, fmt)       # 用对应的线型绘制每条曲线
        self.config_axes()                               # 配置坐标轴
        display.display(self.fig)                        # 显示图形
        display.clear_output(wait=True)                  # 清除旧输出，实现"动画"效果


# ============================================================================
# 第12步：定义单个迭代周期的训练函数
# ============================================================================
def train_epoch_ch3(net, train_iter, loss, updater):
    """
    训练模型一个迭代周期（epoch）。

    一个 epoch = 完整遍历训练集一次。

    参数:
        net:        模型
        train_iter: 训练数据迭代器
        loss:       损失函数
        updater:    参数更新器（可以是 PyTorch 优化器或自定义的 sgd 函数）

    返回:
        (平均训练损失, 平均训练精度)

    训练一个批量的流程：
        (1) 前向传播：y_hat = net(X)，计算预测值
        (2) 计算损失：l = loss(y_hat, y)
        (3) 反向传播：l.sum().backward()，计算梯度
        (4) 更新参数：updater(X.shape[0])，用梯度更新 W 和 b
    """
    if isinstance(net, torch.nn.Module):                 # 如果是 PyTorch 模型，切换到训练模式
        net.train()                                      # 启用 Dropout、BatchNorm 等训练行为
    metric = Accumulator(3)                              # 3个累加变量：[0]损失和, [1]正确数, [2]样本数
    for X, y in train_iter:                              # 逐批遍历训练数据
        y_hat = net(X)                                   # 前向传播：计算预测值
        l = loss(y_hat, y)                               # 计算损失

        # 根据 updater 的类型选择对应的参数更新方式
        if isinstance(updater, torch.optim.Optimizer):   # 如果是 PyTorch 内置优化器（如 SGD, Adam）
            updater.zero_grad()                          # 清零梯度（PyTorch 默认累积梯度）
            l.mean().backward()                          # 对平均损失反向传播
            updater.step()                               # 执行一步参数更新
        else:                                            # 如果是自定义的更新函数（如 d2l.sgd）
            l.sum().backward()                           # 对总损失反向传播（自定义 sgd 内部会除以 batch_size）
            updater(X.shape[0])                          # 调用自定义更新函数，传入 batch_size

        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        # 累加：总损失、预测正确的数量、样本总数

    # 返回平均损失和平均精度
    return metric[0] / metric[2], metric[1] / metric[2]


# ============================================================================
# 第13步：定义完整的训练函数（多个 epoch）
# ============================================================================
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """
    完整训练模型，运行多个 epoch，并实时绘制训练/测试曲线。

    参数:
        net:         模型
        train_iter:  训练数据迭代器
        test_iter:   测试数据迭代器
        loss:        损失函数
        num_epochs:  训练周期数
        updater:     参数更新器

    训练过程：
        for epoch in range(num_epochs):
            (1) train_epoch_ch3(): 训练一个 epoch，得到训练损失和训练精度
            (2) evaluate_accuracy(): 在测试集上评估，得到测试精度
            (3) animator.add(): 将本轮指标添加到实时图表中
    """
    # 创建动画绘图器：x 轴是 epoch 编号，y 轴范围设为 [0.3, 0.9]
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):                      # 进行 num_epochs 轮训练
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)  # 训练一个 epoch
        test_acc = evaluate_accuracy(net, test_iter)     # 在测试集上评估精度
        animator.add(epoch + 1, train_metrics + (test_acc,))  # 绘制：训练损失、训练精度、测试精度

    train_loss, train_acc = train_metrics
    # 训练完成后的断言检查，确保训练效果达标
    assert train_loss < 0.5, train_loss                  # 训练损失应低于 0.5
    assert train_acc <= 1 and train_acc > 0.7, train_acc # 训练精度应在 70% 以上
    assert test_acc <= 1 and test_acc > 0.7, test_acc    # 测试精度应在 70% 以上


# ============================================================================
# 第14步：设置优化器（小批量随机梯度下降 SGD）
# ============================================================================
lr = 0.1                                                # 学习率：控制每次参数更新的步长
                                                         # 太大可能导致不收敛，太小则训练太慢

def updater(batch_size):
    """
    自定义的 SGD 参数更新函数。

    参数:
        batch_size: 当前批量的样本数，用于对梯度做平均（批量越大，梯度求和后需要除以越大的数）

    d2l.sgd() 内部逻辑（简化版）：
        with torch.no_grad():
            param -= lr * param.grad / batch_size  # 沿负梯度方向更新参数
            param.grad.zero_()                      # 清零梯度，为下一轮做准备
    """
    return d2l.sgd([W, b], lr, batch_size)              # d2l.sgd 对给定的参数列表执行 SGD 更新


# ============================================================================
# 第15步：开始训练
# ============================================================================
num_epochs = 10                                         # 训练 10 个 epoch
print("\n开始训练...")
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
# 10 个 epoch 后，训练精度和测试精度通常都在 80% 左右
print("训练完成！")


# ============================================================================
# 第16步：在新数据上做预测
# ============================================================================
def predict_ch3(net, test_iter, n=6):
    """
    展示模型的预测结果。

    参数:
        net:       训练好的模型
        test_iter: 测试数据迭代器
        n:         展示的图片数量

    输出：
        显示 n 张图像，每张图像上方标注真实标签和模型预测标签
    """
    for X, y in test_iter:                               # 取一个批量的数据
        break                                            # 只取第一个批量就退出
    trues = d2l.get_fashion_mnist_labels(y)              # 将数字标签转换为文字标签（如 0→"t-shirt"）
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))  # 模型预测：argmax 取概率最大的类别
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]  # 拼接为"真实\n预测"的标题
    d2l.show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])  # 显示前 n 张图像及其标题

print("\n预测结果展示：")
predict_ch3(net, test_iter)


# ============================================================================
# 补充知识点：数值稳定性改进
# ============================================================================
# 问题1：原始 softmax 中直接对较大的数求 exp 可能溢出（如 exp(50) ≈ 5.18×10^21，超出浮点范围）
# 问题2：交叉熵中对接近 0 的数取 log 会产生 -inf
#
# 解决方法：
#   在 softmax 中，先从输入中减去每行的最大值：
#     softmax(X) = softmax(X - max(X))
#   这在数学上完全等价（因为分子分母同时除以 exp(max)，约掉了），但数值上更稳定。
#   改进后的实现如下：

def softmax_stable(X):
    """数值稳定的 softmax 实现"""
    X_max = X.max(dim=1, keepdim=True).values           # 取每行的最大值
    X_shifted = X - X_max                               # 所有值减去最大值（最大项变成 0）
    X_exp = torch.exp(X_shifted)                        # 再求 exp，最大值为 1，不会溢出
    partition = X_exp.sum(dim=1, keepdim=True)           # 计算分母
    return X_exp / partition                            # 归一化


# ============================================================================
# 总结
# ============================================================================
# 1. Softmax 回归是线性回归在多分类问题上的推广
# 2. softmax 函数将原始分数映射为概率分布（非负且和为1）
# 3. 交叉熵损失衡量预测概率与真实标签之间的差异
# 4. 训练流程：前向传播 → 计算损失 → 反向传播 → 更新参数
# 5. 从零实现能帮助你深入理解每个组件的原理，而不仅仅是调用高级 API
