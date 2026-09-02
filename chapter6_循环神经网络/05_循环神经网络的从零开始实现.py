"""
=============================================================================
循环神经网络的从零开始实现（RNN from Scratch）—— 完整带注释版
=============================================================================
根据《动手学深度学习》第 8 章 sec_rnn 的描述，从头实现一个
字符级语言模型（character-level language model），并在 H.G.Wells 的
《时光机器》(The Time Machine) 数据集上训练。

本节不使用 nn.RNN 等封装好的层，而是手写核心逻辑：
  - 独热编码（one-hot encoding）
  - 参数初始化 get_params
  - 隐状态初始化 init_rnn_state
  - 按时间步循环计算的 rnn 函数
  - 包装类 RNNModelScratch
  - 逐字符预测 predict_ch8、梯度裁剪 grad_clipping
  - 训练循环 train_epoch_ch8 / train_ch8

输入: 时光机器语料（词表 vocab），每个小批量是 (batch, num_steps)
输出: 字符级语言模型；给定一个前缀（字符串），自动生成其后的字符序列

学习目标：真正理解 RNN "按时间步循环、隐状态沿时间传递" 的本质，
以及训练时为什么要初始化状态、为什么要梯度裁剪、如何用困惑度评估。
=============================================================================
"""

# %matplotlib inline —— Jupyter 专用的"IPython 魔法命令"，作用是在笔记本里内联显示图形。
# 纯 .py 脚本 / 命令行下无意义，故按规范略去（仅保留说明）。下面 import 照常保留。
import math                            # 数学库：这里主要用 math.exp 计算困惑度
import torch                           # PyTorch 主库：张量、自动微分（autograd）
from torch import nn                   # 神经网络层、模块基类、损失函数
from torch.nn import functional as F   # 无状态函数式接口：F.one_hot 等
from d2l import torch as d2l           # 《动手学深度学习》官方工具包（数据/计时/绘图/优化器等）


# ============================================================================
# 读取数据集：批量大小与时间步数
# ============================================================================
# batch_size: 每个小批量包含的独立子序列个数（"一次处理几路并行的序列"）
# num_steps:  每个子序列的时间步数（即序列长度，也等于 RNN 的"展开长度" T）
batch_size, num_steps = 32, 35

# d2l.load_data_time_machine：加载 H.G.Wells 的《时光机器》语料（d2l 工具包已封装）。
#   返回:
#     train_iter —— 小批量迭代器，每次产出 (X, Y)：
#                    X 形状 (batch_size, num_steps)，Y 是"下一个时刻"的标签，形状相同
#     vocab     —— 词表对象，内含 token→索引、以及 idx_to_token 反查表
# 说明：真正的训练会通过网络自动下载语料（本文件只做演示，导入后不实际跑训练）。
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)


# ============================================================================
# 独热编码（one-hot encoding）
# ============================================================================
# 回想：train_iter 中每个词元都是用"整数索引"表示的。直接把数字喂给网络，
# 会误导模型去学习"顺序/大小"等不存在的含义。通常我们把每个词元转成更"可表达"的向量。
# 最简单的表示叫独热编码：把索引 i 映射成"第 i 位为 1、其余全为 0"的单位向量。

# F.one_hot(tensor, num_classes)：
#   把每个整数索引映射为一个长度为 num_classes 的独热向量。
#     torch.tensor([0, 2])：两个待编码的数据（索引 0 与 2）
#     len(vocab)：词表大小 N（即独热向量的维度 = 类别数）
#   结果形状：(2, N) —— 2 个样本，每个样本一个 N 维独热向量
#     索引 0 → [1, 0, 0, ...]（第 0 位为 1）
#     索引 2 → [0, 0, 1, ...]（第 2 位为 1）
F.one_hot(torch.tensor([0, 2]), len(vocab))


# 每次采样得到的小批量数据是二维张量：(批量大小 batch, 时间步数 num_steps)。
# 若直接对 (batch, num_steps) 的 X 做 one_hot，得到的最后一维是词表大小：
#     F.one_hot(X, len(vocab))  → (batch, num_steps, vocab)
# 但我们更想要形状为 (时间步数, 批量大小, 词表大小) 的输出：
#   让"最外层维度 = 时间"，这样就能方便地沿最外层维度逐时间步更新隐状态。
# 因此做法是：先转置 X.T（变成 (num_steps, batch)），然后再 one_hot。

X = torch.arange(10).reshape((2, 5))   # X 形状 (2, 5)：2 个样本，每样本 5 个时间步
F.one_hot(X.T, 28).shape               # X.T 形状 (5, 2)；独热后形状为 (5, 2, 28)
#   5   = 时间步数（最外层，即 RNN 循环的维度，逐个时间步处理）
#   2   = 批量大小（同一时刻并行处理 2 个序列）
#   28  = 词表大小（每个时间步对应的独热向量长度）


# ============================================================================
# 初始化模型参数
# ============================================================================
# 隐藏单元数 num_hiddens 是一个可调超参数。
# 训练语言模型时，输入和输出都来自同一个词表，因此二者维度相同（= vocab_size）。

def get_params(vocab_size, num_hiddens, device):
    """
    初始化 RNN 的可学习参数（权重矩阵 W 与偏置向量 b）。

    参数:
        vocab_size:    词表大小 N（= 输入维度 = 输出维度）
        num_hiddens:   隐藏单元数 h
        device:        计算设备（CPU / GPU）

    返回:
        [W_xh, W_hh, b_h, W_hq, b_q] 长度 5 的参数列表（均已开启梯度追踪）
    """
    # 输入、输出维度都等于词表大小（语言模型：预测下一个字符，二者同词表）
    num_inputs = num_outputs = vocab_size

    # normal(shape)：生成指定形状、均值为 0、标准差为 0.01 的随机张量。
    #   torch.randn(size, device) 从标准正态分布 N(0,1) 采样，再乘 0.01 缩小数值。
    #   权重初始化要"小"——太大容易使信号/梯度在传播中爆炸。
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数（把"当前输入 x_t"与"上一时刻隐状态 H_{t-1}"映射到新隐状态 H_t）
    W_xh = normal((num_inputs, num_hiddens))       # (N, h)：输入 → 隐层
    W_hh = normal((num_hiddens, num_hiddens))      # (h, h)：隐层 → 隐层（自循环，体现"记忆"）
    b_h = torch.zeros(num_hiddens, device=device)  # (h,)：隐层偏置
    # 输出层参数（把隐状态 H_t 映射到词表大小的输出 logits）：
    W_hq = normal((num_hiddens, num_outputs))      # (h, N)：隐层 → 输出
    b_q = torch.zeros(num_outputs, device=device)  # (N,)：输出偏置

    # 把全部参数放进一个列表，统一管理、统一计算梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)   # 标记"需要计算梯度"，训练时才能参与反向传播
    return params


# ============================================================================
# 循环神经网络模型
# ============================================================================
# 先定义"初始化隐状态"的函数。它返回一个全 0 张量，形状 (批量大小, 隐藏单元数)。
# 之所以用"元组 (H, )"而不是裸张量，是因为高级 RNN（如 LSTM）的隐状态可能含多个变量
# （cell state + hidden state 等），统一用元组更易扩展。

def init_rnn_state(batch_size, num_hiddens, device):
    # 注意末尾的逗号：让 (张量,) 成为一个"1 元元组"，而不是单纯的括号
    return (torch.zeros((batch_size, num_hiddens), device=device), )


# 下面的 rnn 函数定义了"在一个时间步内"如何计算隐状态和输出。
# 模型通过 inputs 的最外层维度（时间）实现循环，逐时间步更新小批量的隐状态 H。
# 激活函数用 tanh：当输入在实数上均匀分布时 tanh 的均值约为 0，利于数值稳定。

def rnn(inputs, state, params):
    # inputs 的形状：(时间步数量 T, 批量大小 batch, 词表大小 vocab)
    #   —— 这正是前面把 X 转置后再 one_hot 得到的 (T, batch, vocab)
    W_xh, W_hh, b_h, W_hq, b_q = params   # 解包 5 个参数
    H, = state                            # state 是 (H,) 元组，用 `H, =` 取出唯一的 H
    outputs = []                          # 收集每个时间步的输出
    # 循环遍历"每个时间步"：这里的 X 形状 (批量大小 batch, 词表大小 vocab)
    for X in inputs:
        # 隐层递推公式：H_t = tanh( X_t · W_xh + H_{t-1} · W_hh + b_h )
        #   上一时刻的 H 通过 W_hh 参与本时刻计算 → 这就是"循环/记忆"的本质
        #   torch.mm 是矩阵乘法（二维），一次处理整个小批量
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        # 输出层：Y_t = H_t · W_hq + b_q，形状 (batch, vocab)，即各位置的 logits
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)                 # 保存本时间步的输出
    # 沿 dim=0（时间维）拼接所有时间步的输出：(T, batch, vocab) → (T*batch, vocab)
    #   同时把最后一个隐状态 H 作为元组返回，供下一个时间步/下一个小批量继续使用
    return torch.cat(outputs, dim=0), (H,)


# 把所有函数与参数打包成一个类，方便统一调用与状态管理。

class RNNModelScratch:
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        # 记录词表大小与隐藏单元数
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        # 用外部传入的 get_params 生成该模型的全部参数
        self.params = get_params(vocab_size, num_hiddens, device)
        # 保存"隐状态初始化函数"与"前向计算函数"
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        # 前向入口：先把输入的整数索引编码为独热向量。
        #   X 形状 (batch, num_steps) → X.T 形状 (num_steps, batch)
        #   F.one_hot(X.T, vocab_size) → (num_steps, batch, vocab)
        #   .type(torch.float32)：one_hot 默认返回整数张量，需转 float 才能参与矩阵乘
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        # 调用真正的前向函数（即上面手写的 rnn），并传入本类管理的参数
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        # 对外接口：给定批量大小与设备，返回初始隐状态（全 0）
        return self.init_state(batch_size, self.num_hiddens, device)


# 检查输出形状是否正确——例如隐状态的维度是否保持不变。

num_hiddens = 512
# 构建一个从零实现的 RNN：词表大小 len(vocab)，512 个隐藏单元，放到 GPU 上
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
# 用 X 的批量大小（2）初始化隐状态
state = net.begin_state(X.shape[0], d2l.try_gpu())
# 前向计算：Y 形状 (时间步数×批量大小, 词表大小)，new_state 是元组 (H,)
Y, new_state = net(X.to(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape
# 输出形如: (torch.Size([10, 28]), 1, torch.Size([2, 512]))
#   10       = 5 个时间步 × 2 个批量（时间维被拼接）
#   28       = 词表大小
#   1        = new_state 元组里只有 1 个元素
#   (2,512)  = 隐状态（批量×隐藏单元数），与最初 H 的形状保持一致（维度不变）


# ============================================================================
# 预测
# ============================================================================
# 在 prefix 之后生成新字符，prefix 是用户提供的字符串。
# 遍历 prefix 的开头字符时，不断把隐状态传给下一个时间步，但不产生输出——这叫"预热(warm-up)"，
# 让模型借这段上下文自我更新，从而获得比初始值更适合预测的隐状态；预热结束才真正逐字生成。

def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符"""
    # 批量大小为 1：一次只生成一串文本
    state = net.begin_state(batch_size=1, device=device)
    # outputs 存放字符的整数索引；先放入前缀的首字符索引
    outputs = [vocab[prefix[0]]]
    # get_input 构造单个输入张量：取上一个字符的索引，reshape 成 (1,1)（batch=1, 时间步=1）
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期：吃入前缀的剩余字符，只更新隐状态、不输出预测
        _, state = net(get_input(), state)
        outputs.append(vocab[y])   # 记录前缀本身每个字符的索引
    for _ in range(num_preds):     # 预测 num_preds 步（自回归：上一步输出当作下一步输入）
        y, state = net(get_input(), state)
        # argmax(dim=1) 取每个位置概率最大的类别索引；reshape(1) 把 (1,1) 展平为标量
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    # 用 idx_to_token 把索引还原为字符，拼接成一个字符串
    return ''.join([vocab.idx_to_token[i] for i in outputs])


# 用前缀 "time traveller " 生成后续 10 个字符。
# 此时模型尚未训练、参数是随机的，会输出一串"荒谬"的文本（这是正常的预期现象）。
predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())


# ============================================================================
# 梯度裁剪（gradient clipping）
# ============================================================================
# 对长度为 T 的序列，迭代中会在 T 个时间步上反算梯度，形成长约 O(T) 的矩阵乘法链；
# 当 T 较大时容易数值不稳定，导致"梯度爆炸"（梯度很大）或"梯度消失"。
# 梯度裁剪提供一种快速缓解爆炸的手段：把梯度 g 投影回半径 θ 的球内：
#       g ← min(1, θ / ||g||) · g
#   当 ||g|| ≤ θ 时 min(...)=1，梯度保持不变；
#   当 ||g|| > θ 时按 θ/||g|| 缩放（该比值 < 1），梯度方向不变、大小被压到 θ。
# 副作用：限制了任何单个小批量（及其样本）对参数向量的影响，提升训练稳定性。

def grad_clipping(net, theta):
    """裁剪梯度"""
    # net 可能是 nn.Module（由 PyTorch 自动管理参数），也可能是自定义的从零实现类
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params   # 从零实现时，参数集中在 net.params 列表里
    # 计算所有参数梯度的 L2 范数：norm = sqrt( Σ (grad_i)² )
    #   对每个参数的 grad 先平方求和，再加总，最后开方
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            # 对原本地缩放：param.grad[:] *= θ/norm（此时 θ/norm < 1）
            #   用切片 [:]- 实现"原地修改"，不新建张量，保持梯度对象不变
            param.grad[:] *= theta / norm


# ============================================================================
# 训练
# ============================================================================
# 与之前 softmax 从头训练相比有 3 处不同：
#   1. 序列数据的采样方式（随机采样 / 顺序分区）会影响隐状态的初始化；
#   2. 更新参数前要先裁剪梯度，保证即使某处梯度爆炸，模型也不会发散；
#   3. 用困惑度（perplexity）评估，使不同长度的序列具有可比性。

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, d2l.Timer()   # state=当前隐状态；Timer 用于统计速度
    metric = d2l.Accumulator(2)  # 训练损失之和, 词元数量（2 个累加槽）
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 第一次迭代，或使用"随机抽样"：每个周期都要重新初始化隐状态
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # "顺序分区"：只在每个周期开始时初始化一次，之后隐状态跨小批量流水式传递。
            # 但为了降低计算量，在处理每个小批量前先把隐状态从计算图分离（detach），
            # 使隐状态的梯度只局限在当前小批量的时间步内，避免跨整个周期的巨大计算链。
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # 对 nn.GRU：state 是单个张量
                state.detach_()
            else:
                # 对 nn.LSTM 或从零实现模型：state 是元组，逐个元素 detach
                for s in state:
                    s.detach_()
        # 目标标签：Y 形状 (batch, num_steps) → 转置 → 拉平成 (batch*num_steps,) 的长序列
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)    # 把输入与标签搬到 GPU
        y_hat, state = net(X, state)         # 前向：y_hat 形状 (batch*num_steps, vocab)
        l = loss(y_hat, y.long()).mean()     # 交叉熵（默认带 log_softmax）后取均值
        #   y.long()：损失函数要求标签是 int64 长整型，这里显式转类型
        if isinstance(updater, torch.optim.Optimizer):
            # 使用框架内置优化器（如 SGD）：标准"清零→反传→裁剪→更新"四步
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)             # 梯度裁剪，阈值 θ=1
            updater.step()
        else:
            # 使用从零实现的 updater（这里的 d2l.sgd）：同样需要反传与裁剪
            l.backward()
            grad_clipping(net, 1)
            # 因为上面已经对 loss 调用了 mean()，故按 batch_size=1 更新（sgd 会除以 1）
            updater(batch_size=1)
        # 累计：总损失(乘词元数) 与 词元数；metric.add 依次累加到两个槽
        metric.add(l * y.numel(), y.numel())
    # 返回 (困惑度, 每秒处理的词元数)
    #   困惑度 = exp(平均交叉熵)；平均交叉熵 = 总损失 / 总词元数，越小模型越自信
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


# RNN 训练函数同时支持"从零实现"和"高级 API"两种模型。

def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()          # 交叉熵损失（内部含 log_softmax，数值更稳）
    # Animator：d2l 提供的绘图器，横轴 epoch、纵轴 perplexity
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化优化器 / 更新函数
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)         # 高级 API：框架 SGD
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)  # 从零 SGD
    # 预测函数：给定前缀，生成 50 个新字符
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        # 跑一个周期，返回困惑度 ppl 与速度 speed
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:        # 每 10 个周期打印一次预测结果并绘图
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


# 训练 RNN：因为数据集只用了 10000 个词元，模型需要更多周期才能较好地收敛。
# 注意：此处 500 轮、且会下载语料，实际运行耗时较长，
# 本文件只做演示与语法自检（py_compile），并不真正执行训练。
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())


# 最后检查使用"随机抽样"方法的结果：重新造一个模型，并用 use_random_iter=True 训练。
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)


# ============================================================================
# 补充知识点
# ============================================================================

# 1. 独热编码 vs 可学习嵌入
#    独热编码把每个词元映射为相互正交的单位向量，维度 = 词表大小 N，0/1 稀疏。
#    缺点：维度随词表线性增长、且无法表达词与词之间的语义相似度。
#    可学习嵌入（embedding）用低维稠密向量，能习得语义相近的词向量靠近，训练更高效。

# 2. 随机采样 vs 顺序分区（两种小批量取样方式）
#    随机采样：每个子序列在随机位置取，样本之间无顺序关系 → 每个周期都要重置隐状态。
#    顺序分区：各子序列在原文本中相邻，前一个子序列的末位隐状态可初始化下一个子序列，
#              让历史信息在一个周期内"流经"相邻子序列 → 只在周期开始时初始化一次。

# 3. 为什么要 detach（分离梯度）？
#    顺序分区时，若不分离，隐状态的计算依赖同一周期内前面所有小批量，梯度链极长、
#    计算量大。因此处理每个小批量前先 detach 隐状态，把梯度计算限制在单个小批量的时间步内。

# 4. 困惑度（perplexity）的含义
#    困惑度 = exp(平均交叉熵)，它相当于"下一个字符有多少种均匀可选的平均数"。
#    交叉熵越小 → 困惑度越接近 1，说明模型对下一个字符越自信、预测越准确。

# 5. 梯度裁剪能防爆炸，但不能防消失
#    梯度裁剪只限制梯度上界（防爆炸）；梯度消失需要靠门控（LSTM/GRU）、残差连接、
#    恰当的激活函数或梯度截断值等手段来缓解。

# 6. 为什么用 tanh 激活
#    当输入在实数上均匀分布时 tanh 的均值约为 0，有助于后续层数值稳定；
#    且值域 (-1, 1) 有界，便于约束隐状态幅度。（在后面的练习中可用 ReLU 替换观察差异。）


# ============================================================================
# 核心公式回顾
# ============================================================================
#   独热编码: 对索引 i → 一个全 0、仅第 i 位为 1 的 N 维向量（N = 词表大小）
#
#   隐层递推（一个时间步）:
#       H_t = tanh( X_t · W_xh + H_{t-1} · W_hh + b_h )
#         W_xh: (N, h)   输入→隐层
#         W_hh: (h, h)   隐层→隐层（自循环，承载"记忆"）
#         b_h:  (h,)     隐层偏置
#
#   输出层:
#       Y_t = H_t · W_hq + b_q
#         W_hq: (h, N)   隐层→输出
#         b_q:  (N,)     输出偏置
#
#   梯度裁剪（θ 为阈值）:
#       norm = sqrt( Σ_i ‖grad_i‖² )
#       g ← min(1, θ / norm) · g
#
#   困惑度:
#       perplexity = exp( 平均交叉熵 ) = exp( Σ 损失 / Σ 词元数 )


# ============================================================================
# 总结
# ============================================================================
# 1. 从零实现一个字符级语言模型的完整流程：
#    独热编码 → 初始化参数 → 定义隐状态初始化与按时间步计算 → 包装成类 → 预测 → 训练。
# 2. RNN 的核心：隐状态 H_t 既依赖当前输入 X_t，也依赖上一时刻 H_{t-1}，构成时间上的"循环"。
# 3. 输入序列编码成 (时间步, 批量, 词表) 三维张量，最外层是时间，方便逐时间步更新。
# 4. 训练时先初始化隐状态；顺序分区要 detach 隐状态以省算力，随机采样则每周期重置。
# 5. 参数更新前做梯度裁剪（θ=1），防止梯度爆炸导致发散；用困惑度评估生成质量。
# 6. 训练前先"预热"，让模型靠前缀上下文更新隐状态，再逐字符自回归生成。
