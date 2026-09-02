"""
=============================================================================
循环神经网络的简洁实现 —— 完整带注释版
=============================================================================
上一节（sec_rnn_scratch）我们从零手写了一个 RNN：手动初始化 W_xh、W_hh、b_h、
自己写 tanh 隐状态递推公式、用 nn.Parameter 管理所有参数。这样虽然能看清内部机制，
但对日常使用太繁琐。

本节改用 PyTorch 高层 API：用 `nn.RNN` 一个层就完成"隐状态递推"（隐藏层），
再额外接一个 `nn.Linear` 做输出层，把每个时间步的隐状态映射到词表预测。
对比之下：
  - 从零实现：自己写 `H = tanh(X@W_xh + H@W_hh + b_h)`
  - 简洁实现：`rnn_layer = nn.RNN(vocab_size, num_hiddens)`，PyTorch 内部自动完成上式，
              权重维度、初始化、反向传播都已封装好。

任务：在"时光机器"数据集上训练一个字符级语言模型，输入前若干字符预测下一个字符。
输入:  批次中每个序列为 `num_steps` 个字符，每个字符是词表 `vocab` 中的一个索引。
输出:  每个时间步预测下一个字符（在词表上做 softmax 分类）。

本节要点（务必吃透）：
  1. nn.RNN 的构造参数与内部权重维度
  2. RNN 期待的三维输入形状 (num_steps, batch_size, vocab)
  3. 隐状态 state 的形状 (num_layers, batch_size, num_hiddens)
  4. `outputs, state = rnn(X, state)` 返回的两个东西各是什么
  5. 为什么隐层输出后还要接一个 Linear 输出层
  6. 与上一节 RNNModelScratch 的参数对应关系
=============================================================================
"""

# ============================================================================
# 0. 依赖导入与数据读取
# ============================================================================
import torch                        # 张量 / 自动微分 / one_hot 等底层工具
from torch import nn                # 深度学习层容器（nn.RNN / nn.Linear / nn.Module）
from torch.nn import functional as F  # 函数式接口：F.one_hot 等不建状态、直接调用的算子
from d2l import torch as d2l        # D2L 配套工具包（数据加载 / 画图 / 训练循环）

# ----------------------------------------------------------------------------
# 超参数
# ----------------------------------------------------------------------------
batch_size, num_steps = 32, 35      # 每个批次 32 条序列；每条序列长度 35（即截断的 35 个字符）

# 从"时光机器"数据集加载语言模型所需的训练迭代器和词表
#   - train_iter:  迭代器，每次产出一个批次，张量形状为 (batch_size, num_steps)，
#                  即 32 行 × 35 列，每个元素是字符在词表中的整数索引
#   - vocab:       词表对象，len(vocab) = 词表大小（去重后的字符数），
#                  vocab 还负责"索引 <-> 字符"的互查
# 注意：load_data_time_machine 是 d2l 包提供的函数（内部自动下载/读取时光机器文本、
#       做字符切分、随机采样成小批量），这里保持调用并注明来自 d2l。
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)


# ============================================================================
# [**定义模型**]
# ============================================================================
# 高级API提供了循环神经网络的实现。
# 我们构造一个具有256个隐藏单元的单隐藏层的循环神经网络层`rnn_layer`。
# 事实上，我们还没有讨论多层循环神经网络的意义（这将在 sec_deep_rnn 中介绍）。
# 现在仅需要将多层理解为一层循环神经网络的输出被用作下一层循环神经网络的输入就足够了。

# ----------------------------------------------------------------------------
# 构造一个单隐藏层 RNN 层
# ----------------------------------------------------------------------------
num_hiddens = 256                         # 隐藏单元数（隐向量维度），类比 MLP 里隐层神经元个数
rnn_layer = nn.RNN(len(vocab), num_hiddens)  # 输入特征维 = 词表大小，输出隐状态维 = num_hiddens

# --- 关键：nn.RNN(input_size, hidden_size) 到底建了什么？权重长什么样？ ---
# 签名：nn.RNN(input_size, hidden_size, num_layers=1, ...)
#   input_size  = len(vocab)：每个时间步送入的"输入"是一个词表大小的向量（字符的 one-hot）
#   hidden_size = 256        ：每个时间步输出的隐状态是 256 维向量
# 它在内部创建了 4 个可学习参数（默认随机初始化，_l0 表示第 0 层）：
#   weight_ih_l0: 形状 (hidden_size, input_size)  = (256, 词表大小)  ← 对应 W_xh
#   weight_hh_l0: 形状 (hidden_size, hidden_size) = (256, 256)       ← 对应 W_hh
#   bias_ih_l0:  形状 (hidden_size,)             = (256,)           ← 输入偏置
#   bias_hh_l0:  形状 (hidden_size,)             = (256,)           ← 隐状态偏置
# 前向递推公式（PyTorch 内部帮你算）：
#   H_t = tanh(X_t @ W_ih^T + b_ih + H_{t-1} @ W_hh^T + b_hh)
# 也就是说，nn.RNN 一层就把上一节手动写的"隐状态递推"整段封装起来了。
# 这些参数可通过 rnn_layer.weight_ih_l0 / rnn_layer.weight_hh_l0 等访问查看到维度。

# --- 输入形状到底怎么给？---（下一段代码用真实张量验证）
# nn.RNN 默认 batch_first=False，因此它期待的三维输入顺序是：
#   (seq_len / 时间步数num_steps, batch_size, input_size / 词表大小)
# 为什么时间步在最前面？PyTorch 默认以为你会沿时间轴逐帧地遍历输入，
# 这样内部循环按时间步 i 依次取 X[i] 即可，不需要转置。
# 注意：D2L 的 train_iter 给的批次是 (batch_size, num_steps)，需要在 forward 里 .T 转置。

# ----------------------------------------------------------------------------
# 使用张量来初始化隐状态，它的形状是（隐藏层数，批量大小，隐藏单元数）
# ----------------------------------------------------------------------------
# RNN 是"有记忆"的，每一时间步都要用到上一时间步的隐状态。
# 第一个时间步没有"上一时间步"，所以我们需要手动给一个初始隐状态（通常全 0）。
# 形状约定：(num_layers, batch_size, num_hiddens)
#   num_layers  -- 堆叠了多少层 RNN（这里是单层 = 1）
#   batch_size  -- 批次大小
#   num_hiddens -- 隐状态向量维度
# 这里直接用 torch.zeros 造一个全 0 张量即可。
state = torch.zeros((1, batch_size, num_hiddens))
state.shape                        # (1, 32, 256)：1 层，32 个样本，256 维隐状态

# ----------------------------------------------------------------------------
# 通过一个隐状态和一个输入，我们就可以用更新后的隐状态计算输出。
# 需要强调的是，`rnn_layer`的"输出"（`Y`）不涉及输出层的计算：
# 它是指每个时间步的隐状态，这些隐状态可以用作后续输出层的输入。
# ----------------------------------------------------------------------------
# 造一个符合 RNN 输入约定的随机张量：三处维度是 (num_steps, batch_size, vocab)
#   - num_steps    = 35 个时间步
#   - batch_size   = 32 个样本并行
#   - len(vocab)   = 词表大小（每个时间步输入是 one-hot 向量）
# 这就是"喂给 RNN 的 X"的标准形状。真实训练时 X 由 one_hot 编码而来（见 RNNModel.forward）。
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
# 调用 RNN 层：输入 X + 初始隐状态 state，返回 (Y, state_new)
Y, state_new = rnn_layer(X, state)

# --- 返回的两个值分别是什么？---
#   Y:        形状 (num_steps, batch_size, num_hiddens) = (35, 32, 256)
#             它是【每个时间步的隐状态】，即把 X 逐时间步喂进去后，
#             第 t 步算出的隐状态 H_t 被记在 Y[t] 上。这就是上一节里
#             RNNModelScratch 的 `rnn(X, H)` 返回的那个 H。
#   state_new:形状 (num_layers, batch_size, num_hiddens) = (1, 32, 256)
#             它是【最后一个时间步之后】的隐状态，携带整条序列的最终记忆，
#             一般作为下一个序列批次的初始隐状态（跨批次传递记忆）。
# 注意：Y 不是对下一个字符的预测！它只是隐状态，要得到"预测词表概率"还需再接输出层。
Y.shape, state_new.shape

# ----------------------------------------------------------------------------
# 我们为一个完整的循环神经网络模型定义了一个`RNNModel`类。
# 注意，`rnn_layer`只包含隐藏的循环层，我们还需要创建一个单独的输出层。
# ----------------------------------------------------------------------------
# 为什么隐层后面必须接一个 Linear 输出层？
#   nn.RNN 每个时间步只输出 256 维隐状态 H_t，代表"对文本的中间理解"，
#   但我们要的是"下一个字符是谁"——一个 len(vocab) 维的概率分布。
#   因此要用 nn.Linear(num_hiddens, vocab_size) 把 (…,256) 线性变换成 (…,词表大小)。
#   这等价于上一节的 W_hq + b_q。于是整体结构就是：
#     nn.RNN(...) --(每个时间步隐状态)--> nn.Linear --(词表 logits)--> softmax 预测
#   如果用 nn.Sequential 来表达概念，会是：
#     nn.Sequential(nn.RNN(len(vocab), num_hiddens), nn.Linear(num_hiddens, len(vocab)))
#   但 RNN 输出是 3D 的 (时间步,批量,隐维)，而 Linear 要 2D，需要 reshape，因此用自定义类更灵活。
# @save 标记已去除——该定义是自包含的，d2l 包并未内置此模型类。
class RNNModel(nn.Module):
    """循环神经网络模型"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        # nn.Module 的初始化；**kwargs 把多余关键字参数（如 device）透传给父类
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer                  # 保存传入的 nn.RNN 层（只有隐藏循环层）
        self.vocab_size = vocab_size          # 记录词表大小（输出维度）
        # rnn_layer 对象上带有人口属性：hidden_size 就是隐藏单元数
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        # 普通的 nn.RNN 默认单向，num_directions = 1；
        # 双向 RNN 会把正向+反向两个隐状态拼起来，所以输出维度要 ×2。
        if not self.rnn.bidirectional:
            self.num_directions = 1
            # 输出层：把 256 维隐状态映射到词表大小（对应上一节 W_hq,b_q）
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            # 双向时输入给线性层的维度是 2×num_hiddens
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        # inputs 是 d2l 批次数据：形状 (batch_size, num_steps)，元素是字符的整数索引
        # inputs.T -> (num_steps, batch_size)：转置成 RNN 期望的"时间步在前"的顺序
        #   .long() 把索引转成 int64（one_hot 需要整型索引）
        #   F.one_hot(..., self.vocab_size) -> (num_steps, batch_size, vocab_size)
        #   即把每个"整数索引"展开成一个词表大小的独热向量
        #   解释：one-hot 把离散字符变成可参与矩阵乘法的向量，是语言模型把"离散符号
        #         -> 连续表示"的标准手法（另一种更常用的是 nn.Embedding 可学习词表）。
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        # one_hot 返回的是长整型 0/1 张量；矩阵乘需要浮点，所以转成 float32
        X = X.to(torch.float32)
        # 真正跑 RNN：X (时间步,批量,词表) -> Y (时间步,批量,隐维)，同时更新隐状态 state
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        #   Y 是 (num_steps, batch, num_hiddens)，而 nn.Linear 只对最后一维做矩阵乘，
        #   所以用 reshape((-1, Y.shape[-1])) 把 (时间步×批量) 合并到一维：
        #     reshape((num_steps*batch, 256)) -> Linear -> (num_steps*batch, vocab_size)
        #   之后再经 log_softmax / 交叉熵即可得到逐时间步对下一个字符的预测。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        # 生成初始隐状态（供预测/每个序列批次开头使用）
        if not isinstance(self.rnn, nn.LSTM):
            # nn.RNN / nn.GRU 以单个张量作为隐状态
            # 形状：(num_directions × num_layers, batch_size, num_hiddens)
            #   nn.RNN 单层单向时即 (1, batch_size, 256)，全 0 初始化
            return  torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device)
        else:
            # nn.LSTM 以"元组 (隐状态, 细胞状态)"作为记忆，需要返回两个全 0 张量
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))


# ============================================================================
# 训练与预测
# ============================================================================
# 在训练模型之前，让我们基于一个具有随机权重的模型进行预测。

# ----------------------------------------------------------------------------
# 构造模型，放到可用的设备上，然后用随机权重先试预测一段文本
# ----------------------------------------------------------------------------
device = d2l.try_gpu()                        # 查看 GPU 是否可用，返回 'cuda'/'cpu'
net = RNNModel(rnn_layer, vocab_size=len(vocab))  # 把刚才的双层封装成完整模型
net = net.to(device)                           # 把所有参数搬到 device（GPU/CPU）
# 用随机初始化的模型预测：以"time traveller"开头的 10 个字符。
# 因为权重是随机的，输出肯定是胡言乱语——这正说明"性能来自训练，而非架构本身"。
# predict_ch8 是 d2l 包提供的函数（完成预测、采样下一字符、逐步拼接），保持调用并注明来自 d2l。
d2l.predict_ch8('time traveller', 10, net, vocab, device)

# ----------------------------------------------------------------------------
# 很显然，这种模型根本不能输出好的结果。
# 接下来，我们使用上一节 sec_rnn_scratch 中定义的超参数调用`train_ch8`，并且使用高级API训练模型。
# ----------------------------------------------------------------------------
num_epochs, lr = 500, 1                        # 训练 500 轮，学习率 1（字符级任务常用较大 lr）
# 训练循环。train_ch8 是 d2l 包提供的函数：
#   内部完成 train_iter 迭代 -> 前向传播 RNNModel -> 交叉熵回传 -> 打印困惑度,
#   并在最后一轮调 predict_ch8 展示生成效果。
d2l.train_ch8(net, train_iter, vocab, lr, num_epochs, device)

# ----------------------------------------------------------------------------
# 与上一节相比，由于深度学习框架的高级API对代码进行了更多的优化，
# 该模型在较短的时间内达到了较低的困惑度。
# ----------------------------------------------------------------------------
# （以下是原 notebook 的"小结"与"练习"，其中不含代码，此处按规范略去正文。）
# 小结要点：
#   * 高级API提供了循环神经网络层的实现。
#   * 高级API的循环神经网络层返回一个输出和一个更新后的隐状态，我们还需要计算整个模型的输出层。
#   * 相比从零开始实现的循环神经网络，使用高级API实现可以加速训练。


# ============================================================================
# 补充知识点
# ============================================================================

# 1. RNN 输入/输出的四种常见形状（务必分清，极易搞混）
#    - 输入 X：(num_steps, batch_size, vocab)  时间步在前（batch_first=False）
#    - 输出 Y：(num_steps, batch_size, hidden) 每个时间步的隐状态
#    - 初始/最终隐状态：(num_layers, batch_size, hidden)
#    为什么 Y 和 state 形状不同？
#      Y        = 沿时间轴收集"每个时间步"的隐状态 → 时间步在最前
#      state    = 只留"最后一步"的隐状态，并按层组织 → 层数在最前
#   若构造 nn.RNN(..., batch_first=True)，则 X/Y 都变成 (batch, num_steps, ...)，
#   此时 d2l 数据的 .T 转置就不需要了——但 D2L 默认用 batch_first=False。

# 2. nn.RNN 内部权重在哪看？（对 C/C++ 直觉：就是结构体里的成员）
#    rnn_layer.weight_ih_l0.shape -> (hidden, input)   对应 W_xh
#    rnn_layer.weight_hh_l0.shape -> (hidden, hidden)  对应 W_hh
#    rnn_layer.bias_ih_l0.shape   -> (hidden,)        对应 b_ih
#    rnn_layer.bias_hh_l0.base    -> (hidden,)        对应 b_hh
#    与上一节 RNNModelScratch 的参数一一对应（注意转置关系）：
#      W_xh  -> weight_ih_l0.T   (上一节 W_xh 是 (input,hidden)，这里存成 (hidden,input))
#      W_hh  -> weight_hh_l0.T
#      b_h   -> bias_ih_l0 + bias_hh_l0
#      W_hq  -> linear.weight      (形状 (vocab, hidden))
#      b_q   -> linear.bias        (形状 (vocab,))

# 3. 为什么 nn.RNN 不直接输出"预测"而只给隐状态？
#    循环层管"理解序列、记住历史"，输出层管"做出预测"。两者职责不同、且要分别训练。
#    RNN 是通用组件，输出维度 = 隐藏单元数，与具体任务无关；
#    词表预测是"分类问题"，需要把隐状态映射到"词表大小"个类别的 logits。
#    拆分设计让 RNN 可复用于翻译、情感分析等多种任务。

# 4. one_hot 与 Embedding 的取舍
#    one_hot: 每个字符 -> 一个 vocab_size 长的 0/1 向量，简单但稀疏、维度大、无语义。
#    Embedding: 每个字符 -> 一个可学习的低维 dense 向量，训练到语义相似的字靠近。
#    本节为看清结构用 one_hot；实际工程几乎都用 nn.Embedding（下一章现代RNN会讲）。

# 5. 与从零实现 RNNModelScratch 的对照
#    RNNModelScratch             | RNNModel（本节）
#    rnn(x, H) 手动算 H=tanh(...) | nn.RNN 封装该递推
#    get_params 造 W_xh/W_hh/b_h | nn.RNN 内部自动建 weight_ih/weight_hh/bias
#    输出层  Y=H@W_hq+b_q        | self.linear = nn.Linear(hidden, vocab)
#    init_state 造全0隐状态      | begin_state 造全0隐状态
#    区别：一个手写一个封装；本节参数初始化、反向传播全部交给框架，训练更快更稳。

# 6. output 返回形状为何被拍平成 2 维？
#    nn.Linear 对输入最后一维做矩阵乘，(…,in) -> (…,out)。
#    把 (num_steps, batch, hidden) 合并前两维成 (num_steps×batch, hidden) 就能整体一次算完，
#    等价于对每个时间步单独做 linear。d2l 的 train_ch8/predict_ch8 会按需 reshape 回去计算 loss。


# ============================================================================
# 核心公式回顾
# ============================================================================
#   1. 隐状态递推（nn.RNN 内部完成）:
#        H_t = tanh( X_t W_ih^T + b_ih + H_{t-1} W_hh^T + b_hh )
#      对应上一节: H_t = tanh( X_t W_xh + H_{t-1} W_hh + b_h )
#   2. 输入张量形状:
#        X: (num_steps, batch_size, vocab_size)
#        Y: (num_steps, batch_size, num_hiddens)  每个时间步的隐状态
#   3. 输出层（把隐状态映射到词表）:
#        logits = Y @ W_hq^T + b_q ,  W_hq:(vocab, hidden), b_q:(vocab,)
#   4. 每个时间步的预测概率:
#        P(next_char) = softmax( logits )   再逐时间步用交叉熵求损失
#   5. state 形状:
#        (num_layers, batch_size, num_hiddens) ，单层单向即 (1, batch_size, 256)


# ============================================================================
# 总结
# ============================================================================
# 1. nn.RNN(input_size, hidden_size) 一行建出"隐状态递推层"，内部封装权重与反向传播，
#    input_size = 词表大小，hidden_size = 隐向量维度（默认 batch_first=False）。
# 2. RNN 期待输入 (num_steps, batch, vocab)，输出 Y (num_steps, batch, hidden)
#    与 state (num_layers, batch, hidden)；Y 是逐时间步隐状态，state 是最终记忆。
# 3. RNN 层只输出隐状态，不输出预测；必须再接 nn.Linear(hidden, vocab) 输出层把它
#    映射到词表，二者一起才构成完整语言模型。
# 4. RNNModel 把 "nn.RNN 层 + nn.Linear 输出层 + one_hot 编码 + 初始隐状态" 装配成模块，
#    对应上一节 RNNModelScratch，但参数/初始化/训练全部交给框架，效率更高。
# 5. 数据管线：load_data_time_machine 给出 (batch, num_steps) 索引，
#    forward 中转置 + one_hot 成 (num_steps, batch, vocab) 喂给 RNN。
# 6. 本节模型基于随机初始化权重预测时是胡言乱语，训练后困惑度明显下降——
#    印证了"高级API+充足训练"能高效学到字符级语言模型。
