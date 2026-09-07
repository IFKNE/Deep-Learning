"""
=============================================================================
长短期记忆网络（Long Short-Term Memory，LSTM）—— 完整带注释版
=============================================================================
任务：同样是解决"长期信息保存 + 短期输入缺失"的问题，但比 GRU 更早诞生近 20 年，
     也更复杂。灵感来自计算机的逻辑门：LSTM 把隐状态细分为"隐状态 H"和
     "记忆元 C（cell）"两个，并用三扇门（输入门 I、遗忘门 F、输出门 O）控制
     信息的写入、遗忘与读出。

依赖：d2l 包。本文件可独立运行。
=============================================================================
"""

# ============================================================================
# 第1步：导入依赖、加载数据集
# ============================================================================
import torch
from torch import nn
from d2l import torch as d2l

# 超参数：批量大小 32，每个序列窗口 35 个词元
batch_size, num_steps = 32, 35
# 加载《时间机器》字符级数据集（D2L 内置函数）
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)


# ============================================================================
# 第2步：初始化模型参数 get_lstm_params
# ============================================================================
def get_lstm_params(vocab_size, num_hiddens, device):
    """
    生成 LSTM 的全部可学习参数。

    参数:
        vocab_size   : 词表大小（输入/输出维数）
        num_hiddens  : 隐藏单元数 h
        device       : 运行设备

    返回:
        params       : 14 个参数的列表，均已开启梯度
    """
    # 输入、输出维数都等于词表大小
    num_inputs = num_outputs = vocab_size

    # 用标准差 0.01 的高斯分布初始化权重，偏置置 0
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # three() 生成"输入权 W_x、隐状态权 W_h、偏置 b"一组
    #   W_x 形状 (num_inputs, num_hiddens)
    #   W_h 形状 (num_hiddens, num_hiddens)
    #   b   形状 (num_hiddens,)
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    # 输入门参数（I_t）：控制"写入记忆元多少新内容"
    W_xi, W_hi, b_i = three()
    # 遗忘门参数（F_t）：控制"保留多少旧记忆元"
    W_xf, W_hf, b_f = three()
    # 输出门参数（O_t）：控制"从记忆元读出多少给隐状态"
    W_xo, W_ho, b_o = three()
    # 候选记忆元参数（C̃_t）：用于生成"有可能被写入的新记忆"
    W_xc, W_hc, b_c = three()
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))     # (num_hiddens, vocab_size)
    b_q = torch.zeros(num_outputs, device=device)  # (vocab_size,)

    # 共 14 个参数（比 GRU 多一组，因为多了一个"记忆元"通道）
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc,
              b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


# ============================================================================
# 第3步：初始化隐状态 init_lstm_state（注意：有两个状态！）
# ============================================================================
def init_lstm_state(batch_size, num_hiddens, device):
    """
    初始化 LSTM 的"两个隐状态"：H（传给输出层的隐状态）与 C（内部记忆元）。

    与 GRU 只有一个状态 H 不同，LSTM 需要额外维护记忆元 C，
    二者形状都是 (batch_size, num_hiddens)，初始都为 0。
    """
    return (torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))


# ============================================================================
# 第4步：定义 LSTM 前向计算 lstm()
# ============================================================================
def lstm(inputs, state, params):
    """
    LSTM 单层前向计算：按时间步更新 (H, C) 并输出预测。

    参数:
        inputs : 输入序列，每个 X 形状 (batch_size, vocab_size)（one-hot 编码）
        state  : (H, C) 元组，二者形状均 (batch_size, num_hiddens)
        params : 14 个参数

    返回:
        torch.cat(outputs, dim=0) : 所有时间步输出，(num_steps*batch, vocab)
        (H, C)                    : 最后一步的两个状态
    """
    [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c,
     W_hq, b_q] = params
    (H, C) = state  # 同时拿到隐状态 H 和记忆元 C
    outputs = []
    for X in inputs:
        # ---------- ① 输入门 I_t = σ(X_t W_xi + H_{t-1} W_hi + b_i) ----------
        #  决定"把多少新信息写进记忆元"，值在 (0,1)
        I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i)

        # ---------- ② 遗忘门 F_t = σ(X_t W_xf + H_{t-1} W_hf + b_f) ----------
        #  决定"保留多少旧的记忆元内容"。F 接近 1 表示全保留。
        F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f)

        # ---------- ③ 输出门 O_t = σ(X_t W_xo + H_{t-1} W_ho + b_o) ----------
        #  决定"从记忆元读出多少给隐状态"。O 接近 0 时隐状态接近 0。
        O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_o)

        # ---------- ④ 候选记忆元 C̃_t = tanh(X_t W_xc + H_{t-1} W_hc + b_c) ----------
        #  "可能被写入的新记忆"，tanh 把值压到 (-1,1)
        C_tilda = torch.tanh((X @ W_xc) + (H @ W_hc) + b_c)

        # ---------- ⑤ 更新记忆元 C_t = F_t ⊙ C_{t-1} + I_t ⊙ C̃_t ----------
        #  遗忘门 F 决定丢弃多少旧记忆；输入门 I 决定写入多少新候选记忆。
        #  若 F 恒为 1、I 恒为 0，则 C 恒等于 C_{t-1}——旧记忆被永久保存，
        #  这正是缓解梯度消失的关键（梯度有"加法通路"可沿 C 直传）。
        C = F * C + I * C_tilda

        # ---------- ⑥ 更新隐状态 H_t = O_t ⊙ tanh(C_t) ----------
        #  记忆元 C 再经过 tanh 压到 (-1,1)，然后被输出门 O 缩放。
        #  注意：只有 H 进入输出层，记忆元 C 完全属于内部状态。
        H = O * torch.tanh(C)

        # ---------- ⑦ 输出层 ----------
        Y = (H @ W_hq) + b_q   # 形状 (batch_size, vocab_size)
        outputs.append(Y)

    return torch.cat(outputs, dim=0), (H, C)


# ============================================================================
# 第5步：训练与预测
# ============================================================================
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
# d2l.RNNModelScratch：与 GRU 同款的手写模型封装，仅替换参数函数与状态函数
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_lstm_params,
                            init_lstm_state, lstm)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
# 结果示例：
#   perplexity 1.3, 17736.0 tokens/sec on cuda:0
#   time traveller for so it will leong go it we melenot ir cove i s ...


# ============================================================================
# 第6步：简洁实现（高层 API nn.LSTM）
# ============================================================================
num_inputs = vocab_size
# nn.LSTM(input_size, hidden_size)：PyTorch 内置 LSTM 层，内部维护三扇门与记忆元
lstm_layer = nn.LSTM(num_inputs, num_hiddens)
# d2l.RNNModel：高层封装，自动适配 nn.LSTM（含 (H,C) 双状态）
model = d2l.RNNModel(lstm_layer, len(vocab))
model = model.to(device)
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
# 结果示例：
#   perplexity 1.1, 234815.0 tokens/sec on cuda:0
#   time traveller for so it will be convenient to speak of himwas e ...


# ============================================================================
# 补充知识点
# ============================================================================
# 1. 【LSTM 与 GRU 的关键区别】
#    同一份记忆分成两态：LSTM 同时维护 H（对外）与 C（对内）；GRU 只有 H。
#    - LSTM 有"输出门 O"来决定 H 的可见度；GRU 没有输出门，H 直接等于混合结果。
#    - LSTM 有三扇门（I/F/O）；GRU 只有两扇门（R/Z）。
#    - 一般而言 LSTM 表达力更强但参数更多、更慢；GRU 更简、更快、效果相仿。

# 2. 【为什么 C 能缓解梯度消失？】
#    记忆元更新 C_t = F_t ⊙ C_{t-1} + I_t ⊙ C̃_t 中，若遗忘门接近 1，则
#    C_t ≈ C_{t-1}。反向传播时梯度可沿这条"恒等直通"路径传回很久以前，
#    不会像经典 RNN 那样被连乘矩阵逐次缩小。

# 3. 【为什么隐状态还要再套一层 tanh？】
#    记忆元 C 的裸值范围本就不限于 (-1,1)，直接给输出层可能数值过大。
#    先过 tanh 把 C 压到 (-1,1)，再由输出门剪裁，可保证 H_t 始终落在 (-1,1)。

# 4. 【高阶变体】
#    实际中常把 LSTM 堆叠成多层、加残差连接、加 dropout 等正则化。但训练成本高，
#    如今许多任务已被更先进的 Transformer 取代（本书后续章节）。


# ============================================================================
# 核心公式回顾
# ============================================================================
#   输入 X_t ∈ R^{n×d}，上一隐状态 H_{t-1} ∈ R^{n×h}，上一记忆元 C_{t-1} ∈ R^{n×h}。
#
#   ① 输入门：   I_t = σ( X_t W_xi + H_{t-1} W_hi + b_i )
#   ② 遗忘门：   F_t = σ( X_t W_xf + H_{t-1} W_hf + b_f )
#   ③ 输出门：   O_t = σ( X_t W_xo + H_{t-1} W_ho + b_o )
#   ④ 候选记忆元：C̃_t = tanh( X_t W_xc + H_{t-1} W_hc + b_c )
#   ⑤ 记忆元：   C_t = F_t ⊙ C_{t-1} + I_t ⊙ C̃_t
#   ⑥ 隐状态：   H_t = O_t ⊙ tanh(C_t)
#   ⑦ 输出：     Y_t = H_t W_hq + b_q
#
#   只有 H_t 进入输出层；C_t 是纯内部状态。


# ============================================================================
# 练习与思考
# ============================================================================
# 1) 调整超参数，观察困惑度、运行时间与输出字符串的变化。
# 2) 如何改成预测"单词"而非"字符"？（提示：换词元粒度与 embedding）
# 3) 给定隐藏维数，比较 GRU / LSTM / 经典 RNN 的计算成本（训练与推理）。
# 4) 既然候选记忆元 C̃_t 已用 tanh 把值限到 (-1,1)，为何隐状态还要再套 tanh？
#    （答案见"补充知识点 3"）
# 5) 实现一个基于"时间序列"（而非字符序列）做预测的 LSTM。


# ============================================================================
# 总结
# ============================================================================
#  - LSTM 用三扇门（输入/遗忘/输出）+ 一个记忆元 C，精确控制"写/忘/读"。
#  - 隐状态有两份：H 对外可见并进输出层，C 只做内部长期记忆。
#  - 记忆元更新是"加法式"，保留一条梯度直通路径，从而缓解梯度消失。
#  - 与 GRU 相比更强、但更复杂、更慢；两者都是现代 RNN 变体的代表。
