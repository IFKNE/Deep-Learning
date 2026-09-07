"""
=============================================================================
门控循环单元（Gated Recurrent Unit，GRU）—— 完整带注释版
=============================================================================
任务：解决经典 RNN 无法捕捉"长距离依赖"、容易梯度消失/爆炸的问题。
做法：给隐状态加上两扇"门"——重置门（reset gate）与更新门（update gate），
     让网络自己决定"要记住多少旧的 / 要接受多少新的"。两扇门都是可学习的，
     由 sigmoid 输出 (0,1) 之间的值来衡量"保留比例"。

依赖：d2l 包（本书自带的工具库）。本文件可独立运行，需已安装 d2l。
     该环境为 PyTorch 版本；运行会自动使用 CUDA（GPU）。
=============================================================================
"""

# ============================================================================
# 第1步：导入依赖、加载"时光机器"数据集
# ============================================================================
import torch
from torch import nn
from d2l import torch as d2l

# 超参数设置
#   batch_size = 32 : 每个小批量含 32 个样本（每条样本是一个序列窗口）
#   num_steps = 35  : 每个序列窗口的长度（时间步数 / 词元数）
batch_size, num_steps = 32, 35

# d2l.load_data_time_machine 是本书封装好的函数，返回可迭代训练集与词表 vocab
#   它读取《时间机器》小说，做字符级词元化，并把长文本切成 (batch, num_steps)
#   的随机窗口。该函数由 D2L 以 @save 注册进 d2l 包，可直接调用。
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)


# ============================================================================
# 第2步：初始化模型参数
# ============================================================================
def get_params(vocab_size, num_hiddens, device):
    """
    生成 GRU 的全部可学习参数。

    参数:
        vocab_size     : 词表大小（= 输入维数 num_inputs = 输出维数 num_outputs）
        num_hiddens    : 隐藏单元数 h，所有门与候选隐状态的输出维数
        device         : 运行设备（如 'cuda:0' 或 'cpu'）

    返回:
        params         : 11 个参数的列表，每个都已开启 requires_grad=True
    """
    # GRU 中输入、输出都是词表索引，所以输入维数 = 输出维数 = 词表大小
    num_inputs = num_outputs = vocab_size

    # 标准正态采样 × 0.01：让权重的初始标准差为 0.01（很小的初值，避免梯度爆炸）
    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # three() 一次生成一套"输入权 W_x、隐状态权 W_h、偏置 b"（三角套件）。
    #   因为三个门/候选都共享相同的权重形状：
    #   W_x  形状 (num_inputs, num_hiddens)  ：当前输入 X_t 的映射
    #   W_h  形状 (num_hiddens, num_hiddens) ：上一隐状态 H_{t-1} 的映射
    #   b    形状 (num_hiddens,)             ：偏置项
    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    # 更新门参数（Z_t）：决定"用多少旧状态替换新候选"
    W_xz, W_hz, b_z = three()
    # 重置门参数（R_t）：决定"忘掉多少以前的隐状态"
    W_xr, W_hr, b_r = three()
    # 候选隐状态参数（H̃_t）：权重用于计算"可能的下一状态"
    W_xh, W_hh, b_h = three()
    # 输出层参数：把隐状态映射回词表大小，用于预测下一个词元
    W_hq = normal((num_hiddens, num_outputs))     # 形状 (num_hiddens, vocab_size)
    b_q = torch.zeros(num_outputs, device=device)  # 形状 (vocab_size,)

    # 收集所有需要求梯度（训练更新）的参数
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    # 逐个开启梯度追踪：PyTorch 约定，默认新张量 requires_grad=False，
    #   训练时我们要对每个参数做 `.backward()`，所以必须显式打开。
    for param in params:
        param.requires_grad_(True)
    return params


# ============================================================================
# 第3步：定义隐状态初始化函数 init_gru_state
# ============================================================================
def init_gru_state(batch_size, num_hiddens, device):
    """
    初始化 GRU 的隐状态：全零张量，形状 (batch_size, num_hiddens)。

    GRU 的"状态"只有一个 H（不像 LSTM 有 H 和 C 两个）。每个时间步开始，
    隐状态默认为 0。
    """
    # 返回一个长度为 1 的元组 (H,)，与后续 gru() 里的 `H, = state` 解包对应
    return (torch.zeros((batch_size, num_hiddens), device=device),)


# ============================================================================
# 第4步：定义 GRU 前向计算 gru()
# ============================================================================
def gru(inputs, state, params):
    """
    GRU 的单层前向计算：按时间步逐词元地更新隐状态并输出预测。

    参数:
        inputs : 输入序列，形状 (num_steps, batch_size, vocab_size)
                 （这里用 one-hot 编码逐时间步返回，每个 X 是 (batch, vocab)）
        state  : 上一隐状态，元组 (H,)，H 形状 (batch_size, num_hiddens)
        params : get_params 返回的 11 个参数

    返回:
        torch.cat(outputs, dim=0) : 所有时间步的输出拼接，
                                    形状 (num_steps * batch_size, vocab_size)
        (H,)                      : 最后一步的隐状态（供下一个序列窗口继续用）
    """
    # 把 11 个参数解包成 11 个变量，方便下面的公式书写
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state  # 从元组 state 中取出唯一一个隐状态 H（元组解包）
    outputs = []
    # 遍历每一个时间步。inputs 的首维是时间步。每个 X 是一个小批量：
    #   X 的形状 (batch_size, vocab_size)，即一批"词元索引的 one-hot 编码"
    for X in inputs:
        # ---------- ① 更新门 Z_t = σ(X_t W_xz + H_{t-1} W_hz + b_z) ----------
        # sigmoid 把线性结果压到 (0,1)，表示"本轮要保留多少旧状态 H_{t-1}"
        #   (X @ W_xz)  : (batch, vocab) × (vocab, h)      → (batch, h)
        #   (H @ W_hz)  : (batch, h)    × (h, h)           → (batch, h)
        #   @ 是矩阵乘法，等价于 torch.matmul；结果与偏置 b_z 相加（广播到每行）
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)

        # ---------- ② 重置门 R_t = σ(X_t W_xr + H_{t-1} W_hr + b_r) ----------
        # R_t 接近 0 表示"把以前的状态都忘掉"，接近 1 表示"保留旧状态"
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)

        # ---------- ③ 候选隐状态 H̃_t = tanh(X_t W_xh + (R_t ⊙ H_{t-1}) W_hh + b_h) ----------
        #   (R * H)  : 按元素相乘（Hadamard 积，* 在张量间是逐元素相乘），
        #              用重置门 R 去"缩放"旧隐状态 H；R 接近 0 就把旧信息清零
        #   @ W_hh   : (batch, h) × (h, h) → (batch, h)
        #   tanh     : 把结果压到 (-1, 1)，得到"在重置后的新想法"
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)

        # ---------- ④ 最终隐状态 H_t = Z_t ⊙ H_{t-1} + (1 - Z_t) ⊙ H̃_t ----------
        #   这是"凸组合"：Z_t 衡量"保留旧状态"的比例，(1 - Z_t) 衡量"接受新候选"的比例
        #   二者按元素相乘后相加。Z_t 接近 1 → 几乎只保留旧状态（跳过本时间步）；
        #   Z_t 接近 0 → 几乎完全是新的候选状态。
        H = Z * H + (1 - Z) * H_tilda

        # ---------- ⑤ 输出层：把隐状态映射成"下一个词元"的 logits ----------
        # Y 形状 (batch_size, vocab_size)，尚未过 softmax；训练时与交叉熵配合使用
        Y = H @ W_hq + b_q
        outputs.append(Y)

    # 把所有时间步的输出按 dim=0 拼接：
    #   原 outputs 是 list of (batch, vocab)，cat 后 → (num_steps*batch, vocab)
    #   返回最后一步隐状态 H，供下一个 batch 继续（自回归预测时循环使用）
    return torch.cat(outputs, dim=0), (H,)


# ============================================================================
# 第5步：训练与预测
# ============================================================================
# 词表大小、隐藏单元数、运行设备
vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
# num_epochs = 500 : 训练 500 轮；lr = 1 : 学习率（d2l.train_ch8 内部策略下可较大）
num_epochs, lr = 500, 1

# d2l.RNNModelScratch：本书封装的手写模型类，接收以下自定义组件：
#   (1) 词表大小    (2) 隐藏单元数   (3) 设备
#   (4) get_params   —— 参数初始化函数
#   (5) init_gru_state —— 隐状态初始化函数
#   (6) gru          —— 前向计算函数（与 RNNModelScratch 的 forward 约定匹配）
model = d2l.RNNModelScratch(len(vocab), num_hiddens, device, get_params,
                            init_gru_state, gru)

# d2l.train_ch8：本书封装的训练循环，自动做反向传播、优化、困惑度计算与可视化。
#   该函数由 D2L 以 @save 注册进 d2l 包，可直接调用。
#   训练后打印困惑度（perplexity）与速度，并生成一段"time traveller"前缀文本供观察。
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
# 结果示例（分数越低越好，1 为完美）：
#   perplexity 1.1, 19911.5 tokens/sec on cuda:0
#   time traveller firenis i heidfile sook at i jomer and sugard are ...


# ============================================================================
# 第6步：简洁实现（使用高层 API nn.GRU）
# ============================================================================
# 说明：高层 API 把所有"门控"细节封装进编译好的底层算子，训练速度远快于手写版。
# 输入维数 = 词表大小（one-hot 编码时每个词元占一维）
num_inputs = vocab_size
# nn.GRU(input_size, hidden_size)：PyTorch 内置门控循环单元层
#   内部自动维护 更新门/重置门/候选隐状态/输出 的所有权重，无需自己初始化
gru_layer = nn.GRU(num_inputs, num_hiddens)
# d2l.RNNModel：本书提供的"高层模型"封装，自动接入由 nn.GRU 构造的层
#   第一个参数是层本身（nn.GRU），第二个参数是词表大小（决定输出维度）
model = d2l.RNNModel(gru_layer, len(vocab))
# 把模型搬到 GPU（若可用）
model = model.to(device)
# 复用同一个 d2l.train_ch8 完成训练
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)
# 结果示例：nn.GRU 用编译算子，速度显著提升：
#   perplexity 1.0, 109423.8 tokens/sec on cuda:0
#   time travelleryou can show black is white by argument said filby ...


# ============================================================================
# 补充知识点
# ============================================================================
# 1. 【为什么 GRU 能缓解梯度消失？】
#    经典 RNN 的隐状态更新是"乘法叠加"，反向传播时会出现连乘项 Π，因子<1 就消失。
#    GRU 用 H_t = Z_t⊙H_{t-1} + (1-Z_t)⊙H̃_t —— 这是一个"加法"混合，更新门 Z_t
#    接近 1 时，旧状态几乎原样传递下去，梯度的"加法通路"避免被连续乘没。
#    这就是它能把"第一个词元"一路带到序列末尾的原因。

# 2. 【两个门的直观分工】
#   - 重置门 R_t 管"短期依赖"：R 接近 0 → 忘掉旧历史，模型只看当前输入就能
#     快速响应新内容（捕捉短距离变化）。
#   - 更新门 Z_t 管"长期依赖"：Z 接近 1 → 保留长线记忆，把很久以前的信息
#     一路带下来（捕捉长距离关系）。

# 3. 【边界情况】
#   - 当 R_t 全为 1 且 Z_t = 0 时，公式退化成 tanh 版的经典 RNN。
#   - 当 Z_t 全为 1 时，H_t = H_{t-1}，完全"跳过"当前时间步 → 可以用它处理
#     与预测无关的中间词元（如 HTML 标签）。

# 4. 【超参数观察】
#   - 减少 num_epochs 会让文本未收敛；调大 num_hiddens 增加表达力但更慢。
#   - 对比 01（手写）与 06（nn.GRU）的速度与困惑度，理解"编译算子 vs Python 循环"
#     带来的数量级差异。


# ============================================================================
# 核心公式回顾
# ============================================================================
#   输入 X_t ∈ R^{n×d}，上一隐状态 H_{t-1} ∈ R^{n×h}。
#
#   ① 重置门：  R_t = σ( X_t W_xr + H_{t-1} W_hr + b_r )
#   ② 更新门：  Z_t = σ( X_t W_xz + H_{t-1} W_hz + b_z )
#   ③ 候选状态：H̃_t = tanh( X_t W_xh + (R_t ⊙ H_{t-1}) W_hh + b_h )
#   ④ 隐状态：  H_t = Z_t ⊙ H_{t-1} + (1 - Z_t) ⊙ H̃_t
#   ⑤ 输出：    Y_t = H_t W_hq + b_q
#
#   其中 ⊙ 是 Hadamard 积（逐元素相乘），σ 是 sigmoid。


# ============================================================================
# 练习与思考
# ============================================================================
# 1) 如果只想用时间步 t' 的输入预测 t>t' 的输出，每个时间步的重置门/更新门
#    应取何值？（提示：让后续时间步既保留目标信息又忽略无关信息）
# 2) 调整 num_hiddens / num_epochs / lr，观察困惑度、运行时间与输出字符串变化。
# 3) 对比 nn.RNN 与 nn.GRU 对运行时间、困惑度、生成文本的影响。
# 4) 如果只有重置门或只有更新门（去掉另一个），模型会在哪些地方失效？


# ============================================================================
# 总结
# ============================================================================
#  - GRU 通过"重置门 + 更新门"两个可学习的门，灵活控制隐状态"记/忘"。
#  - 重置门捕捉短期依赖，更新门捕捉长期依赖。
#  - 从零实现（get_params / init_gru_state / gru）与高层 API（nn.GRU）结果一致，
#    但后者用编译算子因此快得多。
#  - GRU 是 LSTM 的简化版，二者是本章解决"长距离依赖 + 梯度问题"的一对主力。
