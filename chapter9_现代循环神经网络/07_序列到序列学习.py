"""
=============================================================================
序列到序列学习（Sequence-to-Sequence，seq2seq）—— 完整带注释版
=============================================================================
任务：用"两个循环神经网络"实现上一节的编码器-解码器架构，训练一个英→法翻译模型。
     编码器 RNN 把不定长源序列压成末态隐状态 c；解码器 RNN 用 c 初始化隐状态，
     逐个词元生成目标序列。过程中覆盖：嵌入层、教师强制（teacher forcing）、
     带遮蔽的交叉熵损失、Xavier 初始化、梯度裁剪，以及 BLEU 评估。

依赖：d2l 包。本文件可独立运行，运行会自动下载英法数据集并训练。
=============================================================================
"""

# ============================================================================
# 第1步：导入依赖
# ============================================================================
import collections
import math
import torch
from torch import nn
from d2l import torch as d2l


# ============================================================================
# 第2步：编码器 Seq2SeqEncoder
# ============================================================================
class Seq2SeqEncoder(d2l.Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        # ① 嵌入层：把词表索引映射成稠密特征向量。
        #   nn.Embedding(vocab_size, embed_size)：权重矩阵大小 词表数×特征维数。
        #   给定词元索引 i，它就返回权重矩阵的第 i 行（从 0 开始）作为特征向量。
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # ② 循环层：这里用"多层 GRU"作为编码器（也常换成 LSTM）。
        #    输入特征维 = embed_size，输出隐藏维 = num_hiddens。
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        # X 形状：(batch_size, num_steps)，内容是"词表索引"。
        # 嵌入后 X 形状 → (batch_size, num_steps, embed_size)（每个词元变成一个向量）
        X = self.embedding(X)
        # GRU/LSTM 约定：第一个轴是"时间步"。
        # 但 PyTorch 常见输入是 (batch, time, feat)，因此要 permute 成
        #   (num_steps, batch_size, embed_size) 再喂给循环层。
        X = X.permute(1, 0, 2)
        # 若未显式给初始状态，则循环层默认用全 0 状态。
        # 返回：
        #   output 形状 (num_steps, batch_size, num_hiddens) —— 每个时间步输出
        #   state  形状 (num_layers, batch_size, num_hiddens) —— 最后一层的末态
        output, state = self.rnn(X)
        return output, state


# 实例化编码器并测试张量形状：
#   用 10 个词、嵌入维 8、隐藏 16、2 层 GRU。
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
encoder.eval()  # 设为评估模式（不做 dropout 等）
X = torch.zeros((4, 7), dtype=torch.long)  # 批量 4、时间步 7 的输入
output, state = encoder(X)
# output 形状：(7, 4, 16) = (时间步, 批量, 隐藏单元)
output.shape
# state 形状：(2, 4, 16) = (层数, 批量, 隐藏单元)
#   若用 LSTM，state 里还会额外包含一个"记忆元通道"。
state.shape


# ============================================================================
# 第3步：解码器 Seq2SeqDecoder
# ============================================================================
class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 输入维 = embed_size + num_hiddens：
        #   因为每个时间步要"把编码后的上下文 c 拼到解码输入后面"，
        #   所以循环层的输入维度多出一个隐藏维。
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        # 输出层：把隐状态映射回词表大小，用于预测下一个词元的概率分布
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        # 直接取编码器的"末态"（encoder 返回的 state）作为解码器初始状态。
        #   encoder 的 state 形状 (num_layers, batch, num_hiddens)，
        #   恰好就是解码器 rnn 需要的初态形状。
        return enc_outputs[1]

    def forward(self, X, state):
        # X 形状 (batch_size, num_steps)，嵌入并转置到 (时间步在前)。
        X = self.embedding(X).permute(1, 0, 2)
        # 广播上下文：取解码器初态最后一层（state[-1]，形状 (batch, num_hiddens)），
        # 复制 num_steps 份，使每个时间步都能拼上同一个上下文 c。
        context = state[-1].repeat(X.shape[0], 1, 1)
        # 把"当前词元的嵌入"和"上下文 c"沿最后一维拼接，
        # 得到 X_and_context 形状 (num_steps, batch, embed_size + num_hiddens)。
        X_and_context = torch.cat((X, context), 2)
        # GRU 前向：输出 output 每个时间步的隐状态。
        output, state = self.rnn(X_and_context, state)
        # 过输出层 dense，再转置回 (batch, time, vocab)。
        output = self.dense(output).permute(1, 0, 2)
        # output 形状 (batch_size, num_steps, vocab_size) —— 每个词元的分布
        # state 形状 (num_layers, batch_size, num_hiddens)
        return output, state


# 用与编码器相同的超参数实例化解码器。
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
# output 形状 (4, 7, 10) = (批量, 时间步, 词表大小)
# state  形状 (2, 4, 16) = (层数, 批量, 隐藏单元)
output.shape, state.shape


# ============================================================================
# 第4步：损失函数 —— 用遮蔽（mask）忽略 <pad> 位置
# ============================================================================
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    # X 形状 (batch, num_steps) 或更高。
    maxlen = X.size(1)
    # 构造一个 mask：位置 i 若小于该样本的 valid_len，则为 True（保留），否则 False。
    #   torch.arange(maxlen)[None, :] ：形状 (1, maxlen)，值 0..maxlen-1
    #   valid_len[:, None]             ：形状 (batch, 1)，每样本一个有效长度
    #   逐元素比较，广播成 (batch, maxlen) 的布尔张量。
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    # 把 mask 为 False（即超出有效长度 / 填充）的位置置为 value（默认 0）。
    X[~mask] = value
    return X


# 演示：屏蔽"总共 2 个序列、有效长度分别为 1 和 2"的情况。
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
sequence_mask(X, torch.tensor([1, 2]))
# 输出：
#   tensor([[1, 0, 0],
#           [4, 5, 0]])   —— 第一序列只保留位置 0，第二序列保留前两个位置


# 说明：此函数也可屏蔽"最后几个轴上的所有项"，或使用指定非零值替换。
X = torch.ones(2, 3, 4)
sequence_mask(X, torch.tensor([1, 2]), value=-1)
# 输出：每个样本的有效位置保留 1，其余位置替换为 -1。


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的 softmax 交叉熵损失函数"""
    # pred 的形状：(batch_size, num_steps, vocab_size)
    # label 的形状：(batch_size, num_steps)
    # valid_len 的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        # 初始化掩码为全 1（默认每个词元都参与计算损失）
        weights = torch.ones_like(label)
        # 把 mask 中"填充位置"置 0（只保留有效词元）
        weights = sequence_mask(weights, valid_len)
        # 使用 'none' 聚合：不求和/不求平均，保留每个位置的原始损失值
        self.reduction = 'none'
        # nn.CrossEntropyLoss 期望输入形状是 (N, C, ...)，即"类别维在第 2 维"，
        # 而 pred 是 (batch, num_steps, vocab)，所以要 permute 成 (batch, vocab, num_steps)。
        unweighted_loss = super().forward(pred.permute(0, 2, 1), label)
        # 用掩码"乘"掉填充位置的损失（填充处 weights=0 → 贡献为 0），
        # 再对时间步维度求平均，得到一个"每个样本一个标量损失"。
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


# 代码健全性检查：构造 3 个相同的序列，有效长度分别为 4、2、0。
#   由于 CrossEntropyLoss 对均匀分布（对数概率 = ln(1/10)）的默认损失是 ln(10)≈2.3026，
#   因此：
#     有效长度 4 → 该位置全保留 → 平均仍是 2.3026；
#     有效长度 2 → 4 个位置里只剩 2 个有效 → 损失减半 → 1.1513；
#     有效长度 0 → 全部被 mask → 损失 0。
loss = MaskedSoftmaxCELoss()
loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
     torch.tensor([4, 2, 0]))
# 输出：tensor([2.3026, 1.1513, 0.0000])
#   —— 印证了"第一个序列损失是第二个的两倍，第三个为 0"。


# ============================================================================
# 第5步：训练函数 train_seq2seq
# ============================================================================
def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""
    # 自定义参数初始化函数：对 Linear 和 GRU 层的权重使用 Xavier 均匀初始化，
    #   这有助于梯度平稳、加速收敛。
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            # 遍历 GRU 所有以 "weight" 开头的参数名（如 weight_ih_l0、weight_hh_l0）
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)  # 递归地对每个子层调用初始化函数
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 统计：损失总和、词元数量
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            # 构造解码器输入"dec_input"：
            #   教师强制（teacher forcing）——把"正确的上一词元"而非"预测词元"喂进去。
            #   先建一个 (batch,1) 的 <bos>（序列开始）列，
            #   然后与"去掉最后一个时间步的 Y"（即 Y[:, :-1]）拼接。
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            # 前向：编码器 → 初始化解码器状态 → 解码器生成输出
            Y_hat, _ = net(X, dec_input, X_valid_len)
            # 计算带遮蔽的交叉熵损失，并累加
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # 损失是"每样本一标量"，故取 sum 后再反传
            d2l.grad_clipping(net, 1)  # 梯度裁剪（限制梯度范数，防止爆炸）
            num_tokens = Y_valid_len.sum()  # 本批有效词元总数（用于度量）
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')


# 在机器翻译数据集上创建并训练一个 RNN "编码器-解码器"模型
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                         dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                         dropout)
net = d2l.EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
# 结果示例：
#   loss 0.019, 12745.1 tokens/sec on cuda:0


# ============================================================================
# 第6步：预测 predict_seq2seq
# ============================================================================
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 评估模式（不更新参数、不做 dropout）
    net.eval()
    # ① 把源句子小写 + 按空格切分，转成词表索引（词元化），末尾加上 <eos>()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    # 源序列的有效长度 = 源词元个数
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    # ② 截断/填充到 num_steps，并加一个批量维（unsqueeze(0)）变成 (1, num_steps)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    # ③ 编码：得到编码输出并初始化解码器状态（传入有效长度）
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # ④ 解码初始输入：只有 <bos>
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    # ⑤ 逐时间步自回归：每步用"上一步分数最高的词元"作为下一步输入
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 取该步 logits 中概率最大的词元索引（argmax）
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 若需要，保存当前注意力权重（供注意力章节讨论）
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦预测到 <eos>（序列结束），生成完成，停止
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    # 把预测词索引转回词元文本
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


# ============================================================================
# 第7步：BLEU 评估
# ============================================================================
def bleu(pred_seq, label_seq, k):
    """计算 BLEU（bilingual evaluation understudy）"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    # ① 长度惩罚项：如果预测比标签短，系数 < 1 会压低分数（惩罚过短的输出）。
    score = math.exp(min(0, 1 - len_label / len_pred))
    # ② 逐阶 n-gram 精确度：k 表示最多匹配到多长的 n 元语法
    for n in range(1, k + 1):
        # 统计标签序列中每个 n 元语法的出现次数（用 defaultdict 方便计数）
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        # 统计预测序列中每个 n 元语法被匹配到的次数（有限制：不能超标签出现次数）
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        # 精确度 = 匹配数 / 预测序列中的 n 元语法总数
        #   分母 (len_pred - n + 1) 是预测序列 n 元语法的总数量
        #   权重 0.5^{n}：越长的 n-gram 越难匹配，因此给更高权重
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


# 利用训练好的模型，把几个英语句子翻译成法语并计算 BLEU
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, attention_weight_seq = predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device)
    print(f'{eng} => {translation}, bleu {bleu(translation, fra, k=2):.3f}')
# 结果示例：
#   go . => va !, bleu 1.000
#   i lost . => j'ai perdu ., bleu 1.000
#   he's calm . => il est riche ., bleu 0.658
#   i'm home . => je suis en retard ?, bleu 0.447


# ============================================================================
# 补充知识点
# ============================================================================
# 1. 【教师强制（teacher forcing）的好处】
#    训练时直接把"正确的上一词元"喂给解码器，而不用自回归。好处：
#    - 避免了"一步错、步步错"的累积误差；
#    - 训练可以完全并行（一次把整个序列算完），加速明显；
#    - 缺点是"训练/推理不一致"（exposure bias），推理时只能靠上一步预测。
#    折衷做法是 scheduled sampling（逐步加入自回归输入）。

# 2. 【为什么损失要用遮蔽？】
#    翻译时不同句子长度不同，我们用 <pad> 填充到 num_steps。这些填充位没有任何
#    语义，若参与损失计算会被"误罚"，反而让模型去学"如何预测 <pad>"。用
#    sequence_mask 把这些位置贡献清零，只统计真实有效词元。

# 3. 【Xavier 初始化】
#    权重的初始化直接影响深层网络的收敛。Xavier 保证每层输入/输出方差一致，
#    避免激活值过大/过小导致梯度爆炸或消失。对 Linear 与 GRU 权重统一处理。

# 4. 【BLEU 的两个要点】
#    - n-gram 精确度：预测序列里有几个 n 元语法真的出现在标签里（且不超次）；
#    - 越长的 n-gram 越难匹配，权重越高（0.5^{n}）；
#    - 长度惩罚：预测太短会让 n-gram 易匹配，于是乘一个 <1 的惩罚系数。
#    当预测与标签完全一致时 BLEU = 1。


# ============================================================================
# 核心公式回顾
# ============================================================================
#   编码器：用函数 f 逐步更新隐状态，末态即上下文：
#       h_t = f(x_t, h_{t-1})，  c = q(h_1,...,h_T)（通常取 h_T）
#   解码器：用上下文 c + 已生成前缀预测下一词元：
#       s_{t'} = g(y_{t'-1}, c, s_{t'-1})
#       P(y_{t'} | y_1..y_{t'-1}, c) = softmax(输出层(s_{t'}))
#   训练损失：∑ 带遮蔽的 softmax 交叉熵（只统计有效词元）
#   BLEU = exp(min(0, 1 - len_label/len_pred)) × ∏_{n=1}^{k} p_n^{1/2^n}


# ============================================================================
# 练习与思考
# ============================================================================
# 1) 调整超参数（embed_size / num_hiddens / 层数 / dropout）改善翻译效果。
# 2) 去掉损失里的遮蔽重跑，观察变化并解释原因。
# 3) 若编码器与解码器层数/隐藏单元数不同，如何初始化解码器隐状态？
# 4) 训练时用"上一时间步的预测"替换教师强制，性能有何影响？
# 5) 用 LSTM 替换 GRU 重跑实验。 6) 有无其它设计解码器输出层的方法？


# ============================================================================
# 总结
# ============================================================================
#  - seq2seq 用两个 RNN：编码器压缩源序列为上下文 c，解码器用它逐词元生成目标。
#  - 实现要点：嵌入层、编码器末态初始化解码器、教师强制、遮蔽损失、Xavier 初始化。
#  - 训练用教师强制、推理用自回归，两者差异化是常见调优点。
#  - BLEU 通过 n-gram 匹配 + 长度惩罚来衡量生成序列质量。
#  - 这是"编码器-解码器"架构的第一个完整落地，也为后续注意力机制 / Transformer 垫基。
