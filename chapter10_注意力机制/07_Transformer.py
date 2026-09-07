"""
=============================================================================
Transformer —— 完整带注释版
=============================================================================
任务：完全基于注意力机制（没有卷积层、没有循环神经网络层）的"编码器-解码器"架构。
     把前几节的组件拼装起来：多头自注意力、位置编码、基于位置的前馈网络、
     残差连接 + 层规范化，最终在英法机器翻译任务上端到端训练。

依赖：d2l 包 + pandas。本文件可独立运行（会下载英法数据集并训练）。
=============================================================================
"""

# ============================================================================
# 第1步：导入依赖
# ============================================================================
import math
import pandas as pd     # 用于处理带 NaN 的注意力权重数据（可视化部分）
import torch
from torch import nn
from d2l import torch as d2l


# ============================================================================
# 第2步：基于位置的前馈网络 PositionWiseFFN
# ============================================================================
class PositionWiseFFN(nn.Module):
    """基于位置的前馈网络：对序列中每个位置用同一个两层 MLP 变换"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        # 同一套 MLP 独立作用在"每一个位置"上（即对最后一维做变换）
        return self.dense2(self.relu(self.dense1(X)))


# 演示：输入形状 (2,3,4)，输出最里层维从 4 变成 8。
# 因为所有位置共享同一个 MLP，输入相同时输出也相同。
ffn = PositionWiseFFN(4, 4, 8)
ffn.eval()
ffn(torch.ones((2, 3, 4)))[0]   # 形状 (3,8)，三行完全相同


# ============================================================================
# 第3步：残差连接和层规范化 AddNorm
# ============================================================================
# 说明：批量规范化（BatchNorm）按"批量"统计均值/方差；层规范化（LayerNorm）按
#       "特征维"统计。NLP 序列长度多变，层规范效果通常更好。
# 对比：对 [[1,2],[2,3]]，LayerNorm 沿特征维归一（每行）；BatchNorm 沿批量维归一（每列）。
ln = nn.LayerNorm(2)
bn = nn.BatchNorm1d(2)
X = torch.tensor([[1, 2], [2, 3]], dtype=torch.float32)
print('layer norm:', ln(X), '\nbatch norm:', bn(X))
# layer norm: 每行自己归一 → [[-1,1],[-1,1]]
# batch norm: 每列整体归一 → [[-1,-1],[1,1]]


class AddNorm(nn.Module):
    """残差连接后进行层规范化：输出 = LayerNorm( dropout(Y) + X )"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        # Y 通常是子层输出，X 是子层输入；先 dropout 再加回 X（残差），最后层规范
        return self.ln(self.dropout(Y) + X)


# 残差连接要求 X 与 Y 形状相同（形状 (2,3,4)），加法后仍 (2,3,4)
add_norm = AddNorm([3, 4], 0.5)
add_norm.eval()
add_norm(torch.ones((2, 3, 4)), torch.ones((2, 3, 4))).shape   # torch.Size([2,3,4])


# ============================================================================
# 第4步：Transformer 编码器块 EncoderBlock
# ============================================================================
class EncoderBlock(nn.Module):
    """Transformer编码器块：多头自注意力 + AddNorm + 前馈 + AddNorm"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        # 子层1：多头自注意力（q/k/v 都来自上一层输出）
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        # 子层1的残差+层规范化
        self.addnorm1 = AddNorm(norm_shape, dropout)
        # 子层2：基于位置的前馈网络
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        # 子层2的残差+层规范化
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        # 自注意力（X 同时作 q/k/v）→ 残差加 + 层规范化
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        # 前馈 → 残差加 + 层规范化
        return self.addnorm2(Y, self.ffn(Y))


# 测试：任何编码器层都不改变输入形状 (2,100,24)
X = torch.ones((2, 100, 24))
valid_lens = torch.tensor([3, 2])
encoder_blk = EncoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5)
encoder_blk.eval()
encoder_blk(X, valid_lens).shape   # torch.Size([2, 100, 24])


# ============================================================================
# 第5步：Transformer 编码器 TransformerEncoder
# ============================================================================
class TransformerEncoder(d2l.Encoder):
    """Transformer编码器：嵌入 + 位置编码 + 堆叠 num_layers 个 EncoderBlock"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        # 用 nn.Sequential 堆叠 num_layers 个编码器块
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 EncoderBlock(key_size, query_size, value_size,
                                              num_hiddens, norm_shape,
                                              ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 位置编码值在 -1 和 1 之间，因此嵌入值乘以 √num_hiddens 缩放，
        # 以匹配位置编码的量级，再做加法。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        # 记录每层的注意力权重（用于可视化）
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X


# 创建两层 Transformer 编码器，输出形状 (2,100,24)
encoder = TransformerEncoder(
    200, 24, 24, 24, 24, [100, 24], 24, 48, 8, 2, 0.5)
encoder.eval()
encoder(torch.ones((2, 100), dtype=torch.long), valid_lens).shape   # (2,100,24)


# ============================================================================
# 第6步：Transformer 解码器块 DecoderBlock
# ============================================================================
class DecoderBlock(nn.Module):
    """解码器第 i 个块：掩蔽自注意力 + 编码器-解码器注意力 + 前馈"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        # 子层1：解码器自注意力（q/k/v 都来自解码器上一层输出，且被"掩蔽"）
        self.attention1 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        # 子层2：编码器-解码器注意力（查询来自解码器；键/值来自编码器输出）
        self.attention2 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        # 子层3：基于位置的前馈网络
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段：输出序列所有词元同一时间处理，因此 state[2][i] 初始化为 None；
        # 预测阶段：输出序列逐词元解码，state[2][i] 保存到当前时间步的解码输出。
        if state[2][self.i] is None:
            key_values = X
        else:
            # 把之前已生成的词元与当前词元拼接，作为自注意力的键/值
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            # 训练时用"因果掩蔽"：每个查询只能看到自己及之前的词元位置。
            # dec_valid_lens 形状 (batch, num_steps)，每行是 [1,2,...,num_steps]。
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            # 预测时逐词元进行，无需掩蔽（每次只有一个位置）
            dec_valid_lens = None

        # 子层1：解码器自注意力（用 key_values 作键值，X 作查询）
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # 子层2：编码器-解码器注意力（查询 Y 来自解码器，键值来自编码器输出）
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        # 子层3：前馈 + 残差 + 层规范
        return self.addnorm3(Z, self.ffn(Z)), state


# 测试解码器块：特征维度统一为 num_hiddens=24
decoder_blk = DecoderBlock(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
decoder_blk.eval()
X = torch.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
decoder_blk(X, state)[0].shape   # torch.Size([2, 100, 24])


# ============================================================================
# 第7步：Transformer 解码器 TransformerDecoder
# ============================================================================
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        # 堆叠 num_layers 个解码器块
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 DecoderBlock(key_size, query_size, value_size,
                                              num_hiddens, norm_shape,
                                              ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)  # 输出层

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # 初始状态 = (编码器输出, 编码器有效长度, 每层缓存[None]*num_layers)
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        # 嵌入 + 位置编码
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        # 记录两类注意力权重：解码器自注意力 / 编码器-解码器注意力
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state   # dense 输出 (batch, num_steps, vocab)

    @property
    def attention_weights(self):
        return self._attention_weights


# ============================================================================
# 第8步：训练（英法机器翻译）
# ============================================================================
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
# 结果示例：loss 0.030, 5202.9 tokens/sec on cuda:0


# ============================================================================
# 第9步：翻译并计算 BLEU
# ============================================================================
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
# 示例（这里表现得很好，四条都接近/等于 1）：
#   go . => va !, bleu 1.000 ...


# ============================================================================
# 第10步：可视化注意力权重
# ============================================================================
# 编码器自注意力权重：形状 (层数, 头数, num_steps(查询), num_steps(键值))
enc_attention_weights = torch.cat(net.encoder.attention_weights, 0).reshape(
    (num_layers, num_heads, -1, num_steps))
enc_attention_weights.shape   # torch.Size([2, 4, 10, 10])

# 编码器自注意力：查询、键都来自输入序列；用有效长度避免关注 <pad>。
# 逐行展示两层多头注意力权重，每个头代表不同的表示子空间。
d2l.show_heatmaps(
    enc_attention_weights.cpu(), xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))


# 解码器自注意力与"编码器-解码器"注意力可视化（步骤较多）：
# 把每一步的注意力拼成 pandas DataFrame，用 fillna 把被掩蔽/缺失处填 0。
dec_attention_weights_2d = [head[0].tolist()
                            for step in dec_attention_weight_seq
                            for attn in step for blk in attn for head in blk]
dec_attention_weights_filled = torch.tensor(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
dec_attention_weights = dec_attention_weights_filled.reshape(
    (-1, 2, num_layers, num_heads, num_steps))
# 拆成"解码器自注意力"和"编码器-解码器注意力"两块
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.permute(1, 2, 3, 0, 4)
dec_self_attention_weights.shape, dec_inter_attention_weights.shape
# (2, 4, 6, 10), (2, 4, 6, 10)

# 解码器自注意力是"自回归"的：查询不会关注自己之后的位置（左上三角被掩蔽）。
d2l.show_heatmaps(
    dec_self_attention_weights[:, :, :, :len(translation.split()) + 1],
    xlabel='Key positions', ylabel='Query positions',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))

# 编码器-解码器注意力：查询(解码)不会去关注输入里被填充(<pad>)的位置。
d2l.show_heatmaps(
    dec_inter_attention_weights, xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))


# ============================================================================
# 补充知识点
# ============================================================================
# 1. 【为什么 Transformer 完全不需要 CNN/RNN？】
#    它靠"多头自注意力"捕获任意位置间的依赖（最大路径最短、可并行），
#    靠"位置编码"补充顺序信息，靠"前馈网络 + 残差 + 层规范"提升表达能力与深层训练稳定性。
#    因此它天然适合处理序列，且比 RNN 更高效（可整体并行）。

# 2. 【解码器的"掩蔽自注意力"为什么必要？】
#    序列到序列训练时，解码器一次性看到整个目标序列。若不掩蔽，预测第 t 个词时
#    就能"看到"第 t 个词之后的内容（作弊）。掩蔽把每个查询限制为"只看自己及之前"，
#    从而保留"自回归"性质：预测只依赖已生成的词元。

# 3. 【LayerNorm vs BatchNorm】
#    BatchNorm 沿 batch 维统计均值/方差；LayerNorm 沿"每个样本的特征维"统计。
#    输入序列长度多变时，BatchNorm 的统计量受 batch 组成影响、较不稳定，
#    因此 NLP 里 LayerNorm 通常更合适、效果更好。

# 4. 【残差连接 + 层规范 = 深度网络的"稳定器"】
#    - 残差连接：y = x + sublayer(x)，为梯度提供"直通"路径，缓解深网梯度消失。
#    - 层规范化：把每个位置的激活值归一，避免数值动荡。
#    两者让 Transformer 能堆叠很多层而不崩。

# 5. 【Transformer 的广泛应用】
#    最初为序列到序列而设计，但编码器/解码器常被单独使用（BERT 用编码器；
#    GPT 用解码器）；更推广到视觉、语音、强化学习等领域（如 Vision Transformer）。


# ============================================================================
# 核心公式回顾
# ============================================================================
#   编码器层：  X → MultiHeadSelfAttn(X) → +X → LayerNorm → FFN → +X → LayerNorm
#   解码器块：  ① 掩蔽多头自注意力 ② 编码器-解码器注意力 ③ 前馈网络
#              每个子层都做"残差 + 层规范"
#   缩放点积：  softmax(Q K^T / √d) V
#   多头：      MultiHead = W_o · [head_1; ...; head_h]
#   位置编码：  p_{i,2j}=sin(i/10000^{2j/d})，p_{i,2j+1}=cos(i/10000^{2j/d})
#   复杂度：    O(n²·d)（自注意力）


# ============================================================================
# 练习与思考
# ============================================================================
# 1) 训练更深的 Transformer，对训练速度和翻译效果有何影响？
# 2) 用"加性注意力"替代缩放点积注意力是不是好办法？为什么？
# 3) 对语言模型，应该用 Transformer 的编码器还是解码器，还是都用？
# 4) 输入序列很长时 Transformer 面临什么挑战？为什么？
# 5) 如何提高 Transformer 的计算速度与内存使用效率？
# 6) 不用卷积，如何用 Transformer 做图像分类？（提示：Vision Transformer）


# ============================================================================
# 总结
# ============================================================================
#  - Transformer 是"完全基于注意力"的编码器-解码器：无 CNN/RNN。
#  - 组件：多头自注意力、位置编码、位置前馈、残差+层规范、掩蔽自注意力。
#  - 编码器每层两个子层；解码器每层三个子层（含掩蔽自注意力 + 编码器-解码器注意力）。
#  - 残差+层规范是"能训练很深模型"的关键；掩蔽保留自回归性质。
#  - 它是现代深度学习（GPT/BERT 等）共同的基石。
