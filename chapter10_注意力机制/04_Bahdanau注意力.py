"""
=============================================================================
Bahdanau 注意力（Bahdanau Attention）—— 完整带注释版
=============================================================================
任务：升级第9章的 seq2seq。原 seq2seq 解码器每一步都用"同一个固定的上下文变量 c"，
     但实际翻译某个词时，只有部分源词元有用。Bahdanau 注意力把 c 换成"随解码步
    动态变化的 c_{t'}"，从而让解码器在每一步"选择性关注"源序列的相关部分。

  核心：c_{t'} = Σ_t α(s_{t'-1}, h_t) · h_t
       - 查询 = 上一解码步隐状态 s_{t'-1}
       - 键/值 = 编码器各时间步的隐状态 h_t（键=值）
       - 用加性注意力做评分

依赖：d2l 包。本文件可独立运行（会训练英法机器翻译模型）。
=============================================================================
"""

# ============================================================================
# 第1步：导入依赖
# ============================================================================
import torch
from torch import nn
from d2l import torch as d2l


# ============================================================================
# 第2步：注意力解码器基类 AttentionDecoder
# ============================================================================
class AttentionDecoder(d2l.Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    # 子类必须实现 attention_weights 属性（供可视化注意力权重）
    @property
    def attention_weights(self):
        raise NotImplementedError


# ============================================================================
# 第3步：实现带 Bahdanau 注意力的解码器 Seq2SeqAttentionDecoder
# ============================================================================
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        # 注意力模块：用加性注意力（键/查询/值都投影到 num_hiddens 隐藏空间）
        self.attention = d2l.AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        # 词元嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # GRU 输入维度 = embed_size + num_hiddens（嵌入 + 注意力上下文拼在一起）
        self.rnn = nn.GRU(
            embed_size + num_hiddens, num_hiddens, num_layers,
            dropout=dropout)
        # 输出层：隐状态 → 词表大小的 logits
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # enc_outputs 是编码器输出 (outputs, hidden_state)：
        #   outputs 形状 (num_steps, batch, num_hiddens)——作为注意力"键和值"
        #   hidden_state 形状 (num_layers, batch, num_hiddens)——作为解码器初始隐状态
        outputs, hidden_state = enc_outputs
        # 转置 outputs 到 (batch, num_steps, num_hiddens) 方便后续查询
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # state = (enc_outputs, hidden_state, enc_valid_lens)
        enc_outputs, hidden_state, enc_valid_lens = state
        # X 形状 (batch, num_steps)，嵌入并转置成 (num_steps, batch, embed)
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        # 逐时间步处理（因为注意力要"每一步"动态算，无法批量并行）
        for x in X:
            # query = 上一解码步的最后一层隐状态，形状 (batch, 1, num_hiddens)
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # 注意力：用 query 去"键值对 enc_outputs"上提取上下文。
            #   context 形状 (batch, 1, num_hiddens)——即动态上下文 c_{t'}
            context = self.attention(
                query, enc_outputs, enc_outputs, enc_valid_lens)
            # 把"上下文"和"当前词元嵌入"沿特征维拼接
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 变形为 GRU 需要的 (时序在前)：(1, batch, embed+num_hiddens)
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            # 保存每一层的注意力权重供可视化
            self._attention_weights.append(self.attention.attention_weights)
        # 拼接 outputs 并过输出层，得到 (num_steps, batch, vocab)
        outputs = self.dense(torch.cat(outputs, dim=0))
        # 转回 (batch, num_steps, vocab)，并返回更新后的状态
        return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                          enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights


# ============================================================================
# 第4步：用 7 时间步的 4 序列输入，测试解码器形状
# ============================================================================
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                             num_layers=2)
encoder.eval()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                                  num_layers=2)
decoder.eval()
# X 形状 (batch_size=4, num_steps=7)
X = torch.zeros((4, 7), dtype=torch.long)
# 初始化状态（编码器输出 + 有效长度 None）
state = decoder.init_state(encoder(X), None)
output, state = decoder(X, state)
# output 形状 (4,7,10)；state 含 3 个元素；state[0] 是 encoder 输出（4,7,16）；
# state[1] 是 hidden_state（2 层），state[1][0] 形状 (4,16)
output.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape


# ============================================================================
# 第5步：训练（在英法机器翻译数据集上）
# ============================================================================
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 250, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = d2l.Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
# d2l.train_seq2seq：复用第9章的训练函数（教师强制 + 遮蔽损失 + 梯度裁剪）
# 因为新增了注意力，训练速度比无注意力的 seq2seq 明显慢。
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)
# 结果示例：loss 0.021, 4948.7 tokens/sec on cuda:0


# ============================================================================
# 第6步：翻译几个句子并计算 BLEU
# ============================================================================
engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
# predict_seq2seq 传入 save_attention_weights=True 以保存注意力权重便于可视化
for eng, fra in zip(engs, fras):
    translation, dec_attention_weight_seq = d2l.predict_seq2seq(
        net, eng, src_vocab, tgt_vocab, num_steps, device, True)
    print(f'{eng} => {translation}, ',
          f'bleu {d2l.bleu(translation, fra, k=2):.3f}')
# 结果示例：go . => va !, bleu 1.000（注意力帮助对齐，翻译质量更好）


# 把每一步的注意力权重拼接成可可视化形式
attention_weights = torch.cat(
    [step[0][0][0] for step in dec_attention_weight_seq], 0).reshape((
        1, 1, -1, num_steps))


# ============================================================================
# 第7步：可视化注意力权重
# ============================================================================
# 每个查询都会在键值对上分配不同权重——说明解码时"选择性聚焦"了输入的不同部分。
d2l.show_heatmaps(
    attention_weights[:, :, :, :len(engs[-1].split()) + 1].cpu(),
    xlabel='Key positions', ylabel='Query positions')


# ============================================================================
# 补充知识点
# ============================================================================
# 1. 【Bahdanau 注意力与普通 seq2seq 的区别】
#    - 普通 seq2seq：解码器每一步都用编码器"末态"作为固定上下文 c。
#    - Bahdanau：把 c 换成 c_{t'} = Σ α(s_{t'-1}, h_t) h_t，即"每一步动态对齐"。
#    区别看似只改了一处，却让模型能把注意力"聚焦"到真正有用的源词元。

# 2. 【查询/键/值 在这个模型里是什么】
#    - 查询：上一解码步隐状态 s_{t'-1}（代表"我现在翻译到哪了"）
#    - 键：  编码器各时间步隐状态 h_t（代表"源序列各词元的位置"）
#    - 值：  编码器各时间步隐状态 h_t（代表"源序列各词元的内容"，键=值）

# 3. 【可微对齐】
#    Bahdanau 注意力是学习对齐的早期 breakthrough：不像旧的对齐模型必须
#    强制单向移动，它允许"双向、任意"地对齐，且全程可微、能用反向传播训练。

# 4. 【为什么训练更慢】
#    注意力要在解码器的每一步单独算（逐时间步 for 循环里加一次注意力池化），
#    且加性注意力本身含几个线性层，因此比无注意力的 seq2seq 慢。

# 5. 【改进】
#    可以用缩放点积注意力替换加性注意力（见练习2），训练效率更高；
#    自注意力（下一节之后）甚至能完全去掉循环，实现真正的并行。


# ============================================================================
# 核心公式回顾
# ============================================================================
#   解码器状态：s_{t'} = g(y_{t'-1}, c_{t'}, s_{t'-1})
#   动态上下文：c_{t'} = Σ_{t=1}^T α(s_{t'-1}, h_t) h_t
#   注意力权重：α(s_{t'-1}, h_t) = softmax( a(s_{t'-1}, h_t) )
#   其中 a 用"加性注意力"评分函数；查询=s_{t'-1}，键=值=h_t。
#   自回归约束：解码器每一步只用到"已生成"的词元（<eos> 前）。


# ============================================================================
# 练习与思考
# ============================================================================
# 1) 用 LSTM 替换这里的 GRU 重跑实验。
# 2) 把加性注意力评分换成"缩放点积注意力"，观察训练效率变化。
# 3) 观察注意力热图："每条对角线越亮"，说明解码词与源词如何对应？


# ============================================================================
# 总结
# ============================================================================
#  - Bahdanau 注意力把 seq2seq 的"固定上下文 c"改为"逐步变化的注意力池化输出 c_{t'}"。
#  - 查询=上一解码步隐状态，键=值=编码器各时间步隐状态。
#  - 让模型在解码每个词时"选择性关注"源序列的相关部分，提升翻译质量。
#  - 它是双向、可微的注意力模型，是 seq2seq 时代的突破，也是通往 Transformer 的关键一跨。
