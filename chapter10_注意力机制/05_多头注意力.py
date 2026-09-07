"""
=============================================================================
多头注意力（Multi-Head Attention）—— 完整带注释版
=============================================================================
任务：让模型在"同一组查询、键、值"上，并行学习多种不同的注意力行为，
     从而同时捕获短距离、长距离等不同范围的依赖。做法是把 q/k/v 各做 h 组
     不同的线性投影，做 h 次注意力，再把 h 个输出拼接起来过一层线性投影。

  核心：h_i = f(W_i^q q, W_i^k k, W_i^v v)          （第 i 个头）
        MultiHead = W_o · [h_1; ...; h_h]

依赖：d2l 包。本文件可独立运行。
=============================================================================
"""

# ============================================================================
# 第1步：导入依赖
# ============================================================================
import math
import torch
from torch import nn
from d2l import torch as d2l


# ============================================================================
# 第2步：多头注意力 MultiHeadAttention
# ============================================================================
class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        # 每个头都用"缩放点积注意力"
        self.attention = d2l.DotProductAttention(dropout)
        # 四组线性投影：q/k/v 各一组，最后输出一组 W_o
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        # 初始维度：
        #   queries 形状 (batch, 查询个数, num_hiddens)
        #   keys    形状 (batch, 键值对个数, num_hiddens)
        #   values  形状 (batch, 键值对个数, num_hiddens)
        # 先过线性层，再拆成 num_heads 个头：
        #   → (batch*num_heads, 查询个数, num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 有效长度也要按"头数"重复：每个头的批量都要用同一组有效长度。
            # 把 (batch,) 或 (batch, 查询数) 沿轴0重复 num_heads 次。
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # 并行做 num_heads 次缩放点积注意力：
        #   output 形状 (batch*num_heads, 查询个数, num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # 逆转 transpose_qkv，把 h 个头"拼"回最后一维：
        #   output_concat 形状 (batch, 查询个数, num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        # 最后过一次线性投影 W_o，得到多头注意力输出
        return self.W_o(output_concat)


# ============================================================================
# 第3步：两个形状转置函数（实现"多头的并行计算"）
# ============================================================================
def transpose_qkv(X, num_heads):
    """为多头注意力的并行计算而变换形状"""
    # 输入 X 形状：(batch, 查询/键值对个数, num_hiddens)
    # ① 把最后一维拆成 (num_heads, num_hiddens/num_heads)：
    #    → (batch, n, num_heads, head_dim)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    # ② 把"头"维挪到第 2 维：→ (batch, num_heads, n, head_dim)
    X = X.permute(0, 2, 1, 3)
    # ③ 把 batch 与 head 两维合并成一个大批量：
    #    → (batch*num_heads, n, head_dim)，这样 bmm 就把每个头当作独立批量并行计算
    return X.reshape(-1, X.shape[2], X.shape[3])


def transpose_output(X, num_heads):
    """逆转 transpose_qkv 函数的操作"""
    # 把 (batch*num_heads, n, head_dim) 恢复成 (batch, num_heads, n, head_dim)
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    # 再交换头维 → (batch, n, num_heads, head_dim)
    X = X.permute(0, 2, 1, 3)
    # 合并最后一维 → (batch, n, num_hiddens)
    return X.reshape(X.shape[0], X.shape[1], -1)


# ============================================================================
# 第4步：用键值相同的例子测试
# ============================================================================
num_hiddens, num_heads = 100, 5
attention = MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                               num_hiddens, num_heads, 0.5)
attention.eval()
# 模型结构：内部包含一个 DotProductAttention + 4 个 Linear（q/k/v/o），均无偏置

batch_size, num_queries = 2, 4
num_kvpairs, valid_lens = 6, torch.tensor([3, 2])
# X 作为查询 (2,4,100)，Y 同时充当键和值 (2,6,100)
X = torch.ones((batch_size, num_queries, num_hiddens))
Y = torch.ones((batch_size, num_kvpairs, num_hiddens))
attention(X, Y, Y, valid_lens).shape   # torch.Size([2, 4, 100])


# ============================================================================
# 补充知识点
# ============================================================================
# 1. 【为什么"多头"有用？】
#    单一注意力平均只学到了"一种"匹配模式。多头通过对 q/k/v 各做不同的线性投影，
#    把输入表示送到不同的"子空间"，让不同头学习到不同的关注方式(如某些头关注
#    短距离、某些头关注长距离)，最后拼接融合——表达能力比单一注意力强得多。

# 2. 【头数与维度的设定】
#    为控制参数/计算量，通常令 p_q = p_k = p_v = p_o / h。
#    因此每个头的维度 = num_hiddens / num_heads。
#    要让 num_hiddens 能整除 num_heads（本例 100/5=20）。

# 3. 【transpose_qkv 的动机】
#    如果把 h 个头分别写 for 循环，效率低。把 batch 和 head 合并成一个大批量维，
#    就只需一次 bmm 即可并行计算所有头的缩放点积注意力。

# 4. 【valid_lens 的重复】
#    不同序列的有效长度不同。因为批量维被"乘上 num_heads"，所以 valid_lens
#    也要用 repeat_interleave 重复 num_heads 次，与扩大的批量对齐。


# ============================================================================
# 核心公式回顾
# ============================================================================
#   第 i 个头：h_i = f( W_i^q q, W_i^k k, W_i^v v ) ∈ R^{p_v}
#   多头输出：  MultiHead = W_o · [h_1; ...; h_h] ∈ R^{p_o}
#   其中 f 为缩放点积注意力；各投影 W_i 是独立可学习的线性层。
#   设定 p_q = p_k = p_v = p_o / h 以控制开销并支持并行。


# ============================================================================
# 练习与思考
# ============================================================================
# 1) 分别可视化这个实验中多个头的注意力权重。
# 2) 假设有一个训练好的多头注意力模型，想"剪掉最不重要的头"以提速。
#    如何设计实验来度量每个头的重要性？
# 3) 如果 num_hiddens 不能被 num_heads 整除，会怎样？（维度拆分会报错/溢出）


# ============================================================================
# 总结
# ============================================================================
#  - 多头注意力 = h 组不同的线性投影 + 并行注意力 + 拼接 + 一次线性投影。
#  - 让模型在"同一组 q/k/v"上从不同子空间捕获多种关系，表达更强。
#  - transpose_qkv / transpose_output 是"让多个头并行计算"的关键张量技巧。
#  - 它是 Transformer 中"多头自注意力"的基础组件（下一节与 Transformer 使用）。
