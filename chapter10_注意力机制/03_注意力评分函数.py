"""
=============================================================================
注意力评分函数（Attention Scoring Functions）—— 完整带注释版
=============================================================================
任务：把上一节的注意力池化通用化——核心就是"评分函数 a(q,k)"：
      权重 α(q,k_i) = softmax( a(q,k_i) )，输出是值的加权平均。
     本节介绍两种主流评分函数：
       1. 加性注意力（additive attention）—— 适合查询、键长度不同的情况
       2. 缩放点积注意力（scaled dot-product attention）—— 适合查询、键长度相
          同且更追求效率的情况。
     另外先介绍"掩蔽 softmax"，用于过滤掉填充等无意义位置。

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
# 第2步：掩蔽 softmax 操作 masked_softmax
# ============================================================================
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行 softmax 操作"""
    # X 是 3D 张量，valid_lens 是 1D 或 2D 张量
    if valid_lens is None:
        # 不掩蔽：直接对最后一个轴(键维)做 softmax
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            # 1D valid_len：对所有"查询"都适用相同有效长度。
            # 由于 X 被 reshape 成 2D，valid_lens 需要重复 shape[1]（时间步/查询数）次
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            # 2D valid_len：即"每个查询各自的有效长度"，直接展平即可
            valid_lens = valid_lens.reshape(-1)
        # 把"超出有效长度"的位置替换成一个极大的负数 -1e6，
        # 这样 softmax 后它们趋近于 0（e^{-很大}≈0），相当于被"屏蔽"。
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


# 演示：两个 2×4 矩阵样本，有效长度分别为 2 和 3。
#   超出有效长度的位置(softmax 后)都变成 0。
masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3]))
# 示例输出：每行前 valid_len 个位置归一化，其余为 0。

# 用二维张量给"矩阵样本的每一行分别指定有效长度"：
masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]]))


# ============================================================================
# 第3步：加性注意力 AdditiveAttention
# ============================================================================
class AdditiveAttention(nn.Module):
    """加性注意力（当查询和键是不同长度的矢量时使用）"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        # 三个可学习线性层（都不含偏置，因为数值稳定性/省参）
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)   # 键 → 隐藏
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False) # 查询 → 隐藏
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)          # 隐藏 → 标量
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        # 先做线性变换到"隐藏空间"
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 维度扩展后用"广播"求和，得到每对(查询,键)的逐元素和：
        #   queries 形状 (batch, 查询数, 1, num_hidden)
        #   keys    形状 (batch, 1, 键值对个数, num_hiddens)
        #   相加后 (batch, 查询数, 键值对个数, num_hiddens)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # w_v 只输出一个标量 → squeeze(-1) 去掉最后一个维
        # scores 形状 (batch, 查询数, 键值对个数)
        scores = self.w_v(features).squeeze(-1)
        # 用掩蔽 softmax 得到注意力权重
        self.attention_weights = masked_softmax(scores, valid_lens)
        # 权重对值做加权求和：bmm(权重, values)
        # values 形状 (batch, 键值对个数, 值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)


# 用一个小例子演示 AdditiveAttention：
#   查询形状 (2,1,20)，键 (2,10,2)，值 (2,10,4)，各自的有效长度 [2, 6]
queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
# values：两个值矩阵相同（都是 0..39 排成 (10,4)）
values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
valid_lens = torch.tensor([2, 6])

# 实例化加性注意力，dim key_size=2, query_size=20, num_hiddens=8
attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8,
                              dropout=0.1)
attention.eval()
attention(queries, keys, values, valid_lens)
# 输出：注意力汇聚结果，形状 (2,1,4)（批量, 查询数, 值的维度）

# 由于本例所有键都相同（无法被查询区分），注意力权重是"均匀"的，仅由有效长度决定。
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')


# ============================================================================
# 第4步：缩放点积注意力 DotProductAttention
# ============================================================================
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries 形状 (batch, 查询个数, d)
    # keys    形状 (batch, 键值对个数, d)
    # values  形状 (batch, 键值对个数, 值的维度)
    # valid_lens 形状 (batch,) 或 (batch, 查询个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]   # 特征维度
        # 缩放点积：scores = (Q K^T) / √d
        #   keys.transpose(1,2) 交换后两维，使"矩阵乘法"变成 (batch,n,d)@(batch,d,m)
        #   除以 √d 是为了让"点积的方差"不随 d 增大而膨胀（否则 softmax 梯度消失）
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # 权重对值加权求和
        return torch.bmm(self.dropout(self.attention_weights), values)


# 演示：点积要求查询与键的特征维度相同。
#   查询改为 (2,1,2)（特征维度从 20 改为与键一致=2）。
queries = torch.normal(0, 1, (2, 1, 2))
attention = DotProductAttention(dropout=0.5)
attention.eval()
attention(queries, keys, values, valid_lens)
# 输出与加性注意力相同（(2,1,4)），因为键完全相同 → 权重均匀。

# 同样画热图：得到的也是均匀注意力权重
d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                  xlabel='Keys', ylabel='Queries')


# ============================================================================
# 补充知识点
# ============================================================================
# 1. 【为什么"缩放点积"要除以 √d？】
#    若查询与键的元素都是零均值、单位方差，点积 q·k 的均值为 0、方差为 d。
#    若 d 很大，点积数值会很大 → softmax 的其中一项会"霸权"（接近 1），
#    其它项被压到 0，梯度很小、难训练。除以 √d 把方差拉回 1，数值稳定。

# 2. 【加性注意力 vs 缩放点积注意力】
#    - 加性注意力：a = w_vᵀ tanh(W_q q + W_k k)。用 MLP 把 q、k 先投影到隐藏层
#      再打分。能处理 q、k 长度不同。参数较多、较慢。
#    - 缩放点积注意力：a = q·k/√d。直接算内积，简单高效，但要求 q、k 长度一致。
#      这是 Transformer 用的版本。
#    当 q、k 长度相同时，点积更快；不同时用加性注意力。

# 3. 【为什么"掩蔽"重要】
#    机器翻译里序列被 <pad> 填充到统一长度。这些填充位置不含语义，若让查询去
#    关注它们，会污染注意力。用有效长度把它们 mask 掉，注意力只落在真实词元上。

# 4. 【评分函数的意义】
#    不同的评分函数 a 得到不同的注意力机制。softmax( a(·) ) 永远是(0,1)之间的
#    概率分布，So 注意力输出永远是"值的加权平均"。


# ============================================================================
# 核心公式回顾
# ============================================================================
#   注意力权重：α(q,k_i) = softmax( a(q,k_i) )
#   注意力汇聚：f = Σ_i α(q,k_i) · v_i
#   加性注意力：a(q,k) = w_vᵀ tanh( W_q q + W_k k )
#   缩放点积：  a(q,k) = qᵀ k / √d
#   批量矩阵：  Q Kᵀ / √d → (n,m)；再乘 V → (n,v)


# ============================================================================
# 练习与思考
# ============================================================================
# 1) 修改小例子的键，并可视化注意力权重。加性与点积注意力结果还一样吗？为何？
# 2) 只用矩阵乘法，能否为"不同长度"的查询和键设计新评分函数？
# 3) 当 q、k 长度相同时，"向量求和"作为评分函数是否优于点积？为什么？


# ============================================================================
# 总结
# ============================================================================
#  - 注意力池化的输出 = 值的加权平均；权重由"评分函数 + softmax"给出。
#  - masked_softmax 用于过滤填充/越界位置，是后续所有注意力模型的基础工具。
#  - 加性注意力适合 q、k 长度不同；缩放点积注意力（除以√d）适合长度相同、效率优先。
#  - 评分函数承担"给查询-键对打分"的角色，直接决定了注意力机制的形态。
