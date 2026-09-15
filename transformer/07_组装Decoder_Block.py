"""
=============================================================================
组装 Decoder Block（Pre-RMSNorm + 残差）—— 完整带注释版
=============================================================================
源文件: exercises/block_03_transformer/task_26_decoder_blocks/transformer_blocks.py

任务：Attention 和 FFN 的输入输出都是 (B,T,D)，把它们各放进一条残差支路，
     就得到本章使用的 Pre-Norm decoder block。

  核心：x1 = x  + Attention(RMSNorm1(x))
        x2 = x1 + FFN(RMSNorm2(x1))

        通式（Pre-Norm）： x + sublayer(norm(x))
        对照（Post-Norm）：norm(x + sublayer(x))

        RMSNorm(x) = x / √(mean(x²)+ε) ⊙ w      ← 不减均值、无 bias

输入：x: (B,T,D)
输出：(B,T,D)   —— 贯穿整个 stack 形状不变

依赖：math、torch、torch.nn。本文件可独立运行。
=============================================================================
"""

# ============================================================================
# 第1步：导入依赖
# ============================================================================
import math                       # math.isfinite 用于校验 eps

import torch                      # torch.ones / torch.rsqrt
from torch import nn              # nn.Parameter / nn.Module / nn.ModuleList


# ============================================================================
# 第2步：Pre-Norm 放在哪里
# ============================================================================
# Pre-Norm 的通式是：
#
#     x + sublayer(norm(x))
#
# Post-Norm 则是：
#
#     norm(x + sublayer(x))
#
# 两者都是合法设计。差别在于归一化发生在残差支路的【里面】还是【外面】：
#
#   Pre-Norm:   x ──┬──────────────────────►(+)──► 输出
#                   └─► norm ─► sublayer ──►
#               ↑ 残差是一条干净的直通路，梯度可以原样回传
#
#   Post-Norm:  x ──┬──────────────────────►(+)──► norm ──► 输出
#                   └─► sublayer ──────────►
#               ↑ 归一化堵在残差出口，深层的梯度要穿过它
#
# ── 为什么现代 LLM 普遍用 Pre-Norm ────────────────────────────────────────
# Pre-Norm 的残差是"恒等直通"，反向传播时梯度有一条不经过任何非线性层的
# 高速通道，因此深层堆叠时更稳定，通常也不需要 warmup 那样精细的学习率调度。
# 原论文用的是 Post-Norm，后来的实践逐步转向 Pre-Norm。
#
# 本章的结构图、公式和代码统一采用 Pre-RMSNorm；
# 每层有两个独立 norm，全部 block 之后还有一次 final RMSNorm。
#
# ── 残差相加成立的前提 ────────────────────────────────────────────────────
# sublayer 输出与输入 shape 必须相同：
#
#     x₁ = x  + Attention(RMSNorm₁(x))
#     x₂ = x₁ + FFN(RMSNorm₂(x₁))
#
# 若 FFN 忘了从 hidden_dim 投影回 D，错误会在第二次加法处直接暴露 ——
# 报错位置虽然晚，但症状明确（比如 expected (2,5,16) got (2,5,32)），好定位。
# ============================================================================


# ============================================================================
# 第3步：RMSNorm 与 LayerNorm 的区别
# ============================================================================
# 这里的 RMSNorm 是：
#
#                     x
#     RMSNorm(x) = ─────────────────── ⊙ w
#                  √( mean(x²) + ε )
#
# 它只沿最后一维计算均方根并学习缩放 w：
#   - 【不减均值】—— 所以叫 RMS（Root Mean Square）而不是 LN；
#   - 【没有 bias】—— 只有一份缩放参数 w；
#   - 比 LayerNorm 少一次求均值和一次减法，计算更省。
#
# LayerNorm 通常先减均值，再按方差缩放；名称不能混用。
#
# ── Shape 约定 ────────────────────────────────────────────────────────────
#
#     input:  (...,D)
#     weight: (D,)
#     output: (...,D)
#
# 输入 (B,T,D) 时，每个 token 独立在最后一维归一化，不混合 batch 或 sequence。
# 换句话说：归一化只发生在【特征维 D】上，与 Attention/FFN 的"沿 D 维运算"一致。
#
# ── 与 BatchNorm 的一个关键差别 ───────────────────────────────────────────
# RMSNorm 没有 running statistics（不像 BatchNorm 要维护滑动均值/方差），
# 因此 train/eval 使用相同公式，不需要 model.eval() 来切换行为。
# 这也是它适合变长序列的一个原因。
#
# ── 混合精度下的处理 ──────────────────────────────────────────────────────
# 代码对 FP16/BF16 先用 FP32 计算均方根，再转回输入 dtype，
# 避免小精度下不必要的数值损失（均方、求和这类操作在低精度下容易累积误差）。
# ============================================================================


# ============================================================================
# 第4步：RMSNorm 实现
# ============================================================================
class RMSNorm(nn.Module):
    """均方根归一化：只缩放，不中心化。

    参数:
        dim: 归一化的维度 D（即张量最后一维）
        eps: 稳定项，防止均方根为 0 时除零
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()

        # 入参校验：dim 为正、eps 为正且有限
        # math.isfinite 同时排除 inf 和 nan
        if dim <= 0 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("dim and eps must be positive, with eps finite")

        # nn.Parameter 把张量注册成【可学习参数】：
        #   它会出现在 model.parameters() 里，会被优化器更新，
        #   也会被 model.state_dict() 保存 —— 这三点是普通张量做不到的。
        # torch.ones(dim) 初始化为全 1：训练开始时 RMSNorm 近似于"只做归一化，
        # 不做额外缩放"，让网络从恒等缩放出发，训练更稳。
        self.weight = nn.Parameter(torch.ones(dim))

        # eps 是普通 Python float，不是参数，不会被训练
        self.eps = eps

    def forward(self, x):
        """
        参数:
            x: (...,D) 任意前导维度

        返回:
            与 x 同 shape 同 dtype

        分步说明 / 计算过程：
            (1) 记录输入 dtype，并决定用什么精度来算
            (2) 求均方根并做归一化（用计算精度）
            (3) 乘上可学习的缩放 w
            (4) 转回输入 dtype
        """
        # 记下原始 dtype，最后要转回去
        input_dtype = x.dtype

        # 决定用哪种精度做计算：
        #   输入是半精度（fp16/bf16）→ 用 fp32 算，避免精度损失；
        #   其他情况（fp32/fp64）→ 原样用
        # in (a, b) 是 Python 的成员测试，对元组做包含判断
        compute_dtype = (
            torch.float32
            if input_dtype in (torch.float16, torch.bfloat16)
            else input_dtype
        )

        # 转换到计算精度（若是同一种 dtype，.to 会直接返回原张量，几乎零开销）
        x_compute = x.to(dtype=compute_dtype)

        # ── 核心计算，从里到外读 ────────────────────────────────────────
        # x_compute.square()            逐元素平方 → (...,D)
        # .mean(dim=-1, keepdim=True)   沿最后一维求均值 → (...,1)
        #                               keepdim=True 保留长度为 1 的维，
        #                               这样后面才能与 (...,D) 正确广播
        # + self.eps                    加上稳定项，避免平方和为 0 时除零
        # torch.rsqrt(...)              开平方后取倒数，等价于 1/sqrt(...)
        #                               用 rsqrt 而不是 1/torch.sqrt() 是因为
        #                               它是单个融合算子，更快也更稳
        # x_compute * (...)             广播相乘 → (...,D)
        #
        # 合起来就是 x / √(mean(x²)+ε)，即公式里的归一化项
        normalized = x_compute * torch.rsqrt(
            x_compute.square().mean(dim=-1, keepdim=True) + self.eps
        )

        # 乘上可学习缩放 w，最后转回输入 dtype。
        # self.weight.to(dtype=compute_dtype)：参数通常保持 fp32（混合精度训练时
        # 这是惯例），所以要显式对齐精度才能相乘。
        # 注意是【先归一化 → 再乘 w → 最后转 dtype】的顺序：
        # 缩放用高精度做完再降精度，比先降精度再缩放更准。
        return (normalized * self.weight.to(dtype=compute_dtype)).to(dtype=input_dtype)


# ============================================================================
# 第5步：可注入的 attention 和 FFN
# ============================================================================
class TransformerBlock(nn.Module):
    """Pre-Norm decoder block：两条残差支路，attention 与 FFN 由外部注⼊。

    参数:
        dim:        模型维度 D
        attention:  任意"输入 (B,T,D)、输出 (B,T,D)"的模块
        feed_forward: 同上
    """

    def __init__(self, dim, attention, feed_forward):
        super().__init__()

        # 两个【独立】的 RMSNorm 实例 —— 这一点很重要：
        #   它们各有自己的 weight 参数，norm1 学到的东西不会影响 norm2。
        # 如果误写成共享同一个实例，两个位置就会被迫共用同一套缩放，
        # 表达力下降，而且是一个不会报错的静默错误。
        self.norm1 = RMSNorm(dim)

        # attention 与 feed_forward 直接接收现成的模块对象（不在这里创建）
        # 这叫做"依赖注入"：block 只负责组装，不关心内部实现
        self.attention = attention

        self.norm2 = RMSNorm(dim)
        self.feed_forward = feed_forward

    def forward(self, x):
        """
        参数:
            x: (B,T,D)

        返回:
            (B,T,D)

        分步说明：
            (1) 先 norm，再 attention，最后加到 x 上     ← 第一条残差
            (2) 先 norm，再 ffn，最后加到上一步结果上    ← 第二条残差
        """
        # 第一条支路：x = x + Attention(RMSNorm1(x))
        # 注意 Attention 读的是【归一化后的】x，而加到的是【原始的】x。
        # 这就是 Pre-Norm 的核心：归一化只作用于支路内部，不污染主干。
        x = x + self.attention(self.norm1(x))

        # 第二条支路：x = x1 + FFN(RMSNorm2(x1))
        # 【关键细节】第二个 RMSNorm 读取的是 x1（上一步更新后的结果），
        # 而不是最初的 x。第一条支路已经更新了表示，FFN 接着处理这份新结果。
        # 写成 self.norm2(x) 时，x 已经是被第一条语句重新绑定过的新值 ——
        # 所以这行代码天然就是对的，但要意识到这一点。
        x = x + self.feed_forward(self.norm2(x))

        return x


# ============================================================================
# 第6步：堆叠 block
# ============================================================================
class TransformerStack(nn.Module):
    """把 N 个 decoder block 串起来。

    参数:
        blocks: block 模块的列表/可迭代对象
    """

    def __init__(self, blocks):
        super().__init__()

        # nn.ModuleList 是一个"能正确注册子模块的列表"。
        # 为什么不能直接用 Python list？
        #   普通 list 里的 nn.Module 不会被注册，因此：
        #     model.parameters()  找不到它们 → 优化器不会更新
        #     model.state_dict()  存不下它们 → checkpoint 会缺参数
        #     也不会跟着 .to(device) 一起搬到 GPU
        # ModuleList 解决了全部这些问题。
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        """
        参数:
            x: (B,T,D)

        返回:
            (B,T,D)

        说明:
            每层都保持 (B,T,D)，参数默认【不共享】——
            列表中是多个不同的 block 对象。
            如果重复放入同一实例（比如 [blk]*4），就会变成跨层权重共享，
            那时参数量会骤减但表达力大幅下降。这是一个容易踩的坑。
        """
        # 逐层前向：数据依次流过每个 block
        # 因为每层输入输出 shape 相同，所以可以简单地循环
        for block in self.blocks:
            x = block(x)

        return x


# ============================================================================
# 第7步：这一个脚本演示了什么、没演示什么
# ============================================================================
# TransformerBlock 的构造函数接收两个现成模块：
#
#     block = TransformerBlock(
#         dim,
#         attention=my_attention,
#         feed_forward=my_ffn,
#     )
#
# 这样可以把"两个 norm、两个 sublayer、两次 residual 的组装"
# 与"attention 内部实现"分开观察 —— 本节只关心前者。
#
# ── 重要提醒：本脚本的 smoke test 用了假的 attention ──────────────────────
# 脚本底部用 nn.Linear 与一个小 nn.Sequential 做 smoke test；
# 它没有 RoPE、GQA 或 causal mask。
#
# Task 27 的 DecoderBlock 才固定接入 CausalSelfAttention 和 SwiGLU。
#
# 因此，这里的脚本成功运行【只表明 block 组装方式正常】，
# 不包含对 attention 性质的任何结论。
# 这是一个很值得学习的写法：把"组装"和"组件"分开测试，出错时能立刻定位是哪一层。
# ============================================================================


# ============================================================================
# 第8步：运行与核对
# ============================================================================
#     python exercises/block_03_transformer/task_26_decoder_blocks/transformer_blocks.py
#
# 输出形如：
#
#     stack output: (2, 5, 16)
#     normalization: RMSNorm
#
# Block 组装性质也收录在 Block 3 测试中：
#
#     python -m unittest discover -s tests -p 'test_block3.py' -v
#
# 运行结果与测试覆盖：
#   - RMSNorm 的最后一维计算是否正确；
#   - norm1 与 norm2 是否【独立】（不是同一个对象）；
#   - 两条 residual 的输入来源是否正确（第二条读的是 x1）；
#   - stack 前后的 shape 与反向传播。
#
# Causal、RoPE 与 GQA 在 Task 27 的完整 block 中统一观察。
#
# 参考：Root Mean Square Layer Normalization  https://arxiv.org/abs/1910.07467
# ============================================================================


if __name__ == "__main__":
    # ── 用列表推导式构造 2 个 block ────────────────────────────────────────
    # [ ... for _ in range(2)]：列表推导式，把下面的表达式重复求值 2 次。
    # 下划线 _ 是惯例写法，表示"循环变量用不到"，只是单纯想跑两遍。
    #
    # 每次迭代都会【新建】一个 TransformerBlock 实例，
    # 因此两个 block 的参数是独立的（不是同一个对象被放两次）。
    blocks = [
        TransformerBlock(
            16,   # dim = 16

            # attention 位置注入一个假的模块：nn.Linear(16,16)
            # 它满足"输入 (B,T,D) → 输出 (B,T,D)"的接口，所以能装进去，
            # 但它本质上只是逐 token 的线性变换，没有任何 token 间交互 ——
            # 这正是本节"只验证组装、不验证 attention"的意思。
            attention=nn.Linear(16, 16, bias=False),

            # feed_forward 位置注入一个小型 MLP：16 → 32 → 16
            #   nn.Linear(16, 32)  升维
            #   nn.SiLU()          平滑激活（与 SwiGLU 用的是同一个激活）
            #   nn.Linear(32, 16)  降回 D，保证残差相加 shape 对得上
            # 注意这是普通两层 FFN，不是 SwiGLU（少了一条 up 分支，见 05 节）。
            feed_forward=nn.Sequential(nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 16)),
        )
        for _ in range(2)   # 重复 2 次，得到 2 层
    ]

    # 模拟输入：2 个样本、5 个 token、16 维
    values = torch.randn(2, 5, 16)

    # 依次经过两个 block，输出应保持 (2,5,16) —— 这是能继续堆叠的前提
    print("stack output:", tuple(TransformerStack(blocks)(values).shape))

    # 明确打印出本节使用的归一化方案，避免与 LayerNorm 混淆
    print("normalization: RMSNorm")


# ============================================================================
# 补充知识点：归一化方案对照表
# ============================================================================
#   方案         公式                             是否有均值  是否有 bias  位置
#   ──────────   ──────────────────────────────   ──────────   ──────────   ──────
#   LayerNorm    (x-μ)/√(σ²+ε) ⊙ w + b           是           是           任意
#   RMSNorm      x/√(mean(x²)+ε) ⊙ w             否           否           Pre
#   BatchNorm    沿 batch 维统计                  是           是           少用于 NLP
#
# ── RMSNorm 为什么可以去掉均值 ────────────────────────────────────────────
# 论文的假设是：LayerNorm 的有效性主要来自【缩放不变性】（re-scaling
# invariance），而重新中心化（re-centering）的贡献有限。
# 去掉均值只保留缩放，效果接近但计算更省 —— 这是一次典型的"砍掉低贡献项"的优化。
#
# ── 与注意力的一个呼应 ────────────────────────────────────────────────────
# 注意 04 节里 Q/K 也做了缩放（除以 √d_head）：
# 两者都在控制"数值量级"，只不过一个针对 score，一个针对 hidden state。
# 这解释了为什么 Transformer 里到处是 √ 和 eps —— 深层网络的数值稳定性
# 是靠这些不起眼的小设计堆出来的。
# ============================================================================


# ============================================================================
# 核心公式回顾
# ============================================================================
# (1) Pre-Norm 残差（本章结构）
#
#         x₁ = x  + Attention(RMSNorm₁(x))
#         x₂ = x₁ + FFN(RMSNorm₂(x₁))
#
# (2) Post-Norm（对照）
#
#         x₁ = LayerNorm(x + Attention(x))
#         x₂ = LayerNorm(x₁ + FFN(x₁))
#
# (3) RMSNorm
#
#                        x
#         RMSNorm(x) = ─────────────────── ⊙ w
#                      √( mean(x²) + ε )
#
#         mean 沿最后一维（特征维 D）计算，weight 形状 (D,)
#
# (4) LayerNorm（对照）
#
#                        x - μ
#         LayerNorm(x) = ─────────────── ⊙ w + b
#                        √(σ² + ε)
#
# (5) 混合精度的计算顺序
#
#         x → 转 fp32 → 算均方根 → 归一化 → 乘 w → 转回原 dtype
# ============================================================================


# ============================================================================
# 总结
# ============================================================================
# 1. 一个 decoder block 就两行：
#       x = x + Attention(Norm(x))
#       x = x + FFN(Norm(x))
#    加上最后的 final norm，重复 N 次 —— 这就是 GPT/LLaMA 的整个主干。
#
# 2. Pre-Norm 与 Post-Norm 的区别只在归一化的位置，
#    但 Pre-Norm 让残差成为一条"梯度高速路"，因此深层更稳、更容易训练。
#
# 3. RMSNorm = 只保留 LayerNorm 的缩放、去掉中心化，参数只剩一份 w。
#    它没有 running statistics，所以 train/eval 行为一致，不需要切换模式。
#
# 4. 两个 norm 必须【独立】（不同实例）。共享会让表达力下降，且不会报错 ——
#    这类"静默错误"要靠测试（比较对象身份）来抓。
#
# 5. 第二个 norm 读的是 x₁ 而不是最初的 x，这是残差堆叠的自然结果；
#    写代码时靠"重新绑定同名变量"实现，读代码时要意识到这一点。
#
# 6. 用 nn.ModuleList 而不是 Python list 来存子模块，
#    否则参数不会被注册，优化器和 checkpoint 都会漏掉它们。
#
# 7. 本节用假 attention 做 smoke test 是刻意为之：
#    分开验证"组装"与"组件"，出错时能立刻判断问题出在哪一层。
# ============================================================================
