"""
=============================================================================
Causal 自注意力与 GQA（Grouped-Query Attention）—— 完整带注释版
=============================================================================
源文件: exercises/block_03_transformer/task_23_causal_attention/mha.py

任务：把 Q/K/V 投影、RoPE、两种 mask、多头拼接和输出投影，接成一层可运行的
     causal self-attention。这是 decoder block 里"位置之间交换信息"的那一半。

  核心：scores = Q·Kᵀ / √d_head
        allowed = causal_mask(下三角) & padding_mask
        weights = softmax(scores + M)      ← mask 必须在 softmax 之前
        out     = weights · V

  GQA：保留 query head 的数量，减少 K/V head 的数量。
       n_heads=4, n_kv_heads=2 时，Q0,Q1 共享 KV0，Q2,Q3 共享 KV1。
       共享的是 K/V，不是 query 输出；最终仍有 4 个输出。

输入：x: (B,T,D)
输出：out: (B,T,D)   —— 可直接与 block 输入做残差相加

依赖：math、torch、torch.nn。本文件可独立运行。
=============================================================================
"""

# ============================================================================
# 第1步：导入依赖
# ============================================================================
import math                       # math.sqrt 用于缩放；math.isfinite 校验 rope_base

import torch                      # tril / ones / softmax / masked_fill 等
from torch import nn              # 线性层nn.Linear、基类nn.Module、参数nn.Parameter


# ============================================================================
# 第2步：从 Q/K/V 到输出
# ============================================================================
# 单个 head 在位置 t（query）、位置 j（key）之间的 score 是：
#
#                    q_t · k_jᵀ
#         s_{t,j} = ─────────────
#                    √d_head
#
# 先把禁止位置从 score 中排除，再沿 key 轴做 softmax：
#
#         a_{t,j} = softmax_j( s_{t,j} + M_{t,j} )
#         o_t     = Σ_j a_{t,j} · v_j
#
# ── 一个必须记住的细节：mask 不能等到输出之后才乘 ─────────────────────────
# 若被禁止的 score 参加了 softmax，它仍会分走一部分概率，
# 留下来的权重也会随之改变。
#
# 反例（错误的做法）：
#     weights = softmax(scores)        # 未来位置的 score 也参与了归一化
#     weights = weights * allowed      # 再乘掉 → 每行权重和 < 1，且比例已被污染
#
# 正确做法：softmax 之前就把禁止位置的 score 填成 -∞（实现用 dtype 最小值），
# 这样 exp(-∞)=0，它们对分母的贡献恰好为 0。
#
# ── forward 的 shape 变化（以 D=32, Hq=4, Hkv=2 为例）────────────────────
#
#         head_dim = 32 / 4 = 8
#
#         Q: (B,T,32) -> (B,4,T,8)
#         K: (B,T,16) -> (B,2,T,8)      ← 注意输入是 16 = n_kv_heads × head_dim
#         V: (B,T,16) -> (B,2,T,8)
#
#     Q/K 先应用 RoPE（RoPE 在拆头之后、repeat_kv 之前做，见下）
#     K/V 再沿 head 轴复用给 4 个 query heads
#
#         scores:  (B,4,T,T)
#         weights: (B,4,T,T)
#         heads:   (B,4,T,8)
#         concat:  (B,T,32)
#         output:  (B,T,32)
#
# 输出回到 (B,T,D) 后，可以直接和 block 输入做残差相加。
# ============================================================================


# ============================================================================
# 第3步：MultiHeadSelfAttention 类
# ============================================================================
class MultiHeadSelfAttention(nn.Module):
    """Causal self-attention with RoPE and grouped-query attention (GQA).

    带 RoPE 与分组查询注意力的因果自注意力层。

    参数:
        dim:         模型维度 D
        num_heads:   query head 数量 H（也叫 n_heads）
        num_kv_heads: K/V head 数量 Hkv；默认 None 表示等于 num_heads，即退化为标准 MHA
        rope_base:   RoPE 的频率底数
    """

    def __init__(self, dim, num_heads, num_kv_heads=None, rope_base=10000):
        # nn.Module 的标准写法：先调用父类初始化，注册好内部的参数/子模块机制
        super().__init__()

        # num_kv_heads 不传时就等于 num_heads，此时 kv_repeats=1，就是标准 MHA。
        # 这样同一个类既支持 MHA 也支持 GQA，不用写两份代码。
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        # ── 入参校验：三项整除关系 + 数值合法性，一次检查完 ──────────────
        if num_heads <= 0 or num_kv_heads <= 0:
            raise ValueError("num_heads and num_kv_heads must be positive")
        if not math.isfinite(rope_base) or rope_base <= 0:
            # 与 RoPE 一节的检查一致：base 必须为正且有限
            raise ValueError("rope_base must be positive and finite")
        if dim % num_heads != 0:
            # 否则 D/H 不是整数，拆不出等宽的头
            raise ValueError("dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            # GQA 要求每个 KV head 被整数个 query head 共享，
            # 否则分组会不整齐（4 个 Q 分给 3 个 KV 组就没法均分）
            raise ValueError("num_heads must be divisible by num_kv_heads")

        # ── 把派生尺寸都算好并缓存下来 ────────────────────────────────────
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads      # 单个 head 的宽度 Dh

        if self.head_dim % 2 != 0:
            # RoPE 是两维一个平面，head_dim 必须偶数
            raise ValueError("head_dim must be even for RoPE")

        # 每个 KV head 要被几个 query head 共享：4/2 = 2
        self.kv_repeats = num_heads // num_kv_heads
        self.rope_base = rope_base

        # ── 四组线性投影（全部 bias=False）────────────────────────────────
        # Q 投影输出 n_heads 个头的宽度
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)

        # K/V 投影只输出 n_kv_heads 个头的宽度 —— 这就是 GQA 省参数的地方：
        #   MHA 时 K/V 各需 D×D 参数
        #   GQA 时 K/V 各只需 D×(Hkv·Dh) = D×D×(Hkv/H)，按比例缩小
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)

        # 输出投影：把拼接后的 (B,T,D) 再混合一次。
        # 若没有它，各个 head 的结果只是简单拼接，head 之间无法交流。
        self.out_proj = nn.Linear(dim, dim, bias=False)

    # ========================================================================
    # 第3.1步：_rope —— 把 RoPE 内联进来（静态方法）
    # ========================================================================
    @staticmethod
    # @staticmethod 表示这个方法不需要访问 self，可以直接用类名调用。
    # 写成静态方法是因为它纯粹是数学运算，不依赖这一层的任何状态。
    def _rope(x, start_pos=0, base=10000):
        """对拆好头的 Q/K 施加 RoPE 旋转，等价于 03 节里的 build_rope_cache + apply_rope。

        参数:
            x:         (B, H, T, Dh)
            start_pos: 起始位置，KV Cache 解码时传 past_len
            base:      频率底数

        返回:
            (B, H, T, Dh)，形状与 dtype 不变
        """
        # torch.arange(0, 最后一维, 2) 生成偶数下标 [0,2,...,Dh-2]，共 Dh/2 个
        # 这些就是各个二维平面的编号（步长 2，所以下标值已经是 2i 的形式）
        half = torch.arange(0, x.shape[-1], 2, device=x.device).float()

        # base ** (-half/Dh)：
        #   half 里存的是 2i，除 Dh 得 2i/Dh，取负号，再以 base 为底
        #   → θ_i = base^(-2i/Dh)，与 03 节的 inv_freq 完全一致
        # 直接写 base ** (...) 而不用 1.0/(...)，是因为这里只需要一个"倒数底数"形式
        inv_freq = base ** (-half / x.shape[-1])

        # 位置下标 [start_pos, start_pos+T-1]
        # x.shape[-2] 就是 T（倒数第二维是序列维）
        positions = torch.arange(
            start_pos, start_pos + x.shape[-2], device=x.device
        ).float()

        # 外积：(T,) ⊗ (Dh/2,) → (T, Dh/2)
        # angles[pos, i] = pos · θ_i
        angles = torch.outer(positions, inv_freq)

        # 把每列 cos 值重复两次并插两个 batch/head 轴 → (1,1,T,Dh)
        # .to(x.dtype) 保证与 x 精度一致（张量运算要求两边 dtype 相同）
        cos = torch.repeat_interleave(angles.cos(), 2, dim=-1).to(x.dtype)
        sin = torch.repeat_interleave(angles.sin(), 2, dim=-1).to(x.dtype)

        # 取最后一维的偶数位、奇数位，即每对 (x0, x1)
        even, odd = x[..., 0::2], x[..., 1::2]

        # stack((-odd, even), dim=-1) → (..., Dh/2, 2)
        # flatten(-2)                 → (..., Dh)，即按对交错的 (-x1, x0)
        # 与 03 节的 rotate_half 是同一个操作，只是写在了同一行
        rotated = torch.stack((-odd, even), dim=-1).flatten(-2)

        # 旋转公式：x·cos + rotate_half(x)·sin
        # cos[None,None] 把 (T,Dh) 变成 (1,1,T,Dh)，广播到 (B,H,T,Dh)
        return x * cos[None, None] + rotated * sin[None, None]

    # ========================================================================
    # 第3.2步：_repeat_kv —— 把 K/V heads 复制给所有 query heads
    # ========================================================================
    def _repeat_kv(self, x):
        """把 (B, Hkv, T, Dh) 展开成 (B, H, T, Dh)，不新增任何参数。

        参数:
            x: K 或 V，shape (B, Hkv, T, Dh)

        返回:
            (B, H, T, Dh)，其中每个 KV head 被复制 kv_repeats 次
        """
        if self.kv_repeats == 1:
            # 标准 MHA（Hkv == H）时无事可做，直接返回，省一次拷贝
            return x

        # 解包四个维度：b=batch, h=KV head 数, t=序列长度, d=head_dim
        b, h, t, d = x.shape

        # x[:, :, None]                 → (B, Hkv, 1, T, Dh)   在第 2 维插一个轴
        # .expand(b, h, kv_repeats, t, d) → (B, Hkv, kv_repeats, T, Dh)
        #   expand 是【零拷贝】的广播视图：不分配新内存，只是把 stride 设为 0
        # .reshape(b, num_heads, t, d)  → (B, H, T, Dh)
        #   把 Hkv 与 kv_repeats 两维合并，正好等于 H
        #   顺序上 KV0 出现 kv_repeats 次、接着 KV1 出现 kv_repeats 次……
        #   即 Q0,Q1→KV0；Q2,Q3→KV1 ✓
        # reshape 遇到 expand 出来的非连续张量时会自动先拷贝一份，所以能正常工作
        return (
            x[:, :, None]
            .expand(b, h, self.kv_repeats, t, d)
            .reshape(b, self.num_heads, t, d)
        )

    # ========================================================================
    # 第3.3步：forward —— 一次完整的前向
    # ========================================================================
    def forward(self, x, causal=True, attention_mask=None):
        """
        参数:
            x:              (B,T,D) 输入 hidden states
            causal:         是否施加下三角 causal mask（训练/生成时都是 True）
            attention_mask: (B,T) 的布尔张量，True 表示该位置的 key 可见；
                            用来屏蔽 PAD（padding mask）

        返回:
            (B,T,D) —— 可直接与输入做残差相加
        """
        # 解包：b=batch, t=序列长度, d=模型维度
        b, t, d = x.shape

        # ── (1) 三组投影 + 拆头 ──────────────────────────────────────────
        # .view(b, t, num_heads, head_dim) 把 (B,T,H*Dh) 拆成 (B,T,H,Dh)
        #   —— 相邻的 Dh 个数归一个 head（这正是"一组投影拆成多头"的含义）
        # .transpose(1, 2) 交换 T 和 H 两维 → (B,H,T,Dh)
        #   换到前面是为了后面能直接用 (B,H,T,T) 的批量矩阵乘法一次算完所有 head
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # ── (2) RoPE：只作用于 Q 和 K，V 不旋转 ─────────────────────────
        # 位置信息只影响"匹配分数"，而匹配分数由 Q·Kᵀ 决定
        q = self._rope(q, base=self.rope_base)

        # K 先旋转再 repeat_kv。顺序可以对调（旋转与复制互不干扰），
        # 但"先旋转后复制"更自然：旋转只需要做 Hkv 份，省一点计算。
        k = self._repeat_kv(self._rope(k, base=self.rope_base))

        # V 不旋转，直接复制到与 query head 数量一致
        v = self._repeat_kv(v)

        # ── (3) 计算 score 并缩放 ───────────────────────────────────────
        # k.transpose(-2, -1)：(B,H,T,Dh) → (B,H,Dh,T)，最后两维互换
        # q @ kᵀ： (B,H,T,Dh) @ (B,H,Dh,T) → (B,H,T,T)
        #   scores[i,j] = 位置 i 的 query 与位置 j 的 key 的点积
        # 除以 √head_dim：控制 score 量级，避免 softmax 在高维下饱和（见 01 节）
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)

        # ── (4) 构造 allowed 矩阵：causal mask 与 padding mask 的交集 ────
        # torch.ones(t, t) 先造一个全 True 的 (T,T) 布尔矩阵
        allowed = torch.ones(t, t, device=x.device, dtype=torch.bool)

        if causal:
            # torch.tril 取下三角（含对角线）：
            #   位置 i 只能看 j <= i，即"不能看未来"
            #   这就是 01 节里 M 的实现方式 —— 只不过这里用布尔矩阵表示可见性，
            #   而不是用 0/-∞ 的加法掩码，两者等价
            allowed = torch.tril(allowed)

        # allowed[None, None]  → (1,1,T,T)
        # .expand(b,1,t,t)     → (B,1,T,T)
        #   expand 零拷贝，且第 1 维长度为 1 会自动广播到所有 head
        #   （所有 head 共用同一张 causal 表）
        allowed = allowed[None, None].expand(b, 1, t, t)

        if attention_mask is not None:
            # 约定：attention_mask 形状 (B,T)，True 表示该 key 位置可见
            if attention_mask.shape != (b, t):
                raise ValueError("attention_mask must have shape (batch, seq_len)")

            # 转成 bool 并放到同一设备；.to(dtype=torch.bool) 会把 0/1 解释成 False/True
            mask = attention_mask.to(device=x.device, dtype=torch.bool)

            # mask[:, None, None, :]：(B,T) → (B,1,1,T)
            #   【只屏蔽 key 维】—— 这正是我们要的：
            #   它不删除 PAD query 行（输出仍是 (B,T,D)），
            #   只是让所有 query 都读不到 PAD 位置的 key。
            #   到了语言模型里，PAD query 对应的 target 还会从 loss 中排除。
            # 用 & 取交集：两个条件都满足才允许（既是下三角、又不是 PAD）
            allowed = allowed & mask[:, None, None, :]

        # ── (5) 施加 mask 并做 softmax ──────────────────────────────────
        # torch.finfo(scores.dtype).min 是该 dtype 能表示的最小有限值
        #   （float32 约 -3.4e38）。
        # 为什么不用 -inf？因为 -inf 参与运算容易产生 nan（例如 -inf 与 0 相乘）。
        # 用最小有限值代替：exp(min - max) 在实践中会下溢到 0，效果与 -inf 相同。
        # ~allowed 是布尔取反：把【不允许】的位置填成最小值。
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

        # 沿最后一维（key 方向）做 softmax：
        #   每行变成和为 1 的权重分布
        #   被填成最小值的项 exp 后约等于 0，几乎不占概率
        attn = torch.softmax(scores, dim=-1)

        # ── (6) 处理"全屏蔽行" ──────────────────────────────────────────
        # 若某个样本的所有 key 都不可见（整行都是 PAD），
        # 上面的 softmax 会把一个均匀分布（或者由最小值决定的分布）算出来，
        # 而不是 0。这里再乘一次 allowed，把该行权重强制归零，
        # 于是它的注意力输出恒为 0 —— 有限值，不会产生 NaN。
        # .to(attn.dtype) 把 bool 转成 0.0/1.0 才能做乘法。
        # 这是本教学实现的【显式约定】；生产实现通常依赖 mask 的广播语义来处理。
        attn = attn * allowed.to(attn.dtype)

        # ── (7) 加权求和 ────────────────────────────────────────────────
        # attn @ v： (B,H,T,T) @ (B,H,T,Dh) → (B,H,T,Dh)
        #   out[i] = Σ_j attn[i,j] · v[j]，即"按相关性加权取回内容"
        out = attn @ v

        # ── (8) 合并多头并做输出投影 ────────────────────────────────────
        # .transpose(1, 2)：把 (B,H,T,Dh) 换回 (B,T,H,Dh)，即把 head 维挪到序列维后面
        # .contiguous()：transpose 得到的是非连续张量，view 要求内存连续，
        #                所以先显式拷贝一份（这一步不能让 reshape 自动处理，
        #                因为我们需要保证合并顺序是"按 head 顺序首尾相接"）
        # .view(b, t, d)：把 (H, Dh) 合并回 D = H × Dh → (B,T,D)
        out = out.transpose(1, 2).contiguous().view(b, t, d)

        # 最后过一层输出投影，让不同 head 的信息真正混合
        return self.out_proj(out)


# ============================================================================
# 第4步：GQA 共享的是 K/V，不是 query 输出
# ============================================================================
# 标准 MHA 为每个 query head 各准备一组 K/V。GQA 保留 query heads 的数量，
# 只减少 K/V heads：
#
#     n_heads = 4
#     n_kv_heads = 2
#     kv_repeats = 2
#
#     Q0,Q1 -> KV0
#     Q2,Q3 -> KV1
#
# 图中每个 Q head 只配一组 K/V，最终仍有 O0..O3 四个输出 —— 这一点务必分清：
#   - query 侧没有任何减少，每个 Q head 仍独立地"提问"；
#   - 减少的只是"索引"和"内容"的份数，多个 Q 去查同一份 K/V。
#
# ── _repeat_kv 只做 expand/reshape，不会创建新的参数 ──────────────────────
# 参数只存在于 k_proj / v_proj 里，而它们的输出维度就是 n_kv_heads × head_dim。
# 原始 K/V 投影和后续 KV Cache 都只保存 2 个 heads —— 这正是 GQA 的实际收益：
#   参数量、KV Cache 显存、内存带宽，都按 Hkv/H 的比例减少。
#
# ── 三项整除关系（构造函数里已经检查）────────────────────────────────────
#
#     D % n_heads == 0          拆头要均分
#     n_heads % n_kv_heads == 0 分组要均分
#     head_dim % 2 == 0         RoPE 成对旋转
#
# 非法配置会在【建层时】直接报错，错误位置比 view 或矩阵乘法中的 shape 异常更明确。
# 这一点在调试时很值钱：shape 异常往往发生在离真正原因很远的地方。
# ============================================================================


# ============================================================================
# 第5步：两张 mask，各管一件事
# ============================================================================
#   causal mask   shape 可广播为 (1,1,T,T)  → 负责时间方向（不能看未来）
#   attention_mask shape (B,T) → (B,1,1,T)  → 随样本变化，负责屏蔽 PAD
#
# 两者取交集后才是最终允许矩阵。
#
# ── 几个容易混淆的点 ──────────────────────────────────────────────────────
# (1) 这里的 padding mask 只屏蔽 【key】。
#     它不删除 PAD query 行，输出仍是 (B,T,D)；
#     到语言模型中，PAD query 对应的 target 还会从 loss 中排除（见 06/09 节）。
#     也就是说，"屏蔽 key"影响的是【表示】，"屏蔽 target"影响的是【监督信号】，
#     两者作用不同，缺一不可。
#
# (2) 两张 mask 不能互相替代：
#     causal mask 回答"能不能看未来"，padding mask 回答"能不能读填充值"。
#     只有 causal，模型会把 PAD 当成正常 token 读进去；
#     只有 padding，模型会在训练时偷看后面的答案。
#
# (3) 若一个样本没有任何可见 key，所有 score 都会被填成有限 dtype 的最小值。
#     softmax 后代码再次乘 allowed mask，把该行权重归零，避免 NaN。
#     这个处理是本教学实现的显式约定。
# ============================================================================


# ============================================================================
# 第6步：Causal 性质的数值核对
# ============================================================================
# 固定前缀，任意替换未来：
#
#     x2 = x.clone()
#     x2[:, 4:] = torch.randn_like(x2[:, 4:])
#
#     y1 = attention(x)
#     y2 = attention(x2)
#
# y1[:, :4] 与 y2[:, :4] 在浮点容差内一致 —— 这就是 causal 性质。
#
# 反向对照：改动【前文】通常会改变后面位置的输出。
#
# ── 一个很实用的排错经验 ─────────────────────────────────────────────────
# 如果末位 logits 始终不随前文变化，实际运行的可能仍是逐 token MLP：
# 也就是 attention 的输出被某种写法（比如 residual 绕开、或者维度接错）
# 排除在主干之外了。这条实验能直接把这类结构性错误揪出来。
# ============================================================================


# ============================================================================
# 第7步：运行与核对
# ============================================================================
# mha.py 末尾包含一个 smoke test：
#
#     python exercises/block_03_transformer/task_23_causal_attention/mha.py
#
# 预期输出：
#
#     output: (2, 6, 32)
#     Q heads / KV heads: 4 / 2
#
# 同一组性质也收录在 Block 3 测试中：
#
#     python -m unittest discover -s tests -p 'test_block3.py' -v
#
# 测试覆盖：输入输出 shape、n_kv_heads 对 K/V 投影数量的实际影响、
# 未来 token 对过去位置的隔离、padding mask 的有限输出，
# 以及非法 head 配置和 mask shape 的报错边界。
#
# ── 关于生产实现 ──────────────────────────────────────────────────────────
# mha.py 用基础张量操作展开计算，shape 变化因此比较直观，适合学习。
# 生产代码通常会使用 PyTorch 的 fused SDPA：
#
#     F.scaled_dot_product_attention(q, k, v, attn_mask=..., is_causal=True)
#
# 它把"缩放 → 加 mask → softmax → 加权求和"融成一个 kernel，
# 显存占用更低、速度更快（走 FlashAttention 等优化路径）。
# 换用其他实现时，mask 的布尔语义和 GQA 的 shape 约束仍然不变。
#
# 参考：GQA 论文 https://arxiv.org/abs/2305.13245
#       PyTorch scaled_dot_product_attention
#       https://docs.pytorch.org/docs/stable/generated/
#       torch.nn.functional.scaled_dot_product_attention.html
# ============================================================================


if __name__ == "__main__":
    # 建一层小注意力：D=32，4 个 query head，2 个 KV head
    #   head_dim = 32/4 = 8（偶数 ✓）
    #   kv_repeats = 4/2 = 2（Q0,Q1→KV0；Q2,Q3→KV1）
    layer = MultiHeadSelfAttention(dim=32, num_heads=4, num_kv_heads=2)

    # 模拟一个 batch：2 个样本、6 个 token、32 维
    values = torch.randn(2, 6, 32)

    # 前向：默认 causal=True、attention_mask=None
    # 输出应当回到 (B,T,D) = (2,6,32)，说明可以安全地做残差相加
    print("output:", tuple(layer(values).shape))

    # 打印 Q head 与 KV head 的数量，直观展示 GQA 的分组比例
    print("Q heads / KV heads:", layer.num_heads, "/", layer.num_kv_heads)


# ============================================================================
# 补充知识点：MHA / MQA / GQA 的取舍
# ============================================================================
#   方案    K/V head 数      参数量(K/V投影)   KV Cache 大小     效果
#   ─────   ──────────────   ───────────────   ──────────────   ──────────
#   MHA     H（等于 Q）       2·D·D             最大              最好，最慢
#   GQA     H/2 或 H/4 等     2·D·D·(Hkv/H)     按比例缩小        接近 MHA
#   MQA     Hkv = 1           2·D·D/H           最小（1 份）      略差，最快
#
# ── 为什么现代 LLM 普遍选 GQA ─────────────────────────────────────────────
# 推理时的瓶颈往往不是算力，而是【显存带宽】：每生成一个 token 都要把
# 整个 KV Cache 从显存读一遍。Cache 越小，能塞进的并发请求越多，吞吐越高。
# GQA 用一个连续可调的旋钮（Hkv）在"质量"和"显存"之间取平衡，
# 而 MQA 太激进、MHA 太贵，GQA 落在中间 —— 这是它成为默认选择的原因。
#
# ── 与 KV Cache 一节的直接联系 ────────────────────────────────────────────
# 11_KV_Cache.py 里 cache 的 shape 是 (B, n_kv_heads, past_len, head_dim)，
# 注意是 n_kv_heads 而不是 n_heads —— 存的是【未 repeat_kv】的 K/V。
# 若存展开后的版本，GQA 的省显存效果就完全没有了。
# ============================================================================


# ============================================================================
# 核心公式回顾
# ============================================================================
# (1) 单头 score 与加权
#
#                   q_t · k_jᵀ
#         s_{t,j} = ───────────── ,     a_{t,j} = softmax_j( s_{t,j} + M_{t,j} )
#                   √d_head
#
#         o_t = Σ_j a_{t,j} · v_j
#
# (2) 加法掩码 M
#
#         M_{t,j} = 0        当 j ≤ t（可见）
#         M_{t,j} = -∞       当 j > t（禁止）  → 实现用 finfo(dtype).min
#
# (3) 多头拼接
#
#         MultiHead(x) = W_o · [ head_1 ; head_2 ; … ; head_H ]
#         head_i       = Attention( x·Wq_i , x·Wk_i , x·Wv_i )
#
# (4) GQA 的 head 映射
#
#         kv_repeats = n_heads / n_kv_heads
#         第 i 个 query head 使用第 ⌊i / kv_repeats⌋ 个 KV head
#
# (5) expand 的"零拷贝"语义
#
#         x[:, :, None].expand(b, h, r, t, d).reshape(b, h*r, t, d)
#         expand 时 stride=0（不分配新内存），reshape 时才真正拷贝
# ============================================================================


# ============================================================================
# 总结
# ============================================================================
# 1. 这一层做的事情可以概括成五步：
#     投影拆头 → RoPE → 打分缩放 → mask+softmax → 加权合并 + 输出投影
#    输入输出都是 (B,T,D)，所以能像积木一样插进残差主干。
#
# 2. mask 必须在 softmax 之前生效，且要吃透"两张 mask 各管一件事"：
#      causal (1,1,T,T) 管时间，padding (B,1,1,T) 管填充，取交集才是 allowed。
#
# 3. GQA 只减少 K/V head 的数量，query head 一个不少。
#    _repeat_kv 用 expand/reshape 完成复用，不新增任何参数；
#    KV Cache 里存的也是未展开的版本，这才是省显存的关键。
#
# 4. 三项整除约束在建层时就检查，让错误尽早暴露：
#    D%H==0、H%Hkv==0、head_dim%2==0。
#
# 5. 用 -finfo(dtype).min 而不是 -inf 来填充被屏蔽位置，是为了避免 nan。
#    再用一次 "attn * allowed" 处理全屏蔽行，让输出保持有限值。
#
# 6. 验收这一层是否正确的两个实验：
#      未来不影响过去（固定前缀、改未来的 token，前缀输出不变）；
#      过去影响后来（改前文，末位 logits 会变）。
#    第二个实验能揪出"attention 被架空成逐 token MLP"这类结构性错误。
# ============================================================================
