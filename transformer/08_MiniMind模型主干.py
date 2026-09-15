"""
=============================================================================
MiniMind 模型主干 —— 一个完整的 decoder-only Transformer
=============================================================================
源文件: exercises/block_03_transformer/task_27_minimind_core/minimind_core.py

任务：把前几节各自演示的部件接成一台【可以训练】的微型 decoder-only Transformer。
     包含：RMSNorm、RoPE、causal GQA、SwiGLU、weight tying、padding mask、
          以及逐层 KV Cache 接口。

  数据流：
     input_ids (B,T)
       -> token embedding                       (B,T,D)
       -> N × DecoderBlock
            x = x + CausalSelfAttention(RMSNorm(x))
            x = x + SwiGLU(RMSNorm(x))
       -> final RMSNorm
       -> tied LM head                          (B,T,V)

  重要说明："模型主干齐全"只指数据流 —— RoPE、causal GQA、RMSNorm、SwiGLU、
  残差和 weight tying 都参与 forward。它仍是小型教学模型，
  参数量和训练数据都不能与通用 LLM 相提并论。

依赖：dataclasses、math、torch、torch.nn、torch.nn.functional。本文件可独立运行。
=============================================================================
"""

# ============================================================================
# 第1步：导入依赖与类型别名
# ============================================================================
# 文件顶部的这个字符串是【模块 docstring】，用 help(模块) 或 __doc__ 可以看到。
# 它比函数 docstring 更适合写"这个文件整体在做什么"。
from dataclasses import dataclass    # @dataclass 自动生成 __init__ 等样板代码
import math                          # isfinite 校验

import torch                         # cat / arange / tril 等
from torch import nn                 # 各类层与 nn.Module
from torch.nn import functional as F # F.silu / F.cross_entropy


# ── 类型别名 ──────────────────────────────────────────────────────────────
# KeyValue = tuple[torch.Tensor, torch.Tensor]
#   表示"一对 (K, V) 张量"。
#   tuple[...] 是 Python 3.9+ 的泛型写法，用来标注元组里每个元素的类型。
# 这个别名让函数签名里的 past_key_value: KeyValue | None 一眼就能读懂，
# 比写 tuple[torch.Tensor, torch.Tensor] | None 清爽得多。
# （| 是 Python 3.10+ 的联合类型语法，等价于 Optional[...]）
KeyValue = tuple[torch.Tensor, torch.Tensor]


# ============================================================================
# 第2步：MiniMindConfig —— 用 dataclass 集中管理所有尺寸
# ============================================================================
@dataclass
# @dataclass 装饰器会根据下面的类属性自动生成 __init__：
#   def __init__(self, vocab_size, dim=256, n_layers=4, ...)
# 于是建配置时可以直接写 MiniMindConfig(vocab_size=32, dim=32)，
# 不用手写一堆 self.xxx = xxx。
class MiniMindConfig:
    """模型尺寸配置。

    字段含义：
        vocab_size   词表大小 V
        dim          模型维度 D
        n_layers     DecoderBlock 数量
        n_heads      query heads 数量 H
        n_kv_heads   K/V heads 数量 Hkv（GQA）
        hidden_dim   SwiGLU 中间宽度
        max_seq_len  最大可见窗口
        rope_base    RoPE 的频率底数
        norm_eps     RMSNorm 稳定项
        pad_token_id PAD id；不需要时可设为 None
    """

    # ── 必填字段：没有默认值，建配置时必须给 ──────────────────────────────
    vocab_size: int

    # ── 可选字段：格式是 "名字: 类型 = 默认值" ─────────────────────────────
    # 类型标注（: int / : float）本身不强制检查，只是给人和工具看的说明，
    # 真正的检查由下面的 __post_init__ 完成。
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 8
    n_kv_heads: int = 4          # 只有 n_heads 的一半，典型的 GQA 配置
    hidden_dim: int = 768        # = 3 × dim，比 8/3·d 略大，教学取整方便
    max_seq_len: int = 256
    rope_base: float = 10000.0
    norm_eps: float = 1e-6

    # int | None 表示"整数或 None"。
    # 设为 None 就表示这个数据集没有 PAD（所有位置都有效）。
    pad_token_id: int | None = 0

    def __post_init__(self):
        """dataclass 在自动生成的 __init__ 末尾会调用这个方法（如果定义了）。

        为什么把所有校验放在这里？
        ────────────────────────────────────────────────────────────────
        配置创建时就会检查正整数、head 整除关系、偶数 head_dim、有限的
        rope_base/norm_eps 以及 PAD id 范围。

        这样错误不会拖到第一次 view 或矩阵乘法才出现 ——
        那时报的是 shape 异常，离真正的原因（比如 n_heads 写错了）已经很远，
        调试成本高得多。这是个非常值得养成的习惯：**校验输入，尽早失败**。
        """
        # ── 把需要"正整数"的字段收成一个字典，避免写 7 个 if ──────────────
        positive = {
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "hidden_dim": self.hidden_dim,
            "max_seq_len": self.max_seq_len,
        }
        # .items() 把字典拆成 (键, 值) 对，for 循环里用元组解包一次取两个
        for name, value in positive.items():
            # isinstance(value, int)        必须是整数
            # isinstance(value, bool)       【排除布尔】——
            #   Python 里 True == 1、isinstance(True, int) 也是 True，
            #   所以不显式排除的话，n_heads=True 会被当成 1 通过校验。
            #   这是个很隐蔽的坑，值得单独写一条。
            # value <= 0                    必须为正
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        # ── 三项整除关系 ──────────────────────────────────────────────────
        if self.dim % self.n_heads != 0:
            # 拆头要均分：D / H 必须是整数（head_dim）
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            # GQA 分组要均分：每个 KV head 被 n_heads/n_kv_heads 个 Q head 共享
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if (self.dim // self.n_heads) % 2 != 0:
            # RoPE 两维一个平面，head_dim 必须偶数
            raise ValueError("head_dim must be even for RoPE")

        # ── 数值合法性 ────────────────────────────────────────────────────
        if not math.isfinite(self.rope_base) or self.rope_base <= 0:
            raise ValueError("rope_base must be positive and finite")
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive and finite")

        # ── PAD id 的范围 ─────────────────────────────────────────────────
        if self.pad_token_id is not None:
            # 允许为 None（表示无 PAD），否则必须是 [0, vocab_size) 内的整数
            if (
                not isinstance(self.pad_token_id, int)
                or isinstance(self.pad_token_id, bool)     # 同样排除布尔
                or not 0 <= self.pad_token_id < self.vocab_size
                # 链式比较 0 <= x < V 是 Python 特有的写法，等价于
                # (0 <= x) and (x < V)，比两个条件写起来更清楚
            ):
                raise ValueError("pad_token_id must be an integer inside the vocabulary")


# ============================================================================
# 第3步：RMSNorm（与 07 节相同，此处为了文件自包含再写一份）
# ============================================================================
class RMSNorm(nn.Module):
    """均方根归一化：只缩放，不中心化。

    RMSNorm(x) = x / √(mean(x²)+ε) ⊙ w
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        if dim <= 0 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("dim and eps must be positive, with eps finite")
        # 可学习的缩放参数，初始化为全 1（训练开始时不额外缩放）
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # 记录输入 dtype，最后要转回去
        input_dtype = x.dtype

        # 半精度输入时用 fp32 计算，避免均方/求和累积误差
        compute_dtype = (
            torch.float32
            if input_dtype in (torch.float16, torch.bfloat16)
            else input_dtype
        )
        x_compute = x.to(dtype=compute_dtype)

        # x / √(mean(x²)+ε)：
        #   square() 逐元素平方 → mean(dim=-1, keepdim=True) 沿特征维求均值
        #   → rsqrt 开方取倒数 → 广播相乘
        normalized = x_compute * torch.rsqrt(
            x_compute.square().mean(dim=-1, keepdim=True) + self.eps
        )

        # Parameters normally remain float32 under mixed precision.  Cast only
        # after the affine scale so RMSNorm preserves the activation dtype.
        # （混合精度训练时参数通常保持 fp32；这里在仿射缩放之后才转 dtype，
        #   目的是让 RMSNorm 的输出 dtype 与输入保持一致。）
        return (normalized * self.weight.to(dtype=compute_dtype)).to(dtype=input_dtype)


# ============================================================================
# 第4步：RoPE 相关函数（模块级函数，不属于任何类）
# ============================================================================
def build_rope_cache(seq_len, head_dim, base=10000.0, device=None, dtype=None, start_pos=0):
    """Build distinct sin/cos frequencies for each pair of head dimensions.

    为每一对 head 维度构造【互相不同】的 sin/cos 频率。

    参数:
        seq_len:  需要多少个位置
        head_dim: 单个 head 的维度 Dh，必须正偶数
        base:     频率底数
        device:   目标设备
        dtype:    可选的输出精度
        start_pos: 起始位置；KV Cache 解码时必须传 past_len

    返回:
        cos, sin：均为 (seq_len, head_dim // 2)
    """
    # ── 四项入参校验 ──────────────────────────────────────────────────────
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError("head_dim must be a positive even integer")
    if start_pos < 0:
        raise ValueError("start_pos must be non-negative")
    if not math.isfinite(base) or base <= 0:
        raise ValueError("base must be positive and finite")

    # 偶数下标 [0,2,...,Dh-2]，即各二维平面的编号（值已经是 2i）
    # 显式写 dtype=torch.float32 而不用 .float()，效果相同但更直白
    pair_index = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)

    # θ_i = base^(-2i/Dh)：因为 pair_index 存的是 2i，除以 head_dim 即可
    inv_freq = base ** (-pair_index / head_dim)

    # 位置下标 [start_pos, start_pos+seq_len-1]，fp32
    positions = torch.arange(start_pos, start_pos + seq_len, device=device, dtype=torch.float32)

    # 外积 → (seq_len, Dh/2)，每个位置在每个平面上的旋转角
    angles = torch.outer(positions, inv_freq)

    # 同时求 cos 和 sin
    cos, sin = angles.cos(), angles.sin()

    # 可选转换精度
    if dtype is not None:
        cos, sin = cos.to(dtype=dtype), sin.to(dtype=dtype)

    # 只存 Dh/2 列：每个二维平面一个值，用的时候再 repeat_interleave 展开
    return cos, sin


def rotate_half(x):
    """把最后一维按 (x0,x1) 成对处理，返回 (-x1, x0)。

    这是把二维旋转写成逐元素乘法的关键一步。
    """
    # 取偶数位和奇数位 → 各 (..., Dh/2)
    even, odd = x[..., 0::2], x[..., 1::2]

    # stack 到新的最后一维 → (..., Dh/2, 2)，再 flatten(-2) 合并 → (..., Dh)
    # 结果是按对交错的 (-x1, x0)
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def apply_rope(x, cos, sin):
    """对 x 的最后一维施加 RoPE 旋转。

    参数:
        x:   (B, H, T, Dh) 的 Q 或 K
        cos, sin: (T, Dh//2)

    返回:
        与 x 同 shape 同 dtype
    """
    # 期望的 cache 形状：（序列长度, head_dim/2）
    expected = (x.shape[-2], x.shape[-1] // 2)

    # 一次性检查三件事：最后一维是偶数、cos 与 sin 形状匹配期望
    # 用 or 串起来，任一不满足就报错
    if x.shape[-1] % 2 or cos.shape != expected or sin.shape != expected:
        raise ValueError("RoPE cache shape does not match (seq_len, head_dim // 2)")

    # 每列重复两次与成对分量对齐 → (T,Dh)，再插两个轴 → (1,1,T,Dh)
    # 同时对齐设备与精度，避免 CPU/GPU 或 fp32/fp16 不匹配
    cos = torch.repeat_interleave(cos, 2, dim=-1)[None, None].to(
        device=x.device, dtype=x.dtype
    )
    sin = torch.repeat_interleave(sin, 2, dim=-1)[None, None].to(
        device=x.device, dtype=x.dtype
    )

    # 旋转公式：x·cos + rotate_half(x)·sin
    return x * cos + rotate_half(x) * sin


def repeat_kv(x, repeats):
    """Map ``n_kv_heads`` K/V heads to all query heads without new parameters.

    把 n_kv_heads 个 K/V head 映射到所有 query head，【不产生任何新参数】。

    参数:
        x:       (B, n_kv_heads, seq_len, head_dim)
        repeats: 每个 KV head 要被复制几次 = n_heads / n_kv_heads

    返回:
        (B, n_kv_heads * repeats, seq_len, head_dim)
    """
    # 必须恰好是 4 维，否则后面的解包 b,h,t,d 会错
    if x.ndim != 4:
        raise ValueError("x must have shape (batch, n_kv_heads, seq_len, head_dim)")

    # repeats 必须是正整数（同样排除 bool）
    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats <= 0:
        raise ValueError("repeats must be a positive integer")

    if repeats == 1:
        # 标准 MHA：无需复制，直接返回（省一次拷贝）
        return x

    # 解包四个维度
    b, h, t, d = x.shape

    # x[:, :, None]                  → (B, Hkv, 1, T, Dh)
    # .expand(b, h, repeats, t, d)   → (B, Hkv, repeats, T, Dh)  ← 零拷贝广播
    # .reshape(b, h * repeats, t, d) → (B, H, T, Dh)             ← 这时才真正拷贝
    # 合并顺序保证 Q0,Q1→KV0；Q2,Q3→KV1（当 repeats=2 时）
    return x[:, :, None].expand(b, h, repeats, t, d).reshape(b, h * repeats, t, d)


# ============================================================================
# 第5步：CausalSelfAttention —— 带 RoPE 与 GQA 的因果自注意力
# ============================================================================
class CausalSelfAttention(nn.Module):
    """decoder 自注意力：Q/K/V 全部来自同一序列，Q/K 过 RoPE，可见范围是前缀。

    与 04 节的区别：
      - 从 config 读取所有尺寸，而不是逐个传参；
      - 支持 past_key_value（KV Cache）与 use_cache；
      - causal mask 写成【非方阵】形式（query 与 key 位置不同长）。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()

        # 把常用尺寸存成属性（这里 config 已经校验过，无需重复检查）
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads       # Dh
        self.kv_repeats = config.n_heads // config.n_kv_heads
        self.rope_base = config.rope_base

        # 四组无 bias 投影（LLaMA 风格）
        # Q 输出 n_heads 份，K/V 只输出 n_kv_heads 份 —— GQA 省参数之处
        self.q_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)

    def forward(
        self,
        x,
        attention_mask=None,
        past_key_value: KeyValue | None = None,
        use_cache=False,
    ):
        """
        参数:
            x:              (B,T,D) 当前的输入（缓存在场时 T 通常为 1）
            attention_mask: (B,T) 或 (B,past_len+T) 的布尔掩码，True 表示可见
            past_key_value: 该层的历史 (K,V)，或 None
            use_cache:      是否返回更新后的 (K,V)

        返回:
            use_cache=True  → (out, (K,V))
            use_cache=False → out

        分步说明 / 计算过程：
            (1) Q/K/V 投影 + 拆头
            (2) 校验并解包历史 K/V，得到 past_len
            (3) 构造 RoPE（start_pos = past_len）并旋转 Q/K
            (4) 拼接历史 K/V，得到 present
            (5) 构造【非方阵】causal mask，叠加 padding mask
            (6) repeat_kv 展开、打分、softmax、加权
            (7) 合并多头 + 输出投影
        """
        # 用下划线接住 D（这里用不到，因为投影层自己知道维度）
        b, t, _ = x.shape

        # ── (1) 投影 + 拆头 ──────────────────────────────────────────────
        # view(B,T,H,Dh).transpose(1,2) → (B,H,T,Dh)
        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # ── (2) 校验历史 K/V ─────────────────────────────────────────────
        past_len = 0                     # 历史长度，后面多处要用
        if past_key_value is not None:
            # 元组解包，取出历史 K 和 V
            past_k, past_v = past_key_value

            # past_k.shape[:2] 是 (B, n_kv_heads) —— 前两维必须与当前一致
            # past_v.shape != past_k.shape —— K/V 必须等长
            if past_k.shape[:2] != (b, self.n_kv_heads) or past_v.shape != past_k.shape:
                raise ValueError("past K/V must have shape (batch, n_kv_heads, past_len, head_dim)")
            if past_k.shape[-1] != self.head_dim:
                raise ValueError("past K/V head_dim does not match the model")

            # 倒数第二维就是历史长度
            past_len = past_k.shape[-2]

        # ── (3) RoPE：位置从 past_len 开始（这是最关键的一处）────────────
        # 若缓存已有 past_len 个 token，新 token 的位置就是 past_len，
        # 而不是重新从 0 开始。漏掉这个偏移，shape 全对但 logits 会悄悄偏离。
        cos, sin = build_rope_cache(
            t,                       # 只需要 t 个位置（缓存在场时 t=1）
            self.head_dim,
            base=self.rope_base,
            device=x.device,
            dtype=x.dtype,           # 与 Q/K 精度对齐
            start_pos=past_len,      # ← 位置接在缓存后面
        )

        # 只旋转 Q 和 K，V 不动（位置只影响匹配分数）
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # ── (4) 拼接历史 K/V ─────────────────────────────────────────────
        if past_key_value is not None:
            # torch.cat 沿倒数第二维（序列维）拼接：
            #   (B,Hkv,past_len,Dh) 与 (B,Hkv,t,Dh) → (B,Hkv,past_len+t,Dh)
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        # present 无论是否用缓存都要构造（use_cache=False 时只是不被使用）
        present = (k, v)

        # ── (5) 构造 allowed 矩阵（非方阵！）─────────────────────────────
        # 关键：query 有 t 个（当前 token），key 有 total_len 个（含历史）
        total_len = past_len + t

        # q_positions: (t,1) —— 当前 query 的【绝对】位置 [past_len, past_len+t)
        # [:, None] 在最后一维插轴 → (t,1)
        q_positions = torch.arange(past_len, total_len, device=x.device)[:, None]

        # k_positions: (1,total_len) —— 所有 key 的绝对位置 [0, total_len)
        # [None, :] 在最前面插轴 → (1,total_len)
        k_positions = torch.arange(total_len, device=x.device)[None, :]

        # 广播比较：(t,1) <= (1,total_len) → (t,total_len)
        # 第 i 行第 j 列为 True 表示"第 i 个 query 可以看到第 j 个 key"
        # 这是 causal 条件在"query/key 长度不同"时的正确写法：
        #   比较的是【绝对位置】，而不是简单的下三角。
        #   例如 prefill 5 个 token 后解码第 6 个：q_pos=5，可看 k_pos 0..5 ✓
        # [None, None] → (1,1,t,total_len)，.expand → (B,1,t,total_len)
        allowed = (k_positions <= q_positions)[None, None].expand(b, 1, t, total_len)

        if attention_mask is not None:
            # 转 bool 并同设备
            attention_mask = attention_mask.to(device=x.device, dtype=torch.bool)

            # ── 一个容易踩的坑：缓存在场时 mask 必须覆盖前缀 ──────────────
            # 调用方若只传了当前这一步的 (B,t) mask，历史前缀的信息就丢了。
            # 这里自动在左边补上全 True 的前缀，表示"历史 token 都有效"。
            if attention_mask.shape == (b, t) and past_len:
                prefix = torch.ones((b, past_len), device=x.device, dtype=torch.bool)
                attention_mask = torch.cat((prefix, attention_mask), dim=1)

            # 补完之后长度必须正好等于 total_len
            if attention_mask.shape != (b, total_len):
                raise ValueError("attention_mask must cover current tokens and cached prefix")

            # (B,total_len) → (B,1,1,total_len)，只屏蔽 key 维，与 causal 取交集
            allowed = allowed & attention_mask[:, None, None, :]

        # ── (6) repeat_kv + 打分 + 加权 ─────────────────────────────────
        # 把 Hkv 份 K/V 展开成 H 份，以匹配 query head 数量
        expanded_k = repeat_kv(k, self.kv_repeats)
        expanded_v = repeat_kv(v, self.kv_repeats)

        # (B,H,t,Dh) @ (B,H,Dh,total_len) → (B,H,t,total_len)
        # 除以 √head_dim 控制量级
        scores = q @ expanded_k.transpose(-2, -1) / math.sqrt(self.head_dim)

        # 屏蔽位置填成 dtype 最小值（避免 -inf 带来的 nan）
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

        # softmax 沿 key 方向；先转 fp32 再转回来，减少低精度下的数值误差
        weights = torch.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)

        # This also makes an all-padding row finite: its attention output is zero.
        # （这一步同时让"整行都是 PAD"的行得到有限值：它的注意力输出为 0。）
        weights = weights * allowed.to(weights.dtype)

        # 加权求和：(B,H,t,total_len) @ (B,H,total_len,Dh) → (B,H,t,Dh)
        out = weights @ expanded_v

        # ── (7) 合并多头 + 输出投影 ─────────────────────────────────────
        # transpose 回 (B,t,H,Dh)，contiguous 让内存连续，view 合并成 (B,t,D)
        out = out.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        out = self.out_proj(out)

        # 按需返回：三元表达式 a if cond else b
        return (out, present) if use_cache else out


# ============================================================================
# 第6步：SwiGLU 前馈网络
# ============================================================================
class SwiGLU(nn.Module):
    """门控前馈网络：W_down( SiLU(W_gate·x) ⊙ W_up·x )。

    这里采用 LLaMA 系的命名（gate/up/down）——
    对应 05 节里的 w1/w2/w3，结构完全相同，只是名字不同。
    三组投影都是 bias=False。
    """

    def __init__(self, dim, hidden_dim):
        super().__init__()
        # 门控分支：(B,T,D) → (B,T,H)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        # 内容分支：(B,T,D) → (B,T,H)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        # 输出投影：(B,T,H) → (B,T,D)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # F.silu(gate_proj(x))  门：连续控制放行比例
        # * up_proj(x)          内容：被放行的数值
        # down_proj(...)        压回模型维度 D
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# 第7步：DecoderBlock —— 一个完整的 Transformer 层
# ============================================================================
class DecoderBlock(nn.Module):
    """Pre-Norm decoder block：注意力支路 + FFN 支路，各带一条残差。

    与 07 节的区别：这里的 attention 和 ffn 是【固定接入】的
    （CausalSelfAttention 与 SwiGLU），而不是外部注入。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # 两个独立的 RMSNorm（各有一份 weight，互不影响）
        self.attn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn = SwiGLU(config.dim, config.hidden_dim)

    def forward(self, x, attention_mask=None, past_key_value=None, use_cache=False):
        """一条完整的 decoder 层前向。

        核心三行（展开后）：
            attn_out = attn(attn_norm(x))
            x = x + attn_out
            x = x + ffn(ffn_norm(x))
        """
        # 第一次：归一化 → 注意力
        attn_result = self.attn(
            self.attn_norm(x),           # 注意：attention 读的是归一化后的 x
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        # attention 在 use_cache=True 时返回二元组，否则返回单个张量。
        # 这里统一成 (attn_out, present) 两种形态，方便后面处理。
        if use_cache:
            attn_out, present = attn_result
        else:
            attn_out, present = attn_result, None

        # 残差 1：加到【原始的】x 上（不是归一化后的）
        x = x + attn_out

        # 第二次：归一化 → FFN → 残差 2
        # 注意 ffn_norm 读的是刚更新过的 x —— 这就是 07 节说的"第二个 norm 读 x1"
        # 把两步合成一行写，效果与分开写完全一样
        x = x + self.ffn(self.ffn_norm(x))

        # 同样按需返回
        return (x, present) if use_cache else x


# ============================================================================
# 第8步：MiniMindCore —— 整机组装
# ============================================================================
class MiniMindCore(nn.Module):
    """A compact but structurally complete decoder-only Transformer.

    紧凑但结构完整的 decoder-only Transformer。
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()

        # 保存 config 的引用：forward 里要用 max_seq_len、pad_token_id 等，
        # 外部（如训练脚本）也要通过 model.config 读取词表大小等信息
        self.config = config

        # ── 输入侧：token embedding ──────────────────────────────────────
        # (B,T) 整数 id → (B,T,D)
        # 注意：这里【没有】position_embedding —— 位置信息全部由每层的 RoPE 提供
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)

        # ── 主干：N 个 decoder block ─────────────────────────────────────
        # nn.ModuleList 保证子模块被正确注册（能被 parameters()/state_dict() 看到）
        # 列表推导式每次新建一个 DecoderBlock 实例，参数互不共享
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])

        # ── 输出侧：final norm + LM head ────────────────────────────────
        # 所有 block 之后还有一次归一化，这是 Pre-Norm 结构的标配：
        # 因为每层的残差把未归一化的值一路加到主干上，
        # 到最后一层时主干的数值量级可能偏大，需要收一下再送进 LM head
        self.norm = RMSNorm(config.dim, config.norm_eps)

        # LM head：(B,T,D) → (B,T,V)，无 bias
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # ── 初始化 ──────────────────────────────────────────────────────
        # net.apply(fn) 会【递归遍历】所有子模块并对每个调用 fn。
        # 这是 PyTorch 里做统一初始化的标准手段。
        self.apply(self._init_weights)

        # ── 权值共享（放在初始化之后）───────────────────────────────────
        # Weight tying must share the Parameter object, not merely equal values.
        # 必须是共享 Parameter 【对象】，而不是数值相等。
        # 放在 apply 之后很关键：若顺序反了，apply 会把 lm_head 的权重再随机初始化一次，
        # 虽然引用关系还在（两者仍指向同一对象），但会覆盖掉刚设好的值；
        # 而更重要的是：先绑定再 apply，语义上更清楚"共享是最终状态"。
        self.lm_head.weight = self.token_embedding.weight

    # ── 静态方法：权重初始化 ─────────────────────────────────────────────
    @staticmethod
    def _init_weights(module):
        """对每个子模块做初始化，由 self.apply 递归调用。

        参数:
            module: 当前遍历到的子模块（可能是 Linear、Embedding、RMSNorm 等）
        """
        # isinstance(module, (nn.Linear, nn.Embedding))：
        #   第二个参数是元组时，表示"是其中任意一种类型即可"
        #   只有这两类模块有 weight 需要初始化；RMSNorm 的 weight 初始化为 1，
        #   不应被这里的正态初始化覆盖，所以不加进元组。
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 正态分布 N(mean=0, std=0.02) 原地填充 weight
            #   nn.init.normal_ 的末尾下划线表示【原地操作】，直接改张量内容
            #   std=0.02 是 GPT 系列沿用下来的常见取值：
            #   小方差让初始 logits 接近均匀分布，softmax 不会一开始就极端化。
            #   （注意本模型所有 Linear 都是 bias=False，所以这个分支不会被触发，
            #     但写在这里是为了让初始化函数本身是完整、可复用的。）
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if isinstance(module, nn.Linear) and module.bias is not None:
                # bias 初始化为 0，避免初始输出带上无意义的偏移
                nn.init.zeros_(module.bias)

    # ── 静态方法：从 cache 里读出历史长度 ────────────────────────────────
    @staticmethod
    def _past_length(past_key_values):
        """返回缓存的序列长度；没有缓存时返回 0。

        参数:
            past_key_values: list[(K,V)] 或 None
        """
        if past_key_values is None:
            return 0
        # 取第 0 层的 K：past_key_values[0] 是 (K,V)，[0] 再取 K，
        # .shape[-2] 是倒数第二维，即 past_len
        return past_key_values[0][0].shape[-2]

    # ── forward：整机前向 ────────────────────────────────────────────────
    def forward(
        self,
        input_ids,
        labels=None,
        attention_mask=None,
        past_key_values: list[KeyValue] | tuple[KeyValue, ...] | None = None,
        use_cache=False,
    ):
        """
        参数:
            input_ids:      (B,T) 整数 token id
            labels:         (B,T) 已错开一位的监督目标，或 None
            attention_mask: (B,?) 布尔掩码，True 表示有效
            past_key_values: 每层一对 (K,V) 的列表，或 None
            use_cache:      是否返回新的 K/V

        返回:
            use_cache=True  → (logits, loss, new_past)
            use_cache=False → (logits, loss)

        说明:
            若传入已经错位的 labels，forward 额外返回一个 masked cross-entropy loss；
            若 use_cache=True，还会返回每层的新 K/V。
        """
        # ── 入参形状检查 ─────────────────────────────────────────────────
        # ndim != 2：必须是 (B,T) 的二维张量
        # shape[1] == 0：序列长度为 0 时后面 arange 等操作没有意义
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape (batch, non_empty_seq_len)")

        b, t = input_ids.shape

        # 缓存必须每层一份，数量与层数一致
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("past_key_values must contain one K/V pair per layer")

        # 历史长度（没有缓存时为 0）
        past_len = self._past_length(past_key_values)

        # 历史 + 当前不能超过窗口上限
        if past_len + t > self.config.max_seq_len:
            raise ValueError("sequence length including cache exceeds max_seq_len")

        # ── 自动生成 attention_mask（未显式传入时）───────────────────────
        # 三种情况，顺序不能颠倒：
        if attention_mask is None:
            if past_len:
                # (1) 有缓存：当前只传了一个新 token，但 mask 必须覆盖【前缀】。
                #     这里直接造一个全 True 的 (B, past_len+t)。
                #     注意：这只在"前缀本来就全部有效"时正确；
                #     若 prompt 里含左 PAD，缓存解码必须【显式传完整 mask】
                #     （见 11_KV_Cache.py 的 generate_with_kv_cache）。
                attention_mask = torch.ones((b, past_len + t), device=input_ids.device, dtype=torch.bool)
            elif self.config.pad_token_id is None:
                # (2) 无 PAD 概念：全部位置都有效
                #     torch.ones_like 会继承 input_ids 的 shape 和设备
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                # (3) 有 pad_token_id：把等于 PAD 的位置标为 False
                #     ne 是 "not equal" 的缩写，逐元素比较返回布尔张量
                attention_mask = input_ids.ne(self.config.pad_token_id)
        else:
            # 显式传入：统一 dtype 与设备即可
            attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)

        # ── 逐层前向 ─────────────────────────────────────────────────────
        # 查表得到 (B,T,D)；位置信息由每层的 RoPE 内部注入，这里不加任何东西
        x = self.token_embedding(input_ids)

        # 收集每层返回的 (K,V)
        new_past = []
        # enumerate 同时给出下标和元素 —— 下标用来从 past_key_values 里取该层的缓存
        for layer_index, block in enumerate(self.blocks):
            # 缓存在场时取该层的历史；否则给 None
            layer_past = None if past_key_values is None else past_key_values[layer_index]

            if use_cache:
                # 需要缓存：block 返回 (x, present)
                x, present = block(
                    x,
                    attention_mask=attention_mask,
                    past_key_value=layer_past,
                    use_cache=True,
                )
                # 把这一层的 present 收集起来，最终组成 new_past
                new_past.append(present)
            else:
                # 训练路径：不需要缓存，block 只返回 x。
                # 注意这里【不传 use_cache】，默认 False，省掉构造 present 的开销
                x = block(x, attention_mask=attention_mask, past_key_value=layer_past)

        # ── 输出侧 ───────────────────────────────────────────────────────
        # 先过 final RMSNorm，再经 LM head 投到词表 → (B,T,V)
        logits = self.lm_head(self.norm(x))

        # ── 可选的损失计算 ───────────────────────────────────────────────
        loss = None
        if labels is not None:
            # 标签必须与输入同 shape（因为已经错位，不是 shift）
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")

            # 同样先 clone，避免污染调用方的张量
            # （.clone() 之后再 .to(device=...) 保证与本模型在同一设备）
            targets = labels.to(device=input_ids.device).clone()

            # PAD target 不计损失
            if self.config.pad_token_id is not None:
                targets[targets == self.config.pad_token_id] = -100

            # ── query mask 的取法：只取【当前这一段】───────────────────
            # attention_mask 在缓存在场时长度是 past_len+t，
            # 而 targets 只有当前 t 个位置，所以要用 [-t:] 取最后 t 列对齐。
            # 这是缓存训练时一个很容易写错的地方。
            query_mask = attention_mask[:, -t:]
            # 无效 query 位置（比如 PAD）的预测同样不该被监督
            targets[~query_mask] = -100

            valid = targets.ne(-100)

            # 与 06 节一致：只对有效位置算交叉熵；
            # logits 先转 fp32 再进 CE，低精度下更稳
            # 全无有效 target 时返回 logits.sum()*0.0，值为 0 但仍连着计算图
            loss = (
                F.cross_entropy(logits[valid].float(), targets[valid])
                if valid.any()
                else logits.sum() * 0.0
            )

        # 按 use_cache 返回不同元数
        return (logits, loss, new_past) if use_cache else (logits, loss)

    # ── generate：未缓存的慢速参考生成 ──────────────────────────────────
    @torch.no_grad()
    # @torch.no_grad() 装饰器：整个方法在"不追踪梯度"的上下文里执行。
    #   - 不构建计算图 → 省显存；
    #   - 不会被 autograd 记录 → 速度快。
    # 生成是纯推理，不需要梯度，所以必然加这个装饰器。
    def generate(
        self,
        input_ids,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        top_p=None,
        attention_mask=None,
        eos_token_id=None,
        generator=None,
    ):
        """Reference generation that recomputes the visible context each step.

        参考实现：每一步都重新计算可见窗口内的全部 token。

        Task 30 会给出等价的缓存实现。``temperature=0`` 选中 greedy（贪心）解码；
        正的温度则从 logits 中采样。

        ── 为什么保留这个"慢版本" ────────────────────────────────────────
        它逻辑直白、没有缓存状态，非常适合作为【数值对照】：
        缓存版本的正确性就是靠与它的输出比较来验证的。

        参数:
            input_ids:       (B,T_prompt) 提示词
            max_new_tokens:  最多生成多少个新 token
            temperature:     0 表示贪心；>0 表示按温度采样
            top_k:           只从概率最高的 k 个候选中采样（None 表示不限制）
            top_p:           nucleus 采样阈值，取 (0,1]
            attention_mask:  与 ids 同 shape；变长 batch 必须左 padding
            eos_token_id:    遇到该 id 后该行停止生成
            generator:       torch.Generator，保证采样可复现

        返回:
            (B, T_prompt + 生成数) 的完整 id 序列
        """
        # ── 参数校验 ─────────────────────────────────────────────────────
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature < 0:
            # 负温度在数学上没有定义（会让 softmax 反向放大）
            raise ValueError("temperature must be non-negative")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        if top_p is not None and not 0 < top_p <= 1:
            # 链式比较：等价于 (0 < top_p) and (top_p <= 1)
            raise ValueError("top_p must be in (0, 1]")

        if max_new_tokens == 0:
            # 不生成就原样返回。放前面短路，避免走完后面的流程
            return input_ids

        # result 会不断被追加，最后作为返回值
        result = input_ids

        # ── 准备 attention_mask ──────────────────────────────────────────
        if attention_mask is None:
            # 有 pad_token_id 就按 PAD 推导，否则全部位置有效
            attention_mask = (
                torch.ones_like(result, dtype=torch.bool)
                if self.config.pad_token_id is None
                else result.ne(self.config.pad_token_id)
            )
        else:
            if attention_mask.shape != result.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")
            attention_mask = attention_mask.to(device=result.device, dtype=torch.bool)

        # ── 关键约束：每行最后一位必须是有效 token ────────────────────────
        # 因为生成循环统一取 logits[:, -1]，若最后一列是 PAD，
        # 那取到的就是"预测 PAD 之后是什么"的 logits，完全错了。
        # 所以变长 batch 必须左 padding（把 PAD 放在前面）。
        # torch.all(attention_mask[:, -1]) 检查最后一列是否全为 True
        if not torch.all(attention_mask[:, -1]):
            raise ValueError(
                "each prompt must end in a valid token; left-pad variable-length batches"
            )

        # 记录每行是否已经生成过 EOS。
        # torch.zeros(..., dtype=torch.bool) 全 False，表示都还没结束
        finished = torch.zeros(result.shape[0], device=result.device, dtype=torch.bool)

        # ── 生成主循环：自回归，一次一个 token ───────────────────────────
        for _ in range(max_new_tokens):
            # 只保留最近 max_seq_len 个 token 作为可见窗口。
            # 被滑出窗口的旧 token 仍在 result 里，但不再影响预测。
            # 这是一种明确的截断策略，不是无限上下文。
            context = result[:, -self.config.max_seq_len :]
            context_mask = attention_mask[:, -self.config.max_seq_len :]

            # 完整 forward（无缓存）→ logits (B,T,V)，loss 用不到
            logits, _ = self(context, attention_mask=context_mask)

            # 只取最后一位：它预测的是"下一个 token"
            # 误取 logits[:,0] 会让每一步都根据序列开头生成
            next_logits = logits[:, -1]

            if temperature == 0:
                # ── Greedy：直接取最大 logits 的下标 ─────────────────────
                # argmax 沿最后一维；keepdim=True 保持长度 1 的轴 → (B,1)
                # 这样后面能直接 cat 到 result 上
                next_id = next_logits.argmax(dim=-1, keepdim=True)
            else:
                # ── 采样路径 ─────────────────────────────────────────────
                # 温度缩放：除以 τ。τ<1 分布更尖，τ>1 更平
                next_logits = next_logits / temperature

                if top_k is not None:
                    # k 不能超过词表大小，min 取较小值防止 topk 报错
                    k = min(top_k, next_logits.shape[-1])

                    # topk(...).values 是按降序排好的前 k 个值
                    # [:, -1:] 取第 k 个（也就是阈值），形状 (B,1)
                    # 这样阈值能和 (B,V) 直接广播比较
                    threshold = torch.topk(next_logits, k, dim=-1).values[:, -1:]

                    # 低于阈值的候选填成 -inf，softmax 后概率恰好为 0
                    next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

                if top_p is not None and top_p < 1:
                    # ── nucleus（top-p）采样 ─────────────────────────────
                    # torch.sort(descending=True) 同时返回排序后的值和原始下标，
                    # 下标是后面"把 mask 映射回原位置"的关键
                    sorted_logits, sorted_indices = torch.sort(
                        next_logits, descending=True, dim=-1
                    )

                    # softmax 后做累积和：cumulative 第 i 项 = 前 i+1 个概率之和
                    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

                    # 标出"累计概率已经超过阈值"的位置 —— 这些是要删掉的
                    remove_sorted = cumulative > top_p

                    # ── 右移一格：保证跨过阈值的那个 token 被留下 ────────
                    # 假设概率 [0.6, 0.25, 0.1, 0.05]，top_p=0.7：
                    #   累计 [0.6, 0.85, 0.95, 1.0]，remove 原本是 [F,T,T,T]
                    #   右移后变成 [F,F,T,T] —— 于是保留了前两个，
                    #   正好是"累计到 0.7 所需的最小前缀"。
                    #
                    # 具体做法：把 [:, :-1] 赋给 [:, 1:]，即整体右移一位；
                    # 这里先 .clone() 是因为切片赋值存在重叠，
                    # 不克隆的话源数据会被前一步写坏（经典的自赋值陷阱）。
                    remove_sorted[:, 1:] = remove_sorted[:, :-1].clone()
                    # 第一位强制保留，保证每行至少有一个有限候选，
                    # 否则 softmax 全 -inf 会产生 nan
                    remove_sorted[:, 0] = False

                    # ── 把排序空间的 mask 映射回原始词表顺序 ─────────────
                    # scatter(dim=-1, index=sorted_indices, src=remove_sorted)：
                    #   在 remove 的第 i 行，把 remove_sorted[i] 的值
                    #   按 sorted_indices[i] 的位置写回去。
                    #   scatter 是 gather 的逆操作：
                    #     gather 按 index 取，scatter 按 index 放。
                    # 于是 remove[b, v] = True 表示词表项 v 应当被删除。
                    remove = torch.zeros_like(remove_sorted).scatter(
                        dim=-1, index=sorted_indices, src=remove_sorted
                    )

                    next_logits = next_logits.masked_fill(remove, float("-inf"))

                # 转成概率分布后做多项式采样
                probabilities = torch.softmax(next_logits, dim=-1)

                # torch.multinomial(probabilities, 1)：
                #   按 probabilities 给定的概率分布抽 1 个下标 → (B,1)
                # generator 参数保证可复现（同一 seed → 同一结果）
                next_id = torch.multinomial(probabilities, 1, generator=generator)

            if eos_token_id is not None:
                # 已经结束的行强制继续输出 EOS —— 这样 batch 里各行的
                # 输出长度虽然不同，但都能继续往后拼，不会破坏张量形状
                next_id = torch.where(
                    finished[:, None],                       # 条件：(B,1)
                    torch.full_like(next_id, eos_token_id),  # 已结束 → 填 EOS
                    next_id,                                 # 未结束 → 保留原值
                )
                # |= 是"或赋值"：更新哪些行刚刚生成了 EOS
                # next_id[:, 0] 把 (B,1) 变成 (B,) 才能与 (B,) 的 finished 比较
                finished |= next_id[:, 0].eq(eos_token_id)

            # 把新 token 拼到序列末尾（沿第 1 维）
            result = torch.cat((result, next_id), dim=1)

            # mask 也要同步增长一位（新 token 一定是有效的）
            attention_mask = torch.cat(
                (attention_mask, torch.ones_like(next_id, dtype=torch.bool)), dim=1
            )

            # 所有行都结束后提前退出，不必跑满 max_new_tokens
            if eos_token_id is not None and torch.all(finished):
                break

        return result

    # ── 参数量统计 ───────────────────────────────────────────────────────
    def parameter_count(self):
        """返回模型参数总量。

        numel() 返回张量中元素个数。
        列表推导式把每个参数的元素数收集起来，sum 求和。
        """
        return sum(parameter.numel() for parameter in self.parameters())


# ============================================================================
# 第9步：别名
# ============================================================================
# The reference answer historically used this name; keep it as a readable alias.
# （参考答案历史上用过这个类名，保留它作为一个同样可读的别名。）
# 这只是给同一个类加第二个名字，不是新建一个类 ——
# 所以 isinstance(x, MiniMindModel) 与 isinstance(x, MiniMindCore) 等价。
MiniMindModel = MiniMindCore


# ============================================================================
# 第10步：运行与核对
# ============================================================================
#     python exercises/block_03_transformer/task_27_minimind_core/minimind_core.py
#
# 预期类似：
#
#     logits: (1, 4, 32)
#     parameters: <正整数>
#
# 参数量会随配置变化。三个 logits 维度分别是 batch、sequence、vocabulary。
#
# 相关性质集中在 Block 3 测试中：
#
#     python -m unittest discover -s tests -p 'test_block3.py' -v
#
# 测试覆盖 shape、causal 与上下文敏感性、K/V heads 的实际数量、weight tying、
# PAD key/loss mask、attention 梯度，以及 cache 的层数与 shape。
# 这些性质把"forward 能返回 tensor"进一步细化为可观察的模型行为，
# 也为下一节解读 loss 变化提供基础。
# ============================================================================


if __name__ == "__main__":
    # 固定随机种子，让每次运行的初始化和结果可复现
    # 注意 torch.manual_seed 影响的是全局随机状态（包括后面所有 randn 和初始化）
    torch.manual_seed(0)

    # 建一个很小的配置，保证在 CPU 上秒级跑完：
    #   V=32, D=32, 2 层, 4 个 Q head, 2 个 KV head, FFN 中间 64
    #   校验：32 % 4 == 0 ✓；4 % 2 == 0 ✓；head_dim = 8 为偶数 ✓
    cfg = MiniMindConfig(vocab_size=32, dim=32, n_layers=2, n_heads=4, n_kv_heads=2, hidden_dim=64)

    # 建模型：这一步会触发 _init_weights 与权值共享
    model = MiniMindCore(cfg)

    # 一个 batch、4 个 token 的 id
    ids = torch.tensor([[1, 5, 8, 2]])

    # 前向：use_cache 默认 False，所以只返回两个值 (logits, loss)
    # 这里没传 labels，loss 是 None，用 _ 丢弃
    logits, _ = model(ids)

    # 期望 (1, 4, 32) = (B, T, V)
    print("logits:", tuple(logits.shape))

    # 打印参数量，可以直观感受"教学模型"有多小
    print("parameters:", model.parameter_count())


# ============================================================================
# 补充知识点：整机 forward 的 shape 速查表
# ============================================================================
# 以 cfg = MiniMindConfig(vocab_size=32, dim=32, n_layers=2, n_heads=4,
#                          n_kv_heads=2, hidden_dim=64)，输入 (1,4) 为例：
#
#   input_ids                (1, 4)              整数 id
#   token_embedding(x)       (1, 4, 32)          查表
#   ── 进入 DecoderBlock 0 ────────────────────────────────────────────
#     attn_norm(x)           (1, 4, 32)          RMSNorm 不改变 shape
#     q_proj → (1,4,32) → view (1,4,4,8) → transpose (1,4,4,8)
#     k_proj → (1,4,16) → view (1,4,2,8) → transpose (1,2,4,8)
#     v_proj → (1,4,16) → view (1,4,2,8) → transpose (1,2,4,8)
#     apply_rope(q)          (1, 4, 4, 8)        只旋转，shape 不变
#     repeat_kv(k)           (1, 4, 4, 8)        2 → 4 个 head
#     scores                 (1, 4, 4, 4)        (B,H,T,T)
#     allowed                (1, 1, 4, 4)        下三角（非方阵时更复杂）
#     weights                (1, 4, 4, 4)        softmax 后
#     out = weights @ v      (1, 4, 4, 8)        加权求和
#     transpose+view         (1, 4, 32)          合并多头
#     out_proj               (1, 4, 32)
#     x = x + attn_out       (1, 4, 32)          残差 1
#     ffn_norm(x)            (1, 4, 32)
#     gate/up_proj           (1, 4, 64)          升到 hidden_dim
#     SiLU ⊙                 (1, 4, 64)
#     down_proj              (1, 4, 32)          降回 D
#     x = x + ffn_out        (1, 4, 32)          残差 2
#   ── DecoderBlock 1 同上 ────────────────────────────────────────────
#   final norm               (1, 4, 32)
#   lm_head                  (1, 4, 32)          ← 期望 (1,4,32)，因为 V=32
#                                                 恰好与 D 相同，别被巧合迷惑
#
# 建议亲手在 forward 里插 print(x.shape) 走一遍，比对着表看记得牢得多。
# ============================================================================


# ============================================================================
# 核心公式回顾
# ============================================================================
# (1) 整机数据流
#
#         h₀ = Embed(input_ids)
#         hᵢ = DecoderBlock_i(h_{i-1}),   i = 1..N
#         logits = LMHead(RMSNorm(h_N))
#
#         其中 DecoderBlock(x) = x + FFN(RMSNorm(x + Attn(RMSNorm(x))))
#
# (2) 权重共享
#
#         lm_head.weight ≡ token_embedding.weight      （同一 Parameter 对象）
#
# (3) 非方阵 causal 条件（KV Cache 场景）
#
#         allowed[i,j] = (j ≤ past_len + i)  且  attention_mask[j]
#         i ∈ [0, t),  j ∈ [0, past_len + t)
#
# (4) 采样概率
#
#         p = softmax( filtered(z) / τ )
#         其中 filtered 已应用 top-k 与 top-p
#
# (5) 损失
#
#         loss = CE( logits[valid], targets[valid] )
#         valid = (targets ≠ -100)
#         targets: PAD → -100，无效 query → -100
# ============================================================================


# ============================================================================
# 总结
# ============================================================================
# 1. 这一个文件是前面所有部件的汇合点：RMSNorm（07）、RoPE（03）、
#    causal GQA（04）、SwiGLU（05）、权值共享（06），
#    以及新增的 KV Cache 接口（11）。
#
# 2. MiniMindConfig 用 dataclass + __post_init__ 把"输入校验"集中到一处：
#    三项整除关系、偶数 head_dim、有限的 base/eps、PAD id 范围。
#    这样非法配置在建模型前就报错，而不是在第一次 view 时报 shape 异常。
#    注意 isinstance(x, bool) 的排除 —— 否则 n_heads=True 会被当成 1 通过。
#
# 3. 模型【没有】learned position embedding，位置信息全部来自每层的 RoPE。
#    这是与 06 节 TinyLanguageModel 的关键区别，两套位置方案不要叠加。
#
# 4. 非方阵 causal mask 是缓存解码的核心细节：
#    query 的绝对位置是 [past_len, past_len+t)，key 是 [0, past_len+t)，
#    用"k_positions <= q_positions"比较，而不是简单的 torch.tril。
#
# 5. attention_mask 有【两个用途】，不要混淆：
#    传进 attention → 屏蔽 PAD key（影响表示）
#    用于筛 loss    → 把无效 query 的 target 置 -100（影响监督）
#    缓存解码时还要注意 mask 必须覆盖历史前缀，且 query_mask 取 [-t:] 对齐。
#
# 6. forward 有三种返回形态：(logits, loss) / (logits, loss, new_past)。
#    use_cache=False 时连 present 都不构造，训练路径因此没有额外开销。
#
# 7. 内置的 generate() 是【未缓存的参考实现】—— 慢，但每次重算，
#    逻辑直白，正好作为 11 节缓存实现的数值对照物。保留它是有意的设计。
#
# 8. 初始化顺序值得记一下：apply(_init_weights) → 绑定权值 → 载入 state_dict。
#    先建好引用关系再 load，checkpoint round-trip 后共享关系才不会丢。
# ============================================================================
