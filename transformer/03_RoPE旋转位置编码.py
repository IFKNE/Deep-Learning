"""
=============================================================================
RoPE 旋转位置编码（Rotary Position Embedding）—— 完整带注释版
=============================================================================
源文件: exercises/block_03_transformer/task_22_rope_position/rope.py

任务：不再把位置向量加到 embedding 上，而是把 attention 中的 Q、K 按位置"旋转"，
     V 不旋转 —— 这样位置关系直接影响"当前 query 与哪个 key 匹配"的分数。

  核心：把 Q/K 的相邻两维 (x0,x1) 看成一个二维平面，
        位置 m 对应一次旋转角 mθ_i，其中 θ_i = base^(-2i/Dh)，base = 10000：

            [x0']   [ cos(mθ)  -sin(mθ) ] [x0]
            [x1'] = [ sin(mθ)   cos(mθ) ] [x1]

        展开得：x0' = x0·cos(mθ) - x1·sin(mθ)
                x1' = x1·cos(mθ) + x0·sin(mθ)
        这正是 rotate_half(x) = (-x1, x0) 然后 x·cos + rotate_half(x)·sin 的由来。

输入：x: (B, H, T, Dh)          —— 已经拆好头的 Q 或 K
输出：(B, H, T, Dh)             —— shape 与 dtype 都不变，只是数值被旋转

依赖：math、torch。本文件可独立运行。
=============================================================================
"""

# ============================================================================
# 第1步：导入依赖
# ============================================================================
import math                       # 用到 math.isfinite 做 base 的合法性检查

import torch                      # arange / outer / cos / sin / repeat_interleave


# ============================================================================
# 第2步：思路转换 —— 从"加法"到"旋转"
# ============================================================================
# 正弦位置编码把位置向量【加到】embedding 上（见 02_正弦位置编码.py）。
# RoPE 换了一个入口：embedding 保持不动，attention 中的 Q、K 按位置【旋转】，V 不旋转。
#
# ── 为什么是 Q/K 而不是 V ────────────────────────────────────────────────
# 位置关系需要改变"当前 query 与哪个 key 匹配"——
# 这是【匹配分数】的事，而匹配分数只由 Q 和 K 决定。
# value 仍负责提供被汇总的内容，它与"谁和谁更相关"无关，所以不需要旋转。
#
# ── 旋转而不是加法，好处在哪 ─────────────────────────────────────────────
# 加法是"往向量里塞一个位置签名"，而旋转是【保长的】：
#   ||R_m·x|| = ||x||（旋转矩阵是正交矩阵，不改变向量长度）
# 因此位置信息不会稀释 token 本身的表示强度。
# 更关键的是它让点积显式地带上了相对位移（见第 5 步的推导）。
# ============================================================================


# ============================================================================
# 第3步：把相邻两维看成一个平面
# ============================================================================
# 取 Q 或 K 的一对分量 (x0, x1)。位置 m 对应一次二维旋转：
#
#     ⎡x'₀⎤   ⎡ cos(mθ)  -sin(mθ) ⎤ ⎡x₀⎤
#     ⎣x'₁⎦ = ⎣ sin(mθ)   cos(mθ) ⎦ ⎣x₁⎦
#
# 展开第一、二行：
#
#     x'₀ = x₀·cos(mθ) - x₁·sin(mθ)
#     x'₁ = x₁·cos(mθ) + x₀·sin(mθ)
#
# 这两式可以合写成代码里的形式：
#
#     rotate_half(x) = (-x₁, x₀)              # 取出"交换并取负"的那一项
#     out            = x·cos + rotate_half(x)·sin
#
# 验证第一维： x₀·cos + (-x₁)·sin = x₀·cos - x₁·sin  ✓
# 验证第二维： x₁·cos + ( x₀)·sin = x₁·cos + x₀·sin  ✓
#
# 输入通常是 (B,H,T,Dh)，旋转只发生在【最后一维】，输出 shape 仍为 (B,H,T,Dh)。
# 注意 Dh 必须是偶数：每一对 (0,1)、(2,3)、… 是一个独立的二维平面。
# ============================================================================


# ============================================================================
# 第4步：build_rope_cache —— 构造 cos/sin 表
# ============================================================================
def build_rope_cache(
    seq_len,
    head_dim,
    base=10000,
    device=None,
    dtype=None,
    start_pos=0,
):
    """Return one cosine/sine value for every two-dimensional RoPE plane.

    Column ``i`` uses angular frequency ``base ** (-2 * i / head_dim)``;
    adjacent two-dimensional planes therefore do *not* share one frequency.
    ``start_pos`` is needed when decoding after a KV cache prefix.

    参数:
        seq_len:  需要多少个位置（通常等于当前序列长度 T）
        head_dim: 单个 head 的维度 Dh，必须为正偶数（两两成平面）
        base:     频率底数，默认 10000；典型的"长上下文"改法就是调大它
        device:   目标设备
        dtype:    若指定，把 cos/sin 转成该精度（通常与 Q/K 一致）
        start_pos: 起始位置偏移；KV Cache 解码时必须传 past_len

    返回:
        cos, sin：均为 (seq_len, head_dim // 2)
                  ——【每一对维度只存一个值】，用的时候再重复两次

    分步说明 / 计算过程：
        (1) inv_freq  → (head_dim//2,)，每个二维平面的角速度 θ_i
        (2) positions → (seq_len,)，各位置的整数下标（从 start_pos 开始）
        (3) angles = outer(positions, inv_freq) → (seq_len, head_dim//2)
        (4) cos/sin 逐元素求值，可选地转 dtype
    """
    # ── 入参校验：把所有边界一次检查完，避免错误拖到后面变成难懂的 shape 异常 ──
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if head_dim <= 0 or head_dim % 2 != 0:
        # 必须偶数：RoPE 是"两维一个平面"，奇数维无法配对
        raise ValueError("head_dim must be a positive even integer for RoPE")
    if start_pos < 0:
        # 位置索引不能为负
        raise ValueError("start_pos must be non-negative")
    if not math.isfinite(base) or base <= 0:
        # math.isfinite 同时排除 inf 和 nan；
        # base 必须为正且有限，否则 base ** x 会产生 inf/0，整张表失效
        raise ValueError("base must be positive and finite")

    # torch.arange(0, head_dim, 2) 生成偶数下标 [0,2,4,...,Dh-2]，共 Dh/2 个 —— 即平面编号 i
    # .float() 后除以 head_dim，得到 i/Dh
    # base ** (-i/Dh) 就是 θ_i = base^(-2i/Dh)（因为 arange 步长 2，i 已经是 2×平面号）
    # 1.0 / (...) 写成倒数，与公式 θ_i = base^(-2i/Dh) 完全一致
    # 结果 shape (head_dim//2,)：每个二维平面一个角速度，各不相同
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )

    # 位置下标从 start_pos 到 start_pos+seq_len-1
    #   start_pos=0     → [0,1,...,seq_len-1]      完整 forward 的常规情形
    #   start_pos=5     → [5,6,...,5+seq_len-1]    KV Cache 已存 5 个 token 时的续写
    positions = torch.arange(start_pos, start_pos + seq_len, device=device).float()

    # torch.outer(a, b) 计算外积：结果第 i 行第 j 列 = a[i] * b[j]
    # (seq_len,) ⊗ (head_dim//2,) → (seq_len, head_dim//2)
    # freqs[pos, i] = pos · θ_i，即位置 pos 在第 i 个平面上转过的角度
    freqs = torch.outer(positions, inv_freq)

    # 对每个角度同时算出 cos 和 sin，两者形状都是 (seq_len, head_dim//2)
    cos, sin = torch.cos(freqs), torch.sin(freqs)

    # 可选地把结果转成目标精度（例如与 fp16 的 Q/K 对齐，避免中间被提升为 fp32）
    if dtype is not None:
        cos, sin = cos.to(dtype=dtype), sin.to(dtype=dtype)

    # 注意：这里【没有】repeat_interleave 到 head_dim。
    # 只存 Dh/2 个值是刻意的 —— 存储省一半，pair 的语义也更清楚。
    return cos, sin


# ============================================================================
# 第5步：rotate_half —— 实现"交换并取负"
# ============================================================================
def rotate_half(x):
    """
    把最后一维按 (x0,x1) 成对处理，返回 (-x1, x0)。

    参数:
        x: (..., Dh) 的张量，Dh 为偶数

    返回:
        与 x 同 shape，内容为每对分量交换后取负

    说明:
        这是把"二维旋转"写成逐元素乘法的关键一步。
        旋转公式 x'₀ = x₀cos - x₁sin 里那个 "-x₁sin"，
        正是靠 (-x₁, x₀) 这一项配合 x·cos 拼出来的。
    """
    # x[..., 0::2] 取最后一维的所有偶数下标 → (..., Dh/2)，即所有的 x0
    x1 = x[..., 0::2]     # 先取偶数位（注意变量名 x1 只是习惯叫法，含义是"偶数位"）
    x2 = x[..., 1::2]     # 再取奇数位 → (..., Dh/2)，即所有的 x1

    # torch.stack((-x2, x1), dim=-1)：
    #   把 -x2 和 x1 沿新的最后一维堆起来 → (..., Dh/2, 2)
    #   stack 是"新增一维"，与 cat（在已有维上拼接）不同
    # .flatten(-2)：把最后两维合并 → (..., Dh)
    #   于是第 0 位是 -x2[0] = -x1，第 1 位是 x1[0] = x0，
    #   紧接着第 2 位是 -x2[1]，第 3 位是 x1[1]…… 正好是按对交错排列的 (-x1, x0)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


# ============================================================================
# 第6步：apply_rope —— 把旋转真正应用上去
# ============================================================================
def apply_rope(x, cos, sin):
    """
    对 x 的最后一维施加 RoPE 旋转。

    参数:
        x:   (B, H, T, Dh)，已经拆好头的 Q 或 K
        cos: (T, Dh//2)，由 build_rope_cache 生成
        sin: 同 cos

    返回:
        与 x 同 shape、同 dtype 的张量

    注意:
        cos/sin 每列要【重复两次】才能与 x 的成对分量对齐：
            cos[:, 0] 对应 x 的第 0、1 维
            cos[:, 1] 对应 x 的第 2、3 维
        ...
    """
    # 最后一维不是偶数就没法两两分平面，直接报错
    if x.shape[-1] % 2 != 0:
        raise ValueError("the last dimension must be even for RoPE")

    # 形状必须严格匹配：cos/sin 的 seq_len 要对上 x 的 T，列数要是 Dh//2
    # 这里不做广播宽容处理 —— 序列长度对不上时立即报错，比静默广播出错误结果好得多
    if cos.shape != sin.shape or cos.shape != (x.shape[-2], x.shape[-1] // 2):
        raise ValueError("cos/sin must have shape (seq_len, head_dim // 2)")

    # repeat_interleave(cos, repeats=2, dim=-1)：
    #   (T, Dh/2) → (T, Dh)，每列【复制两次】而不是整块复制
    #   例：cos = [[c0, c1]]  →  [[c0, c0, c1, c1]]
    #   这样才能与 x 的 (x0,x1,x2,x3...) 成对对齐：c0 管 x0x1，c1 管 x2x3
    # [None, None, :, :] 在前后各插一个长度 1 的轴：
    #   (T, Dh) → (1, 1, T, Dh)，于是可与 (B,H,T,Dh) 广播
    cos = torch.repeat_interleave(cos, repeats=2, dim=-1)[None, None, :, :]
    sin = torch.repeat_interleave(sin, repeats=2, dim=-1)[None, None, :, :]

    # 显式对齐设备与精度：cos/sin 由 build_rope_cache 生成，
    # 可能建在 CPU/fp32 上，而 x 在 GPU/fp16 —— 不转的话乘法会直接报错
    cos = cos.to(device=x.device, dtype=x.dtype)
    sin = sin.to(device=x.device, dtype=x.dtype)

    # RoPE 的最终形式：
    #   x·cos + rotate_half(x)·sin
    # 展开后正是第 3 步里那两行旋转公式。
    # 全程是逐元素运算，没有矩阵乘法，shape 与 dtype 都保持不变。
    return x * cos + rotate_half(x) * sin


# ============================================================================
# 第7步：点积为何出现相对位移
# ============================================================================
# 这是 RoPE 最重要的性质，值得单独推一遍。
#
# 把位置 m 的 query 写成 R_m·q，位置 n 的 key 写成 R_n·k（R 是旋转矩阵）：
#
#     (R_m·q)ᵀ (R_n·k) = qᵀ R_mᵀ R_n k = qᵀ R_{n-m} k
#
# 推导用到的两条性质：
#   (a) (R_m·q)ᵀ = qᵀ R_mᵀ          —— 转置穿进乘积并交换顺序
#   (b) R_mᵀ R_n = R_{n-m}          —— 旋转矩阵正交：R_mᵀ = R_m⁻¹，
#                                      而 R_m⁻¹ R_n = R_{n-m}（角度相减）
#
# 结论：位置项通过 (n - m) 进入点积。
#
# 因此较准确的表述是：
#     **RoPE 让旋转后的 Q/K 点积显式依赖相对位移。**
#
# ── 但不要过度解读 ────────────────────────────────────────────────────────
# 这【不等于】"模型输出对绝对位置完全不变"。
# 内容本身、causal mask（可见范围随绝对位置变化）、上下文窗口、以及后续网络层
# 仍然会影响结果；上式只说明旋转后的匹配分数具有怎样的位置结构。
#
# ── 与正弦编码的区别 ──────────────────────────────────────────────────────
# 02 节的方案里，相对位置是"能由线性变换得到"（存在性）；
# RoPE 是"点积里直接就是相对位移"（结构性）。
# 后者更强，不需要网络额外去学那个变换。
# ============================================================================


# ============================================================================
# 第8步：start_pos 是为缓存解码准备的
# ============================================================================
# 完整 forward 的位置从 0 开始。若 KV Cache 已保存 5 个 token，
# 新 token 应使用位置 5，而不是重新使用位置 0：
#
#     cos, sin = build_rope_cache(
#         seq_len=1,
#         head_dim=head_dim,
#         start_pos=5,
#     )
#
# 两种取法的结果一致：
#
#     cache(start_pos=5, seq_len=1) == cache(start_pos=0, seq_len=6)[5:6]
#
# ── 为什么这是一个"静默错误"的高发点 ─────────────────────────────────────
# 漏掉这个偏移时，代码可能仍能运行，shape 也完全正常，生成循环也照样跑完，
# 但 cached logits 会偏离完整 forward —— 因为 Q/K 的旋转角度错了。
# 没有数值对照就发现不了。
# Task 30（本模块第 11 节）的数值等价测试会覆盖这类情形：
# 它专门比较"全量 forward"与"cached 逐步 forward"的最大绝对误差。
# ============================================================================


# ============================================================================
# 第9步：代码结构与边界
# ============================================================================
# rope.py 有三个函数：
#
#     build_rope_cache  构造各位置、各维度对的 cos/sin
#     rotate_half       (x0,x1) -> (-x1,x0)
#     apply_rope        把旋转应用到 (B,H,T,Dh)
#
# 实现中的边界如下：
#
#   - Dh 为正偶数；
#   - 位置 0 的 cos 全为 1、sin 全为 0，因此向量不变（自检点！）；
#   - 位置 1 的不同列不相同（说明各平面角速度不同）；
#   - apply_rope 保持 shape 和 dtype；
#   - cos/sin 的 shape 与输入序列长度不符时立即报错。
#
# 运行：
#
#     python exercises/block_03_transformer/task_22_rope_position/rope.py
#
# 输出形如：
#
#     input/output: (2, 4, 6, 8) (2, 4, 6, 8)
#     pair frequencies differ: True
#
# RoPE 的边界也收录在 Block 3 测试中：
#
#     python -m unittest discover -s tests -p 'test_block3.py' -v
#
# 运行与测试会呈现这些性质：位置 0 不旋转，维度对频率不同，shape/dtype 不变，
# 奇数 head_dim 会触发入参错误，start_pos 切片与完整 cache 对齐。
#
# 参考：RoFormer: Enhanced Transformer with Rotary Position Embedding
#       https://arxiv.org/abs/2104.09864
# ============================================================================


if __name__ == "__main__":
    # 构造一个模拟的 Q/K：batch=2, heads=4, 序列长度=6, head_dim=8
    # 这里用 randn 只是为了有个具体张量来验证 shape 与性质，数值本身无意义
    values = torch.randn(2, 4, 6, 8)

    # 为 6 个位置、head_dim=8 构造 cos/sin 表
    # 返回 shape 均为 (6, 4) —— 注意是 8/2=4 列，每个二维平面一个值
    rope_cos, rope_sin = build_rope_cache(6, 8)

    # 施加旋转：输出的 shape 应与输入完全一致
    rotated = apply_rope(values, rope_cos, rope_sin)

    # 确认 shape 没有变化（RoPE 不改变张量形状，只改变数值）
    print("input/output:", tuple(values.shape), tuple(rotated.shape))

    # rope_cos[1, 0] 是位置 1、第 0 个平面的 cos 值
    # rope_cos[1, 1] 是位置 1、第 1 个平面的 cos 值
    # 若两者相同，说明所有平面共用一个频率 —— 那 RoPE 就退化成了"整体旋转"，
    # 失去区分不同维度对的能力。torch.allclose 若为 False 则 not 为 True，符合预期。
    print("pair frequencies differ:", not torch.allclose(rope_cos[1, 0], rope_cos[1, 1]))


# ============================================================================
# 补充知识点：RoPE 的三个自检点
# ============================================================================
# 写完或读到一个 RoPE 实现时，按顺序检查这三件事就能定位绝大多数 bug：
#
#   (1) 位置 0 应该"什么都不做"
#         cos = 1, sin = 0  →  x·1 + rotate_half(x)·0 = x
#       跑一下 apply_rope(x, cache[0:1])，结果应等于 x 本身。
#
#   (2) 不同平面的角速度必须不同
#         cos[1, 0] != cos[1, 1]
#       若全部相等，说明 inv_freq 算错了（比如误把 head_dim 写成了 2，
#       或者 arange 的步长漏了 2）。
#
#   (3) start_pos 必须接在 past_len 上
#         cache(start_pos=k, seq_len=1) == cache(start_pos=0, seq_len=k+1)[k:k+1]
#       这是 KV Cache 数值正确的前提。
#
# ── base 参数能做什么 ────────────────────────────────────────────────────
# base=10000 是原论文的取值。base 越大，各平面的角速度整体越慢，
# 相邻位置之间的旋转差异越小 —— 这使得模型在超出训练长度时衰减更平缓，
# 因此长上下文模型常用 base=500000 或更大（配合"YaRN""NTK 插值"等方案）。
# 本节代码把 base 暴露成参数，正是为这种调整留的口子。
# ============================================================================


# ============================================================================
# 核心公式回顾
# ============================================================================
# (1) 二维旋转（位置 m，平面 i）
#
#         ⎡x'₀⎤   ⎡ cos(mθᵢ)  -sin(mθᵢ) ⎤ ⎡x₀⎤
#         ⎣x'₁⎦ = ⎣ sin(mθᵢ)   cos(mθᵢ) ⎦ ⎣x₁⎦
#
# (2) 角速度（每个二维平面一个）
#
#         θᵢ = base^(-2i/Dh),   i = 0,1,…,Dh/2-1,   base = 10000
#
#     inv_freq = 1.0 / ( base ** (arange(0, Dh, 2) / Dh) )
#
# (3) 代码实现形式
#
#         rotate_half(x) = (-x₁, x₀)          # 交换相邻两维并取负
#         out = x · cos + rotate_half(x) · sin
#
# (4) 相对位移性质
#
#         (R_m q)ᵀ (R_n k) = qᵀ R_mᵀ R_n k = qᵀ R_{n-m} k
#
# (5) 缓存对齐条件
#
#         cache(start_pos=k, seq_len=1) == cache(start_pos=0, seq_len=k+1)[k:k+1]
# ============================================================================


# ============================================================================
# 总结
# ============================================================================
# 1. RoPE 的入口和正弦编码不同：不碰 embedding，只旋转 attention 里的 Q 和 K。
#    原因是位置要影响的是"匹配分数"，而匹配分数只由 Q/K 决定，V 无关。
#
# 2. 实现就三步：
#      build_rope_cache  算每个位置的 cos/sin（每对维度只存一个值）
#      rotate_half       把 (x0,x1) 变成 (-x1,x0)
#      apply_rope        x·cos + rotate_half(x)·sin
#    没有矩阵乘法，因此 shape 和 dtype 完全不变 —— 这也是它好实现、好插入的原因。
#
# 3. 旋转的核心收益是保长（正交变换）与相对性（点积里出现 n-m），
#    而且位置信息在【每一层】都生效，不像正弦编码只在输入端注入一次。
#
# 4. cos/sin 表存 Dh/2 列而不是 Dh 列，用的时候 repeat_interleave(2) 展开。
#    理解这个"存一半、用两次"的约定，是读懂 KV Cache 那节的前提之一。
#
# 5. start_pos 是本节最容易被忽略、后果又最隐蔽的参数：
#    漏掉它代码照样跑，shape 全对，只有 logits 悄悄偏离。
#    记住顺序：prefill 从 0 开始，之后每一步的位置都是当前 cache 长度。
#
# 6. 三个自检点务必记住：位置 0 不旋转、各平面频率不同、start_pos 可切片对齐。
# ============================================================================
