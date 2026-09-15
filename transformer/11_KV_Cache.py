"""
=============================================================================
KV Cache —— 逐层缓存解码与数值等价检验
=============================================================================
源文件: exercises/block_03_transformer/task_30_kv_cache/kv_cache.py

任务：未缓存生成在每一步重算整个可见窗口 ——
     Prompt 长度为 5 时：第 1 步 forward 5 tokens，第 2 步 6 tokens，第 3 步 7 tokens…

     旧 token 在每层得到的 K/V 不会因为后来又生成一个 token 而改变，
     因此推理时可以把它们留下。下一步只计算新 token 的 Q/K/V，
     再让新 Q 读取"历史 K/V + 当前 K/V"。

  重要说明：KV Cache 用于【自回归推理】，不用于本章的并行训练 forward。

  本文件的三个核心函数：
     prefill                 一次处理整个 prompt，返回 logits + 每层 K/V
     decode_one              只处理一个新 token，复用历史 K/V
     cache_equivalence_error 全量 forward 与 cached 逐步 forward 的最大绝对误差

依赖：argparse、pathlib、sys、torch，以及同模块的 train / generate。
      本文件可独立运行。
=============================================================================
"""

# ============================================================================
# 第1步：导入依赖与路径处理
# ============================================================================
import argparse                      # 命令行参数
from pathlib import Path             # 路径操作
import sys                           # 改 sys.path

import torch                         # 张量运算


# ── 让本文件能 import 到 train（load_checkpoint）────────────────────────
BLOCK = Path(__file__).resolve().parents[1]
TRAINING = BLOCK / "task_28_next_token_training"
SAMPLING = BLOCK / "task_29_generate_sampling"

# 整理到 my_learn/transformer/ 之后的回退路径（原始逻辑保持不变）
if not TRAINING.exists():
    TRAINING = Path(
        r"d:\桌面\深度学习\quickly_access_to_deeplearning-main"
        r"\exercises\block_03_transformer\task_28_next_token_training"
    )
if not SAMPLING.exists():
    SAMPLING = Path(
        r"d:\桌面\深度学习\quickly_access_to_deeplearning-main"
        r"\exercises\block_03_transformer\task_29_generate_sampling"
    )

# 元组 + for 循环：一次把两个目录都加进 sys.path
for folder in (TRAINING, SAMPLING):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))


# ============================================================================
# 第2步：Cache 保存什么
# ============================================================================
# 每个 decoder layer 有自己的 K/V 投影，cache 因此【按层保存】：
#
#     past_key_values = [
#         (K_layer0, V_layer0),
#         (K_layer1, V_layer1),
#         ...
#     ]
#
# 每个张量的 shape 是：
#
#     (B, n_kv_heads, past_len, head_dim)
#
# ── 注意这里是 n_kv_heads 而不是 n_heads ─────────────────────────────────
# 存的是【尚未 repeat_kv】的 K/V。
# 若模型有 8 个 query heads、2 个 KV heads，cache 只保留 2 份 K/V，
# 计算 attention 时再按组展开到 8 份。
# 这也是 GQA 减少缓存体积的直接来源 ——
# 如果存展开后的版本，GQA 省显存的效果就完全没有了。
#
# ── 缓存并没有把 attention 变成常数时间 ──────────────────────────────────
# 单步新 query 仍要和所有可见 keys 计算 score：
#
#     Q:                  (B, n_heads, 1, head_dim)
#     K(repeat_kv 展开后): (B, n_heads, past_len+1, head_dim)
#     scores:             (B, n_heads, 1, past_len+1)
#
# 它省掉的是【旧 token 的 block 计算和 K/V 投影】；
# cache 本身的存储会随上下文长度增长。
# 所以这是一个"用显存换计算"的经典权衡。
# ============================================================================


# ============================================================================
# 第3步：sample_next_token —— 采样原语（本文件内的副本）
# ============================================================================
def sample_next_token(
    logits, temperature=1.0, top_k=None, top_p=None, generator=None
):
    """Local copy of task 29's tiny sampling primitive for import safety.

    task 29 那个小采样原语的本地副本，目的是让 import 更安全。

    逻辑与 10_自回归生成与采样.py 中的版本完全一致，这里简要注释，
    详细说明请参见第 10 节的同名函数。
    """
    # 形状必须是 (B,V)
    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocab_size)")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    # 贪心分支：argmax 直接返回，不再过滤
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    # 温度缩放（在副本上做，不改调用方的张量）
    filtered = logits / temperature

    if top_k is not None:
        # 取第 k 大的值作阈值，低于阈值的填 -inf
        k = min(top_k, filtered.shape[-1])
        threshold = torch.topk(filtered, k, dim=-1).values[:, -1:]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))

    if top_p is not None and top_p < 1:
        # nucleus：排序 → 累积 → 标记超阈值 → 右移一格 → scatter 回原顺序
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove_sorted = cumulative > top_p
        # 右移一格：保留跨过阈值的那个 token，并保证每行至少有一个候选。
        # .clone() 必不可少 —— 切片赋值区域重叠，不克隆会写坏源数据。
        remove_sorted[:, 1:] = remove_sorted[:, :-1].clone()
        remove_sorted[:, 0] = False
        remove = torch.zeros_like(remove_sorted).scatter(
            dim=-1, index=sorted_indices, src=remove_sorted
        )
        filtered = filtered.masked_fill(remove, float("-inf"))

    # softmax 后按分布抽一个下标
    return torch.multinomial(torch.softmax(filtered, dim=-1), 1, generator=generator)


# ============================================================================
# 第4步：_default_mask —— 按 PAD 推导默认掩码
# ============================================================================
def _default_mask(model, input_ids):
    """没传 mask 时，根据 pad_token_id 造一个。

    参数:
        model:     MiniMindCore（要读它的 config.pad_token_id）
        input_ids: (B,T)

    返回:
        (B,T) 布尔张量
    """
    pad_id = model.config.pad_token_id
    # pad_id 为 None 表示没有 PAD 概念 → 全 True
    # 否则不等于 PAD 的位置为 True
    # 三元表达式写成一行，因为是简单的一对一映射
    return torch.ones_like(input_ids, dtype=torch.bool) if pad_id is None else input_ids.ne(pad_id)


# ============================================================================
# 第5步：prefill —— 一次处理整个 prompt
# ============================================================================
@torch.no_grad()
def prefill(model, input_ids, attention_mask=None):
    """Process a prompt once and return its logits plus one K/V pair per layer.

    一次性处理 prompt，返回它的 logits 和每层一对 K/V。

    参数:
        model:          MiniMindCore
        input_ids:      (B, T_prompt)
        attention_mask: (B, T_prompt)，None 时按 PAD 自动生成

    返回:
        (logits, cache)
        logits: (B, T_prompt, V)   【所有】位置的 logits
        cache:  list[(K,V)]        每层一份，K/V shape (B, n_kv_heads, T_prompt, head_dim)

    说明:
        实际生成第一个 token 只用 logits[:, -1]，
        保留全部 logits 是为了后面的数值对照（见第 7 步）。
    """
    if attention_mask is None:
        attention_mask = _default_mask(model, input_ids)

    # use_cache=True 让 forward 返回三元组 (logits, loss, new_past)
    # 这里不需要 loss，用 _ 接住
    logits, _, cache = model(input_ids, attention_mask=attention_mask, use_cache=True)
    return logits, cache


# ============================================================================
# 第6步：decode_one —— 只处理一个新 token
# ============================================================================
@torch.no_grad()
def decode_one(model, input_id, past_key_values, attention_mask=None):
    """Decode one new input token while reusing all prefix K/V tensors.

    解码一个新 token，同时复用全部前缀 K/V 张量。

    参数:
        model:          MiniMindCore
        input_id:       (B, 1) —— 必须是单个 token
        past_key_values: 上一轮返回的 cache
        attention_mask: (B, past_len+1)，必须覆盖缓存前缀 + 当前 token

    返回:
        (logits, cache)
        logits: (B, 1, V)
        cache:  每层的 K/V 长度都从 past_len 变成 past_len+1
    """
    # 形状检查：(B,1) 正好两维，且第 1 维长度为 1。
    # 为什么强制要求单个 token？
    #   因为缓存的语义就是"每次只增量一个位置"；
    #   若一次传多个，causal mask 的逻辑虽然仍成立，
    #   但就失去了"避免重复计算"的意义，也容易掩盖调用方的错误。
    if input_id.ndim != 2 or input_id.shape[1] != 1:
        raise ValueError("input_id must have shape (batch, 1)")

    # 从缓存的第 0 层 K 里读出历史长度（倒数第二维）
    past_len = past_key_values[0][0].shape[-2]

    if attention_mask is None:
        # 缺省时把【整个前缀】视为有效。
        #
        # 注意：Prompt 含左侧 PAD 时，完整 mask 会随新 token 保留并追加 True；
        # 如果使用这个缺省值，原有 PAD 位置会被视为有效 —— 这是一个陷阱。
        # 因此生成接口（generate_with_kv_cache）总是显式传完整 mask。
        attention_mask = torch.ones(
            (input_id.shape[0], past_len + 1), device=input_id.device, dtype=torch.bool
        )

    logits, _, cache = model(
        input_id,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        use_cache=True,
    )
    return logits, cache


# ============================================================================
# 第7步：logits_with_kv_cache —— 用缓存算出"所有位置的 logits"
# ============================================================================
@torch.no_grad()
def logits_with_kv_cache(model, input_ids, attention_mask=None):
    """Return every position's logits by prefill(1) + repeated cached decoding.

    通过"prefill 一个 token + 反复缓存解码"，得到每个位置的 logits。

    这条路径好检查、好对照 —— 它完全不走并行 forward，
    而是把序列一个 token 一个 token 地喂进去，最后拼起来。

    参数:
        model:          MiniMindCore
        input_ids:      (B,T)
        attention_mask: (B,T)

    返回:
        (B,T,V) —— 与普通并行 forward 的输出形状相同，可直接逐元素比较
    """
    # 输入不能超过窗口上限（否则预填充就会报错）
    if input_ids.shape[1] > model.config.max_seq_len:
        raise ValueError("input is longer than max_seq_len")

    if attention_mask is None:
        attention_mask = _default_mask(model, input_ids)

    # ── 第一步：只 prefill 第 0 个位置 ───────────────────────────────────
    # 这里刻意只喂 1 个 token，而不是整个序列：
    #   这样后面每一个位置都走 decode_one，
    #   能完整覆盖"缓存路径"的每一环（拼接、mask、RoPE 的 start_pos）。
    #   如果先 prefill 整段再解码一个，缓存路径只被验证了一次。
    first_logits, cache = prefill(model, input_ids[:, :1], attention_mask[:, :1])

    # pieces 收集每个位置的 logits，最后沿序列维拼起来
    pieces = [first_logits]

    # ── 第二步：逐个位置解码 ─────────────────────────────────────────────
    # range(1, T) 生成 1,2,...,T-1
    for position in range(1, input_ids.shape[1]):
        step_logits, cache = decode_one(
            model,
            # input_ids[:, position:position+1]：
            #   切片保留长度为 1 的维，得到 (B,1) 而不是 (B,) ——
            #   这正是 decode_one 要求的形状
            input_ids[:, position : position + 1],
            cache,
            # mask 只取到当前位置，覆盖 [0, position]
            attention_mask=attention_mask[:, : position + 1],
        )
        pieces.append(step_logits)

    # 沿第 1 维（序列维）拼接：(B,1,V) × T → (B,T,V)
    return torch.cat(pieces, dim=1)


# ============================================================================
# 第8步：cache_equivalence_error —— 数值等价检验（本文件的核心验收手段）
# ============================================================================
@torch.no_grad()
def cache_equivalence_error(model, input_ids, attention_mask=None):
    """Maximum absolute error between ordinary causal forward and cached forward.

    普通因果 forward 与缓存 forward 之间的最大绝对误差。

    这个对照同时覆盖：
        - 每层 cache 是否对应正确；
        - K/V 是否沿序列轴追加；
        - RoPE 的 start_pos 是否正确；
        - 非方阵 causal mask 是否对齐当前 query；
        - padding mask 是否覆盖历史前缀。

    因此，仅观察"代码里有一个 list"或"生成能跑完"无法区分上述实现细节 ——
    必须做数值比较。
    """
    if attention_mask is None:
        attention_mask = _default_mask(model, input_ids)

    # 路径 A：一次并行 forward（训练时用的那条路）
    full_logits, _ = model(input_ids, attention_mask=attention_mask)

    # 路径 B：prefill + 逐步解码（推理时用的那条路）
    cached_logits = logits_with_kv_cache(model, input_ids, attention_mask)

    # 逐元素相减，取绝对值，再取全局最大值（.max() 返回 0 维张量，.item() 取出数值）
    return (full_logits - cached_logits).abs().max().item()


# ============================================================================
# 第9步：RoPE 的位置接着 past_len
# ============================================================================
# 缓存已有 past_len 个位置时，新 token 的位置索引就是 past_len：
#
#     cos, sin = build_rope_cache(
#         seq_len=1,
#         head_dim=head_dim,
#         start_pos=past_len,
#     )
#
# ── 一个典型的"静默错误"──────────────────────────────────────────────────
# 若每一步又从位置 0 开始，shape、cache 长度甚至生成循环都可能看似正常，
# 但 Q/K 的旋转角度已经偏离 —— 输出会变得没有意义，却不会有任何报错。
#
# cached/full logits 的数值比较会直接暴露这种静默错误，
# 这就是第 8 步那个函数存在的最大价值。
#
# 这处代码在 08_MiniMind模型主干.py 的 CausalSelfAttention.forward 里：
#     cos, sin = build_rope_cache(
#         t, self.head_dim, base=self.rope_base,
#         device=x.device, dtype=x.dtype,
#         start_pos=past_len,          # ← 就是这一行
#     )
#
# 等价的表述（03 节里验证过的性质）：
#     cache(start_pos=past_len, seq_len=1) == cache(start_pos=0, seq_len=past_len+1)[past_len:]
# ============================================================================


# ============================================================================
# 第10步：缓存生成循环
# ============================================================================
@torch.no_grad()
def generate_with_kv_cache(
    model,
    input_ids,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    top_p=None,
    attention_mask=None,
    eos_token_id=None,
    generator=None,
):
    """Generate with real per-layer caches, rebuilding only after window rollover.

    用真实的逐层缓存生成；只在窗口滚动后才重建缓存。

    流程：
        prompt -> prefill -> 最后一位 logits + cache
        repeat:
            sample next_id
            append next_id
            decode_one(next_id, cache)      ← 只算一个新 token

    参数与 10 节的 generate 完全一致，便于对照。

    返回:
        (B, T_prompt + 生成数) 的完整 id 序列
    """
    # ── 参数校验 ─────────────────────────────────────────────────────────
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    if max_new_tokens == 0:
        return input_ids

    result = input_ids

    # ── attention_mask ───────────────────────────────────────────────────
    if attention_mask is None:
        attention_mask = _default_mask(model, result)
    elif attention_mask.shape != result.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")
    else:
        attention_mask = attention_mask.to(device=result.device, dtype=torch.bool)

    # 每行最后一位必须是有效 token（理由同 10 节）
    if not torch.all(attention_mask[:, -1]):
        raise ValueError(
            "each prompt must end in a valid token; left-pad variable-length batches"
        )

    # ── 初始窗口与预填充 ─────────────────────────────────────────────────
    # 只取最近 max_seq_len 个 token 作为可见窗口
    visible = result[:, -model.config.max_seq_len :]
    visible_mask = attention_mask[:, -model.config.max_seq_len :]

    # prefill 得到最后一位 logits（以及全部位置的，但我们只用最后一位）
    # 和每一层的 K/V 缓存
    logits, cache = prefill(model, visible, visible_mask)

    finished = torch.zeros(result.shape[0], device=result.device, dtype=torch.bool)

    # ── 主循环 ───────────────────────────────────────────────────────────
    # range(max_new_tokens) 从 0 开始，step 是"已经生成了几个"
    for step in range(max_new_tokens):
        # 从当前 logits 的最后一位采样
        next_id = sample_next_token(
            logits[:, -1],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )

        if eos_token_id is not None:
            # 已结束的行持续输出 EOS，保持 batch 形状一致
            next_id = torch.where(
                finished[:, None],
                torch.full_like(next_id, eos_token_id),
                next_id,
            )
            finished |= next_id[:, 0].eq(eos_token_id)

        # 拼到结果序列上
        result = torch.cat((result, next_id), dim=1)
        # mask 同步增长
        attention_mask = torch.cat(
            (attention_mask, torch.ones_like(next_id, dtype=torch.bool)), dim=1
        )

        # 全部结束就提前退出
        if eos_token_id is not None and torch.all(finished):
            break

        # 已经是最后一个 token 了，不需要再做无用的 decode
        if step + 1 == max_new_tokens:
            break

        # ── 判断缓存是否还有增长空间 ─────────────────────────────────────
        # 从第 0 层的 K 读出当前缓存长度
        cached_len = cache[0][0].shape[-2]

        if cached_len < model.config.max_seq_len:
            # ── 正常路径：增量解码一个 token ─────────────────────────────
            # visible_mask 也要同步增长一位，才能覆盖"历史前缀 + 新 token"
            # （这是第 6 步强调的那个陷阱：mask 必须覆盖缓存前缀）
            visible_mask = torch.cat(
                (visible_mask, torch.ones_like(next_id, dtype=torch.bool)), dim=1
            )
            logits, cache = decode_one(
                model, next_id, cache, attention_mask=visible_mask
            )
        else:
            # ── 窗口已满：重新 prefill ───────────────────────────────────
            # Sliding the window changes which token is position zero. Re-prefill
            # the new window so RoPE positions exactly match ordinary generation.
            # （滑动窗口会改变"哪个 token 是位置 0"。重新预填充新窗口，
            #   才能让 RoPE 的位置与普通生成完全一致。）
            #
            # 为什么不能直接裁掉最早的 K/V？
            #   因为旧 cache 中的 RoPE 位置基于【滚动前】的索引。
            #   窗口滑动后，参考实现把新窗口的第一个 token 重新视为位置 0，
            #   而旧 cache 里它的旋转角是按旧位置算的 —— 直接裁掉得不到
            #   新位置语义。所以只能重算。
            #
            # 那一步暂时失去增量计算的优势，但能保证两条教学实现一致。
            # （这是本仓库选择的有限窗口策略，不代表已实现长期位置外推
            #   或 ring-buffer cache 等更复杂的方案。）
            visible = result[:, -model.config.max_seq_len :]
            visible_mask = attention_mask[:, -model.config.max_seq_len :]
            logits, cache = prefill(model, visible, visible_mask)

    return result


# ============================================================================
# 第11步：窗口满时为什么重新 prefill（再说清楚一点）
# ============================================================================
# Cache 长度不能超过 max_seq_len。窗口已满又生成新 token 时，
# 参考实现会保留最近窗口，并把其中第一个 token 重新视作位置 0：
#
#     visible = result[:, -max_seq_len:]
#
# 旧 cache 中的 RoPE 位置基于滚动前的索引，
# 直接裁掉最早 K/V 并不能得到参考实现的新位置语义。
# 因此缓存版本在窗口滚动时重新 prefill。
#
# ── 换个角度看这件事 ─────────────────────────────────────────────────────
# 位置语义是【全局约定】的：模型训练时见过的是"第 n 个 token 用位置 n"。
# 一旦窗口滑动，绝对的"第几个"就变了。要么接受位置语义的改变（需要重算），
# 要么引入更复杂的相对位置 / 位置插值方案。
# 本教学实现选择"重算以保持一致"，是最保守也最容易验证的做法。
# ============================================================================


# ============================================================================
# 第12步：命令行接口
# ============================================================================
def parse_args():
    """定义命令行参数。

    注意默认 temperature=0.0（贪心）—— 与 10 节默认 0.8 不同。
    原因是本脚本的用途是【验证缓存正确性】，
    贪心解码结果确定，便于与未缓存版本逐 token 比较。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="清晨，")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    """脚本入口：先打印数值误差，再做缓存生成。"""
    args = parse_args()

    # 函数内 import，理由同 10 节（避免与无关的 train 模块冲突）
    from train import load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 这里用 _ 丢弃 checkpoint 字典（不需要 step 元数据）
    model, tokenizer, _ = load_checkpoint(args.checkpoint, device=device)
    model.eval()

    # 编码 prompt 并保留 batch 维 → (1, T_prompt)
    ids = torch.tensor(
        [tokenizer.encode(args.prompt, add_bos=True)], dtype=torch.long, device=device
    )

    generator = torch.Generator(device=device).manual_seed(args.seed)

    # ── 第一行输出：数值误差 ─────────────────────────────────────────────
    # :.3e 是科学计数法保留 3 位小数（如 1.234e-07）。
    # CPU float32 下通常小于 1e-6；具体尾数取决于设备和浮点运算次序。
    print(f"cached/full max_abs_error={cache_equivalence_error(model, ids):.3e}")

    generated = generate_with_kv_cache(
        model,
        ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
        generator=generator,
    )
    # 第二行是生成文本；默认模型很小，文本流畅度不用来判断 cache 是否正确
    print(tokenizer.decode(generated[0].tolist()))


# ============================================================================
# 第13步：运行与核对
# ============================================================================
# Checkpoint 可由 Task 28 的训练脚本产生：
#
#     python .../task_28_next_token_training/train.py --steps 80 \
#       --checkpoint /tmp/minimind_demo.pt
#
# Greedy 缓存生成的命令是：
#
#     python .../task_30_kv_cache/kv_cache.py \
#       --checkpoint /tmp/minimind_demo.pt --prompt "清晨，" --max-new-tokens 20
#
# 第一行形如：
#
#     cached/full max_abs_error=1.234e-07
#
# CPU float32 下通常小于 1e-6。第二行是生成文本。
#
# 也可以检查缓存采样参数：
#
#     python .../kv_cache.py --checkpoint /tmp/minimind_demo.pt --prompt "清晨，" \
#       --max-new-tokens 20 --temperature 0.8 --top-k 20 --top-p 0.9 --seed 0
#
# 缓存性质也收录在 Block 3 测试中：
#
#     python -m unittest discover -s tests -p 'test_block3.py' -v
#
# 测试覆盖：cache 层数与 n_kv_heads shape、每次 decode 后的长度变化、
# cached/full logits 容差、两种 greedy 生成的 token 一致性、
# padding mask 沿 cache 的传递，以及跨过 max_seq_len 后与未缓存窗口策略的一致性。
#
# 参考：Hugging Face Caching
#       https://huggingface.co/docs/transformers/main/cache_explanation
#       https://huggingface.co/docs/transformers/kv_cache
# ============================================================================


if __name__ == "__main__":
    main()


# ============================================================================
# 补充知识点：KV Cache 的收益与代价
# ============================================================================
# ── 复杂度对照（生成 n 个 token，上下文长度 L）────────────────────────────
#                     每步计算量        生成 n 步总计算量      显存
#   ───────────────   ──────────────    ────────────────────   ──────────────
#   未缓存            O(L)              O(n·L) ≈ O(n²)         只存当前激活
#   有缓存            O(L)（attention   O(n·L) 但省掉了         额外 O(L) 的
#                     本身没变，省的是   n 次 K/V 投影与          K/V 存储
#                     投影与 block 计算） block 计算
#
# 准确的表述是：**缓存没有降低单步 attention 的复杂度**（新 query 仍要与
# 全部 key 算 score），它省掉的是旧 token 的 K/V 投影和逐层 block 计算。
# 在 L 很大时，这部分省的运算量相当可观。
#
# ── 真正的瓶颈往往是显存带宽 ─────────────────────────────────────────────
# 每生成一个 token 都要把整个 KV Cache 从显存读一遍。
# Cache 大小 ≈ 2 × n_layers × n_kv_heads × L × head_dim × B × 精度字节数。
# 这也是 GQA（减少 n_kv_heads）和量化 KV Cache 这类技术存在的理由。
#
# ── 三个容易写错的地方 ───────────────────────────────────────────────────
#   1. RoPE 的 start_pos 没接上 past_len  → shape 全对，logits 悄悄偏离
#   2. mask 只传了当前一步，丢了历史前缀  → 前缀中的 PAD 会被当成有效内容
#   3. 窗口满时直接裁掉旧 K/V 而不重算    → 位置语义与参考实现不一致
# 这三个都属于"不报错但结果错"的类型，只有数值对照能抓出来。
# ============================================================================


# ============================================================================
# 核心公式回顾
# ============================================================================
# (1) 缓存结构
#
#         past_key_values = [ (K_l, V_l) for l in 0..n_layers-1 ]
#         K_l, V_l: (B, n_kv_heads, past_len, head_dim)      ← 未 repeat_kv
#
# (2) prefill 与 decode 的输入
#
#         prefill:    input_ids (B, L_prompt)   → logits (B,L_prompt,V), cache
#         decode_one: input_id  (B, 1)          → logits (B,1,V),          cache
#         每次 decode 后：past_len ← past_len + 1
#
# (3) 非方阵 causal 条件（缓存解码时）
#
#         q_positions = [past_len, past_len + t)
#         k_positions = [0, past_len + t)
#         allowed[i,j] = (k_positions[j] ≤ q_positions[i]) 且 mask[j]
#
# (4) RoPE 位置对齐
#
#         start_pos = past_len
#         cache(start_pos=p, seq_len=1) == cache(start_pos=0, seq_len=p+1)[p:p+1]
#
# (5) 等价性判据
#
#         max | full_logits - cached_logits |  ~ 1e-7 ~ 1e-6   （CPU float32）
#         greedy token ids 则要求【逐项完全相同】
#
#     注意这是"量级判据"，不是一个硬阈值：
#     实测同一个 checkpoint 可能是 1.4e-06，取决于浮点累加次序与算子实现。
#     真正的判据是"误差远小于 logits 本身的量级"，
#     以及 greedy 解码出来的 token 序列逐项一致。
# ============================================================================


# ============================================================================
# 总结
# ============================================================================
# 1. KV Cache 的道理一句话：旧 token 的 K/V 不会因为新 token 出现而改变，
#    所以推理时可以把它们留下来复用。
#    省掉的是旧 token 的 K/V 投影与逐层 block 计算，不是 attention 本身。
#
# 2. cache 按层保存，形状是 (B, n_kv_heads, past_len, head_dim) ——
#    注意是 n_kv_heads 而不是 n_heads，存的是【未 repeat_kv】的版本。
#    这正是 GQA 能直接缩小缓存体积的原因。
#
# 3. 三个核心函数的分工很清楚：
#      prefill        一次处理整个 prompt
#      decode_one     只处理一个 (B,1) 的新 token
#      logits_with_kv_cache  串起前两者，产出与并行 forward 可比的输出
#
# 4. 验收标准是【数值等价】，不是"能跑完"：
#     max_abs_error 通常在 1e-7 ~ 1e-6 量级（CPU float32，实测可能是 1.4e-06），
#     greedy ids 应逐项相同。
#     这个对照一次性覆盖了 cache 对应关系、序列轴追加、RoPE 的 start_pos、
#     非方阵 mask 对齐、padding mask 传递五件事 ——
#     仅靠"代码里有个 list"或"生成长度对"是区分不出来的。
#
# 5. start_pos = past_len 是本节最关键的细节。
#    漏掉它不会报错、shape 全对、生成也能跑完，只有 logits 悄悄偏离 ——
#    典型的静默错误，只能靠数值对照发现。
#
# 6. mask 必须覆盖缓存前缀 + 当前 token。
#    decode_one 的缺省值会把整个前缀视为有效，这在 prompt 含左 PAD 时是错的，
#    所以生成循环总是显式传完整 mask。
#
# 7. 窗口满时重新 prefill：滑动窗口会改变"谁是位置 0"，
#    旧 cache 的 RoPE 位置基于旧索引，裁掉得不到新语义。
#    这一步暂时失去增量优势，但保证了与未缓存实现的数值一致。
#
# 8. 与 10 节的关系：10 节是【数值对照物】，本节是【优化实现】。
#    两者用同一套采样规则，因此可以直接比较 token 序列。
# ============================================================================
