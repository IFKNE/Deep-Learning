"""
=============================================================================
自回归生成与采样（Greedy / Temperature / Top-k / Top-p）
=============================================================================
源文件: exercises/block_03_transformer/task_29_generate_sampling/generate.py

任务：训练时一次 forward 返回所有位置的 logits；生成时未来 token 还不存在，
     只能把刚生成的 id 接回序列，再算下一步：

         当前上下文 -> 最后一位 logits -> 选 token -> 追加 -> 下一轮

     这一节先实现【未缓存】版本。它会重复计算上下文，但逻辑直白，
     也为 11 节的 KV Cache 提供数值对照。

  输入：一个 checkpoint、一段 prompt
  输出：prompt + 生成文本

依赖：argparse、pathlib、sys、torch，以及同模块的 train（见下方路径处理）。
      本文件可独立运行。
=============================================================================
"""

# ============================================================================
# 第1步：导入依赖与路径处理
# ============================================================================
import argparse                      # 命令行参数
from pathlib import Path             # 路径操作
import sys                           # 改 sys.path

import torch                         # 张量、Generator、multinomial


# ── 让本文件能 import 到 train（从而拿到 load_checkpoint）────────────────
# BLOCK 是上一级目录（原始仓库里的 block_03_transformer）
BLOCK = Path(__file__).resolve().parents[1]
TRAINING = BLOCK / "task_28_next_token_training"

# 整理到 my_learn/transformer/ 之后，同级目录没有 task_28，
# 因此增加一层回退，指向原始仓库位置；原始逻辑保持不变。
if not TRAINING.exists():
    TRAINING = Path(
        r"d:\桌面\深度学习\quickly_access_to_deeplearning-main"
        r"\exercises\block_03_transformer\task_28_next_token_training"
    )

if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))


# ============================================================================
# 第2步：为什么取 logits[:, -1]
# ============================================================================
#     input_ids: (B,T)
#     logits:    (B,T,V)
#
# 第 t 行 logits 预测的是"该位置【之后】的 token"。当前上下文停在 T-1，
# 所以下一个 token 来自：
#
#     next_logits = logits[:, -1]     # (B,V)
#
# ── 一个常见错误 ──────────────────────────────────────────────────────────
# 误取 logits[:, 0] 会让每一步都根据序列开头来生成 ——
# 代码照样能跑，但输出会莫名其妙地重复开头的内容。
#
# next_logits 是 (B,V)：每个 batch 行一份词表分布。
# sample_next_token 为每行返回一个 id，shape 为 (B,1)，
# 再把它拼回 input_ids 的第 1 维。
# ============================================================================


# ============================================================================
# 第3步：sample_next_token —— 采样规则的核心
# ============================================================================
def sample_next_token(
    logits, temperature=1.0, top_k=None, top_p=None, generator=None
):
    """Sample one id from ``(batch, vocab)`` logits; temperature 0 is greedy.

    从 (batch, vocab) 的 logits 中采样一个 id；temperature=0 时退化为贪心。

    参数:
        logits:      (B,V) 最后一个位置的未归一化分数
        temperature: τ，0 表示贪心；>0 表示按温度采样
        top_k:       只保留概率最高的 k 个候选（None 表示不过滤）
        top_p:       nucleus 阈值，落在 (0,1]
        generator:   torch.Generator，控制可复现性

    返回:
        (B,1) 的整数张量

    分步说明 / 计算过程：
        (1) 参数校验
        (2) τ=0 → argmax 直接返回
        (3) 除以 τ 做温度缩放
        (4) top-k 过滤
        (5) top-p（nucleus）过滤
        (6) softmax → multinomial 采样
    """
    # ── (1) 参数校验 ─────────────────────────────────────────────────────
    # ndim 是张量的维数，(B,V) 应该是 2 维
    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, vocab_size)")
    if temperature < 0:
        # 负温度没有定义：除以负数会把高 logit 压成低概率，语义反转
        raise ValueError("temperature must be non-negative")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive")
    if top_p is not None and not 0 < top_p <= 1:
        # 链式比较等价于 (0 < top_p) and (top_p <= 1)
        raise ValueError("top_p must be in (0, 1]")

    # ── (2) 贪心分支 ─────────────────────────────────────────────────────
    if temperature == 0:
        # argmax 沿最后一维取最大值的下标；
        # keepdim=True 保留长度 1 的轴，输出 (B,1) 而不是 (B,) ——
        # 这样才能直接 cat 到 (B,T) 的序列上
        #
        # 注意：这个分支【直接返回】，不再应用 top-k/top-p。
        # 这是有意的设计：贪心本身已经是确定性的选择，
        # 过滤只是缩小候选集，不改变 argmax 的结果。
        return logits.argmax(dim=-1, keepdim=True)

    # ── (3) 温度缩放 ─────────────────────────────────────────────────────
    # 下面所有操作都在 filtered 上进行，不修改调用方传入的 logits
    filtered = logits / temperature
    # 回顾 softmax(z/τ)：
    #   0 < τ < 1 → z/τ 被放大 → 分布更尖，更偏向高 logit
    #   τ > 1     → z/τ 被压缩 → 分布更平，低 logit 更容易被采到
    # 温度只改变同一组 logits 的随机性，不能补回模型没有学到的知识。

    # ── (4) top-k 过滤 ───────────────────────────────────────────────────
    if top_k is not None:
        # k 不能超过词表大小，min 取较小值防止 topk 报错
        k = min(top_k, filtered.shape[-1])

        # topk(...).values 返回按降序排好的前 k 个值
        # [:, -1:] 取第 k 个（即最小值，也就是阈值），形状 (B,1)
        # 用 (B,1) 的阈值与 (B,V) 广播比较
        threshold = torch.topk(filtered, k, dim=-1).values[:, -1:]

        # 低于阈值的候选填成 -inf：softmax 后 exp(-inf)=0，概率恰好为 0，
        # 也就永远不会被 multinomial 抽到
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))

        # 严格来说，若阈值处出现相同 logit，这个"按阈值"实现会保留所有并列项，
        # 因此候选数可能略多于 k。最高 logit 唯一时，top_k=1 与 greedy 相同。

    # ── (5) top-p（nucleus）过滤 ─────────────────────────────────────────
    if top_p is not None and top_p < 1:
        # 先按概率【降序】排列。
        # sort 同时返回排序后的值和它们的原始下标 ——
        # 下标后面要用来把 mask 映射回原位置。
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)

        # softmax 后累积求和：cumulative 第 i 项 = 前 i+1 个概率之和
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # 先标出"累计概率已经超过阈值"的位置 —— 这些都是要被删掉的
        remove_sorted = cumulative > top_p

        # Shift right so the first token crossing p is retained. This also
        # guarantees that every row keeps at least one finite candidate.
        # （右移一位，让跨过阈值的那个 token 被保留；同时保证每行
        #   至少留下一个有限候选。）
        #
        # 举例：probabilities = [0.60, 0.25, 0.10, 0.05]，top_p = 0.70
        #   累计 = [0.60, 0.85, 0.95, 1.00]
        #   remove 原本 = [False, True, True, True]
        #   右移后      = [False, False, True, True]
        #   于是保留前两项 —— 正好是"累计到 0.70 所需的最小前缀"。
        #   若不做右移，会连 0.60 那一项也删掉，得到空集、softmax 全是 nan。
        #
        # 具体做法：把 [:, :-1] 赋给 [:, 1:]，整体向右挪一格。
        # .clone() 是必需的 —— 切片赋值时源和目标是同一块内存的重叠区域，
        # 不克隆的话源数据会被前一步写坏（经典的自赋值陷阱）。
        remove_sorted[:, 1:] = remove_sorted[:, :-1].clone()
        # 第一位强制保留：保证每行至少有一个候选
        remove_sorted[:, 0] = False

        # ── 把排序空间的 mask 映射回原始词表顺序 ─────────────────────────
        # scatter(dim=-1, index=sorted_indices, src=remove_sorted)：
        #   在 remove 的第 i 行，把 remove_sorted[i] 的每个值
        #   按 sorted_indices[i] 指定的位置写回去。
        #   scatter 是 gather 的逆操作：gather 按 index 取，scatter 按 index 放。
        # 结果：remove[b, v] = True 表示词表项 v 应当被删除。
        remove = torch.zeros_like(remove_sorted).scatter(
            dim=-1, index=sorted_indices, src=remove_sorted
        )

        filtered = filtered.masked_fill(remove, float("-inf"))

    # ── (6) softmax + 采样 ───────────────────────────────────────────────
    # -inf 经过 softmax 后是 0，所以被过滤掉的候选概率恰好为 0
    probabilities = torch.softmax(filtered, dim=-1)

    # torch.multinomial 按给定概率分布抽 1 个下标 → (B,1)
    # generator 参数保证可复现：同一 seed + 同一分布 → 同一结果
    return torch.multinomial(probabilities, 1, generator=generator)


# ============================================================================
# 第4步：四种选择规则的对照
# ============================================================================
# ── Greedy（贪心）────────────────────────────────────────────────────────
#     next_id = logits.argmax(dim=-1, keepdim=True)
#
#   - 每一步都取概率最大的那个，结果【完全确定】；
#   - 不一定给出最自然的文本（容易陷入重复），
#     但适合核对最后位置、EOS 和 cached/uncached 等价性。
#
# ── Temperature（温度）───────────────────────────────────────────────────
#   设 z 为最后一位的 logits 向量（(B,V) 中的一行，shape 为 (V,)）：
#
#                    exp(z_i / τ)
#         p_i = ────────────────────────,      p = softmax(z / τ)
#                Σ_j exp(z_j / τ)
#
#   - 0 < τ < 1：分布更尖，更偏向高 logit；
#   - τ > 1    ：分布更平，低 logit 更容易被采到；
#   - τ < 0    ：没有定义，代码抛出 ValueError。
#
# ── Top-k ────────────────────────────────────────────────────────────────
#   实现先找到第 k 大 logit 的阈值，把更低的候选设为 -inf，再 softmax。
#
#         top_k=None  不过滤
#         top_k<=0    报错
#         top_k>V     截到 V
#
# ── Top-p / nucleus ──────────────────────────────────────────────────────
#   Top-p 先按概率降序排列，再保留累计概率达到 p_top 所需的【最小前缀】。
#
#         probabilities = [0.60, 0.25, 0.10, 0.05]
#         top_p = 0.70
#         → 第一个候选只累计到 0.60，所以还要保留第二个，
#           采样集合为前两项。
#
#   实现先标出累计概率超过阈值的位置，再把删除 mask 右移一格，
#   保证跨过 p_top 的那个 token 留下。
#
#         top_p=None 或 1  不过滤
#         0 < top_p < 1    nucleus 过滤
#         top_p<=0 或 >1   报错
#
# ── 同时设置 top-k 与 top-p 时 ───────────────────────────────────────────
#   代码先做 top-k，再在剩余候选中做 top-p。
#   顺序不能颠倒：若先做 top-p，再做 top-k，两者作用的候选集不同，
#   结果会不同。
# ============================================================================


# ============================================================================
# 第5步：generate —— 未缓存的自回归主循环
# ============================================================================
@torch.no_grad()
def generate(
    model,
    input_ids,
    max_new_tokens,
    temperature=1.0,
    top_k=None,
    top_p=None,
    eos_token_id=None,
    generator=None,
    attention_mask=None,
):
    """Readable non-cached reference loop; task 30 removes repeated work.

    可读性优先的未缓存参考循环；11 节会消除其中的重复计算。

    参数:
        model:            MiniMindCore 实例（已 eval）
        input_ids:        (B,T_prompt)
        max_new_tokens:   最多生成多少个新 token
        temperature/top_k/top_p: 采样参数
        eos_token_id:     终止 token；生成它之后该行固定输出 EOS
        generator:        随机数发生器
        attention_mask:   与 input_ids 同 shape

    返回:
        (B, T_prompt + 生成数) 的完整 id 序列

    注意:
        max_new_tokens 是【生成上限】。若始终没有 EOS：
            output_length = prompt_length + max_new_tokens
        Tokenizer 解码时会隐藏 BOS/EOS/PAD，因此检查长度要看 token ids，
        不能只数字符串。
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
        # 短路返回，避免走完整个流程
        return input_ids

    result = input_ids

    # ── 准备 attention_mask ──────────────────────────────────────────────
    if attention_mask is None:
        # 从模型配置里读 PAD id（这里与 08 节的写法略有不同：
        # 08 节用 self.config，本函数是模块级函数，所以从 model.config 取）
        pad_id = model.config.pad_token_id
        attention_mask = (
            torch.ones_like(result, dtype=torch.bool)
            if pad_id is None
            else result.ne(pad_id)
        )
    elif attention_mask.shape != result.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")
    else:
        attention_mask = attention_mask.to(device=result.device, dtype=torch.bool)

    # ── 变长 batch 与上下文窗口 ──────────────────────────────────────────
    # 生成统一读取每行【最后位置】，因此变长 batch 接受左 padding：
    #
    #     [PAD, PAD, BOS, t0, t1]   # 支持
    #     [BOS, t0, t1, PAD, PAD]   # 拒绝（右 padding 会让最后一位落在 PAD 上）
    #
    # 接口约定 attention_mask 与 ids 同 shape，最后一列全为有效 token。
    if not torch.all(attention_mask[:, -1]):
        raise ValueError(
            "each prompt must end in a valid token; left-pad variable-length batches"
        )

    # 每行是否已生成过 EOS。全 False 起步。
    finished = torch.zeros(result.shape[0], device=result.device, dtype=torch.bool)

    # ── 主循环 ───────────────────────────────────────────────────────────
    for _ in range(max_new_tokens):
        # 只保留最近 max_seq_len 个 token 作为可见窗口：
        #     context = result[:, -model.config.max_seq_len:]
        # 完整 result 仍保留所有生成 id，但被滑出窗口的旧 token 不再影响预测。
        # 这是一种明确的截断策略，不是无限上下文。
        context = result[:, -model.config.max_seq_len :]
        context_mask = attention_mask[:, -model.config.max_seq_len :]

        # 完整 forward（无缓存）：每一步都要重算整个可见窗口
        logits, _ = model(context, attention_mask=context_mask)

        # 取最后一位的分布，交给采样函数
        next_id = sample_next_token(
            logits[:, -1],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )

        if eos_token_id is not None:
            # 已经结束的行强制继续输出 EOS：
            # 这样 batch 里各行的长度虽然不同，但形状始终一致，能继续拼接。
            # torch.where(条件, A, B)：条件为 True 取 A，否则取 B。
            # finished[:, None] 把 (B,) 变 (B,1) 才能与 (B,1) 的 next_id 广播。
            next_id = torch.where(
                finished[:, None],
                torch.full_like(next_id, eos_token_id),
                next_id,
            )
            # 更新哪些行刚刚生成了 EOS。
            # next_id[:, 0] 把 (B,1) 压成 (B,) 才能与 finished 比较。
            finished |= next_id[:, 0].eq(eos_token_id)

        # 拼到序列末尾
        result = torch.cat((result, next_id), dim=1)
        # mask 同步增长一位（新 token 一定是有效的）
        attention_mask = torch.cat(
            (attention_mask, torch.ones_like(next_id, dtype=torch.bool)), dim=1
        )

        # 所有行都结束时提前退出，不必跑满 max_new_tokens
        if eos_token_id is not None and torch.all(finished):
            break

    return result


# ============================================================================
# 第6步：随机性怎么控制
# ============================================================================
# CLI 创建显式 generator：
#
#     generator = torch.Generator(device=device).manual_seed(seed)
#
# 在设备、软件环境、模型、prompt 和采样参数都相同时，
# 同一初始 seed 会得到相同序列。
#
# 显式 generator 的好处是：这一性质【不受全局随机状态干扰】。
# 如果用 torch.manual_seed 设置全局种子，
# 那么任何一次额外的随机调用（比如别处的一次 randn）都会让结果漂移。
# ============================================================================


# ============================================================================
# 第7步：命令行接口
# ============================================================================
def parse_args():
    """定义命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)

    # required=True：必须提供 checkpoint，否则 argparse 直接报错退出
    parser.add_argument("--checkpoint", type=Path, required=True)
    # 默认 prompt 取语料开头，保证有意义的续写
    parser.add_argument("--prompt", default="清晨，")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    # top-p 默认 None（不启用），help 里说明取值范围
    parser.add_argument(
        "--top-p", type=float, default=None, help="optional nucleus threshold in (0, 1]"
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    """脚本入口：载入 checkpoint，编码 prompt，生成并打印文本。"""
    args = parse_args()

    # CLI-only dependency: keeping it local makes this module safe to import in
    # a larger pytest process that may already contain an unrelated `train`.
    # （只在 CLI 路径下才 import，这样本模块被 pytest 等进程 import 时更安全 ——
    #   那个进程里可能已经有一个不相干的名为 train 的模块，
    #   放在函数内部可以推迟到真正用到时才解析。）
    from train import load_checkpoint

    # 有 GPU 就用 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load_checkpoint 重建模型与分词器（见 09 节）
    model, tokenizer, checkpoint = load_checkpoint(args.checkpoint, device=device)
    # 切到评估模式：关闭 Dropout 等训练期行为
    model.eval()

    # ── 编码 prompt ──────────────────────────────────────────────────────
    # add_bos=True：与训练时一致，序列以 <bos> 开头
    # （训练里 CharacterTokenizer.encode 也是 add_bos=True, add_eos=True）
    prompt_ids = tokenizer.encode(args.prompt, add_bos=True)

    # 用双层列表构造 (1, T_prompt) 的批量维度。
    # 即使只有一条输入也要保留 batch 维，因为模型要求输入是 (B,T)。
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    # 显式 generator，保证同一 seed 结果可复现
    generator = torch.Generator(device=device).manual_seed(args.seed)

    output = generate(
        model,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_token_id,
        generator=generator,
    )

    # 打印 checkpoint 的训练步数，确认用的是哪一个存档
    print(f"checkpoint_step={checkpoint['step']}")
    # output[0] 取第一条样本，.tolist() 转成 Python 列表再解码
    # 解码时会自动隐藏 BOS/EOS/PAD
    print(tokenizer.decode(output[0].tolist()))


# ============================================================================
# 第8步：运行与核对
# ============================================================================
# Checkpoint 可由 Task 28 的训练脚本产生：
#
#     python exercises/block_03_transformer/task_28_next_token_training/train.py \
#       --steps 80 --checkpoint /tmp/minimind_demo.pt
#
# 对应的采样命令是：
#
#     python exercises/block_03_transformer/task_29_generate_sampling/generate.py \
#       --checkpoint /tmp/minimind_demo.pt --prompt "清晨，" \
#       --max-new-tokens 20 --temperature 0.8 \
#       --top-k 20 --top-p 0.9 --seed 0
#
# 第一行形如：
#
#     checkpoint_step=80
#
# 随后打印 prompt 与生成文本。
# 【提醒】默认语料和训练步数很小，文本流畅度不用来判断采样逻辑是否正确。
#
# 采样边界也收录在 Block 3 测试中：
#
#     python -m unittest discover -s tests -p 'test_block3.py' -v
#
# 测试覆盖：greedy/argmax 一致性、无并列时的 top-1、nucleus 候选范围、
# 非法参数、固定 generator 的可复现性、EOS 与输出长度、右 padding 报错，
# 以及窗口滚动时 ids 与 mask 的同步截取。
#
# 参考：The Curious Case of Neural Text Degeneration（nucleus sampling）
#       https://arxiv.org/abs/1904.09751
# ============================================================================


if __name__ == "__main__":
    main()


# ============================================================================
# 补充知识点：采样参数怎么调
# ============================================================================
# ── 常见组合与效果 ────────────────────────────────────────────────────────
#   τ      top-k   top-p   效果
#   ────   ─────   ─────   ─────────────────────────────────────────────
#   0      -       -       完全确定，适合做数值对照与复现实验
#   0.7    40      0.9     保守，文本稳定但多样性一般
#   0.8    20      -       教学演示的常用组合
#   1.0    0/None  0.95    较活泼，创意写作常用
#   1.2+   0/None  0.95    发散，容易跑题或产生不连贯内容
#
# ── top-k 与 top-p 的分工 ────────────────────────────────────────────────
#   top-k 是【固定数量】的截断：无论分布多平，都只留 k 个。
#     它的问题在于 k 是绝对的：分布很尖时 k 太大（引入了不该有的低概率词），
#     分布很平时 k 又太小（可能切掉了合理的候选）。
#   top-p 是【自适应】的截断：按累计概率决定保留多少，分布尖时自动留得少，
#     平的时候自动留得多。这是它比 top-k 更常用的原因。
#
# ── 一个反直觉的点 ───────────────────────────────────────────────────────
# 温度不能"创造"知识。若模型对某个续写毫无概念，所有 logits 都很低且接近，
# 提高温度只会让采样更随机，不会让输出更合理。
# 温度调整的是"在已知分布上探索多少"，不是"知道多少"。
# ============================================================================


# ============================================================================
# 核心公式回顾
# ============================================================================
# (1) 下一步的分布来自最后一位
#
#         next_logits = logits[:, -1]          (B,V)
#
# (2) 温度缩放
#
#                        exp(z_i / τ)
#         p_i = ──────────────────────────
#                Σ_j exp(z_j / τ)
#
# (3) top-k 阈值
#
#         threshold = kth_largest(z)
#         z_i ← -∞   当 z_i < threshold
#
# (4) top-p（nucleus）
#
#         排序后累计： cumulative_i = Σ_{j≤i} softmax(z)_j
#         保留满足 cumulative_i - p_top ≤ 0 的最小前缀
#         实现：remove = cumulative > p_top，再右移一格
#
# (5) 采样
#
#         next_id ~ Multinomial( softmax(filtered) )
#         generator 固定 ⇒ 结果可复现
# ============================================================================


# ============================================================================
# 总结
# ============================================================================
# 1. 自回归生成的全部秘密就是三行：
#      context = result[:, -max_seq_len:]
#      logits, _ = model(context, attention_mask=...)
#      next_id = sample_next_token(logits[:, -1], ...)
#    然后拼回去、循环。
#
# 2. 取 logits[:, -1] 是因为第 T-1 行预测的正是"下一个" token。
#    误取 [:, 0] 会让每一步都基于序列开头生成 —— 能跑，但结果莫名其妙。
#
# 3. 四种规则的层次关系：
#      温度  → 调整整体分布形状（缩放）
#      top-k → 按排名截断（固定数量）
#      top-p → 按累计概率截断（自适应数量）
#    先 top-k 再 top-p，顺序固定。
#
# 4. top-p 实现里"右移一格"的那一行是整段代码最精妙的地方：
#    它把"累计超过阈值"变成"跨过阈值的那个 token 保留下来"，
#    同时天然保证每行至少留一个候选，避免 softmax 全 -inf 产生 nan。
#
# 5. 用显式 torch.Generator 而不是全局种子，可复现性才不会被别处的随机调用破坏。
#
# 6. 变长 batch 必须【左 padding】，因为循环统一取 logits[:, -1]；
#    右 padding 会让最后一位落在 PAD 上，接口直接拒绝。
#
# 7. max_new_tokens 是上限。某个 batch 行先遇到 EOS 后持续填 EOS，
#    其余行继续，直到全部结束或达到上限。
#
# 8. 这一节的实现是【未缓存】的：每步重算整个窗口。
#    它的价值在于逻辑直白、可作为 11 节 KV Cache 的数值对照物。
# ============================================================================
