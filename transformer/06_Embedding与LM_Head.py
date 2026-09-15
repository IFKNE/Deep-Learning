"""
=============================================================================
Embedding、LM Head 与 Weight Tying —— 完整带注释版
=============================================================================
源文件: exercises/block_03_transformer/task_25_embedding_lm_head/language_model.py

任务：说清语言模型的"入口"和"出口"——
     Tokenizer 把字符变成整数 id；embedding 把 id 查成向量；
     模型处理完后，LM head 再把每个位置的向量投影回词表得到 logits。

  核心：一张矩阵 E: (V,D) 有两种用法
        查表（按 id 取行）        → token embedding
        乘转置（H 投到 V 维）     → LM head，logits = H · Eᵀ

        权值共享（weight tying）：让两个模块【引用同一个 Parameter 对象】
            self.lm_head.weight = self.token_embedding.weight

输入：input_ids: (B,T)  整数 id
输出：logits: (B,T,V)   每个位置上对词表中每个词的未归一化分数
      loss:   标量或 None

依赖：torch、torch.nn、torch.nn.functional。本文件可独立运行。
=============================================================================
"""

# ============================================================================
# 第1步：导入依赖
# ============================================================================
import torch                               # tensor 用于 smoke test
from torch import nn                       # nn.Embedding / nn.Linear / nn.Module
from torch.nn import functional as F       # F.cross_entropy


# ============================================================================
# 第2步：各张量的形状约定
# ============================================================================
# 设词表大小为 V，hidden dimension 为 D：
#
#     token_embedding.weight: (V,D)       ← 每一行是一个 token 的向量
#     input_ids:              (B,T)       ← 整数 id
#     hidden:                 (B,T,D)     ← 查表后
#     lm_head.weight:         (V,D)       ← 与 embedding 同一张表
#     logits:                 (B,T,V)     ← 每个位置对全词表的打分
#
# ── logits 的含义 ─────────────────────────────────────────────────────────
# logits[b,t,v] 是样本 b 在位置 t 预测词表项 v 的【未归一化分数】。
#
#   - 训练时：cross_entropy 内部会做 log_softmax，所以不需要手动 softmax；
#   - 生成时：才由采样函数处理最后一个位置的 logits
#            （见 10_自回归生成与采样.py）。
#
# 这一点很重要：如果训练时手动 softmax 之后又丢给 cross_entropy，
# 就相当于做了两次 softmax，梯度会变得很小、训练异常缓慢。
# ============================================================================


# ============================================================================
# 第3步：一张矩阵，两种用法
# ============================================================================
# Embedding 表 E ∈ R^(V×D) 按 token id 查行。
# LM head 则使用同一张表的转置，把 hidden vector 投到 V 个词表分数：
#
#     H: (B,T,D),          H · Eᵀ : (B,T,V)
#
# 形状推导： (B,T,D) @ (D,V) → (B,T,V)
#   而 Eᵀ 的形状是 (D,V)，因为 E 是 (V,D)。
#
# ── PyTorch 的一个便利之处 ────────────────────────────────────────────────
# nn.Linear(D,V) 把 weight 存成 (V,D)（注意是 (out,in) 的顺序），
# 与 nn.Embedding(V,D) 的 weight 形状【完全相同】。
# 因此可以直接让两个模块引用同一个 Parameter：
#
#     self.lm_head.weight = self.token_embedding.weight
#
# 这一行是【赋值引用】，不是拷贝 —— 两个模块从此共用一个张量对象。
#
# ── 千万不要写成数值复制 ──────────────────────────────────────────────────
# 下面这种写法是错的（达不到权值共享的效果）：
#
#     self.lm_head.weight.data.copy_(self.token_embedding.weight.data)
#
# 它得到的只是"初始数值相同的两份参数"。训练开始后两者各自更新，很快就不一样了。
#
# ── 怎么验证真的共享了 ────────────────────────────────────────────────────
# 用 is 判断对象身份，而不是用 == 比较数值：
#
#     model.lm_head.weight is model.token_embedding.weight      # True
#
# ── 共享之后梯度怎么办 ────────────────────────────────────────────────────
# 来自输入 embedding 路径和输出分类路径的梯度会【累积到同一个张量】，
# optimizer 也只维护这一份 Parameter。
# 这正是权值共享能省参数、又能让两者互相约束的原因。
# ============================================================================


# ============================================================================
# 第4步：TinyLanguageModel —— 一个刻意简化的模型
# ============================================================================
class TinyLanguageModel(nn.Module):
    """用于演示 embedding / LM head / weight tying / loss mask 的最小模型。

    重要说明：这个模型【没有 attention】。
    为了让不同位置不至于完全相同，它额外加入可学习的位置表。
    它适合检查 embedding、LM head、权值共享和 loss mask 这几件事，
    但【不能】读取其他 token 的内容，不能当作语言模型主干来评价上下文能力。
    """

    def __init__(self, vocab_size, dim, max_seq_len, pad_token_id=0):
        """
        参数:
            vocab_size:   词表大小 V
            dim:          模型维度 D
            max_seq_len:  支持的最大序列长度（位置表的行数）
            pad_token_id: 填充 token 的 id，默认 0
        """
        super().__init__()

        # 存下来备用：forward 里要用 max_seq_len 做长度检查，
        # 用 pad_token_id 做 loss mask
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        # token embedding 表：(V,D)
        # nn.Embedding 就是一个可学习的查表：输入整数 id，输出对应的 D 维向量
        # 注意输入是整数张量（dtype 必须是 long），不是浮点
        self.token_embedding = nn.Embedding(vocab_size, dim)

        # 可学习的位置表：(max_seq_len, D)
        # 与 02 节的正弦编码不同，这张表【参与训练】，是 nn.Parameter。
        # 因为本模型没有 attention，不加位置信息的话，同一 token 在任何位置
        # 得到的表示都完全一样，模型连"位置"这个概念都没有。
        self.position_embedding = nn.Embedding(max_seq_len, dim)

        # LM head：把 (B,T,D) 投到 (B,T,V)
        # nn.Linear(D,V) 的 weight 形状是 (V,D)，与 token_embedding.weight 一致
        # bias=False：输出层不需要偏置，也为了让权值共享更干净
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # ── 权值共享（本文件的核心）────────────────────────────────────────
        # Both modules reference the same Parameter, rather than copying its value.
        # 注意是"引用同一个 Parameter"，而不是复制它的值。
        # 执行这一行之后：
        #     model.lm_head.weight is model.token_embedding.weight   →  True
        # 之后无论哪个模块更新这份权重，另一个都会同步变化。
        self.lm_head.weight = self.token_embedding.weight

    # ========================================================================
    # forward
    # ========================================================================
    def forward(self, input_ids, labels=None, attention_mask=None):
        """
        参数:
            input_ids:      (B,T) 整数 token id
            labels:         (B,T) 已经【错开一位】的监督目标，或 None（仅推理）
            attention_mask: (B,T) 布尔张量，True 表示该位置有效

        返回:
            logits: (B,T,V)
            loss:   标量（labels 为 None 时为 None）

        分步说明 / 计算过程：
            (1) 长度检查
            (2) token 查表 + 位置查表，两路相加得到 hidden
            (3) LM head 投影出 logits
            (4) 若有 labels，构造 mask 并算交叉熵
        """
        # 解包：(B,T)
        b, t = input_ids.shape

        # 位置表只有 max_seq_len 行，超长的序列会查表越界
        if t > self.max_seq_len:
            raise ValueError("sequence length exceeds max_seq_len")

        # torch.arange(t) 生成 [0,1,...,t-1]，即当前位置下标
        # device 与 input_ids 保持一致，否则后面相加会报设备不匹配
        pos = torch.arange(t, device=input_ids.device)

        # (1) self.token_embedding(input_ids) → (B,T,D)，按 id 查行
        # (2) self.position_embedding(pos)    → (T,D)，只按位置查，与 batch 无关
        #     [None, :, :] 在第 0 维插一个长度 1 的轴 → (1,T,D)
        #     这样 (B,T,D) + (1,T,D) 会沿 batch 维【广播】：
        #     同一位置对所有样本使用同一份位置向量
        # (3) 相加而不是拼接：所以位置维度必须等于 D
        x = self.token_embedding(input_ids) + self.position_embedding(pos)[None, :, :]

        # LM head 投影：(B,T,D) @ (D,V)ᵀ… 实际是 Linear，输出 (B,T,V)
        # 由于权值共享，这里用的就是 embedding 那张表
        logits = self.lm_head(x)

        # loss 默认为 None：调用方只想要 logits 时（推理）不需要算损失
        loss = None

        if labels is not None:
            # ── 构造 loss mask：把不该参与监督的位置标成 -100 ───────────────
            # .clone() 很重要：不能直接改调用方传进来的 labels 张量，
            # 否则会意外污染外部数据（比如 DataLoader 复用的 batch）
            targets = labels.clone()

            # label 等于 pad_token_id 的位置不算损失 —— PAD 是补齐用的，
            # 让它参与训练等于逼模型去预测"填充符"，纯属噪声
            targets[targets == self.pad_token_id] = -100

            if attention_mask is not None:
                # query 位置无效时，它产生的预测同样不该被监督
                # ~attention_mask.bool() 是先转 bool 再取反：
                #   ~True = False（有效位置保持原样）
                #   ~False = True（无效位置被标成 -100）
                targets[~attention_mask.bool()] = -100

            # 布尔张量：标记哪些位置是真正参与计算的
            valid = targets.ne(-100)

            # ── 只对有效位置算交叉熵 ────────────────────────────────────────
            # logits[valid]  : (N_valid, V)，被选中的行按布尔掩码取出
            # targets[valid]: (N_valid,)
            # F.cross_entropy 内部做 log_softmax + NLL，所以传入的是原始 logits
            #
            # 为什么可以手动筛选之后再算 CE？
            #   因为 CE 的 reduction='mean' 是对"传入的样本"取平均，
            #   先筛掉无效位置再取平均，与"先算再按权重平均"结果一致，
            #   但避免了给无效位置分配计算量。
            #
            # if valid.any() else ...：整个 batch 都没有有效 target 时，
            #   logits[valid] 是空张量，cross_entropy 会返回 nan。
            #   这里返回 logits.sum() * 0.0：
            #     数值上是 0（有限的），
            #     但仍然【连在计算图上】—— loss.backward() 不会因为它是
            #     常量而报"does not require grad"的错。
            #   这是一个很实用的小技巧。
            loss = (
                F.cross_entropy(logits[valid], targets[valid])
                if valid.any()
                else logits.sum() * 0.0
            )

        # 返回两个值，与后续 MiniMindCore 的接口保持一致
        return logits, loss


# ============================================================================
# 第5步：TinyLanguageModel 的作用范围（再说一遍，因为很容易误用）
# ============================================================================
# language_model.py 这一小模型没有 attention。为了让不同位置不至于完全相同，
# 它额外加入可学习的位置表：
#
#     token embedding + learned position embedding
#
# 它适合检查 embedding、LM head、weight tying 和 loss mask，
# 却不能读取其他 token 的内容，不能当作语言模型主干来评价上下文能力。
#
# ── 与 Task 27 的关系 ─────────────────────────────────────────────────────
# Task 27 的 MiniMindCore 【不使用】这张 learned position table；
# 完整模型在每层 attention 内用 RoPE 处理 Q/K。
#
# 两套位置方案分别属于两个示例，没有同时叠加 ——
# 换句话说：MiniMindCore 里既没有 position_embedding，也没有正弦编码表，
# 位置信息的唯一来源就是 RoPE。
# ============================================================================


# ============================================================================
# 第6步：Next-token 标签由 Dataset 错开
# ============================================================================
# forward 【不自动移动 labels】，它接收的是已经错开一位的序列：
#
#     tokens: [BOS, t0, t1, t2, EOS]
#     input:  [BOS, t0, t1, t2]
#     label:  [t0,  t1, t2, EOS]
#
#     位置 0 的输入是 BOS，要预测的目标是 t0
#     位置 1 的输入是 t0， 要预测的目标是 t1
#     ...
#     位置 3 的输入是 t2， 要预测的目标是 EOS
#
# ── 如果直接传 labels=input_ids 会怎样 ────────────────────────────────────
# 监督目标会变成"复制当前位置的 token"。
# 模型学到的是恒等映射，loss 很快降得很低，但完全没有语言建模能力 ——
# 这是一个非常典型的"看起来在训练，其实什么都没学到"的陷阱。
#
# ── 谁负责错位 ────────────────────────────────────────────────────────────
# Task 28 的 NextTokenDataset 会统一完成错位（见 09_Next_token训练.py）：
#
#     input_ids, labels = block[:-1], block[1:]
#
# 因此训练循环不需要再 shift 一次。
# 接口上把"错位"这件事交给数据侧，模型侧保持简单，是一个清晰的分工。
# ============================================================================


# ============================================================================
# 第7步：PAD 在 loss 中怎样消失
# ============================================================================
# TinyLanguageModel.forward 会把以下 target 改成 -100：
#
#   - label 等于 pad_token_id；
#   - 当前 query 的 attention_mask 为 False。
#
# ── 为什么是 -100 ─────────────────────────────────────────────────────────
# PyTorch cross-entropy 忽略 index 为 -100 的目标（这是框架的约定，
# 与 ignore_index 参数的默认值一致）。
# 所以 -100 相当于给这些位置贴上"不计分"的标签。
#
# ── 全 PAD 时的处理 ───────────────────────────────────────────────────────
# 若整个 batch 都没有有效 target，代码返回 logits.sum() * 0.0：
# 数值为 0，同时仍与计算图相连（见第 4 步的注释）。
#
# ── 一处容易混淆的地方 ────────────────────────────────────────────────────
# 这里的 attention_mask 只用于【筛 loss】，因为这个局部模型没有 attention。
# 完整模型（MiniMindCore）还会把它传给 attention，用来【屏蔽 PAD key】。
# 二者作用不同：
#     传进 attention → 影响表示（不让模型读到 PAD 的内容）
#     用于筛 loss    → 影响监督信号（不让模型去预测 PAD）
# 两者都要做，缺一不可。
# ============================================================================


# ============================================================================
# 第8步：运行与核对
# ============================================================================
#     python exercises/block_03_transformer/task_25_embedding_lm_head/language_model.py
#
# 输出形如：
#
#     logits: (1, 4, 40)
#     weights shared: True
#
# Embedding、weight tying 和 loss mask 也收录在 Block 3 测试中：
#
#     python -m unittest discover -s tests -p 'test_block3.py' -v
#
# 运行结果与测试覆盖以下性质：
#   - logits 为 (B,T,V)；
#   - 共享关系用 is 观察为真；
#   - 输入超过 max_seq_len 时会报错；
#   - PAD target 不参与 loss；
#   - 全 PAD 时返回有限的 0 loss。
#
# 参考：Using the Output Embedding to Improve Language Models
#       https://arxiv.org/abs/1608.05859
# ============================================================================


if __name__ == "__main__":
    # 建一个小模型：词表 40 个 token、维度 16、最长 8 个位置
    model = TinyLanguageModel(vocab_size=40, dim=16, max_seq_len=8)

    # 一个 batch、4 个 token 的 id 序列
    # torch.tensor([[...]]) 用双层列表构造出 (1,4) 的二维整数张量
    # 注意 dtype 默认由内容推断，这里全整数所以是 int64（long），
    # 正好满足 nn.Embedding 对整数输入的要求
    ids = torch.tensor([[1, 7, 3, 2]])

    # 只取 logits，loss 用 _ 接住丢弃（这里没传 labels，loss 就是 None）
    logits, _ = model(ids)

    # 期望 (1, 4, 40) = (B, T, V)
    # 三个维度分别对应 batch、sequence、vocabulary
    print("logits:", tuple(logits.shape))

    # 用 is 判断对象身份 —— 这是检验权值共享是否真的生效的正确方法
    # 若这里打印 False，说明写成了数值复制而不是引用共享
    print("weights shared:", model.lm_head.weight is model.token_embedding.weight)


# ============================================================================
# 补充知识点：权值共享为什么有效
# ============================================================================
# ── 省参数 ────────────────────────────────────────────────────────────────
# 词表 V 通常很大（LLaMA 是 32000，GPT-2 是 50257）。
# 不共享时，embedding 和 LM head 各占 V×D 个参数；
# 共享后直接省掉一半，对 V=32000、D=4096 来说就是 1.3 亿参数。
#
# ── 语义上的合理性 ────────────────────────────────────────────────────────
# 输入侧：id → 向量，"这个 token 是什么意思"
# 输出侧：向量 → id 打分，"哪个 token 与当前上下文匹配"
# 这两件事本质上是同一个语义空间的正反两向，共用一张表相当于加了一个
# 强正则：要求"能读懂某个词"和"能预测某个词"用同一套表示。
# 论文的结论是在小数据上效果明显、大数据上差异不大，但省参数是确定的。
#
# ── 实现上的两个坑 ────────────────────────────────────────────────────────
# 1. 一定要用赋值引用（=），不要用 .data.copy_()；
# 2. 先建好引用关系再 load_state_dict —— 若顺序反了，
#    load 会把两份权重各自覆盖，共享关系可能被破坏。
#    Task 27 的 MiniMindCore 里就是"先 apply 初始化 → 再绑定 → 再载入"的顺序。
# ============================================================================


# ============================================================================
# 核心公式回顾
# ============================================================================
# (1) 查表与投影
#
#         embedding:  E[id]           → (D,)      按 id 取行
#         lm_head:    logits = H · Eᵀ → (V,)      投影回词表
#
#         批量形式：  H:(B,T,D)  →  H·Eᵀ :(B,T,V)
#
# (2) 权值共享
#
#         self.lm_head.weight = self.token_embedding.weight      ← 引用同一对象
#         验证： model.lm_head.weight is model.token_embedding.weight
#
# (3) 交叉熵（框架内部做的事）
#
#         CE = -log softmax(logits)[target]
#         被 ignored 的位置：target == -100
#
# (4) 全无效 batch 的安全损失
#
#         loss = logits.sum() * 0.0     ← 值为 0，但仍在计算图上
# ============================================================================


# ============================================================================
# 总结
# ============================================================================
# 1. 一张 (V,D) 的矩阵承担了两个角色：
#      按 id 取行   → token embedding
#      乘转置投到 V → LM head
#    PyTorch 的 nn.Linear(D,V) 与 nn.Embedding(V,D) 的 weight 形状一致，
#    所以可以用一行赋值实现权值共享。
#
# 2. 权值共享的判定标准是【对象身份】（is），不是数值相等。
#    用 .data.copy_() 只是初始化成一样的数值，训练后就会分叉。
#
# 3. logits 是未归一化的分数，训练时直接交给 cross_entropy，
#    不要再手动 softmax —— 那会变成两次 softmax，训练会非常慢。
#
# 4. 标签错位由 Dataset 负责，模型不自动 shift。
#    传 labels=input_ids 会让模型学成恒等映射 —— 一个典型而隐蔽的陷阱。
#
# 5. PAD 要从两条路径上排除：
#      attention_mask 传给 attention → 屏蔽 PAD key（影响表示）
#      target 置 -100               → 排除出 loss（影响监督）
#    两者作用不同，不能互相替代。
#
# 6. 全 PAD batch 用 logits.sum() * 0.0 兜底：
#    值是 0，但保持与计算图相连，backward 不会报错。
# ============================================================================
