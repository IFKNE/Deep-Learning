# ============================================================================
#  Transformer 深度学习总结 —— 章节索引
#  （对应源课程 block_03_transformer：从 Attention 到 KV Cache 的完整解码器）
# ============================================================================
#  本模块不讲 CNN 也不讲 RNN，而是回答一个问题：
#
#      "如何让序列中的任意两个位置直接交换信息，并且顺序不会丢失？"
#
#  卷积用固定窗口读局部，RNN 把状态沿时间顺序传递，两者在长距离依赖上都吃力。
#  Self-attention 把这件事写成矩阵乘法：
#
#      Attention(Q,K,V) = softmax( Q·Kᵀ / √d_head + M ) · V
#
#  其中 Q/K 决定"看哪里"，V 决定"拿回来什么"，M 是屏蔽未来与 PAD 的掩码。
#
#  ── 模型全景（decoder-only，本模块最终产物 MiniMindCore）──────────────────
#
#     input_ids (B,T)
#       -> token embedding                        (B,T,D)   ← 任务25：权值共享
#       -> N × DecoderBlock
#            x = x + CausalRoPEGQA(RMSNorm(x))              ← 任务22/23/26
#            x = x + SwiGLU(RMSNorm(x))                     ← 任务24/26
#       -> final RMSNorm
#       -> tied LM head                           (B,T,V)   ← 与 embedding 同参数
#
#  ── 学习主线 ─────────────────────────────────────────────────────────────
#     理论(20) → 位置编码(21,22) → 注意力(23) → 前馈(24) → 词嵌入与输出(25)
#     → 组装 Block(26) → 拼装整机(27) → 训练(28) → 生成采样(29) → KV Cache(30)
#
#  ── 与 D2L 第10章的对应关系 ──────────────────────────────────────────────
#     D2L 第10章讲的是"原论文版 Transformer"：encoder-decoder + 加性位置编码
#     + LayerNorm + Post-Norm + ReLU FFN。
#     本模块讲的是"现代 LLM 版 Transformer"：decoder-only + RoPE + RMSNorm
#     + Pre-Norm + SwiGLU + GQA + KV Cache —— 也就是 LLaMA 一系的结构。
#     两者骨架相同，差别集中在上面这几处替换上，对照学习效果最好。
# ============================================================================

# ============================================================================
# 1. 理论：从 Attention 到 Decoder-only
#    源文件: task_20_transformer_theory/README.md（纯理论，无 .py）
#    目标文件: 01_Transformer理论_从Attention到Decoder_only.py
# ----------------------------------------------------------------------------
#  内容要点:
#    - 为什么需要 self-attention：文本依赖可能隔很远，卷积窗口和 RNN 状态都受限
#    - Q = X·Wq（要匹配什么）/ K = X·Wk（怎样被匹配）/ V = X·Wv（取回什么）
#    - 缩放点积注意力公式；√d_head 缩放的来历：点积方差随维度增长，
#      不缩放会让 softmax 过早饱和（梯度消失）
#    - 单头 → 多头的 shape 变化：(B,T,D) → (B,H,T,Dh) → (B,H,T,T) → (B,T,D)
#    - causal mask：下三角可见，防止位置 t 偷看 t+1 的答案
#    - 原论文 encoder-decoder 与本章 decoder-only 的对照
#  学习目标:
#    把 Q/K/V、缩放、掩码、多头这四件事的数学形式和 shape 变化一次说清
#  前置知识:
#    D2L 第10章注意力机制（查询-键-值框架、缩放点积注意力、多头注意力）
#  学习建议:
#    重点记 shape。后续 9 个代码文件的每一步，都是这里的 shape 在具体化
# ============================================================================

# ============================================================================
# 2. 正弦位置编码
#    源文件: task_21_sinusoidal_position/position.py
#    目标文件: 02_正弦位置编码.py
# ----------------------------------------------------------------------------
#  内容要点:
#    - self-attention 本身不含顺序信息："小猫追小狗"与"小狗追小猫"对它一样
#    - PE(pos,2i)=sin(pos/10000^(2i/D))，PE(pos,2i+1)=cos(...)
#    - 相邻偶奇维共用同一频率，不同维度对频率不同（低维变化快、高维变化慢）
#    - 位置表不参与训练，是固定常量；加到 embedding 上：x = emb + PE[:T]
#    - pos=0 时结果恒为 [0,1,0,1,...]，可作为快速自检点
#    - 和角公式 sin(a+b)=sin a cos b + cos a sin b 说明相对位置可由线性变换得到
#  学习目标:
#    能手推 PE 的第 0 行，理解"多频率"是这张表的精髓
#  前置知识:
#    三角函数、广播机制、embedding 的形状约定
#  学习建议:
#    先跑通输出，确认 position 0 是 0/1 交替，再去看 RoPE 的动机
# ============================================================================

# ============================================================================
# 3. RoPE 旋转位置编码
#    源文件: task_22_rope_position/rope.py
#    目标文件: 03_RoPE旋转位置编码.py
# ----------------------------------------------------------------------------
#  内容要点:
#    - 思路转换：不再把位置加到 embedding，而是把 Q/K 按位置"旋转"，V 不动
#    - 把相邻两维 (x0,x1) 看成一个二维平面，位置 m 对应旋转角 mθ_i
#    - θ_i = base^(-2i/Dh)，base=10000，每个维度对角速度不同
#    - rotate_half(x) = (-x1, x0)，于是旋转 = x·cos + rotate_half(x)·sin
#    - 关键性质：(R_m q)ᵀ(R_n k) = qᵀ R_{n-m} k —— 点积显式依赖相对位移
#    - start_pos 参数：KV Cache 解码时新 token 必须用位置 past_len，不能从 0 重来
#  学习目标:
#    理解 RoPE 为什么只旋转 Q/K，以及 start_pos 漏掉会产生的"静默错误"
#  前置知识:
#    第2节正弦位置编码、二维旋转矩阵、广播
#  学习建议:
#    亲手验证 cache(start_pos=5, seq_len=1) == cache(start_pos=0, seq_len=6)[5:6]
# ============================================================================

# ============================================================================
# 4. Causal 注意力与 GQA
#    源文件: task_23_causal_attention/mha.py
#    目标文件: 04_Causal注意力与GQA.py
# ----------------------------------------------------------------------------
#  内容要点:
#    - 把 Q/K/V 投影、RoPE、两种 mask、多头拼接、输出投影接成一层完整注意力
#    - 两张 mask 各管一件事：causal mask (1,1,T,T) 管时间方向，
#      padding mask (B,T) 管样本内填充；取交集才是最终 allowed 矩阵
#    - mask 必须在 softmax *之前* 施加，否则被禁位置仍会分走概率
#    - GQA：保留 query head 数量，减少 K/V head 数量；Q0,Q1 共享 KV0
#    - _repeat_kv 只用 expand/reshape，不新增参数，KV Cache 也只存 2 份
#    - 全屏蔽行（全 PAD 样本）的处理：softmax 后再乘 allowed，把该行归零避免 NaN
#  学习目标:
#    能独立写出 causal self-attention，并说清三种 head 整除约束的来历
#  前置知识:
#    RoPE（第3节）、softmax、张量广播与转置
#  学习建议:
#    用"固定前缀、改未来 token，前缀输出不变"这个实验来验证 causal 性质
# ============================================================================

# ============================================================================
# 5. SwiGLU 前馈网络
#    源文件: task_24_swiglu_ffn/ffn.py
#    目标文件: 05_SwiGLU前馈网络.py
# ----------------------------------------------------------------------------
#  内容要点:
#    - FFN 在单 token 内部做特征变换，不沿序列轴混合信息
#    - 普通 FFN：W2·ReLU(W1·x)，中间维度 4D，ReLU 是"开/关"二值闸门
#    - SwiGLU：W_down( SiLU(W_gate·x) ⊙ W_up·x )，两路投影 + 一扇连续的门
#    - SiLU(z) = z·σ(z) 处处平滑，允许小幅负值通过（z=-1 时约 -0.27）
#    - 参数量配平：3dm ≈ 8d² ⟹ m ≈ 8d/3 ≈ 2.67d，所以不再取 4D
#      （LLaMA d=4096 时算得 10923，再对齐到 256 倍数 → 11008）
#    - 命名对照：ffn.py 用 w1/w2/w3，LLaMA 系用 gate_proj/up_proj/down_proj
#  学习目标:
#    理解门控机制相对 ReLU 的增益，并记住 8/3·d 这个经验比例的推导
#  前置知识:
#    ReLU/GELU 激活函数、线性层参数量计算
#  学习建议:
#    注意 "Linear→SiLU→Linear" 不是 SwiGLU，它少了一整条 up 分支
# ============================================================================

# ============================================================================
# 6. Embedding、LM Head 与权值共享
#    源文件: task_25_embedding_lm_head/language_model.py
#    目标文件: 06_Embedding与LM_Head.py
# ----------------------------------------------------------------------------
#  内容要点:
#    - 一张 E:(V,D) 两种用法：按 id 查行 = embedding；乘 Eᵀ = LM head
#    - 权值共享必须共享 Parameter 对象：self.lm_head.weight = self.token_embedding.weight
#      用 .data.copy_() 只是数值相同，不是共享；用 is 判断对象身份
#    - 梯度从输入路径和输出路径累积到同一张量，优化器只维护一份
#    - next-token 标签由 Dataset 错开一位，forward 不自动 shift
#    - loss mask：label 为 PAD 或 query mask 为 False 的位置置 -100，被 CE 忽略
#    - 全批次无有效 target 时返回 logits.sum()*0.0（值为 0 但仍连在计算图上）
#  学习目标:
#    建立"词表矩阵既做输入又做输出"的直觉，理解 -100 的语义
#  前置知识:
#    交叉熵损失、nn.Embedding、nn.Linear 的 weight 形状约定
#  学习建议:
#    先跑一遍看 "weights shared: True"，再用 is 在交互式环境里验证
# ============================================================================

# ============================================================================
# 7. 组装 Decoder Block
#    源文件: task_26_decoder_blocks/transformer_blocks.py
#    目标文件: 07_组装Decoder_Block.py
# ----------------------------------------------------------------------------
#  内容要点:
#    - Pre-Norm 通式：x + sublayer(norm(x))；Post-Norm 是 norm(x + sublayer(x))
#    - 本章统一用 Pre-RMSNorm，每层 2 个 norm，全部 block 后还有 1 次 final norm
#    - RMSNorm(x) = x / √(mean(x²)+ε) ⊙ w —— 不减均值、无 bias，只有缩放
#    - RMSNorm 无 running statistics，train/eval 同公式
#    - FP16/BF16 下先用 FP32 算均方根再转回，避免小精度数值损失
#    - TransformerBlock 的 attention/FFN 可注入，便于把"组装"与"内部实现"分开观察
#    - TransformerStack 用 nn.ModuleList 堆叠，参数默认不共享
#  学习目标:
#    分清 Pre-Norm/Post-Norm，理解残差成立的前提是 shape 不变
#  前置知识:
#    第4/5节的 attention 与 FFN（输入输出均为 (B,T,D)）
#  学习建议:
#    留意"第二个 RMSNorm 读的是 x1 而不是最初的 x"这个细节
# ============================================================================

# ============================================================================
# 8. MiniMind 模型主干
#    源文件: task_27_minimind_core/minimind_core.py
#    目标文件: 08_MiniMind模型主干.py
# ----------------------------------------------------------------------------
#  内容要点:
#    - 把前 6 个部件接成可训练的微型 decoder-only Transformer
#    - MiniMindConfig 用 dataclass 集中管理尺寸，__post_init__ 一次校验所有约束
#    - 没有 learned position embedding，位置信息全部来自每层的 RoPE
#    - forward 三种返回形态：(logits, loss) / (logits, loss, new_past)
#    - 非方阵 causal mask：query 位置是 [past_len, total_len)，key 位置是 [0, total_len)
#    - 内置未缓存的慢速参考 generate()：greedy / temperature / top-k / top-p / EOS
#    - 初始化：正态 N(0, 0.02)，bias 清零；先绑定权值再载入 state_dict
#  学习目标:
#    能读懂整机 forward 的每一步 shape，理解 config 校验为什么放在建模型之前
#  前置知识:
#    第4~7节全部内容
#  学习建议:
#    把 forward 里的每个 shape 注释抄一遍，这是模块最核心的一节
# ============================================================================

# ============================================================================
# 9. Next-token 训练
#    源文件: task_28_next_token_training/train.py
#    目标文件: 09_Next_token训练.py
# ----------------------------------------------------------------------------
#  内容要点:
#    - 字符级 tokenizer：4 个特殊 token（pad/unk/bos/eos）+ 排序后的字符表
#    - 分词器只在 train text 上 fit，验证集未见字符映射到 <unk>，保证真正留出
#    - NextTokenDataset：每次取 seq_len+1 个 id，输入是 block[:-1]，标签是 block[1:]
#      起点按 0, seq_len, 2*seq_len 前进，相邻块共享一个边界 token，不重复训练
#    - 训练循环：forward → masked CE → zero_grad → backward → clip(1.0) → AdamW
#    - validation 的分母：按有效 token 数加权，而不是按 batch 数平均
#      （最后一批 PAD 多，按 batch 平均会偏）
#    - checkpoint 保存 config/model/tokenizer/step/val_loss，round-trip 后 logits 一致
#  学习目标:
#    打通 text → ids → 错位标签 → mask → loss → 反向 → 验证 → 存档 的完整链路
#  前置知识:
#    第6节的 loss mask、第8节的 MiniMindCore、DataLoader/Dataset 基础
#  学习建议:
#    --steps 4 跑一次 smoke test，确认 loss 有限、checkpoint 落盘即可
# ============================================================================

# ============================================================================
# 10. 自回归生成与采样
#    源文件: task_29_generate_sampling/generate.py
#    目标文件: 10_自回归生成与采样.py
# ----------------------------------------------------------------------------
#  内容要点:
#    - 生成只能取最后一位：next_logits = logits[:, -1]，shape (B,V)
#    - Greedy：temperature=0 时 argmax，且不再应用 top-k/top-p
#    - Temperature：p = softmax(z/τ)；τ<1 更尖，τ>1 更平，τ<0 报错
#    - Top-k：取第 k 大 logit 作阈值，低于阈值置 -inf 再 softmax
#    - Top-p（nucleus）：按概率降序保留累计到 p_top 的最小前缀；
#      实现上把删除 mask 右移一格，保证跨过阈值的那个 token 被留下
#    - 显式 generator：torch.Generator().manual_seed(seed)，不受全局随机态干扰
#    - 变长 batch 必须左 padding，因为循环统一取 logits[:, -1]
#  学习目标:
#    能说清四种采样规则的差异，并解释"为什么每一步只看最后一位"
#  前置知识:
#    softmax、多项式采样、第9节的 checkpoint 格式
#  学习建议:
#    先用 temperature=0 跑，结果确定，便于核对；再逐步开启 top-k/top-p
# ============================================================================

# ============================================================================
# 11. KV Cache
#    源文件: task_30_kv_cache/kv_cache.py
#    目标文件: 11_KV_Cache.py
# ----------------------------------------------------------------------------
#  内容要点:
#    - 未缓存生成每步重算整个窗口：第1步 5 个 token，第2步 6 个，第3步 7 个…
#    - 旧 token 的 K/V 不会因新 token 出现而改变，所以可以留下来复用
#    - cache 按层保存：list[(K,V)]，每个 shape (B, n_kv_heads, past_len, head_dim)
#    - 存的是未 repeat_kv 的 K/V，GQA 因此直接把缓存体积按比例缩小
#    - prefill 一次处理整个 prompt；decode_one 只接受 (B,1) 的单个新 token
#    - RoPE 的 start_pos=past_len：漏掉会让 logits 偏离，但 shape 完全正常
#    - 等价测试：全量 forward 与 cached 逐步 forward 的最大绝对误差在
#      1e-7 ~ 1e-6 量级（CPU fp32；实测某个 checkpoint 为 1.4e-06），
#      greedy 解码出的 token 序列则逐项完全相同
#    - 窗口满时重新 prefill，因为滚动后第一个 token 要重新视为位置 0
#  学习目标:
#    理解 cache 省掉的是"K/V 投影与旧 token 的 block 计算"，而不是 attention 本身
#  前置知识:
#    第3节 RoPE 的 start_pos、第8节的 use_cache 接口、第9/10节
#  学习建议:
#    先看 cache_equivalence_error 的输出，再看生成循环，理解"数值等价"才是验收标准
# ============================================================================

# ============================================================================
#                           模块学习路线图
# ============================================================================
#
#                       ┌─────────────────────────────┐
#                       │ 01 理论：Q/K/V、缩放、掩码   │
#                       │      多头、decoder-only 路线 │
#                       └──────────────┬──────────────┘
#                                      │
#              ┌───────────────────────┴───────────────────────┐
#              ▼                                               ▼
#   ┌──────────────────────┐                        ┌──────────────────────┐
#   │ 02 正弦位置编码       │                        │ 04 Causal注意力+GQA  │
#   │    PE(pos,2i)=sin…   │                        │    scores/mask/拼接  │
#   └──────────┬───────────┘                        └──────────┬───────────┘
#              │ 动机对照                                      │
#              ▼                                               │
#   ┌──────────────────────┐                                   │
#   │ 03 RoPE              │─────────── Q/K 旋转 ──────────────┘
#   │    start_pos 为缓存  │
#   └──────────────────────┘
#                                                              │
#   ┌──────────────────────┐                                   │
#   │ 05 SwiGLU            │─────── 门控 FFN ──────────────────┤
#   │    8/3·d 的来历      │                                   │
#   └──────────────────────┘                                   │
#                                                              ▼
#   ┌──────────────────────┐                        ┌──────────────────────┐
#   │ 06 Embedding+LM Head │──── 权值共享 ─────────▶│ 07 组装 Decoder Block│
#   │    一张表两种用法     │                        │    Pre-RMSNorm×2     │
#   └──────────────────────┘                        └──────────┬───────────┘
#                                                              │
#                                                              ▼
#                                                   ┌──────────────────────┐
#                                                   │ 08 MiniMind 模型主干 │
#                                                   │    整机 forward      │
#                                                   └──────────┬───────────┘
#                                                              │
#                                      ┌───────────────────────┴────────────┐
#                                      ▼                                    ▼
#                          ┌──────────────────────┐            ┌──────────────────────┐
#                          │ 09 Next-token 训练    │            │ 10 生成与采样         │
#                          │    数据/掩码/存档     │───ckpt───▶ │    greedy/τ/top-k/p  │
#                          └──────────────────────┘            └──────────┬───────────┘
#                                                                         │ 数值对照
#                                                                         ▼
#                                                              ┌──────────────────────┐
#                                                              │ 11 KV Cache          │
#                                                              │    prefill+decode    │
#                                                              └──────────────────────┘
# ============================================================================

# ============================================================================
#                           推荐学习路径
# ============================================================================
#   【初学路径】1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11
#       严格按依赖顺序。每节先读 README 的"为什么"，再读代码，最后跑 __main__ 看输出。
#       到第 8 节时回头看第 1 节的 shape 表，会发现每个 shape 都落在了具体代码上。
#
#   【先跑通路径】9 → 10 → 11 → 再回头补 1~8
#       想先看"能生成中文"的效果，可以先跑 train.py --steps 80 拿到 checkpoint，
#       再用 generate.py 采样。有直观感受后再补结构细节，读代码会快很多。
#
#   【速通路径】1 → 3 → 4 → 8 → 11
#       只抓四个关键点：注意力公式与 mask、RoPE 的旋转与 start_pos、
#       整机组装、推理加速。适合面试前快速回顾。
#
#   【对照 D2L 路径】先读 my_learn/chapter10_注意力机制（原论文版），
#       再读本模块（现代 LLM 版），重点对照这五处替换：
#           加性位置编码 → RoPE            LayerNorm    → RMSNorm
#           Post-Norm     → Pre-Norm       ReLU FFN     → SwiGLU
#           MHA           → GQA
# ============================================================================

if __name__ == "__main__":
    print("=" * 76)
    print(" Transformer 深度学习总结 —— 章节索引")
    print(" 源课程: exercises/block_03_transformer（task 20 ~ task 30）")
    print("=" * 76)
    print(" 01  理论：从 Attention 到 Decoder-only      task_20  无代码，纯理论")
    print(" 02  正弦位置编码                            task_21  position.py")
    print(" 03  RoPE 旋转位置编码                       task_22  rope.py")
    print(" 04  Causal 注意力与 GQA                     task_23  mha.py")
    print(" 05  SwiGLU 前馈网络                         task_24  ffn.py")
    print(" 06  Embedding、LM Head 与权值共享           task_25  language_model.py")
    print(" 07  组装 Decoder Block                      task_26  transformer_blocks.py")
    print(" 08  MiniMind 模型主干                       task_27  minimind_core.py")
    print(" 09  Next-token 训练                         task_28  train.py")
    print(" 10  自回归生成与采样                        task_29  generate.py")
    print(" 11  KV Cache                                task_30  kv_cache.py")
    print("-" * 76)
    print(" 建议起点：先读 01 建立 shape 直觉，再按 02→11 顺序推进。")
    print(" 最短跑通：python 09_Next_token训练.py --steps 80")
    print("           python 10_自回归生成与采样.py --checkpoint minimind_demo.pt")
    print("=" * 76)
