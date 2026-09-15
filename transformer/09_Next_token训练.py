"""
=============================================================================
Next-token 训练 —— 从文本到 checkpoint 的完整数据路径
=============================================================================
源文件: exercises/block_03_transformer/task_28_next_token_training/train.py

任务：模型结构接好以后，文本还要变成监督样本。这条数据路径同时处理
     训练集与验证集的隔离、标签错位、attention/loss 两处 PAD 屏蔽，
     以及 checkpoint 恢复后的 logits 一致性。

  完整链路：
     text -> tokenizer -> dataset -> masks -> Transformer
          -> loss -> backward -> validation -> checkpoint

  设计取舍：内置一段 UTF-8 中文短文，也接受 --text 指向本地文件。
           默认不联网，不依赖第三方 tokenizer，目的是让整条数据路径
           能在一个文件里读完。

依赖：argparse、dataclasses、pathlib、sys、torch、torch.utils.data，
      以及同模块的 minimind_core（见下方路径处理）。本文件可独立运行。
=============================================================================
"""

# ============================================================================
# 第1步：导入依赖
# ============================================================================
import argparse                      # 命令行参数解析
from dataclasses import asdict       # 把 dataclass 实例转成普通字典，便于序列化
from pathlib import Path             # 面向对象的路径操作，比字符串拼接更安全
import sys                           # 需要改 sys.path 才能 import 到同级模块

import torch                         # 张量、随机数、保存/载入
from torch.utils.data import DataLoader, Dataset
# DataLoader：批量、打乱、并行加载的封装
# Dataset   ：自定义数据集要继承的基类


# ============================================================================
# 第2步：让本文件能 import 到模型定义
# ============================================================================
# ── 原始仓库中的写法 ──────────────────────────────────────────────────────
# CORE = Path(__file__).resolve().parents[1] / "task_27_minimind_core"
#
#   __file__            当前文件路径
#   .resolve()          转成绝对路径（消除 .. 和符号链接）
#   .parents[1]         往上两级：task_28_next_token_training → block_03_transformer
#   再拼上 "task_27_minimind_core" 就得到模型所在的目录
#
# ── 为什么需要插入 sys.path ───────────────────────────────────────────────
# Python 的 import 只会搜索 sys.path 里的目录。
# 同级的兄弟目录默认【不在】搜索路径里（只有"当前目录"和"包安装目录"在），
# 所以要把模型目录手动加进去。
#
# ── 本整理版的调整 ────────────────────────────────────────────────────────
# 这份文件被整理到 my_learn/transformer/ 下，同级目录里不再有 task_27，
# 因此增加一个"找不到就回退到原始仓库位置"的分支。
# 原始的那一行逻辑保持不动，只是多了一层兜底，便于两个位置都能直接运行。
CORE = Path(__file__).resolve().parents[1] / "task_27_minimind_core"
if not CORE.exists():
    # exists() 判断路径是否真实存在，不存在就换用原仓库的绝对路径
    CORE = Path(
        r"d:\桌面\深度学习\quickly_access_to_deeplearning-main"
        r"\exercises\block_03_transformer\task_27_minimind_core"
    )

if str(CORE) not in sys.path:
    # 先判断再加，避免重复插入（多次插入虽然无害，但会让 sys.path 越来越长）
    # insert(0, ...) 插到最前面，保证优先从这里查找
    sys.path.insert(0, str(CORE))

# 这行 import 必须放在 sys.path 修改【之后】，
# 因此 linter 会抱怨"导入不在文件顶部"（noqa: E402 就是用来抑制这条告警的）
from minimind_core import MiniMindConfig, MiniMindCore  # noqa: E402
# 说明：my_learn/transformer/08_MiniMind模型主干.py 与本模块内容是同一份代码，
# 但文件名含中文，无法直接 import，所以这里仍然从原仓库导入。


# ============================================================================
# 第3步：内置语料
# ============================================================================
# 一段自洽的中文短文：讲一个孩子过石桥、上课、又回到石桥。
# 用同一个屋檐滴水作为开头和结尾，语言重复度高、词汇量小 ——
# 这样一个小模型在几十步内就能看到明显的 loss 下降，
# 适合做 smoke test。真实语料不需要这种"刻意好记"的性质。
DEFAULT_CORPUS = """
清晨，小城的雨停了。屋檐上的水滴慢慢落下，街边的树叶在风里发亮。
一个孩子背着书包走过石桥，他数着河面上的波纹，也数着远处的钟声。
老人打开面包店的木门，暖气带着麦香涌到街上。早起的人们互相问好，然后走向各自的一天。
中午，阳光穿过云层。孩子在课堂上写下：学习像搭桥，每一块石头都要放在正确的位置。
老师说，好问题比快答案更重要。先观察，再猜想，然后用实验检查猜想。
傍晚，河水映着灯光。孩子回到石桥，他发现早晨的问题已经有了新的答案。
他也明白，答案不是终点。当一个问题被说清楚，下一个问题就会从它身后走出来。
夜里又下起小雨，屋檐上的水滴慢慢落下。小城安静了，但新的故事正在开始。
""".strip()
# .strip() 去掉首尾的换行和空白，
# 否则开头的 \n 会成为一个无意义的字符 token，末尾的 \n 同理。
# 三引号字符串 """...""" 可以跨多行，是写长文本常量的标准做法。


# ============================================================================
# 第4步：字符级 Tokenizer
# ============================================================================
# ── 为什么用字符级 ────────────────────────────────────────────────────────
# 这里采用字符 tokenizer 是为了【透明】：一个字符对应一个 id，
# 编码解码都能肉眼核对。它没有 BPE、SentencePiece 的压缩效率，
# 也不代表实际大规模语言模型的数据处理方式。
#
# 本模块的语料只有几百个字符，字符级完全够用，而且省去了装 tokenizer 库的麻烦。
class CharacterTokenizer:
    """一个最简的字符级分词器。

    词表 = 4 个特殊 token + 训练文本中出现过的所有字符（排序后）。
    """

    # ── 四个特殊 token 的字符串形式 ──────────────────────────────────────
    # 写成类属性（而不是实例属性），所有实例共享同一组常量，
    # 且可以通过 CharacterTokenizer.PAD 访问
    PAD = "<pad>"     # 补齐：让同一 batch 内的序列长度一致
    UNK = "<unk>"     # 未知：验证集里出现的、训练集没见过的字符
    BOS = "<bos>"     # 序列开始
    EOS = "<eos>"     # 序列结束

    def __init__(self, token_to_id):
        """从"token → id"的映射建分词器。

        参数:
            token_to_id: dict，如 {"<pad>": 0, "<unk>": 1, ...}
        """
        # dict(...) 做一次浅拷贝：避免外部后续修改那个字典影响到本对象
        self.token_to_id = dict(token_to_id)

        # ── 反向映射：用一个列表按下标存 token ───────────────────────────
        # [None] * n 先造一个长度为 n、全是 None 的列表，
        # 然后按 id 填入对应的 token —— 这就是"用下标查值"的 O(1) 结构，
        # 比反向字典更省内存（token 是字符串，存两份没必要）
        self.id_to_token = [None] * len(self.token_to_id)
        for token, index in self.token_to_id.items():
            self.id_to_token[index] = token

        # ── 校验四个特殊 token 都在 ──────────────────────────────────────
        # 元组 + any(...) 的组合：只要有一个缺失就报错
        # 生成器表达式 (token not in ... for token in required) 逐个求值，
        # any 遇到第一个 True 就短路返回
        required = (self.PAD, self.UNK, self.BOS, self.EOS)
        if any(token not in self.token_to_id for token in required):
            raise ValueError("tokenizer vocabulary is missing a special token")

    @classmethod
    # @classmethod 让方法接收 cls（类本身）而不是 self（实例），
    # 因此可以在【还没有实例】的时候调用：CharacterTokenizer.fit(text)
    def fit(cls, text):
        """从文本构造分词器：特殊 token 在前，出现过的字符按序在后。

        参数:
            text: 训练文本

        返回:
            CharacterTokenizer 实例
        """
        # 特殊 token 先占住 0~3 这四个 id
        specials = [cls.PAD, cls.UNK, cls.BOS, cls.EOS]

        # sorted(set(text)) 的妙处，从里往外读：
        #   set(text)        去重 —— 得到"文本里出现过的所有字符"（集合无序）
        #   sorted(...)      排序 —— 让顺序确定，不依赖 Python 哈希的随机性
        #   list 拼接        特殊 token 在前 + 字符在后
        # 若少了 sorted，每次运行词表顺序可能不同，checkpoint 就没法复现了。
        vocabulary = specials + sorted(set(text))

        # 列表推导式 + enumerate 生成 {token: 下标} 的字典
        # {token: index for index, token in enumerate(vocabulary)}
        return cls({token: index for index, token in enumerate(vocabulary)})

    # ── 便捷属性：用 @property 把"查表"包装成属性访问 ─────────────────────
    # @property 让方法可以像属性一样调用（写 tok.pad_token_id 而不是 tok.pad_token_id()），
    # 读起来更自然，也避免了调用方忘记加括号
    @property
    def pad_token_id(self):
        return self.token_to_id[self.PAD]

    @property
    def bos_token_id(self):
        return self.token_to_id[self.BOS]

    @property
    def eos_token_id(self):
        return self.token_to_id[self.EOS]

    @property
    def vocab_size(self):
        # 反向列表的长度就是词表大小
        return len(self.id_to_token)

    # ── 编码：文本 → id 列表 ─────────────────────────────────────────────
    def encode(self, text, add_bos=False, add_eos=False):
        """把文本编码成 id 列表。

        参数:
            text:                输入文本
            add_bos:             是否在开头插入 <bos>
            add_eos:             是否在结尾追加 <eos>

        返回:
            Python 整数列表（不是张量）
        """
        # dict.get(键, 默认值) 的用法：
        #   self.token_to_id.get(char, self.token_to_id[self.UNK])
        #   若字符在词表里就取它的 id，否则取 <unk> 的 id。
        # 这就是"训练集未见字符映射到 <unk>"的实现方式。
        ids = [self.token_to_id.get(char, self.token_to_id[self.UNK]) for char in text]

        if add_bos:
            # list.insert(0, x) 在开头插入
            ids.insert(0, self.bos_token_id)
        if add_eos:
            # list.append(x) 在末尾追加
            ids.append(self.eos_token_id)

        return ids

    # ── 解码：id 列表 → 文本 ─────────────────────────────────────────────
    def decode(self, ids, skip_special_tokens=True):
        """把 id 列表还原成文本。

        参数:
            ids:                 整数序列（可以是 Python 列表或张量）
            skip_special_tokens: 是否隐藏 PAD/UNK/BOS/EOS

        说明:
            解码时会隐藏特殊 token，因此检查生成长度要看 token id 的数量，
            不能只数字符串的字符数。
        """
        specials = {self.PAD, self.UNK, self.BOS, self.EOS}
        tokens = []
        for index in ids:
            # int(index) 是必要的：传入张量时元素是 0 维张量，
            # 直接当下标用虽然多数情况可行，但显式转换更稳
            token = self.id_to_token[int(index)]

            if skip_special_tokens and token in specials:
                # continue 跳过本次循环剩下的语句，直接进入下一轮
                continue
            tokens.append(token)

        # "".join(...) 把字符列表拼成一个字符串
        # 用 "" 而不是 " "，因为这是字符级分词，每个元素就是一个字符
        return "".join(tokens)

    # ── 序列化：让 tokenizer 能随 checkpoint 一起保存 ────────────────────
    def state_dict(self):
        """返回可被 torch.save 序列化的状态。"""
        # 只存正向映射就够了 —— 反向映射可以由 __init__ 重建
        return {"token_to_id": self.token_to_id}

    @classmethod
    def from_state_dict(cls, state):
        """从 state_dict 重建分词器。

        用 @classmethod + cls(...) 而不是写死 CharacterTokenizer(...)，
        这样子类继承时也能正确构造出子类实例。
        """
        return cls(state["token_to_id"])


# ============================================================================
# 第5步：NextTokenDataset —— 错位标签的核心
# ============================================================================
class NextTokenDataset(Dataset):
    """Fixed-size blocks; neighbors share one token and the final block is padded.

    定长分块；相邻两块共享一个 token，最后一块不足时补 PAD。

    ── 错位是怎么发生的 ──────────────────────────────────────────────────
    NextTokenDataset 每次取 seq_len+1 个 id：

        block: [BOS, t0, t1, t2, EOS]
        input: [BOS, t0, t1, t2]
        label: [t0,  t1, t2, EOS]

    起点按 0, seq_len, 2*seq_len, ... 前进。
    因为每块取 seq_len+1 个 id，相邻两块会共享一个边界 token。
    它在前一块是最后一个 label，在后一块是第一个 input，
    所以每条 next-token 转移只训练一次 —— 这是这个切分方案的关键设计。

    最后一块不足时补 PAD。这个方案没有高性能预训练常见的复杂 packing，
    但 input、label 和 padding 的对应关系容易手工检查。
    """

    def __init__(self, token_ids, seq_len, pad_token_id):
        """
        参数:
            token_ids:    完整的 id 列表
            seq_len:      每块的输入长度 T（标签会多一位）
            pad_token_id: 补齐用的 id
        """
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if len(token_ids) < 2:
            # 少于两个 token 就没有任何 next-token 转移可学
            raise ValueError("a split needs at least two tokens")

        # 一次性转成张量，避免 __getitem__ 每次转换（会被频繁调用）
        # dtype=torch.long 是 nn.Embedding 要求的整数类型
        self.ids = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.pad_token_id = pad_token_id

        # ── 预先算好所有块的起点 ─────────────────────────────────────────
        # range(0, len-1, seq_len)：从 0 开始，步长 seq_len，到 len-1 为止
        #   上界是 len-1 而不是 len，因为每块需要 seq_len+1 个 id，
        #   最后一块至少要能取到一个 token 才有意义（len < 1 时直接放弃）
        # 用 list(...) 物化，这样 __len__ 可以 O(1) 返回，不用每次重算
        self.starts = list(range(0, len(token_ids) - 1, seq_len))

    def __len__(self):
        # DataLoader 靠这个知道一共多少个样本（BatchSampler 要用）
        return len(self.starts)

    def __getitem__(self, index):
        """取第 index 个块，返回三元组 (input_ids, labels, attention_mask)。

        参数:
            index: 样本下标，由 DataLoader 传入

        返回:
            input_ids:      (seq_len,)
            labels:         (seq_len,)
            attention_mask: (seq_len,)，True 表示非 PAD
        """
        start = self.starts[index]

        # 切出 seq_len+1 个 id（多取一个是为了错位出标签）
        # 切片越界时 PyTorch 不报错，只返回能取到的部分 —— 这里依赖这个行为
        block = self.ids[start : start + self.seq_len + 1]

        if block.numel() < self.seq_len + 1:
            # numel() 返回元素个数
            # 不足时补 PAD：
            #   block.new_full(shape, value) 用 block 的 dtype/device 新建一个
            #   填满指定值的张量 —— 这样不用手写 dtype 和 device
            padding = block.new_full((self.seq_len + 1 - block.numel(),), self.pad_token_id)
            # cat 沿第 0 维拼接，（n,）+（m,）→（n+m,）
            block = torch.cat((block, padding))

        # ── 错位：这就是 next-token 监督的全部秘密 ──────────────────────
        # block[:-1]：去掉最后一个 → 输入
        # block[1:] ：去掉第一个   → 标签
        # 于是 input[i] 的下一个 token 正好是 label[i]
        input_ids, labels = block[:-1], block[1:]

        # ── attention_mask ───────────────────────────────────────────────
        # ne 是 "not equal"：input 里不等于 PAD 的位置为 True
        # 注意一个细节：EOS 对应的输入位置仍然有效（mask 为 True），
        # 它的下一个 target 是 PAD，所以【loss 端】才被排除。
        # 也就是说 attention mask 和 loss mask 在这里是分离的两件事。
        attention_mask = input_ids.ne(self.pad_token_id)

        return input_ids, labels, attention_mask


# ============================================================================
# 第6步：语料切分
# ============================================================================
def split_corpus(text, train_fraction=0.85):
    """把文本按字符位置切成训练/验证两段。

    参数:
        text:           完整文本
        train_fraction: 训练集占比，必须落在 [0.5, 1.0)

    返回:
        (train_text, val_text)

    说明:
        当前演示按字符位置切分：
            前 85% -> train text
            后 15% -> validation text
        两段不重叠。

        连续切分适合这个 smoke test，但【不是严谨的语料评估方案】。
        处理真实文档时通常还要按文档去重、避免同源文本跨 split，
        并单独保留 test set。
    """
    # 链式比较：0.5 <= train_fraction < 1.0
    # 上界取不到 1.0 —— 否则验证集为空，evaluate 会除零
    if not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be in [0.5, 1.0)")
    if len(text) < 40:
        # 太短的文本切出来的验证集连两个 token 都不够
        raise ValueError("the corpus is too short for independent train/validation splits")

    # int() 向下取整到整数位置
    split = int(len(text) * train_fraction)
    # 切片：text[:split] 是前 split 个字符，text[split:] 是剩下的
    return text[:split], text[split:]


# ============================================================================
# 第7步：构造 DataLoader
# ============================================================================
def make_dataloaders(text, seq_len=48, batch_size=8, seed=0):
    """完成"切分 → 拟合分词器 → 编码 → 建 Dataset → 建 DataLoader"的全过程。

    参数:
        text:       完整语料
        seq_len:    每块的输入长度
        batch_size: 批大小
        seed:       随机种子（用于训练集打乱的可复现）

    返回:
        (tokenizer, train_loader, val_loader)
    """
    train_text, val_text = split_corpus(text)

    # Fitting only on the training split keeps validation genuinely held out.
    # （只在训练集上拟合分词器，验证集才是真正留出的。）
    #
    # 这是一个很容易被忽略的细节：如果分词器在全量文本上 fit，
    # 验证集里的字符就已经"被模型见过"了（至少在词表层面），
    # 严格来说是一种信息泄漏。
    tokenizer = CharacterTokenizer.fit(train_text)

    # 编码时加 BOS/EOS：BOS 让模型知道"序列从这里开始"，
    # EOS 给它一个"可以停下了"的信号 —— 生成时才能正常终止。
    # 这里返回的是 Python 列表，Dataset 内部会转成张量。
    train_ids = tokenizer.encode(train_text, add_bos=True, add_eos=True)
    val_ids = tokenizer.encode(val_text, add_bos=True, add_eos=True)

    train_data = NextTokenDataset(train_ids, seq_len, tokenizer.pad_token_id)
    val_data = NextTokenDataset(val_ids, seq_len, tokenizer.pad_token_id)

    # ── 显式 Generator，让打乱顺序可复现 ────────────────────────────────
    # torch.Generator() 是一个独立的随机数发生器，
    # manual_seed(seed) 之后，同样的 seed 会给出同样的打乱顺序。
    # 用一个独立对象而不是全局随机状态，好处是不受其他地方的
    # torch.manual_seed 或随机调用影响。
    generator = torch.Generator().manual_seed(seed)

    # 训练集：shuffle=True 打乱顺序，避免同一 batch 全是相邻文本块
    #        （相邻块高度相似，会让梯度方向过于一致）
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)

    # 验证集：shuffle=False —— 顺序固定，两次评估的结果才可比
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return tokenizer, train_loader, val_loader


# ============================================================================
# 第8步：验证损失
# ============================================================================
@torch.no_grad()
# 验证不需要梯度，加这个装饰器可以省显存、提速
def evaluate(model, data_loader, device="cpu"):
    """在验证集上计算按有效 token 加权的平均损失。

    参数:
        model:       要评估的模型
        data_loader: 验证集 DataLoader
        device:      设备字符串

    返回:
        float，加权平均的验证损失

    ── 为什么不能简单按 batch 平均 ───────────────────────────────────────
    模型返回的是【有效 target token 上的平均 loss】。
    验证集最后一批往往含更多 PAD（因为只剩一点数据，补齐得多），
    如果按 batch 数简单平均，小批量里的少数有效 token 会被赋予过大的权重。
    正确做法是按【有效 token 数】加权。
    """
    # model.eval() 切换到评估模式。
    # 本模型没有 Dropout/BatchNorm，所以行为上差别不大，
    # 但这是必须养成的习惯 —— 换成含 Dropout 的模型时，漏掉它会得到错误的评估结果。
    model.eval()

    # 累加器：分母 token_count 初始设为 0 会在最后除零，
    # 所以末尾用 max(token_count, 1) 兜底
    loss_total, token_count = 0.0, 0

    # DataLoader 迭代时自动完成组批，每个 batch 是一个三元组
    # for 循环里用元组解包一次取三个值
    for input_ids, labels, attention_mask in data_loader:
        # 搬到目标设备（GPU 训练时是必需的）
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)

        # 只取 loss，logits 用 _ 丢弃
        _, loss = model(input_ids, labels, attention_mask=attention_mask)

        # ── 重新算一遍有效 token 数 ──────────────────────────────────────
        # 为什么不能直接用 attention_mask.sum()？
        #   因为 attention_mask 标记的是"输入位置是否有效"，
        #   而 loss 看的是"标签是否有效"。两者在 EOS 那里不同：
        #   EOS 位置 attention_mask=True，但它的 label 是 PAD（不计损失）。
        #
        # attention_mask.bool() 先确保是布尔类型（Dataset 里已经是了，
        #   但显式转换更稳）
        # labels.ne(-100)：模型内部已经把无效位置标成 -100，
        #   不过它是在 clone 出来的 targets 上做的，原 labels 不变 ——
        #   所以这里还要自己判断一次
        valid = attention_mask.bool() & labels.ne(-100)
        if model.config.pad_token_id is not None:
            # 再排除 label 本身是 PAD 的位置
            valid = valid & labels.ne(model.config.pad_token_id)

        # sum() 返回 0 维张量，int(...) 转成 Python 整数
        valid_count = int(valid.sum())

        # The model returns the mean over non-masked target tokens.  Aggregate
        # by that same denominator; weighting by batch size biases validation
        # whenever the last sequence contains more padding.
        # （模型返回的是非屏蔽 target 上的均值。用同一个分母来聚合；
        #   若按 batch size 加权，最后一批 PAD 较多时验证损失会失真。）
        #
        # loss * valid_count 相当于把这个 batch 的"总和"还原出来，
        # 最后再除以总 token 数 —— 这就是标准的"加权平均"。
        # float(loss) 把 0 维张量转成 Python float，避免把计算图带进累加器。
        loss_total += float(loss) * valid_count
        token_count += valid_count

    # max(token_count, 1) 防止一个有效 token 都没有时除零
    return loss_total / max(token_count, 1)


# ============================================================================
# 第9步：Checkpoint 的保存与载入
# ============================================================================
def save_checkpoint(path, model, optimizer, tokenizer, step, val_loss):
    """保存一次完整的训练状态。

    保存内容包括：
        config          用 dataclass 重建模型结构
        model_state     模型参数
        optimizer_state 优化器状态（动量等）
        tokenizer       词表映射
        step            已训练步数
        val_loss        最后一次验证损失

    说明:
        CLI 暂未提供 --resume。虽然文件中保存了 optimizer_state，
        继续训练时仍需调用者创建 optimizer 并显式 load_state_dict；
        "能保存优化器"不等于"已实现完整断点续训"。
    """
    path = Path(path)

    # mkdir(parents=True) 会连同缺失的父目录一起创建；
    # exist_ok=True 表示目录已存在时不报错。
    # 没有这两行的话，保存到 /tmp/a/b/c.pt 这类路径会直接失败。
    path.parent.mkdir(parents=True, exist_ok=True)

    # torch.save 用 pickle 序列化一个字典。
    # 只存 state_dict（参数张量）而不是整个模型对象，是推荐做法：
    #   1. 文件更小、更干净；
    #   2. 载入时不依赖原来的类定义路径，更健壮；
    #   3. 不会把代码里的临时属性一起存进去。
    torch.save(
        {
            # asdict() 把 dataclass 实例递归转成普通 dict，
            # 这样载入时可以用 MiniMindConfig(**checkpoint["config"]) 重建
            "config": asdict(model.config),
            "model_state": model.state_dict(),
            # 三元表达式：optimizer 为 None 时存 None，避免 AttributeError
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "tokenizer": tokenizer.state_dict(),
            # 显式转成 Python 原生类型（int/float），
            # 避免存进去的是 numpy 或 torch 的标量类型
            "step": int(step),
            "val_loss": float(val_loss),
        },
        path,
    )


def load_checkpoint(path, device="cpu"):
    """载入 checkpoint，重建模型与分词器。

    参数:
        path:   checkpoint 路径
        device: 载入到哪个设备

    返回:
        (model, tokenizer, checkpoint_dict)
    """
    try:
        # weights_only=False 表示允许反序列化任意对象（这里存了 dict 和 float，
        # 新版 PyTorch 默认 weights_only=True 会拒绝）。
        # map_location=device 把张量直接映射到目标设备，避免先载入 CPU 再搬。
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch before the weights_only argument existed.
        # 旧版 PyTorch 没有 weights_only 这个参数，传了会抛 TypeError。
        # 用 try/except 做版本兼容，是处理这种 API 差异的常见手法。
        checkpoint = torch.load(path, map_location=device)

    tokenizer = CharacterTokenizer.from_state_dict(checkpoint["tokenizer"])

    # ── 重建模型：** 是字典解包 ──────────────────────────────────────────
    # MiniMindConfig(**checkpoint["config"]) 把字典展开成关键字参数，
    # 等价于 MiniMindConfig(vocab_size=..., dim=..., ...)。
    # .to(device) 把模型所有参数搬到目标设备。
    model = MiniMindCore(MiniMindConfig(**checkpoint["config"])).to(device)

    # load_state_dict 按 key 名字把参数填回去。
    # 注意：MiniMindCore.__init__ 里已经建立了权值共享关系，
    # 而 state_dict 中 lm_head.weight 与 token_embedding.weight 指向同一个张量，
    # 载入后共享关系仍然成立 —— 这是 08 节强调"先绑定再载入"的原因。
    model.load_state_dict(checkpoint["model_state"])

    return model, tokenizer, checkpoint


# ============================================================================
# 第10步：训练循环
# ============================================================================
def train_model(
    text=DEFAULT_CORPUS,
    steps=80,
    seq_len=48,
    batch_size=8,
    device="cpu",
    seed=0,
    checkpoint_path=None,
):
    """完整的训练流程。

    参数:
        text:            语料
        steps:           训练步数（不是 epoch 数 —— 这里是按步训练）
        seq_len:         每块长度
        batch_size:      批大小
        device:          "cpu" 或 "cuda"
        seed:            随机种子
        checkpoint_path: 保存路径，None 表示不保存

    返回:
        (model, tokenizer, val_loss)
    """
    # 固定全局随机种子：影响权重初始化和 dropout 等一切随机行为
    torch.manual_seed(seed)

    tokenizer, train_loader, val_loader = make_dataloaders(
        text, seq_len=seq_len, batch_size=batch_size, seed=seed
    )

    # ── 模型配置 ─────────────────────────────────────────────────────────
    # vocab_size 必须来自分词器 —— 词表大小决定了 embedding 和 LM head 的形状
    config = MiniMindConfig(
        vocab_size=tokenizer.vocab_size,   # 例如 4 + 字符种数
        dim=64,                            # 模型维度（教学模型取小，CPU 上跑得动）
        n_layers=2,                        # 2 层 decoder block
        n_heads=4,                         # 4 个 query head
        n_kv_heads=2,                      # 2 个 K/V head（GQA）
        hidden_dim=128,                    # SwiGLU 中间宽度 = 2×dim
        max_seq_len=seq_len,               # 窗口正好等于块长
        pad_token_id=tokenizer.pad_token_id,
    )
    # 校验一遍：64 % 4 == 0 ✓；4 % 2 == 0 ✓；head_dim = 16 为偶数 ✓

    model = MiniMindCore(config).to(device)

    # ── 优化器：AdamW ────────────────────────────────────────────────────
    # AdamW = Adam + 解耦的权重衰减（decoupled weight decay）。
    # 传统 Adam 把 L2 正则混进梯度里，会被自适应学习率缩放；
    # AdamW 把权重衰减单独作用在参数上，效果更符合预期，是现在的主流默认选择。
    #
    # lr=2e-3 对这么小的模型是合适的；大模型通常用 1e-4 ~ 3e-4 量级。
    # 2e-3 是科学计数法，等于 0.002。
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)

    # ── 手工迭代 DataLoader ──────────────────────────────────────────────
    # iter(loader) 得到一个迭代器，next() 一次取一个 batch。
    # 为什么不用 for epoch in range(...)？
    #   因为这里是"按步训练"：总共跑 steps 步，而不是跑完整 epoch。
    #   语料很小，跑一个 epoch 可能只有十几步。
    iterator = iter(train_loader)

    # 切到训练模式（本模型无 Dropout，但保持习惯）
    model.train()

    for step in range(1, steps + 1):
        # ── 取一个 batch；取完就重新开一轮 ──────────────────────────────
        try:
            input_ids, labels, attention_mask = next(iterator)
        except StopIteration:
            # 迭代器耗尽时会抛 StopIteration。
            # 这里重新构造迭代器（相当于进入下一个 epoch），保证循环不会中断。
            # 注意 train_loader 的 shuffle 用的是同一个 generator，
            # 所以每轮的顺序仍然可复现。
            iterator = iter(train_loader)
            input_ids, labels, attention_mask = next(iterator)

        # 搬到设备
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_mask = attention_mask.to(device)

        # ── 前向：模型内部完成 loss mask ────────────────────────────────
        # 传入的 labels 已经是错位好的（由 Dataset 负责），
        # 模型不会自动 shift。这点在 06 节强调过。
        _, loss = model(input_ids, labels, attention_mask=attention_mask)

        # ── 反向传播的标准四步 ──────────────────────────────────────────
        # (1) 清空梯度
        #     PyTorch 默认【累加】梯度，不清零的话上次的梯度会叠加进来。
        #     set_to_none=True 把 .grad 设为 None 而不是填 0：
        #       省一次写显存，且后续 backward 会直接覆盖而非累加，略快。
        optimizer.zero_grad(set_to_none=True)

        # (2) 反向传播：自动微分算出所有参数的 .grad
        loss.backward()

        # (3) 梯度裁剪
        #     clip_grad_norm_ 把所有参数的梯度拼成一个向量，若其 L2 范数超过
        #     max_norm=1.0，就整体等比缩小到这个范数。
        #     作用：防止个别 batch 产生巨大梯度把参数"打飞"，
        #     对 Transformer 这种深层结构几乎是必备操作。
        #     注意末尾的下划线：它是【原地】修改参数的 .grad。
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # (4) 更新参数
        optimizer.step()

        # ── 打印进度 ────────────────────────────────────────────────────
        # step == 1：第一步一定打印（便于确认训练真的在跑）
        # step % max(steps // 4, 1) == 0：大约每 1/4 的进度打印一次
        #   max(..., 1) 防止 steps<4 时 steps//4 变成 0（对 0 取模会报错）
        # f-string 里的 :03d 表示"补零到 3 位"，:.4f 表示保留 4 位小数
        if step == 1 or step % max(steps // 4, 1) == 0:
            print(f"step={step:03d} train_loss={loss.item():.4f}")
            # loss.item() 取出 0 维张量的 Python 数值。

    # ── 训练结束后的验证 ────────────────────────────────────────────────
    val_loss = evaluate(model, val_loader, device)
    print(f"validation_loss={val_loss:.4f}")

    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, model, optimizer, tokenizer, steps, val_loss)
        # 打印保存的路径，方便下一步（10 节的生成脚本）直接复制使用
        print(f"checkpoint={Path(checkpoint_path)}")

    return model, tokenizer, val_loss


# ============================================================================
# 第11步：命令行接口
# ============================================================================
def parse_args():
    """定义并解析命令行参数。

    argparse 会自动生成 --help 并做类型转换和基本校验。
    """
    # description=__doc__ 把模块 docstring 作为帮助文本，
    # 这样 --help 时能直接看到文件开头的说明
    parser = argparse.ArgumentParser(description=__doc__)

    # type=Path 让 argparse 自动把字符串转成 Path 对象。
    # 不传 --text 时是 None，走内置语料（离线可用）。
    parser.add_argument("--text", type=Path, help="optional UTF-8 corpus; built-in prose is used offline")

    # type=int 会把字符串转成整数，非法输入时 argparse 自己报错
    parser.add_argument("--steps", type=int, default=80)
    # 注意 argparse 把下划线自动转成短横线：--seq_len 要写成 --seq-len
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=Path("minimind_demo.pt"))

    # parse_args() 返回一个 Namespace 对象，用 args.steps 这样访问
    return parser.parse_args()


def main():
    """脚本入口：解析参数、选设备、开始训练。"""
    args = parse_args()

    if args.steps <= 0:
        raise ValueError("steps must be positive")

    # read_text(encoding="utf-8") 显式指定编码 ——
    # 在 Windows 上默认编码往往不是 utf-8，读中文文件必须写清楚，
    # 否则会出现乱码或 UnicodeDecodeError。
    # 三元表达式：给了 --text 就读文件，否则用内置语料。
    text = args.text.read_text(encoding="utf-8") if args.text else DEFAULT_CORPUS

    # torch.cuda.is_available() 判断当前环境是否真的能用 GPU
    # （装了 CUDA 版 PyTorch 且有可用显卡才为 True）
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 打印设备与语料长度，便于确认输入正确
    print(f"device={device} corpus_chars={len(text)}")

    train_model(
        text=text,
        steps=args.steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
    )


# ============================================================================
# 第12步：运行与核对
# ============================================================================
# 短实验（几秒钟就能跑完，用来确认整条链路通畅）：
#
#     python exercises/block_03_transformer/task_28_next_token_training/train.py \
#       --steps 4 --seq-len 24 --batch-size 2 \
#       --checkpoint /tmp/minimind_smoke.pt
#
# 终端输出形如：
#
#     device=... corpus_chars=...
#     step=001 train_loss=...
#     ...
#     validation_loss=...
#     checkpoint=/tmp/minimind_smoke.pt
#
# 这段输出中的 loss 为有限数，最后会写出 checkpoint 文件。
#
# 80 步的默认演示命令是：
#
#     python .../train.py --steps 80 --checkpoint /tmp/minimind_demo.pt
#
# 换成本地文本：
#
#     python .../train.py --text path/to/corpus.txt --steps 200 \
#       --checkpoint /tmp/minimind_demo.pt
#
# 文本过短时脚本会报错；train/validation 两侧都需要至少两个 token。
#
# 同一组性质也收录在 Block 3 测试中：
#
#     python -m unittest discover -s tests -p 'test_block3.py' -v
#
# 测试覆盖：tokenizer 的训练集拟合范围、split 样本隔离、input/label 错位、
# causal/padding/loss 三类 mask、attention 非零梯度、固定小 batch 的过拟合现象、
# 按有效 token 加权的 validation，以及 checkpoint round-trip 前后的 eval logits。
#
# 这条管线验证的是：
#
#     text -> tokenizer -> dataset -> masks -> Transformer
#          -> loss -> backward -> validation -> checkpoint
#
# 它没有大规模语料、成熟 tokenizer、分布式训练、系统评测或可用的对话能力，
# 因此这里的结果代表【微型训练管线闭环】，不等同于从头训练通用 LLM。
# ============================================================================


if __name__ == "__main__":
    # 只有在直接运行本文件时才执行 main()；
    # 被其他文件 import 时不会触发（10 节的 generate.py 就会 import load_checkpoint）
    main()


# ============================================================================
# 补充知识点：训练一个 Transformer 的最小闭环
# ============================================================================
# ── 五个必须做对的环节 ────────────────────────────────────────────────────
#   1. 标签错位    input 与 label 差一位，否则模型学成恒等映射
#   2. 两条 mask   attention_mask 屏蔽 PAD key；target 置 -100 排除出 loss
#   3. 梯度清零    PyTorch 默认累加梯度，不清零等于变相增大 batch
#   4. 梯度裁剪    防止偶发的大梯度破坏训练（Transformer 几乎必备）
#   5. 正确的验证分母  按有效 token 数加权，而不是按 batch 数平均
#
# ── 怎么判断 attention 真的参与了学习 ─────────────────────────────────────
# 第一层 q_proj.weight.grad 在非退化 batch 上应当有非零梯度。
# 如果 loss 在变化而 attention 梯度恒为 0，说明损失下降不能归因于 Transformer 主干 ——
# 很可能 attention 被某种写法架空了（比如残差绕过了它）。
# 这是一个非常高效的"结构性错误"检测手段。
#
# ── checkpoint 能恢复什么 ─────────────────────────────────────────────────
# 一次完整 round-trip 应呈现为：
#     - 保存前 eval logits == 加载后 eval logits（浮点容差内）
#     - weight tying 仍成立
#     - tokenizer 映射一致
#     - step / val_loss 元数据一致
# ──────────────────────────────────────────────────────────────────────────
# 特别提醒：这几十步的小语料实验【不能】用来推断泛化能力。
# validation loss 在这里的作用只是确认"验证通路是通的"，
# 不是"模型学得好不好"的证据。
# ============================================================================


# ============================================================================
# 核心公式回顾
# ============================================================================
# (1) 标签错位
#
#         block  = ids[start : start + seq_len + 1]
#         input  = block[:-1]
#         label  = block[1:]
#         ⇒ label[i] = input[i+1]
#
# (2) 两条 mask 的分工
#
#         attention_mask[i] = (input_ids[i] != pad_id)        ← 屏蔽 PAD key
#         target[i] = -100  当 label[i] == pad_id 或 query 无效 ← 排除出 loss
#
# (3) 加权验证损失
#
#                    Σ_batch  loss_batch × n_valid_tokens
#         val_loss = ───────────────────────────────────────
#                           Σ_batch  n_valid_tokens
#
# (4) 训练四步
#
#         loss.backward() → clip_grad_norm_(params, 1.0) → step()
#         且每步之前 optimizer.zero_grad(set_to_none=True)
#
# (5) AdamW 更新（简化）
#
#         θ ← θ - lr · ( m̂ / (√v̂ + ε) ) - lr · λ · θ
#                                          ↑ 解耦的权重衰减项
# ============================================================================


# ============================================================================
# 总结
# ============================================================================
# 1. 这一节把"文本"变成"监督信号"，共三步：
#     切分（前 85% 训练 / 后 15% 验证）
#     拟合分词器（只在训练集上 fit，避免泄漏）
#     错位切块（每块 seq_len+1 个 id，输入去掉最后一个，标签去掉第一个）
#
# 2. 字符级 tokenizer 的选择是为了透明，不是为了性能。
#    dict.get(char, unk_id) 一行就实现了"未见字 → <unk>"。
#
# 3. 相邻块共享一个边界 token，保证每条 next-token 转移只训练一次。
#    这是"定长分块 + 错位"方案里最容易被忽略的设计点。
#
# 4. Padding 要走两条路径：
#     attention_mask 屏蔽 PAD key（影响表示）；
#     target 置 -100（影响监督）。
#    图中 EOS 对应的输入位置 attention mask 为 1，但它的下一个 target 是 PAD，
#    所以 loss mask 为 0 —— 这正是代码的行为。
#
# 5. 验证损失必须按有效 token 数加权。模型返回的是"非屏蔽 target 上的均值"，
#    所以要用同一个分母来聚合，否则最后一批 PAD 多的时候会失真。
#
# 6. Checkpoint 存 config/model/tokenizer/step/val_loss，
#    其中 config 用 dataclass 转 dict 保存，载入时用 ** 解包重建模型 ——
#    这是"配置与权重分离"的一个很干净的示范。
#
# 7. 这一节验证的是"微型训练管线闭环"，不是"训出了一个 LLM"。
#    链路通了、loss 有限、checkpoint 可 round-trip，目标就达到了。
#    下一步（10 节）转向 logits 到 token 的逐步生成过程。
# ============================================================================
