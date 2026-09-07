"""
=============================================================================
机器翻译与数据集（Machine Translation & Dataset）—— 完整带注释版
=============================================================================
任务：机器翻译把"源语言序列"转成"目标语言序列"，是序列转换模型的核心问题。
     本文件准备"英-法"翻译数据集：下载、预处理、词元化、建词表、截断/填充，
     最终产出可迭代的小批量训练数据。

与第 8 章语言模型的差别：这里是"一对平行语料"（源 + 目标），
而非单一语言的长语料，预处理方式因此完全不同。

依赖：d2l 包。本文件可独立运行。
=============================================================================
"""

# ============================================================================
# 第1步：导入依赖
# ============================================================================
import os
import torch
from d2l import torch as d2l


# ============================================================================
# 第2步：下载并读取数据集
# ============================================================================
# d2l.DATA_HUB 是本书维护的"已下载数据集索引"。这里注册'mfra-eng'的下载地址与
#   校验值（SHA-1）。
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')


def read_data_nmt():
    """载入“英语－法语”数据集"""
    # d2l.download_extract 负责下载并解压 fra-eng.zip（若已存在则直接复用）
    data_dir = d2l.download_extract('fra-eng')
    # 以 UTF-8 读取数据集文件 fra.txt，每行是一对"英文\法文"（用制表符分隔）
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
              encoding='utf-8') as f:
        return f.read()


# 读取全部原始文本并打印前 75 个字符，看看长什么样
raw_text = read_data_nmt()
print(raw_text[:75])
# 输出示例：
#   Go.	Va !
#   Hi.	Salut !
#   Run!	Cours !
#   ...


# ============================================================================
# 第3步：预处理
# ============================================================================
def preprocess_nmt(text):
    """预处理“英语－法语”数据集"""
    # no_space：判断某个字符是否是"需要前面加空格的标点"
    #   若 char 是 ,.!? 之一，且它的前一个字符不是空格就返回 True
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # ① 用普通空格替换"不间断空格"（  与 \xa0，避免被当作词元的一部分）
    # ② 全部转小写（统一大小写，减少词表规模）
    text = text.replace(' ', ' ').replace('\xa0', ' ').lower()
    # ③ 在单词与标点之间插入空格：对每个位置 i，如果当前字符是标点且前一个字符
    #    不是空格，则在其前面加一个空格。这样 "go." 会变成 "go ."。
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


text = preprocess_nmt(raw_text)
print(text[:80])
# 输出示例：
#   go .	va !
#   hi .	salut !
#   ...


# ============================================================================
# 第4步：词元化（单词级）
# ============================================================================
def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    # 按换行逐行处理；若指定了 num_examples 则只处理前那么多个样本
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        # 每行用制表符 '\t' 切分成 [英文部分, 法文部分]
        parts = line.split('\t')
        if len(parts) == 2:  # 只保留"格式正确"的行（恰好两个字段）
            # 各自按空格切分，得到"单词级"的词元列表
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target


# 对全部文本做词元化，展示前 6 对结果
source, target = tokenize_nmt(text)
source[:6], target[:6]
# 输出示例：
#   ([['go', '.'], ['hi', '.'], ['run', '!'], ...],
#    [['va', '!'], ['salut', '!'], ['cours', '!'], ...])


# ============================================================================
# 第5步：绘制"每个序列词元数量"的直方图
# ============================================================================
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """绘制列表长度对的直方图"""
    # d2l.set_figsize：设置绘图画布尺寸
    d2l.set_figsize()
    # 用 plt.hist 画两组长度的直方图（X 和 Y 各一组）
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    # 给第二组（target）柱子上加斜线纹理，便于区分
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)


# 查看统计：大多数英-法句对词元数少于 20 个
show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', source, target);


# ============================================================================
# 第6步：建词表（源、目标各一个）
# ============================================================================
# 单词级词表比字符级大很多，这里把"出现次数 < 2"的词统一视作 <unk>（未知词）。
# 另外保留三个特殊词元：
#   <pad> ：批量中用于把不同长度序列填充到相同长度
#   <bos> ：序列开始（begin of sequence）
#   <eos> ：序列结束（end of sequence）
src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
# 源词表大小（示例约 10012）
len(src_vocab)


# ============================================================================
# 第7步：加载数据集（截断与填充）
# ============================================================================
def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断：只保留前 num_steps 个词元
    return line + [padding_token] * (num_steps - len(line))  # 填充：补 <pad>


# 示例：把 source 的第一个句子的索引，截断/填充到长度 10
#   source[0] = ['go', '.'] 对应词表索引，结果补了很多 <pad>
truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
# 示例输出：[47, 4, 1, 1, 1, 1, 1, 1, 1, 1]


def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    # ① 词元 → 词表索引（词元化后的 token list → 每个词元对应的整数索引）
    lines = [vocab[l] for l in lines]
    # ② 每个序列末尾追加 <eos>，标序列的结束
    lines = [l + [vocab['<eos>']] for l in lines]
    # ③ 对每个序列做截断/填充，使其长度恰好为 num_steps，再转成张量
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    # ④ 有效长度：统计每个序列中"不等于 <pad>"的元素个数（去掉填充部分）
    #    .type(torch.int32)：转成整型；.sum(1)：按行求和 → 每个序列的有效长度
    valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
    return array, valid_len


# ============================================================================
# 第8步：训练模型 —— 封装成 load_data_nmt
# ============================================================================
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表"""
    # 完整流水线：预处理 → 词元化 → 建源/目标词表 → 转张量
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    # 打包成四个字段：源序列、源有效长度、目标序列、目标有效长度
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    # d2l.load_array：按数组打包成 DataLoader 可迭代小批量
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab


# 读出第一个小批量数据（演示形状与内容）
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', X.type(torch.int32))
    print('X的有效长度:', X_valid_len)
    print('Y:', Y.type(torch.int32))
    print('Y的有效长度:', Y_valid_len)
    break
# 示例输出：
#   X: tensor([[ 7, 43,  4,  3,  1,  1,  1,  1], [44, 23,  4,  3, ...]], ...)
#   这里 1 是 <pad> 的索引，有效长度提示每个序列真正有效（非填充）的词元数


# ============================================================================
# 补充知识点
# ============================================================================
# 1. 【为什么用"单词级"词元化？】
#    语言模型常用字符级（小词表），但翻译更倾向于单词级 token 化——
#    词能直接对应语义。代价是词表远大于字符集，因此用 min_freq=2 把低频词
#    归并为 <unk>，控制词表规模。
#    （现代最先进模型会用 BPE/WordPiece 等子词 token 化，属更高级方案。）

# 2. 【截断与填充的必要性】
#    不同句子长度不同，GPU 批量计算要求同一批长度一致。用 <pad> 补短、截断长，
#    即可用统一 num_steps 的小批量。同时用 valid_len 记录"真实长度"，
#    供后续损失函数忽略 <pad> 位置。

# 3. 【<bos>/<eos> 的作用】
#    <eos>：告诉解码器"输出到此结束"；<bos>：作为解码器的第一个输入。
#    训练时用 teacher forcing（把正确上一步当输入），预测时遇到 <eos> 才停。


# ============================================================================
# 核心公式回顾
# ============================================================================
#   数据流：raw_text → preprocess_nmt → tokenize_nmt(source, target)
#          → 各建 Vocab → build_array_nmt → load_array(data_iter)
#   每个样本 = (源序列索引, 源valid_len, 目标序列索引, 目标valid_len)
#   长度统一 = num_steps；valid_len = 实际有效（非 <pad>）词元数。


# ============================================================================
# 练习与思考
# ============================================================================
# 1) 在 load_data_nmt 中尝试不同的 num_examples 值，看它对源/目标词表大小的影响。
# 2) 中文、日文等语言没有空格这种"词边界指示符"，单词级词元化还是好主意吗？
#    （提示：通常需要分词工具或改用字符/子词级。）


# ============================================================================
# 总结
# ============================================================================
#  - 机器翻译 = 序列转换任务，数据由"源语言-目标语言"文本序列对组成。
#  - 与语言模型不同，需一套专门预处理：下载/清洗/词元化/建词表/截断填充。
#  - 单词级词表更大 → 用低频词归并 <unk> 缓解。
#  - 用 <pad> 统一长度 + valid_len 记录真实长度，为下一节 seq2seq 打基础。
