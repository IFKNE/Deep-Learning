"""
=============================================================================
文本预处理（Text Preprocessing）—— 完整带注释版
=============================================================================
本节是《动手学深度学习》第8章"循环神经网络"的前置数据处理环节。
循环神经网络（RNN）处理的是序列数据，而文本是最常见的序列数据之一。
但神经网络模型只能吃"数字"，不能直接吃"字符串"，
所以我们必须先把文本翻译成一串数字（词元索引），模型才能真正动手。

本节完整流水线（4 步）：
  1. 读取数据集：把整本书（Text 文件）读进内存，变成"若干行字符串"的列表；
  2. 词元化（tokenize）：把每一行字符串拆成更小的词元（token，单词或字符）；
  3. 构建词表（vocabulary）：统计所有唯一词元、按词频排序，为每个词元分配一个
     数字索引，建立"字符 ↔ 数字"的双向映射；
  4. 整合所有功能：把上面三步打包成 load_corpus_time_machine 函数，
     直接把整本书变成一串数字索引（corpus），喂给后面章节的 RNN 模型。

输入 : H.G.Wells 的《时间机器》文本（timemachine.txt，约 3 万多单词）
输出 : corpus（词元索引列表，用于训练）+ vocab（词表对象，做 字符串<->索引 翻译）
关键数据结构：词法上的"字典/哈希表"——token_to_idx 与 idx_to_token 互为镜像。
=============================================================================
"""

import collections  # collections.Counter：统计词元出现次数的工具
import re           # 正则表达式，用来清洗文本（去掉标点、统一小写）
from d2l import torch as d2l   # d2l 工具包：提供 DATA_HUB / download 等下载加载能力


# ============================================================================
# 读取数据集
# ============================================================================
# 目标：把《时间机器》加载成"由若干文本行组成的列表"，每行是一个字符串。
# 为简化问题，这里忽略标点符号和大写（统一转小写，非字母一律替换成空格）。
# 注意：d2l 包的 d2l.DATA_HUB 是"数据集注册表"，用来把数据集的下载地址和
#       md5 校验码登记进去，之后用 d2l.download('time_machine') 就能按需下载。
#       这里直接修改全局字典 DTA_HUB，给它新增名叫 'time_machine' 的一条记录。

# 向 d2l 的全局数据字典注册《时间机器》数据集。
#   元组第 1 项 = 完整下载地址（d2l.DATA_URL + 文件名）
#   元组第 2 项 = 文件的 md5 校验码，用于下载后校验完整性（防损坏/防篡改）
# d2l.DATA_URL 是 d2l 内置的官方数据仓库前缀，这里直接拼上文件名即可。
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')


# 说明：d2l 包其实已经内置了 d2l.read_time_machine 这个同名函数，
#       但为了忠实地还原本 notebook（并保持文件自包含、可独立阅读），
#       这里仍然把函数定义写在本地。调用时仍用本地的 read_time_machine()，
#       功能与 d2l 包内置版本完全一致（本地定义覆盖/等价于 d2l 的版本）。
def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    # d2l.download('time_machine')：若本地无该文件则自动下载，返回本地文件路径；
    #   'r' 模式以只读方式打开文本文件；with ... as ... 是上下文管理器（类似 C++ RAII），
    #   离开 with 块时自动关闭文件句柄，即使发生异常也能保证文件被关闭。
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()   # 一次性读出所有行，得到一个字符串列表（每行含换行符）

    # 对每一行做清洗，下面用列表推导式逐行处理，相当于 C 里的 for 循环 + push_back：
    #   re.sub('[^A-Za-z]+', ' ', line)
    #       用正则把"所有不属于 A-Za-z 的连续字符"（即标点、数字、换行等）替换成一个空格。
    #       [^...] 表示"取反字符集"，+ 表示"一个或多个"。效果：非字母全部压成单个空格。
    #   .strip()   去掉行首行尾的多余空白（因为 replace 可能产生前导/尾随空格）。
    #   .lower()   把所有字母统一转成小写，从而忽略大小写差异。
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


# 实际调用：把整本书读取为 "lines"，每条是一个字符串。
lines = read_time_machine()
print(f'# 文本总行数: {len(lines)}')
print(lines[0])    # 打印第一行，看清洗后的样子
print(lines[10])   # 打印第 11 行，验证结果


# ============================================================================
# 词元化
# ============================================================================
# 目标：把"一行字符串"进一步切成"最小的词元（token）"。
# 词元是文本的基本单位——可以是单词（word），也可以是字符（char）。
# 返回一个"词元列表的列表"：外层对应每一行，内层是那一行的词元序列。
def tokenize(lines, token='word'):   # token 参数选择词元类型，默认按单词切分
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        # str.split() 默认按"任意空白符"切分字符串，返回单词列表。
        # [line.split() for line in lines]：对每一行都切一次，得到"行的列表"。
        return [line.split() for line in lines]
    elif token == 'char':
        # list(line)：把字符串拆成单个字符的列表（每个字符是一个词元）。
        # 例如 "abc" → ['a', 'b', 'c']。
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)


# 用默认的"按单词"方式对刚才读到的文本做词元化，得到 "tokens"。
tokens = tokenize(lines)
# 打印前 11 个词的词元列表，直观感受切分结果。
for i in range(11):
    print(tokens[i])


# ============================================================================
# 词表
# ============================================================================
# 目标：建立"字符串 词元 <-> 数字索引"的双向映射。
# 词元是字符串，模型不能直接用；我们要给每个词元编一个从 0 开始的整数号。
# 步骤：把训练集所有文档合并 → 统计每个唯一词元出现的次数（即"语料 corpus"）
#       → 按词频从高到低排序 → 分配索引。
# 低频词元通常会被丢弃（降低词表大小）；语料里没见过或被删掉的词元统一映射到
# 特殊词元 "<unk>"（unknown，未知词）。此外还可预留一些特殊词元：
#   '<pad>' 填充（补齐长度）、'<bos>' 序列开始、'<eos>' 序列结束。
class Vocab:   # 词表类：本质是一对互为镜像的字典 + 排序后的词频列表
    """文本词表"""
    # __init__ 是构造函数，创建对象时自动调用（类似 C++ 的构造器）。
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        # 防止调用方传 None 造成后续遍历出错：给一个默认空列表。
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        # 统计所有词元的出现频率，得到形如 {'the': 105, 'a': 50, ...} 的计数。
        counter = count_corpus(tokens)

        # 把 (词元, 词频) 组成的 items() 按词频 freq(x[1]) 从大到小排序。
        #   key=lambda x: x[1]：匿名函数，对每个元组取第 2 个元素（词频）作为排序关键字；
        #   reverse=True：降序——词频最高的排最前，先分配的小索引号。
        # 结果 self._token_freqs 是"按词频降序的 (词元, 词频) 列表"。
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)

        # 索引 → 词元 的列表。约定：未知词元 <unk> 的索引固定为 0。
        # 列表前部先放 <unk>，再放用户预留的特殊词元（如 <pad>、<bos>、<eos>）。
        self.idx_to_token = ['<unk>'] + reserved_tokens

        # 词元 → 索引 的字典。用字典推导式：
        #   enumerate(self.idx_to_token) 同时给出 (下标 idx, 词元 token)，
        #   于是建立 {token: idx} 的"查名字得编号"映射。
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

        # 遍历排序后的 (词元, 词频)，把"够常见"的词元补进词表。
        for token, freq in self._token_freqs:
            if freq < min_freq:   # 低于最小词频就停止（列表已按降序，后续只会更小）
                break
            if token not in self.token_to_idx:   # 该词元还没被登记过
                # 追加到"索引→词元"列表，新索引 = 当前列表长度 - 1。
                self.idx_to_token.append(token)
                # 同步登记"词元→索引"。len(...) - 1 就是刚才 append 进去那个单词的下标。
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    # __len__ 是 Python 特殊方法：len(vocab) 时自动调用，返回"词表大小"（词元总数）。
    def __len__(self):
        return len(self.idx_to_token)

    # __getitem__ 是 Python 特殊方法：vocab[token] 或 vocab[tokens] 时自动调用。
    # 它让"词元的索引查找"可以用方括号下标语法完成，很像 C++ 的重载 operator[]。
    def __getitem__(self, tokens):
        # 若传入的是单个词元（不是列表/元组），直接查字典返回其索引；
        #   查不到时用 .get(tokens, self.unk) 兜底返回 0，也就是 <unk> 的索引。
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)

        # 若传入的是一批词元（列表/元组），递归地让每个词元都走一遍上面的逻辑，
        # 返回"索引组成的列表"。
        return [self.__getitem__(token) for token in tokens]

    # 反向翻译：给定"索引"列表/单个索引，返回对应的"词元"。
    # 这是 idx_to_token 方向的查询，用于把模型输出索引还原成可读文本。
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]   # 单个索引 → 单个词元
        return [self.idx_to_token[index] for index in indices]   # 列表 → 词元列表

    @property
    def unk(self):   # @property 装饰器：让 vocab.unk 像访问属性一样（无需加括号）
        """未知词元 <unk> 的索引固定为 0"""
        return 0

    @property
    def token_freqs(self):   # @property：让外部能只读地拿到"按词频排序的 (词元,词频) 列表"
        return self._token_freqs


# count_corpus：统计词元出现频率，是 Vocab 的底层依赖。
def count_corpus(tokens):   # #@save 标记已去除，函数保持自包含
    """统计词元的频率"""
    # 这里 tokens 可能是 1D 列表（已展平的一系列词元），也可能是 2D 列表（"行→词元"的嵌套）。
    # 若第一个元素本身是 list，说明是 2D 嵌套结构，需要先展平成一维。
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 双层列表推导式：外层 for line 逐行，内层 for token 逐个词元，
        # 把所有词元"平铺"进同一个一维列表。
        tokens = [token for line in tokens for token in line]

    # collections.Counter(iterable) 返回一个"词元 → 次数"的计数器对象，
    #   能像字典一样 dict(token) 查次数，也自带 .items() / .most_common() 等工具。
    return collections.Counter(tokens)


# --- 使用时光机器数据集作为语料构建词表 ---
# 刚才 tokenize 得到的是"词元列表的列表"，这里直接喂给 Vocab 构造函数，
# 构造过程会自动统计词频、排序、给每个词元分配索引。
vocab = Vocab(tokens)
# 打印词表里编号最小的前 10 个 (词元, 索引) 对，验证：最前面的是 <unk> + 高频词。
print(list(vocab.token_to_idx.items())[:10])


# --- 把每条文本行转换成数字索引列表 ---
# 遍历两条测试行（第 0 行和第 10 行），演示"字符串 → 索引"的可逆翻译。
for i in [0, 10]:
    print('文本:', tokens[i])              # 原始的"词元列表"
    # vocab[tokens[i]]：__getitem__ 收到一个"列表"，递归地把每个词元翻译成索引。
    print('索引:', vocab[tokens[i]])


# ============================================================================
# 整合所有功能
# ============================================================================
# 目标：把"读取 → 词元化 → 建词表 → 转索引"整合成一个函数，直接输出可训练的数据。
# 相比前面单独调用，这里做了两处改动：
#   (1) 用"字符"而不是"单词"做词元化，简化后面章节的训练（字符级词表通常几十个词元）；
#   (2) 返回的 corpus 是一个"单个扁平列表"（而不是多层嵌套）——因为《时间机器》里
#       的每一行未必是完整句子/段落，可能只是一个单词，无需保留行结构。
def load_corpus_time_machine(max_tokens=-1):   # max_tokens=-1 表示"不限长度"（默认全要）
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()          # 第 1 步：读取文本行列表
    tokens = tokenize(lines, 'char')     # 第 2 步：按"字符"词元化
    vocab = Vocab(tokens)                # 第 3 步：构建字符级词表
    # 第 4 步：把所有行的所有词元"展平"成一条长的一维索引序列。
    #   双层推导式：内层 token 是每条文本行里一个个词元，vocab[token] 取到它的索引。
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:                   # 若指定了最大长度，就截断前 max_tokens 个
        corpus = corpus[:max_tokens]
    return corpus, vocab                 # 返回 (索引序列, 词表对象)


# 调用整合函数，得到整个《时间机器》的字符索引序列与对应词表。
corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)   # 返回 (字符总数, 词表大小)——即语料长度与词表规模


# ============================================================================
# 补充知识点
# ============================================================================

# 1. 词法预处理整条流水线
#    原始文本 --read_time_machine--> lines(行列表)
#                    --tokenize--> tokens(词元列表的列表)
#                         --Vocab--> vocab(词表) + corpus(索引序列)
#    一句话串联：文本 -> 行 -> 词元 -> (统计/排序/编号) -> 索引。RNN 之后吃的就是 token 编号。

# 2. collections.Counter 是什么？
#    它是一个"哈希计数器"，统计可哈希元素（字符串、数字等）各自出现了多少次。
#    Counter(['a','b','a']) 会得到 Counter({'a': 2, 'b': 1})。
#    它继承自 dict，所以可用 .items()、.get()，也可像普通字典一样索引。

# 3. Python 特殊方法（魔法方法 / dunder 方法）
#    以双下划线开头结尾的方法由 Python 解释器"勾住"，在特定语法出现时自动调用：
#       __init__   → 创建对象时调用（构造函数）
#       __len__    → 内建函数 len(obj) 时调用
#       __getitem__→ 下标 obj[x] 时调用（让对象支持方括号索引，可读性极佳）
#    这正是 Vocab 能用 vocab['the']、len(vocab) 这种"像字典又像列表"的语法的原因。

# 4. index_to_token 与 token_to_idx 双向映射
#    两者互为"逆映射"，缺一不可：
#      token_to_idx : 词元 → 索引（正向，推理时把输入文本翻译成数字）
#      idx_to_token : 索引 → 词元（逆向，解码时把模型输出数字翻译回文本）
#    放在同一个类里成对维护，保证两侧始终一致，不会出现"有词没号/有号没词"。

# 5. <unk>（未知词）与特殊词元
#    <unk>（索引 0）：语料里从未出现、或频次过低被删除的词，统一映射到它。
#    这样模型遇到"生词"也不会报错，而是得到一个固定的占位索引。
#    <pad> 填充 / <bos> 序列开始 / <eos> 序列结束：处理"长度不一"的序列批与起止标志，
#    本 notebook 未用到，但词表预留了 reserved_tokens 参数以支持后续章节。

# 6. 字符级 vs 词级词表
#    字符级：词元=单个字符。词表很小（常见语言就几十个），不会出现"未登录词"，
#           能处理任意拼写，但序列更长、模型要学"拼写成词"的规律。
#    词级：词元=单词。序列更短、语义更整，但词表很大、且有大量低频词/未登录词，
#         常需要 <unk> 兜底，还需处理大小写与形态变化。
#    本 notebook 最后用字符级词元化，就是为了把语料压扁成一个简洁的一维索引序列，方便训练。

# 7. 为什么 lines 里的每一行不一定是句子？
#    《时间机器》原文的行，既可能是整句/整段，也可能只是一个单词。
#    所以 load_corpus_time_machine 不保留"行"的结构，直接展平成一个长列表，
#    避免模型在"行边界"上产生多余的假设。


# ============================================================================
# 核心公式回顾
# ============================================================================
#   词频统计（Counter）：     count(token) = 词元在语料中出现的次数
#   词表排序关键字：          sorted(..., key=lambda x: x[1], reverse=True)  ← 按词频降序
#   索引分配：                idx(token) = 该词元在"按词频降序排列的列表"中的位置
#   词元→索引查表：           vocab[token] = token_to_idx.get(token, unk=0)
#   索引→词元还原：           vocab.to_tokens(idx) = idx_to_token[idx]
#   字符级语料展平：          corpus = [vocab[token] for line in tokens for token in line]


# ============================================================================
# 总结
# ============================================================================
# 1. 文本是序列数据最典型的形式，RNN 前必须先做预处理，把字符串翻译成数字索引。
# 2. 预处理三步曲：读取数据集 → 词元化（tokenize）→ 构建词表（Vocab）。
# 3. Vocab 用双向字典（token_to_idx / idx_to_token）实现"词元 <-> 索引"映射，
#    并用 Counter + 排序得到一个按词频降序的稳定编号，高频词编号小。
# 4. 特殊词元 <unk> 兜底"未知/被删词"，保留了 reserved_tokens 以支持 <pad>/<bos>/<eos>。
# 5. load_corpus_time_machine 把整条流水线打包，输出"字符级索引序列 + 词表"，
#    供第8章后续的 RNN 模型直接消费。
