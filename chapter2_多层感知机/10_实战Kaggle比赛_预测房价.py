"""
=============================================================================
实战Kaggle比赛：预测房价 —— 完整数据处理与模型训练（完整带注释版）
=============================================================================
数据集：Kaggle House Prices - Advanced Regression Techniques
        包含2006-2010年亚利桑那州埃姆斯市的房屋销售数据
        训练集：1460条样本，测试集：1459条样本
        每条样本有约80个特征（包括数值型和类别型）
目标：  预测房屋销售价格（SalePrice）
评估：  使用对数均方根误差（Log RMSE）作为评价指标
=============================================================================
本脚本涵盖以下深度学习知识点：
  1. 真实数据集的下载、缓存与完整性校验（SHA-1）
  2. 混合数据类型（数值型+类别型）的完整预处理流程
  3. 标准化（零均值、单位方差）、缺失值填充
  4. 独热编码（One-Hot Encoding）处理离散类别特征
  5. K折交叉验证（K-Fold Cross Validation）
  6. 对数RMSE —— 为什么用对数衡量房价误差
  7. Adam优化器与权重衰减
  8. 基线模型设计（线性回归作为起始点）
  9. Kaggle竞赛提交流程
=============================================================================
"""

# ============================================================================
# 第1步：导入依赖库
# ============================================================================
import hashlib          # 用于计算文件的SHA-1哈希值，校验文件完整性
import os               # 用于文件和目录操作（路径拼接、目录创建等）
import tarfile           # 用于解压 .tar / .tar.gz 格式的压缩文件
import zipfile           # 用于解压 .zip 格式的压缩文件
import requests          # 用于发送HTTP请求，从网络下载数据集
import numpy as np       # 科学计算库，提供高效的多维数组操作
import pandas as pd      # 数据分析库，提供 DataFrame 结构，方便处理表格数据
import torch             # PyTorch深度学习框架核心库
from torch import nn     # 导入PyTorch的神经网络模块（包含各种层和损失函数）
from d2l import torch as d2l  # 《动手学深度学习》官方工具包
                               # 封装了绘图、数据加载等常用功能

# ============================================================================
# 第2步：构建数据集下载与缓存系统
# ============================================================================
# DATA_HUB: 全局字典，将数据集名称映射到(url, sha1哈希值)二元组
#           key=数据集名称字符串, value=(下载链接, SHA-1校验码)
# SHA-1校验码用于验证下载文件的完整性，防止文件损坏或被篡改
DATA_HUB = dict()
# DATA_URL: 所有D2L数据集的统一托管地址（Amazon S3加速域名）
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

# --------------------------------------------------------------------------
# download(name, cache_dir): 下载数据集并缓存到本地
#   参数: name        -- DATA_HUB中注册的数据集名称
#         cache_dir   -- 本地缓存目录，默认为 ../data
#   返回: 本地文件的完整路径
# --------------------------------------------------------------------------
def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    # 断言：确保请求的数据集名称已注册在DATA_HUB中
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    # 从DATA_HUB中取出该数据集对应的下载URL和预期的SHA-1哈希值
    url, sha1_hash = DATA_HUB[name]
    # 递归创建缓存目录（exist_ok=True表示目录已存在时不报错）
    os.makedirs(cache_dir, exist_ok=True)
    # 从URL中提取文件名（取最后一个'/'之后的部分），拼接到缓存目录下
    fname = os.path.join(cache_dir, url.split('/')[-1])

    # --- 检查缓存：如果文件已存在，验证其SHA-1是否匹配 ---
    if os.path.exists(fname):
        sha1 = hashlib.sha1()                         # 创建SHA-1哈希计算器对象
        with open(fname, 'rb') as f:                   # 以二进制只读模式打开文件
            while True:
                data = f.read(1048576)                 # 每次读取1MB数据（1048576字节）
                if not data:                           # 读不到数据说明已到文件末尾
                    break
                sha1.update(data)                      # 将读取的数据块喂给哈希计算器
        # 将计算出的哈希值与DATA_HUB中记录的预期值对比
        if sha1.hexdigest() == sha1_hash:
            return fname  # 哈希匹配 → 缓存命中，直接返回本地路径，无需重新下载

    # --- 缓存未命中：从网络下载 ---
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)    # 发送GET请求，stream=True流式下载大文件
    with open(fname, 'wb') as f:                       # 以二进制写入模式打开文件
        f.write(r.content)                             # 将响应内容一次性写入文件
    return fname                                        # 返回下载后文件的本地路径

# --------------------------------------------------------------------------
# download_extract(name, folder): 下载并解压zip/tar压缩文件
#   参数: name  -- 数据集名称
#         folder-- 解压后的目标子文件夹名（可选）
#   返回: 解压后的目录路径
# --------------------------------------------------------------------------
def download_extract(name, folder=None):  #@save
    """下载并解压zip/tar文件"""
    fname = download(name)                              # 先调用download下载压缩文件
    base_dir = os.path.dirname(fname)                   # 获取文件所在目录
    data_dir, ext = os.path.splitext(fname)             # 分离文件名和扩展名（如 .zip, .tar）
    # 根据扩展名选择解压方式
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')                # 创建ZipFile对象用于解压.zip
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')                   # 创建TarFile对象用于解压.tar / .tar.gz
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)                             # 将所有文件解压到base_dir目录
    # 如果指定了子文件夹，返回其完整路径；否则返回去掉扩展名的路径
    return os.path.join(base_dir, folder) if folder else data_dir

# --------------------------------------------------------------------------
# download_all(): 批量下载DATA_HUB中注册的所有数据集
# --------------------------------------------------------------------------
def download_all():  #@save
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)


# ============================================================================
# 第3步：注册Kaggle房价数据集并加载
# ============================================================================
# 将Kaggle训练集和测试集注册到DATA_HUB中
# 每个条目包含：数据集在DATA_URL上的文件名、以及对应的SHA-1校验码
DATA_HUB['kaggle_house_train'] = (                      #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')         # 训练集的SHA-1哈希
DATA_HUB['kaggle_house_test'] = (                       #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')         # 测试集的SHA-1哈希

# pd.read_csv(): 读取CSV文件并将其转换为pandas的DataFrame对象
# DataFrame是一个二维表格型数据结构，类似Excel表格，有行索引和列名
# download()返回下载后文件的本地路径
train_data = pd.read_csv(download('kaggle_house_train'))   # 训练集DataFrame，大小 (1460, 81)
test_data = pd.read_csv(download('kaggle_house_test'))     # 测试集DataFrame，大小 (1459, 80)
                                                            # 测试集比训练集少一列：SalePrice（标签列）

# 查看数据形状：.shape返回 (行数, 列数) 元组
print(train_data.shape)  # 输出: (1460, 81) — 1460条记录，81列（含Id和SalePrice）
print(test_data.shape)   # 输出: (1459, 80) — 1459条记录，80列（含Id，无SalePrice）

# 查看前4条样本的部分列：
#   .iloc[行索引切片, 列索引列表] — 基于整数位置的索引
#   [0, 1, 2, 3, -3, -2, -1] 分别代表第0、1、2、3列和倒数第3、2、1列
#   这些列涵盖了：Id(标识), MSSubClass(住宅子类), MSZoning(分区类型),
#                LotFrontage(临街面宽度), SaleType(销售类型),
#                SaleCondition(销售条件), SalePrice(房价标签)
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])


# ============================================================================
# 第4步：特征合并 —— 将训练集和测试集的特征放在一起统一预处理
# ============================================================================
# 为什么要合并？因为预处理（标准化、独热编码）需要使用所有可用数据的统计量
# 如果分别处理训练集和测试集，可能导致两边的编码不一致（例如类别值不同）。
#
# pd.concat([DataFrame列表]): 沿轴拼接DataFrame，默认axis=0（纵向拼接，增加行）
#
# train_data.iloc[:, 1:-1]:
#   抛出第0列（Id列——它只是标识符，不包含任何预测信息，会让模型"记住"编号）
#   和第-1列（SalePrice列——这是标签，不是特征，测试集中也没有）
#   取第1列到倒数第2列的所有特征列
#
# test_data.iloc[:, 1:]:
#   同样抛出第0列（Id列），但test_data本就没有SalePrice列，所以只需取第1列到末尾
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# 此时 all_features 包含了训练集和测试集的所有特征数据（共2919行，79列原始特征）


# ============================================================================
# 第5步：数据预处理 —— 标准化数值特征
# ============================================================================
# 【知识点：标准化（零均值、单位方差）】
# 将每个特征值x变换为 z = (x - mean) / std
# 变换后的特征：均值为0，标准差为1（即"零均值单位方差"或"Z-score标准化"）
#
# 为什么需要标准化？
#   (1) 不同特征的量纲差异巨大（如房屋面积~2000，卧室数~3，年份~2000）
#       如果不做标准化，梯度下降时不同参数的梯度尺度差异大，优化不稳定
#   (2) 在使用正则化（如L2正则/权重衰减）时，如果不做标准化，
#       量纲大的特征会被惩罚得更多（这不公平——我们事先不知道哪个特征重要）
#   (3) 许多优化算法（如Adam、SGD）对输入尺度敏感，标准化有助于更快收敛
#
# 为什么不在填充缺失值之前先做标准化？
#   因为fillna(0)在标准化之后执行：标准化后均值=0，所以用0填充等价于用均值填充
#   如果先填充再标准化，填充的值会改变均值和标准差的计算

# all_features.dtypes: 返回每一列的数据类型（如 int64, float64, object(字符串/类别)）
# all_features.dtypes != 'object': 筛选出非对象类型的列 → 即数值型特征
# .index: 获取筛选后的列名列表
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index

# .apply(lambda x: ...): 对DataFrame的每一列（Series）应用匿名函数
#   lambda x: (x - x.mean()) / (x.std())
#   对每一列计算：(该列每个值 - 该列均值) / 该列标准差
#   x.mean() — 该列的均值μ；x.std() — 该列的标准差σ
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 此时数值特征已经过标准化：均值为0，标准差为1

# fillna(0): 将所有缺失值（NaN）填充为0
# 因为标准化后均值为0，用0填充等价于"用均值填充缺失值"
# 这是巧妙利用标准化性质的一步——标准化之后均值就是0
all_features[numeric_features] = all_features[numeric_features].fillna(0)


# ============================================================================
# 第6步：数据预处理 —— 独热编码处理离散（类别）特征
# ============================================================================
# 【知识点：独热编码 (One-Hot Encoding)】
# 将类别型特征转换为二值（0或1）的指示向量
# 例如：MSZoning列有 "RL", "RM", "FV" 三种值 →
#       生成三列：MSZoning_RL, MSZoning_RM, MSZoning_FV
#       如果某个样本的MSZoning="RL"，则MSZoning_RL=1，其余两列为0
#
# 为什么要独热编码？
#   类别值之间没有大小关系（"RL"不比"RM"大），不能直接用数字编码
#   (0="RL", 1="RM", 2="FV")，否则模型会认为"FV">"RM">"RL"
#   独热编码消除了这种虚假的序关系，将每个类别当作独立维度
#
# 为什么在标准化之后再编码？
#   数值特征先做标准化；类别特征后做独热编码——两类特征处理独立
#
# pd.get_dummies(DataFrame, dummy_na=True):
#   将DataFrame中所有类别列（object类型）自动转换为独热编码
#   dummy_na=True: 将缺失值(NaN)也视为一个有效类别，单独创建一列
#   例如：如果"MSZoning"某些行的值为NaN，会额外生成"MSZoning_nan"列
#   这样做的好处是：模型可以学习到"数据缺失"本身也是一个有意义的信号
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)  # 输出: (2919, 331)
# 特征数从79爆增到331——这是独热编码的典型代价：特征维度膨胀


# ============================================================================
# 第7步：将pandas数据转换为PyTorch张量
# ============================================================================
# n_train: 训练样本数量（用于切分合并后的all_features）
n_train = train_data.shape[0]

# all_features[:n_train].values: 取前1460行（训练集）的原始数值，返回NumPy数组
# torch.tensor(..., dtype=torch.float32): 将NumPy数组转换为PyTorch的float32张量
#   深度学习模型通常使用float32，因为GPU对float32的加速最好
#   张量形状: (1460, 331) — 1460个样本，每个样本331个特征
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)

# all_features[n_train:].values: 取后1459行（测试集）
#   张量形状: (1459, 331) — 1459个样本，331个特征
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)

# 训练标签（房价）: 从原始train_data中取出SalePrice列
#   train_data.SalePrice.values: 取SalePrice列的值（NumPy一维数组）
#   .reshape(-1, 1): 变形为列向量（-1表示自动推断该维度，1表示1列）
#   形状从 (1460,) 变为 (1460, 1) — 每行一个房价标签
#   PyTorch的损失函数（nn.MSELoss）要求标签形状与预测输出匹配
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)


# ============================================================================
# 第8步：定义模型 —— 线性回归基线
# ============================================================================
# 【知识点：线性回归作为基线 (Baseline)】
# 基线模型是最简单的模型，用于"健全性检查"：
#   如果连最简单的模型都能学到有用的东西，说明数据本身是有信息的
#   如果基线模型的误差很大，很可能是数据预处理出错了
#   基线还提供了一个参考点：复杂的模型必须比它做得更好才有意义

# nn.MSELoss(): 均方误差损失函数
#   MSE = (1/n) * sum((y_pred - y_true)^2)
#   reduction默认为'mean'，对批次内所有样本取平均
#   用于回归问题中衡量预测值与真实值之间的平方误差
loss = nn.MSELoss()

# 输入特征的维度（331个特征，经过独热编码后）
in_features = train_features.shape[1]

def get_net():
    """构建线性回归模型
    nn.Sequential: 容器模块，按顺序将多个层串联起来
       前一层输出自动成为后一层的输入 —— 类似流水线
    nn.Linear(in_features, 1): 全连接层（线性层）
       本质是 y = Wx + b（矩阵乘法 + 偏置）
       in_features=331: 输入维度（经过预处理后的特征数）
       out_features=1:  输出维度（预测一个房价数值）
       内部包含: 权重矩阵W的形状(1, 331)和偏置b的形状(1,)
    """
    net = nn.Sequential(nn.Linear(in_features, 1))
    return net


# ============================================================================
# 第9步：定义评估指标 —— 对数均方根误差 (Log RMSE)
# ============================================================================
# 【知识点：为什么用对数RMSE衡量房价预测？】
# 房价的误差应该用"相对误差"而非"绝对误差"来衡量
#   例1：预测10万美元，真实值12.5万美元 → 偏差2.5万，相对误差20%（很糟糕）
#   例2：预测390万美元，真实值400万美元 → 偏差10万，相对误差2.5%（还不错）
#   例1的绝对误差(2.5万)小于例2(10万)，但例1的预测质量更差
#   取对数后：log(12.5万)-log(10万) ≈ 0.223, log(400万)-log(390万) ≈ 0.025
#   对数将"倍数关系"转换为"差关系"，天然衡量相对误差
#
# Log RMSE公式：
#   sqrt( (1/n) * sum_i ( log(y_i) - log(y_pred_i) )^2 )
#
# 性质：
#   |log y - log y_pred| < delta 等价于 e^(-delta) < y_pred/y < e^delta
#   即对数误差delta对应价格的相对误差范围

def log_rmse(net, features, labels):
    """计算对数均方根误差
    参数:
        net:      模型
        features: 输入特征张量
        labels:   真实标签张量
    返回:
        float类型的对数RMSE值
    """
    # torch.clamp(input, min, max): 将输入张量的所有元素限制在 [min, max] 范围内
    #   net(features) 输出模型预测的房价值（可能为负数——这在物理上不合理）
    #   将小于1的值截断为1：因为 log(负数) 无定义，log(0) = -∞
    #   min=1 确保取对数时输入 > 0（真实房价 > 0，预测值也应该 > 0）
    #   max=float('inf') 表示不限制上界（预测再大也允许）
    clipped_preds = torch.clamp(net(features), 1, float('inf'))

    # torch.log(): 计算自然对数（以e为底）
    #   loss(clipped的对数预测, 真实的对数标签) → 计算对数空间中的MSE
    #   torch.sqrt(): 对MSE取平方根，得到RMSE
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    # .item(): 将只包含一个元素的张量转换为Python标量（float）
    #   用于从计算图中取出数值（不参与梯度计算）
    return rmse.item()


# ============================================================================
# 第10步：定义训练函数
# ============================================================================
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    """训练模型的通用函数
    参数:
        net:            模型
        train_features: 训练特征
        train_labels:   训练标签
        test_features:  验证/测试特征（可为None，表示无验证集）
        test_labels:    验证/测试标签（可为None）
        num_epochs:     训练轮数（完整遍历数据集的次数）
        learning_rate:  学习率（控制参数更新步长）
        weight_decay:   权重衰减系数（L2正则化强度，0表示不使用正则化）
        batch_size:     批次大小（每批处理的样本数）
    返回:
        (train_ls, test_ls): 每轮训练和验证的对数RMSE列表
    """
    train_ls, test_ls = [], []  # 分别存储每轮训练集和测试集的损失值

    # d2l.load_array(data, batch_size): 将(特征, 标签)元组包装为PyTorch数据迭代器
    #   每次迭代返回一个批次 (batch_size个样本的特征, batch_size个样本的标签)
    #   shuffle默认为True（训练时打乱数据顺序，有助于模型学习）
    train_iter = d2l.load_array((train_features, train_labels), batch_size)

    # 【知识点：Adam优化器】
    # torch.optim.Adam: 自适应矩估计优化器
    #   Adam结合了两种思想：
    #     (1) Momentum（动量）: 加速收敛，抑制震荡
    #     (2) RMSProp: 为每个参数自适应调整学习率
    #   优势：对初始学习率不太敏感，大多数情况下默认设置就能很好地工作
    #         是深度学习中最常用的优化器之一
    #
    # weight_decay: 权重衰减 = L2正则化
    #   在每次参数更新时，额外减去 weight_decay * 参数值
    #   等价于在损失函数中加上 (weight_decay/2) * ||w||^2
    #   作用：防止权重过大 → 防止过拟合 → 提升泛化能力
    #   weight_decay=0 表示不使用L2正则化
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)

    for epoch in range(num_epochs):                          # 外层循环：遍历每一轮
        for X, y in train_iter:                              # 内层循环：遍历每个批次
            optimizer.zero_grad()                            # 清除上一轮的梯度（否则梯度会累积）
            l = loss(net(X), y)                              # 前向传播：计算当前批次的损失
            l.backward()                                     # 反向传播：计算损失对每个参数的梯度
            optimizer.step()                                 # 参数更新：根据梯度更新权重
        # 每轮结束后，在整个训练集上计算对数RMSE（评估模型当前表现）
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:                          # 如果有验证集，同步计算验证集RMSE
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


# ============================================================================
# 第11步：K折交叉验证
# ============================================================================
# 【知识点：K折交叉验证 (K-Fold Cross Validation)】
# 目的：在固定数据集上更可靠地评估模型性能、选择合适的超参数
# 原理：
#   1. 将训练集均匀分成K份（折）
#   2. 每次取其中1份作为验证集，其余K-1份作为训练集
#   3. 重复K次，每次使用不同的那份作为验证集
#   4. 将K次的验证结果取平均，作为最终的性能估计
# 优点：
#   - 每个样本都有机会被用作验证，评估更稳定
#   - 比简单划分训练/验证集更充分利用数据（尤其数据量少时）
# 本例中K=5，即5折交叉验证

def get_k_fold_data(k, i, X, y):
    """获取第i折的K折交叉验证数据
    参数:
        k: 折数（如k=5）
        i: 当前折的索引（0到k-1），第i折作为验证集
        X: 全部特征数据
        y: 全部标签数据
    返回:
        (X_train, y_train, X_valid, y_valid): 训练数据和验证数据
    """
    assert k > 1  # k必须大于1，至少需要2折
    fold_size = X.shape[0] // k  # 每折的样本数（整除，可能舍弃尾部少量样本）

    X_train, y_train = None, None  # 训练集初始为空
    for j in range(k):
        # slice(start, stop): 创建一个切片对象，用于索引
        # 例如：k=5, fold_size=292, j=0 → slice(0, 292)
        idx = slice(j * fold_size, (j + 1) * fold_size)
        # X[idx, :]: 取出第j折的所有特征（所有列）
        # y[idx]:    取出第j折的所有标签
        X_part, y_part = X[idx, :], y[idx]

        if j == i:                                          # 当前折 == 指定的验证折
            X_valid, y_valid = X_part, y_part               # → 作为验证集
        elif X_train is None:                               # 第一份非验证数据
            X_train, y_train = X_part, y_part               # → 直接赋给训练集
        else:                                               # 后续的非验证数据
            # torch.cat([张量列表], dim): 沿指定维度拼接张量
            # dim=0 表示沿第0维（行/样本维度）拼接，即纵向堆叠
            X_train = torch.cat([X_train, X_part], 0)       # 将当前折拼入训练集
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    """执行K折交叉验证，返回平均训练和验证误差
    每次折都会：
      1. 从零开始创建一个全新的模型（不共享权重）
      2. 用该折的训练数据训练模型
      3. 在训练集和验证集上评估对数RMSE
      4. 第1折时绘制损失曲线，方便可视化训练过程
    """
    train_l_sum, valid_l_sum = 0, 0  # 累加所有折的训练/验证最终损失
    for i in range(k):
        # 获取第i折的数据划分
        data = get_k_fold_data(k, i, X_train, y_train)
        # 每折都创建一个全新的模型（避免折间信息泄露）
        net = get_net()
        # *data: 解包元组，等效于 train(net, X_train, y_train, X_valid, y_valid, ...)
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        # train_ls[-1]: 最后一轮（epoch）的训练损失
        # valid_ls[-1]: 最后一轮的验证损失
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]

        # 仅在第1折时绘制损失曲线（方便观察训练趋势，不必为每折都画图）
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
            # yscale='log': y轴用对数刻度，便于观察训练初期损失快速下降的过程

        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')

    # 返回K折平均的训练和验证损失
    return train_l_sum / k, valid_l_sum / k


# ============================================================================
# 第12步：模型选择 —— 超参数设定与交叉验证
# ============================================================================
# 超参数（Hyperparameters）是训练前人工设定的参数，模型不会自动学习它们：
#   k=5:           5折交叉验证
#   num_epochs=100: 训练100轮（遍历整个训练集100次）
#   lr=5:          学习率5（Adam优化器下可以使用较高的学习率）
#   weight_decay=0: 不使用权重衰减（基线模型，先不做正则化）
#   batch_size=64:  每批处理64个样本

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64

# 执行K折交叉验证，获取平均训练误差和平均验证误差
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)

# 打印最终的K折平均结果
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')


# ============================================================================
# 第13步：在全量训练数据上训练最终模型并生成Kaggle提交
# ============================================================================
def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    """使用全部训练数据训练最终的模型，并对测试集做预测
    注意：此时不再划分验证集，用全部训练数据来训练
    这是竞赛中的常见做法：交叉验证用于选超参数，全量训练用于最终提交
    """
    net = get_net()  # 构建一个新模型（从头训练）
    # 注意test_labels=None：不使用验证集，全量数据用于训练
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)

    # 绘制训练过程中的对数RMSE变化曲线
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')

    # net(test_features): 将测试集特征输入模型，得到预测值（PyTorch张量）
    # .detach(): 将预测值从计算图中分离，得到一个不跟踪梯度的张量
    #   【重要】：我们只需要预测结果，不需要计算梯度
    #   不调用detach()会导致numpy()失败（带梯度的张量不能直接转NumPy）
    # .numpy(): 将PyTorch张量转换为NumPy数组（用于pandas操作）
    preds = net(test_features).detach().numpy()

    # 将预测结果存入test_data的SalePrice列
    # pd.Series(preds.reshape(1, -1)[0]):
    #   preds的形状是(1459, 1)，reshape(1, -1)变成(1, 1459)
    #   [0]取出第一行，得到一个长度为1459的一维数组
    #   创建为pandas的Series对象（一维带标签数组）
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])

    # 构建提交文件：只需要Id列和预测的SalePrice列
    # pd.concat([DataFrame, DataFrame], axis=1): axis=1表示横向拼接（增加列）
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    # .to_csv(): 导出为CSV文件
    #   index=False: 不保存行索引（Kaggle提交格式要求不包含行号）
    submission.to_csv('submission.csv', index=False)
    print('预测结果已保存到 submission.csv')


# 使用交叉验证中确定的超参数，在全量数据上训练并生成提交文件
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)


# ============================================================================
# ============================================================================
# 补充知识点
# ============================================================================
# ============================================================================

# ============================================================================
# 补充知识点1：数据预处理最佳实践
# ============================================================================
"""
在真实世界的数据科学项目中，数据预处理通常占据80%以上的时间。
以下是一些关键的最佳实践：

1. 【探索性数据分析（EDA）先行】
   - 在写预处理代码之前，先用 .describe(), .info(), .isna().sum() 了解数据概况
   - 画直方图查看数值特征的分布（是否有偏态？是否有异常值？）
   - 画柱状图查看类别特征的分布（某些类别是否太少？）

2. 【缺失值处理的多种策略】
   本例用了"用均值填充"（通过fillna(0)利用标准化后的零均值实现）。
   其他常见策略：
   - 中位数填充：当数据有严重偏态或异常值时，中位数比均值更鲁棒
   - 众数填充：适用于类别特征
   - 前向/后向填充：适用于时间序列数据
   - 模型预测填充：用其他特征训练一个小模型来预测缺失值（KNN、随机森林等）
   - 创建"缺失指示器"：额外添加一列标记该值是否缺失（pandas中dummy_na=True实现）

3. 【标准化 vs 归一化】
   - 标准化 (Standardization): z = (x - μ) / σ，结果服从均值为0、标准差为1的分布
     适用于：数据大致服从正态分布，或使用梯度下降优化的模型
   - 归一化 (Normalization/MinMax): x' = (x - min) / (max - min)，结果在[0,1]之间
     适用于：数据有严格的上下界，或基于距离的方法（如KNN、K-Means）
   - 选择依据：如果不确定，标准化通常是更安全的选择

4. 【特征工程要点】
   - 对于房屋面积、价格等长尾分布的特征，取对数可能有助于模型学习
   - 日期特征（如建造年份）可以转换为"房龄"（当前年份-建造年份）
   - 可以组合现有特征创造新特征（如：总面积 = 一层面积 + 二层面积 + 地下室面积）
   - 注意：特征工程需要在交叉验证之前完成，避免数据泄露

5. 【训练/验证/测试集的划分陷阱】
   - 预处理必须在划分数据之后分别执行（或如本例，用全量数据统计量再分别应用）
   - 绝对不能"偷看"测试集的信息（如用测试集的均值来标准化训练集）
   - 交叉验证中的标准化应该在每一折的训练集上重新计算统计量（更严格的做法）
"""

# ============================================================================
# 补充知识点2：Adam优化器简介
# ============================================================================
"""
Adam (Adaptive Moment Estimation) 是当前深度学习中最流行的优化算法之一。

核心思想：
  Adam = Momentum + RMSProp 的结合
  - Momentum: 用梯度的指数加权移动平均（一阶矩）来平滑更新方向，加速收敛、减少震荡
  - RMSProp: 用梯度平方的指数加权移动平均（二阶矩）为每个参数自适应调整学习率
    梯度大的参数给小的学习率，梯度小的参数给大的学习率

更新公式（简化）：
  m_t = β₁ * m_{t-1} + (1-β₁) * g_t        # 一阶矩估计（动量项）
  v_t = β₂ * v_{t-1} + (1-β₂) * g_t²       # 二阶矩估计（缩放项）
  θ_t = θ_{t-1} - lr * m_t / (√v_t + ε)    # 参数更新

默认超参数（通常无需调整）：
  lr=0.001:  学习率（本例中用lr=5，因为数据量和模型都很小，可以用较大学习率）
  β₁=0.9:    一阶矩衰减系数
  β₂=0.999:  二阶矩衰减系数
  ε=1e-8:    防止除零的小常数

为什么Adam受欢迎？
  (1) 对初始学习率不敏感 —— 即使不精细调整，也能得到不错的结果
  (2) 每个参数有独立的学习率 —— 适合稀疏特征（梯度更新频率不同的参数）
  (3) 在实践中通常优于SGD（尤其是训练初期收敛更快）
  (4) 几乎是"开箱即用"的默认选择

与其他优化器的对比：
  - SGD + Momentum: 需要仔细调整学习率和学习率衰减策略，但调好后可能泛化更好
  - Adam: 收敛快、鲁棒性好，但有时泛化性能略逊于精心调参的SGD
  - AdamW: Adam的改进版，将权重衰减与自适应学习率解耦，通常效果更好
"""

# ============================================================================
# 补充知识点3：为什么用对数RMSE衡量房价预测？
# ============================================================================
"""
场景对比：
  房屋A：真实价12.5万美元，预测价10.0万美元 → 误差2.5万（绝对误差更小，但预测很差）
  房屋B：真实价400万美元，预测价390万美元  → 误差10万（绝对误差更大，但预测更好）

  直觉：2.5万/12.5万 = 20% 的误差  >>> 10万/400万 = 2.5% 的误差

数学推导：
  对数误差：|log(y_pred) - log(y_true)|
          = |log(y_pred / y_true)|
          = |log(1 + (y_pred-y_true)/y_true)|
          ≈ |(y_pred-y_true)/y_true|    （当相对误差较小时，log(1+x)≈x）

  因此对数RMSE近似衡量的是"相对误差"的均方根

  确切关系：
    若 |log(y_pred) - log(y_true)| < δ，则：
    e^(-δ) < y_pred / y_true < e^δ
    即预测值在真实值的 [e^(-δ), e^δ] 倍数范围内

  例如 δ=0.1: 预测值在真实值的 [0.905, 1.105] 倍之间（约 ±10%）
        δ=0.2: 预测值在真实值的 [0.819, 1.221] 倍之间（约 ±22%）

为什么不用RMSE（普通均方根误差）？
  - RMSE对大价格的房屋惩罚过重（因为绝对误差尺度大）
  - 模型会"偏向"大房价的样本，忽略低价房屋的预测精度
  - 对数RMSE让每个样本的相对贡献更均匀

为什么对小于1的值做clamp？
  torch.clamp(net(features), 1, float('inf'))
  - 负预测值或0预测值会导致 log(负数) = NaN, log(0) = -∞
  - 真实房价最小也是正数（通常 > 1），截断到1是合理的
  - 等价于假设最低预测房价为1美元
"""

# ============================================================================
# 补充知识点4：K折交叉验证原理详解
# ============================================================================
"""
经典流程（以5折为例，数据有1460条样本）：

  原始训练集 (1460条)
  ├─ 折1 (0-291)    ├─ 折2 (292-583)  ├─ 折3 (584-875)
  ├─ 折4 (876-1167)  └─ 折5 (1168-1459)

  第1轮：折1=验证集, 折2+折3+折4+折5=训练集 (1168条训练, 292条验证)
  第2轮：折2=验证集, 折1+折3+折4+折5=训练集
  ...
  第5轮：折5=验证集, 折1+折2+折3+折4=训练集

  最终得分 = (验证误差1 + 验证误差2 + ... + 验证误差5) / 5

为什么使用K折交叉验证？
  (1) 单次划分可能"运气好"或"运气差"——K折平均减少了方差
  (2) 数据量小时，只用一份验证集会浪费训练数据
  (3) 每个样本都被用作验证一次，评估更全面

K值的选择：
  - K=5 或 K=10 是最常用的选择
  - K太小（如2）：验证集太大，训练集太小，模型训练不充分
  - K太大（如N，留一法）：计算代价极高，且方差大
  - 实践中K=5或10是性价比最高的选择

分层K折 (Stratified K-Fold):
  - 当数据类别不平衡时，应使用分层采样，保证每折中的类别比例与总体一致
  - 本例是回归任务，不涉及类别不平衡，普通K折即可

注意事项：
  - 每折必须从头训练新模型（不能在不同折之间共享权重）
  - 如果要做特征工程/标准化，严格来说应在每折的训练集上重新计算统计量
  - 交叉验证用于选择超参数，最终模型应在全量训练数据上训练
"""

# ============================================================================
# 补充知识点5：PyTorch关键API速查
# ============================================================================
"""
+---------------------+-------------------------------------------------------+
| API                 | 说明                                                  |
+---------------------+-------------------------------------------------------+
| nn.Linear(d_in,d_out)| 全连接层（线性变换层）                                |
| nn.Sequential(...)  | 将多个层按顺序串联为一个模块（前一层输出→后一层输入） |
| nn.MSELoss()        | 均方误差损失函数，用于回归任务                        |
| torch.clamp(x,min,max)| 将张量x的值限制在[min,max]范围内                   |
| torch.log(x)        | 逐元素计算自然对数                                    |
| torch.cat(list,dim) | 沿指定维度拼接张量列表                                |
| tensor.detach()     | 从计算图中分离张量（不跟踪梯度）                      |
| tensor.item()       | 将单元素张量转换为Python标量                          |
| tensor.numpy()      | 将张量转换为NumPy数组（需先detach()）                 |
| tensor.reshape()    | 改变张量形状（不改变数据本身）                        |
| optimizer.zero_grad()| 清零所有参数的梯度（每批训练前必须调用）             |
| loss.backward()     | 反向传播计算梯度                                      |
| optimizer.step()    | 根据梯度更新参数                                      |
+---------------------+-------------------------------------------------------+

pandas速查：
+---------------------+-------------------------------------------------------+
| API                 | 说明                                                  |
+---------------------+-------------------------------------------------------+
| pd.read_csv(file)   | 读取CSV文件为DataFrame                                |
| pd.concat(list,axis)| 沿指定轴拼接多个DataFrame                             |
| pd.get_dummies(df)  | 将类别列自动转换为独热编码                            |
| .iloc[row, col]     | 基于整数位置的索引（类似NumPy切片）                   |
| .apply(func)        | 对每列/每行应用函数                                   |
| .fillna(value)      | 用指定值填充所有缺失值（NaN）                         |
| .dtypes             | 返回每列的数据类型                                    |
| .shape              | 返回DataFrame的(行数, 列数)                           |
| .values             | 返回底层NumPy数组表示                                 |
+---------------------+-------------------------------------------------------+
"""

# ============================================================================
# 补充知识点6：常见问题与练习
# ============================================================================
"""
Q1: 如果提交给Kaggle，这个模型的表现如何？
    A: 这个基线线性模型的对数RMSE通常在0.15~0.17左右
       这大约对应价格预测在真实值的 [e^-0.17, e^0.17] ≈ [84%, 119%] 范围内
       对于基线来说还不错，但排行榜上的顶级模型可以做到0.10以下

Q2: 能否直接最小化价格的对数来改进模型？
    A: 可以尝试两种方式：
       (a) 预测 log(SalePrice) 而非 SalePrice（即模型输出取对数后的价格）
       (b) 使用对数损失函数（nn.MSELoss on log space）
       方式(a)的问题是：最终需要exp()还原价格，而exp会对误差有放大效应
       方式(b)与本节的方法等价

Q3: 用平均值替换缺失值总是好主意吗？
    A: 不一定。考虑以下场景：
       - 如果数据不是"随机缺失"而是"有结构地缺失"
         （如：豪宅主人更不愿意填写某些信息），
         均值填充会引入系统性偏差
       - 如果缺失比例很高（>50%），均值填充几乎等同于添加噪声
       - 更好的做法：添加"是否缺失"的指示特征 + 用其他特征预测缺失值
       - 本例中缺失比例不高，且使用dummy_na=True添加了缺失指示器

Q4: 如何通过调整超参数提高Kaggle得分？
    可以尝试的改进：
    - 调整学习率：尝试 lr ∈ {0.1, 1, 3, 5, 10}
    - 添加权重衰减：尝试 weight_decay ∈ {1e-5, 1e-4, 1e-3, 1e-2}
    - 调整训练轮数：num_epochs ∈ {100, 200, 500}
    - 调整批次大小：batch_size ∈ {32, 64, 128, 256}

Q5: 如何改进模型本身？
    - 增加隐藏层：nn.Sequential(nn.Linear(331, 64), nn.ReLU(), nn.Linear(64, 1))
    - 添加Dropout：在隐藏层后加 nn.Dropout(0.5) 防止过拟合
    - 使用BatchNorm：加 nn.BatchNorm1d(64) 加速训练、稳定收敛
    - 尝试更深网络：2-3个隐藏层，每层神经元递减（如 256→128→64→1）
    - 添加残差连接（跳跃连接）

Q6: 如果不标准化连续数值特征会发生什么？
    - 梯度下降会非常不稳定：量纲大的特征梯度大，量纲小的特征梯度小
    - 收敛速度极慢：优化路径呈之字形（zig-zag），因为不同方向的梯度尺度不一致
    - 权重衰减（L2正则）失效：对量纲大的特征惩罚更多，不公平
    - 数值问题：可能导致梯度爆炸或梯度消失
    - 结论：对于使用梯度优化的模型，标准化几乎总是必需的！
"""

# ============================================================================
# 总结
# ============================================================================
"""
本脚本完整实现了Kaggle房价预测竞赛的端到端流程，涵盖了：

数据侧：
  - 数据集下载、缓存与SHA-1完整性校验
  - 混合数据类型（数值型+类别型）的预处理
  - 零均值单位方差标准化 + 缺失值填充
  - 独热编码（含缺失值指示器）
  - pandas DataFrame → PyTorch Tensor

模型侧：
  - 线性回归基线模型（nn.Linear）
  - 对数RMSE评估指标（torch.log + torch.sqrt）
  - Adam优化器 + 权重衰减正则化
  - K折交叉验证（5折）
  - 训练/验证损失曲线可视化

实践侧：
  - Kaggle提交流程（生成 submission.csv）
  - 超参数选择的方法论
  - 基线模型的思想（先跑通，再改进）

核心收获：
  > 数据预处理的每一行代码背后都有统计学和机器学习原理
  > 交叉验证是模型选择的金标准
  > 对数RMSE让模型公平对待不同价位的房屋
  > 从简单基线开始，逐步迭代改进 —— 这是数据科学的正确姿势
"""
