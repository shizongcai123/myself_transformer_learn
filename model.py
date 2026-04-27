import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ======================================================================
# 1. 位置编码 (Positional Encoding)
# ======================================================================
class PositionalEncoding(nn.Module):
    """
    【为什么需要位置编码？】
    Transformer 内部的注意力机制是"无序的"——它只关心词与词之间的相关性，
    本身感知不到"第1个词"和"第5个词"在位置上的差异。
    位置编码就是给每个位置叠加一个独特的"坐标信号"，让模型知道顺序。

    【编码方式：正弦/余弦公式】
    对于位置 pos、第 i 个维度：
        PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))

    直觉：不同维度的频率不同（从高频到低频），
    就像时钟的秒针、分针、时针一样，组合后每个位置的编码都唯一。
    """
    def __init__(self, d_model, max_seq_len=200):
        super().__init__()

        # 假设 d_model=4, max_seq_len=3，用具体数字说明每一步的形状变化

        # pe: [max_seq_len, d_model] = [3, 4]
        # 初始化全0，后面逐个填入 sin/cos 值
        pe = torch.zeros(max_seq_len, d_model)

        # position: [max_seq_len] -> unsqueeze(1) -> [max_seq_len, 1] = [3, 1]
        # 内容: [[0],
        #        [1],
        #        [2]]
        # 意义：每一行代表一个位置的序号
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()

        # div_term: [d_model/2] = [2]
        # arange(0, d_model, 2) = [0, 2]（取偶数索引）
        # 计算: exp([0, 2] * (-ln(10000) / 4))
        #     = exp([0, -ln(100)])
        #     = [1.0, 0.01]
        # 意义：频率衰减因子，第0对维度频率高（变化快），后面维度频率低（变化慢）
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # position * div_term:
        #   [3, 1] * [2] -> 广播 -> [3, 2]
        #   = [[0*1.0,  0*0.01],     pos=0
        #      [1*1.0,  1*0.01],     pos=1
        #      [2*1.0,  2*0.01]]     pos=2
        #   = [[0.0,    0.0 ],
        #      [1.0,    0.01],
        #      [2.0,    0.02]]

        # 偶数维度(0,2)填 sin，奇数维度(1,3)填 cos
        # pe[:, 0::2] 取第0,2列（偶数列）：形状 [max_seq_len, d_model/2] = [3, 2]
        # pe[:, 1::2] 取第1,3列（奇数列）：形状 [max_seq_len, d_model/2] = [3, 2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 填完后的 pe: [3, 4]
        # = [[sin(0.0),  cos(0.0),  sin(0.0),  cos(0.0) ],   pos=0
        #    [sin(1.0),  cos(1.0),  sin(0.01), cos(0.01)],   pos=1
        #    [sin(2.0),  cos(2.0),  sin(0.02), cos(0.02)]]   pos=2
        # = [[0.000,     1.000,     0.000,     1.000    ],
        #    [0.841,     0.540,     0.010,     0.999    ],
        #    [0.909,    -0.416,     0.020,     0.999    ]]
        #
        # 观察规律：
        #   前两列（高频）：值变化很快，pos=0,1,2 差异明显
        #   后两列（低频）：值变化很慢，pos=0,1,2 几乎一样
        #   每一行的数值组合都是唯一的 → 每个位置有独特的"指纹"

        # 注册为 buffer：不是模型参数（不参与梯度更新），但跟随模型保存/加载
        # unsqueeze(0): [max_seq_len, d_model] -> [1, max_seq_len, d_model]
        # 加 batch=1 维度，方便后续和 [batch, seq_len, d_model] 广播相加
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x 输入形状: [batch, seq_len, d_model]
        #   例如 [2, 3, 4] = 2个样本，每个3个token，每个token 4维向量
        #
        # self.pe: [1, max_seq_len, d_model] = [1, 200, 4]
        # self.pe[:, :seq_len, :]: 切片取前 seq_len 个位置 -> [1, 3, 4]
        #
        # 广播相加: [2, 3, 4] + [1, 3, 4] -> [2, 3, 4]
        #   batch 维度广播：同一批次内每个样本加的位置编码相同
        #   这是合理的：第0个位置无论在哪个样本里，位置编码都一样
        return x + self.pe[:, :x.size(1), :]


# ======================================================================
# 2. 多头注意力 (Multi-Head Attention)
# ======================================================================
class MultiHeadAttention(nn.Module):
    """
    【注意力机制的直觉】
    把"查询"比作在图书馆找书：
    - Query (Q)：你的查询需求，"我想找 transformer 相关的书"
    - Key   (K)：每本书的标签，"这本书的标签是 深度学习、NLP"
    - Value (V)：书的实际内容

    步骤：拿 Q 和所有 K 做匹配打分 → softmax 归一化 → 用分数对 V 加权求和
    结果：根据你的需求，把最相关的内容聚合出来

    【多头的意义】
    用多个不同的"视角"（head）同时做注意力，
    例如：一个头关注语法关系，另一个头关注语义相似性。
    最终把所有头的结果拼接，综合多种信息。
    """
    def __init__(self, d_model=128, n_head=4):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"

        self.d_model = d_model
        self.n_head  = n_head
        self.d_k     = d_model // n_head  # 每个头的维度，例如 128/4=32

        # 四个线性投影层，权重矩阵均为 [d_model, d_model]
        # 没有 bias，遵循原论文设计
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # 多头拼接后的输出投影, 做一次 [128, 128] 的线性变换，让 4 个头的信息混合融合，输出一个真正整合了多头信息的向量

    def forward(self, q, k, v, mask=None):
        """
        参数：
            q: [batch, seq_q, d_model]  Query 序列
            k: [batch, seq_k, d_model]  Key 序列
            v: [batch, seq_k, d_model]  Value 序列（和 Key 来自同一序列）
            mask: [batch, 1, seq_q, seq_k] 或 None

        注意：
            自注意力时：q=k=v=同一个序列
            交叉注意力时：q 来自解码器，k/v 来自编码器输出
        """
        batch = q.size(0)
        seq_q = q.size(1)
        seq_k = k.size(1)

        # ------ Step 1: 线性投影 ------
        # 每个输入 [batch, seq, d_model] 乘以各自的权重矩阵 [d_model, d_model]
        # 输出形状不变: [batch, seq, d_model]
        Q = self.W_q(q)  # [batch, seq_q, d_model]
        K = self.W_k(k)  # [batch, seq_k, d_model]
        V = self.W_v(v)  # [batch, seq_k, d_model]

        # ------ Step 2: 切分多头 ------
        # 把最后的 d_model 维拆成 [n_head, d_k]，再把 head 维提到前面
        # [batch, seq, d_model]
        #   -> view  -> [batch, seq, n_head, d_k]
        #   -> transpose -> [batch, n_head, seq, d_k]
        Q = Q.view(batch, seq_q, self.n_head, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_k, self.n_head, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_k, self.n_head, self.d_k).transpose(1, 2)
        # 此时: Q=[batch, n_head, seq_q, d_k]  K=V=[batch, n_head, seq_k, d_k]

        # ------ Step 3: 计算注意力分数 ------
        # Q @ K^T:
        #   [batch, n_head, seq_q, d_k] @ [batch, n_head, d_k, seq_k]
        #   -> [batch, n_head, seq_q, seq_k]  虽然前面是按照头去切了d_k，但是当计算了QK后，这个score的最后两维seq_q, seq_k，依旧是整个句子token的长度。意思每个头都在提取整个句子的特征，只是提取的信息不同。
        # scores[b, h, i, j] = 第 b 个样本，第 h 个头，位置 i 对位置 j 的相关性
        #
        # 除以 sqrt(d_k)：防止点积值过大导致 softmax 梯度消失
        # （维度越高点积越大，需要缩放）
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: [batch, n_head, seq_q, seq_k]

        # ------ Step 4: 应用 Mask ------
        if mask is not None:
            # mask=0 的位置（PAD 或未来 token）填充极小值.  Causal Mask只有decoder需要  tgt_mask: [batch, 1, tgt_len, tgt_len]，这里就能将未来的token遮掩(即设置为非常小的值)，一个倒三角的样子。
            # softmax 后这些位置的权重趋近于 0，即被"忽略"
            scores = scores.masked_fill(mask == 0, -1e9) 

        # ------ Step 5: Softmax 归一化 ------
        # 在 seq_k 维度做归一化，每个 Query 位置的注意力权重之和为 1
        attn_weights = F.softmax(scores, dim=-1)  # [batch, n_head, seq_q, seq_k]

        # ------ Step 6: 加权求和 ------
        # attn_weights @ V:
        #   [batch, n_head, seq_q, seq_k] @ [batch, n_head, seq_k, d_k]
        #   -> [batch, n_head, seq_q, d_k]
        # 每个 Query 位置的输出 = 对所有 Value 的加权平均
        output = torch.matmul(attn_weights, V)  # [batch, n_head, seq_q, d_k]

        # ------ Step 7: 拼接多头 ------
        # 把 n_head 和 d_k 合并回 d_model
        # [batch, n_head, seq_q, d_k]
        #   -> transpose -> [batch, seq_q, n_head, d_k]
        #   -> contiguous().view -> [batch, seq_q, d_model]
        output = output.transpose(1, 2).contiguous().view(batch, seq_q, self.d_model) #这里的 dmodel = 128 ,只是顺序排布，但是没有融合[ head0的32维 | head1的32维 | head2的32维 | head3的32维 ]

        # ------ Step 8: 输出投影 ------
        # [batch, seq_q, d_model] @ [d_model, d_model] -> [batch, seq_q, d_model] 让 4 个头的信息混合融合，输出一个真正整合了多头信息的向量。
        return self.W_o(output)

    def forward_cached(self, q, cache_k, cache_v, update_cache=True):
        """
        【带 KV Cache 的注意力计算】推理专用，每次只处理 1 个新 token。

        普通 forward 每步处理完整序列（长度随步数增长，重复计算历史 K/V），
        forward_cached 每步只处理 1 个新 token 的 Q，
        历史 token 的 K/V 直接从 cache 里读取，不重复计算。

        参数：
            q:            [batch, 1, d_model]              新 token 的输入
            cache_k:      [batch, n_head, seq_so_far, d_k] 历史 K
            cache_v:      [batch, n_head, seq_so_far, d_k] 历史 V
            update_cache: 是否把新 token 的 K/V 追加进 cache
                          - 自注意力：True（每步产生新的 K/V 需要存入 cache）
                          - 交叉注意力：False（K/V 来自固定的 memory，一次算好就不变了）

        返回：
            output:               [batch, 1, d_model]
            new_cache_k/cache_v:  更新后的 cache（或原 cache）
        """
        batch = q.size(0)

        # 只计算新 token 的 Q
        # [batch, 1, d_model] -> [batch, n_head, 1, d_k]
        Q = self.W_q(q).view(batch, 1, self.n_head, self.d_k).transpose(1, 2)

        if update_cache:
            # 自注意力：计算新 token 的 K/V，追加到 cache
            # [batch, 1, d_model] -> [batch, n_head, 1, d_k]
            K_new = self.W_k(q).view(batch, 1, self.n_head, self.d_k).transpose(1, 2)
            V_new = self.W_v(q).view(batch, 1, self.n_head, self.d_k).transpose(1, 2)

            # 拼接到 cache：[batch, n_head, seq_so_far, d_k]
            #            -> [batch, n_head, seq_so_far+1, d_k]
            K = torch.cat([cache_k, K_new], dim=2)
            V = torch.cat([cache_v, V_new], dim=2)
            new_cache_k, new_cache_v = K, V
        else:
            # 交叉注意力：直接用预计算好的 K/V，不更新 cache
            K = cache_k   # [batch, n_head, src_len, d_k]
            V = cache_v
            new_cache_k, new_cache_v = cache_k, cache_v

        # 注意力分数
        # Q: [batch, n_head, 1, d_k]
        # K: [batch, n_head, seq_so_far+1 或 src_len, d_k]
        # scores: [batch, n_head, 1, seq_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 推理时不需要 mask：
        #   自注意力：cache 里只有历史 token，不存在"未来"，因果性天然满足
        #   交叉注意力：可以看全部 src，不需要屏蔽
        attn_weights = F.softmax(scores, dim=-1)  # [batch, n_head, 1, seq_k]

        # 加权求和：[batch, n_head, 1, d_k]
        output = torch.matmul(attn_weights, V)

        # 拼接多头 + 输出投影：[batch, 1, d_model]
        output = output.transpose(1, 2).contiguous().view(batch, 1, self.d_model)
        return self.W_o(output), new_cache_k, new_cache_v


# ======================================================================
# 2.5 分组多头注意力 (Grouped Query Attention, GQA)
# ======================================================================
class GroupedQueryAttention(nn.Module):
    """
    【GQA 与标准 MHA 的区别】

    标准 MHA（Multi-Head Attention）：
        Q 有 8 个头，K 有 8 个头，V 有 8 个头
        每个 Q 头和自己对应的 K/V 头配对，1 对 1

    GQA（Grouped Query Attention）：
        Q 有 8 个头，K 只有 4 个头，V 只有 4 个头
        每 2 个 Q 头共享同一组 K/V 头，多 对 1

    【为什么要这样做？】
    推理时 KV Cache 是显存瓶颈：
        标准 MHA：cache 存 8 组 K/V → 显存大
        GQA：     cache 只存 4 组 K/V → 显存省一半！

    效果：接近 MHA 的质量，但 KV Cache 更小，推理更快。
    LLaMA 2/3、Mistral、Gemma 等主流大模型都用 GQA。

    【特殊情况】
        n_kv_head = n_head  → 退化为标准 MHA（每个 Q 头独享一组 KV）
        n_kv_head = 1       → 退化为 MQA（Multi-Query Attention，所有 Q 头共享一组 KV）

    【参数量对比】（d_model=128, n_head=8, n_kv_head=4, d_k=16）
        标准 MHA:
            W_q: [128, 128]   W_k: [128, 128]   W_v: [128, 128]   共 49152 参数
        GQA:
            W_q: [128, 128]   W_k: [128, 64]    W_v: [128, 64]    共 40960 参数
                                     ↑ 只投影到 n_kv_head * d_k = 4*16 = 64 维
    """
    def __init__(self, d_model=128, n_head=8, n_kv_head=4):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"
        assert n_head % n_kv_head == 0, "n_head 必须能被 n_kv_head 整除"

        self.d_model   = d_model
        self.n_head    = n_head       # Q 的头数（完整）
        self.n_kv_head = n_kv_head    # K/V 的头数（更少）
        self.d_k       = d_model // n_head   # 每个头的维度，例如 128/8=16
        self.n_rep     = n_head // n_kv_head # 每组 KV 被多少个 Q 头共享，例如 8/4=2

        # Q 投影：输出维度 = n_head * d_k = 8*16 = 128（和标准 MHA 一样）
        self.W_q = nn.Linear(d_model, n_head * self.d_k, bias=False)
        # K/V 投影：输出维度 = n_kv_head * d_k = 4*16 = 64（比标准 MHA 小！）
        self.W_k = nn.Linear(d_model, n_kv_head * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_head * self.d_k, bias=False)
        # 输出投影：输入是 n_head 个 Q 头拼接的结果，维度 = n_head * d_k = d_model
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def _repeat_kv(self, x):
        """
        把 KV 头"复制"成和 Q 头一样多，让矩阵乘法对齐。

        输入 x: [batch, n_kv_head, seq, d_k]   例如 [2, 4, 10, 16]
        输出:   [batch, n_head,    seq, d_k]   例如 [2, 8, 10, 16]

        过程（n_kv_head=4, n_rep=2）：
            原始 4 组 KV:    [KV0, KV1, KV2, KV3]
            每组复制 2 次:   [KV0, KV0, KV1, KV1, KV2, KV2, KV3, KV3]
            结果 8 组，正好和 8 个 Q 头一一对应

        实现细节:
            [batch, 4, seq, 16]
              ↓ unsqueeze(2)   在头维度和 seq 之间插入一维
            [batch, 4, 1, seq, 16]
              ↓ expand          沿新维度复制 n_rep=2 次（不分配新内存，只是视图）
            [batch, 4, 2, seq, 16]
              ↓ reshape         合并前两个头维度 4*2=8
            [batch, 8, seq, 16]

        注意：expand 不真的复制数据，只创建一个"虚拟视图"，
        所以显存开销几乎为零。这是 GQA 高效的关键！
        """
        if self.n_rep == 1:
            # n_kv_head == n_head，已经是标准 MHA，不需要复制
            return x
        batch, n_kv_head, seq, d_k = x.shape
        x = x.unsqueeze(2)                                  # [batch, n_kv_head, 1, seq, d_k]
        x = x.expand(batch, n_kv_head, self.n_rep, seq, d_k)  # [batch, n_kv_head, n_rep, seq, d_k]
        x = x.reshape(batch, self.n_head, seq, d_k)            # [batch, n_head, seq, d_k]
        return x

    def forward(self, q, k, v, mask=None):
        """
        GQA 前向传播

        参数：
            q: [batch, seq_q, d_model]
            k: [batch, seq_k, d_model]
            v: [batch, seq_k, d_model]
            mask: [batch, 1, seq_q, seq_k] 或 None

        【与标准 MHA 的数据流对比】（d_model=128, n_head=8, n_kv_head=4, d_k=16）

          标准 MHA:                           GQA:
          ─────────                           ────
          Q: [batch,seq,128]                  Q: [batch,seq,128]       ← 相同
             ↓ W_q [128,128]                     ↓ W_q [128,128]
             [batch,seq,128]                     [batch,seq,128]
             ↓ view+transpose                    ↓ view+transpose
             [batch, 8, seq, 16]                 [batch, 8, seq, 16]   ← Q 头数相同

          K: [batch,seq,128]                  K: [batch,seq,128]
             ↓ W_k [128,128]                     ↓ W_k [128,64]       ← 投影更小！
             [batch,seq,128]                     [batch,seq,64]
             ↓ view+transpose                    ↓ view+transpose
             [batch, 8, seq, 16]                 [batch, 4, seq, 16]  ← 只有 4 组
                                                 ↓ _repeat_kv         ← 复制扩展
                                                 [batch, 8, seq, 16]  ← 对齐到 8 组

          后续 scores → softmax → output 完全一致
        """
        batch = q.size(0)
        seq_q = q.size(1)
        seq_k = k.size(1)

        # ------ Step 1: 线性投影 ------
        # Q 投影到 n_head * d_k = 128 维
        Q = self.W_q(q)   # [batch, seq_q, n_head * d_k] = [batch, seq_q, 128]
        # K/V 投影到 n_kv_head * d_k = 64 维（比 Q 少！）
        K = self.W_k(k)   # [batch, seq_k, n_kv_head * d_k] = [batch, seq_k, 64]
        V = self.W_v(v)   # [batch, seq_k, n_kv_head * d_k] = [batch, seq_k, 64]

        # ------ Step 2: 切分多头 ------
        # Q: [batch, seq_q, 128] -> [batch, 8, seq_q, 16]   （8 个 Q 头）
        Q = Q.view(batch, seq_q, self.n_head, self.d_k).transpose(1, 2)
        # K/V: [batch, seq_k, 64] -> [batch, 4, seq_k, 16]  （4 个 KV 头）
        K = K.view(batch, seq_k, self.n_kv_head, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_k, self.n_kv_head, self.d_k).transpose(1, 2)

        # ------ Step 3: 复制 KV 头，对齐到 Q 头数量 ------
        # [batch, 4, seq_k, 16] -> [batch, 8, seq_k, 16]
        # Q头 0,1 共享 KV头 0；Q头 2,3 共享 KV头 1；...
        K = self._repeat_kv(K)
        V = self._repeat_kv(V)

        # ------ Step 4: 计算注意力（和标准 MHA 完全相同）------
        # Q: [batch, 8, seq_q, 16]
        # K^T: [batch, 8, 16, seq_k]
        # scores: [batch, 8, seq_q, seq_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)  # [batch, 8, seq_q, seq_k]

        # [batch, 8, seq_q, seq_k] @ [batch, 8, seq_k, 16] -> [batch, 8, seq_q, 16]
        output = torch.matmul(attn_weights, V)

        # ------ Step 5: 拼接 + 输出投影 ------
        # [batch, 8, seq_q, 16] -> [batch, seq_q, 128]
        output = output.transpose(1, 2).contiguous().view(batch, seq_q, self.d_model)
        return self.W_o(output)  # [batch, seq_q, d_model]

    def forward_cached(self, q, cache_k, cache_v, update_cache=True):
        """
        【带 KV Cache 的 GQA】推理专用，每次只处理 1 个新 token。

        【和标准 MHA cache 的关键区别】
        标准 MHA cache:
            cache_k: [batch, n_head,    seq_so_far, d_k] = [batch, 8, seq, 16]
            显存: 8 * seq * 16 = 128 * seq

        GQA cache:
            cache_k: [batch, n_kv_head, seq_so_far, d_k] = [batch, 4, seq, 16]
            显存: 4 * seq * 16 = 64 * seq   ← 省了一半！

        Cache 始终以 n_kv_head 的形式存储（紧凑），
        计算注意力时才 _repeat_kv 扩展到 n_head（不占额外显存）。

        参数：
            q:            [batch, 1, d_model]                   新 token
            cache_k:      [batch, n_kv_head, seq_so_far, d_k]   历史 K（紧凑格式）
            cache_v:      [batch, n_kv_head, seq_so_far, d_k]   历史 V（紧凑格式）
            update_cache: True=自注意力（追加新KV），False=交叉注意力（KV不变）

        返回：
            output:      [batch, 1, d_model]
            new_cache_k: [batch, n_kv_head, seq_so_far+1, d_k]  更新后的 K cache
            new_cache_v: [batch, n_kv_head, seq_so_far+1, d_k]  更新后的 V cache
        """
        batch = q.size(0)

        # Q 投影：新 token → n_head 个 Q 头
        # [batch, 1, d_model] -> [batch, n_head, 1, d_k] = [batch, 8, 1, 16]
        Q = self.W_q(q).view(batch, 1, self.n_head, self.d_k).transpose(1, 2)

        if update_cache:
            # 自注意力：计算新 token 的 K/V（只有 n_kv_head 个头！）
            # [batch, 1, d_model] -> [batch, n_kv_head, 1, d_k] = [batch, 4, 1, 16]
            K_new = self.W_k(q).view(batch, 1, self.n_kv_head, self.d_k).transpose(1, 2)
            V_new = self.W_v(q).view(batch, 1, self.n_kv_head, self.d_k).transpose(1, 2)

            # 追加到 cache（保持 n_kv_head 格式，不扩展）
            # [batch, 4, seq_so_far, 16] cat [batch, 4, 1, 16]
            # -> [batch, 4, seq_so_far+1, 16]
            K = torch.cat([cache_k, K_new], dim=2)
            V = torch.cat([cache_v, V_new], dim=2)
            new_cache_k, new_cache_v = K, V
        else:
            # 交叉注意力：直接用预算好的 K/V
            K = cache_k
            V = cache_v
            new_cache_k, new_cache_v = cache_k, cache_v

        # 计算注意力前，把 KV 头扩展到和 Q 头对齐
        # [batch, 4, seq, 16] -> [batch, 8, seq, 16]
        K_expanded = self._repeat_kv(K)
        V_expanded = self._repeat_kv(V)

        # Q: [batch, 8, 1, 16] @ K^T: [batch, 8, 16, seq] -> scores: [batch, 8, 1, seq]
        scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)

        # [batch, 8, 1, seq] @ [batch, 8, seq, 16] -> [batch, 8, 1, 16]
        output = torch.matmul(attn_weights, V_expanded)

        # 拼接 + 输出投影: [batch, 8, 1, 16] -> [batch, 1, 128] -> [batch, 1, d_model]
        output = output.transpose(1, 2).contiguous().view(batch, 1, self.d_model)

        # 返回的 cache 保持 n_kv_head 格式（紧凑！不是扩展后的 n_head 格式）
        return self.W_o(output), new_cache_k, new_cache_v


# ======================================================================
# 3. 前馈网络 (Feed-Forward Network)
# ======================================================================
class FeedForward(nn.Module):
    """
    【FFN 的作用】
    注意力层负责"跨位置的信息聚合"（不同 token 互相交换信息）。
    FFN 层负责"对每个位置单独做非线性变换"（提炼和变换特征）。

    结构：先升维（d_model -> d_ff）再降维（d_ff -> d_model）
    中间用 ReLU 引入非线性。
    原论文 d_ff = 4 * d_model（本实现可自定义）。
    """
    def __init__(self, d_model=128, d_ff=256):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)   # 升维，权重: [d_model, d_ff]
        self.linear2 = nn.Linear(d_ff, d_model)   # 降维，权重: [d_ff, d_model]

    def forward(self, x):
        # x: [batch, seq, d_model]
        # -> linear1 -> [batch, seq, d_ff]
        # -> relu    -> [batch, seq, d_ff]   （激活，让网络有非线性表达能力）
        # -> linear2 -> [batch, seq, d_model]
        return self.linear2(F.relu(self.linear1(x)))


# ======================================================================
# 4. 编码器层 (Encoder Layer)
# ======================================================================
class EncoderLayer(nn.Module):
    """
    【编码器层结构】

        输入 x: [batch, src_len, d_model]
            ↓
        [Self-Attention]  ← x 中每个位置都可以关注 x 中所有其他位置
            ↓
        [Add & LayerNorm] ← 残差连接：把注意力的"增量"加回原始 x，防止梯度消失
            ↓
        [Feed Forward]    ← 每个位置独立做非线性变换
            ↓
        [Add & LayerNorm]
            ↓
        输出: [batch, src_len, d_model]（形状与输入相同）

    【残差连接的意义】
    output = LayerNorm(x + sublayer(x))
    即使 sublayer 学得不好，至少能保留原始 x 的信息（恒等映射）。
    """
    def __init__(self, d_model=128, n_head=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.ffn       = FeedForward(d_model, d_ff)
        self.norm1     = nn.LayerNorm(d_model)  # 参数: gamma, beta，形状均为 [d_model]
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        # x: [batch, src_len, d_model]

        # 自注意力：q=k=v=x，每个位置都能"看"全部位置
        attn_out = self.self_attn(q=x, k=x, v=x, mask=src_mask)  # [batch, seq_q, d_model] 
        # 残差 + 归一化（先加后归一化，原论文风格）
        x = self.norm1(x + self.dropout(attn_out))  # [batch, src_len, d_model]

        # 前馈网络
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))   # [batch, src_len, d_model]

        return x


# ======================================================================
# 5. 解码器层 (Decoder Layer)
# ======================================================================
class DecoderLayer(nn.Module):
    """
    【解码器层结构】
    比编码器多一个 Cross-Attention，用来"查询"编码器的输出：

        输入 x: [batch, tgt_len, d_model]    （目标序列当前状态）
        memory: [batch, src_len, d_model]    （编码器输出，固定不变）
            ↓
        [Masked Self-Attention]  ← tgt_mask 保证位置 i 只能看 0..i（不能看未来）
            ↓
        [Add & LayerNorm]
            ↓
        [Cross-Attention]        ← Q 来自解码器，K/V 来自编码器 memory
                                   "用当前翻译状态，去查询源语言的信息"
            ↓
        [Add & LayerNorm]
            ↓
        [Feed Forward]
            ↓
        [Add & LayerNorm]
            ↓
        输出: [batch, tgt_len, d_model]

    【Cross-Attention 的核心】
    Q = 解码器当前输出（"我现在翻译到哪了，需要什么信息"）
    K = V = 编码器输出（"源语言的全部信息"）
    通过 Q 和 K 的匹配，决定从源语言的哪些位置提取 V 的内容。
    """
    def __init__(self, d_model=128, n_head=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_head)  # Masked 自注意力
        self.cross_attn = MultiHeadAttention(d_model, n_head)  # 交叉注意力
        self.ffn        = FeedForward(d_model, d_ff)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        x:        [batch, tgt_len, d_model]  解码器输入（目标序列）
        memory:   [batch, src_len, d_model]  编码器输出（源序列的语义表示）
        src_mask: 屏蔽源序列中 PAD token 的 mask
        tgt_mask: 因果 mask，屏蔽 PAD 且防止看未来
        """
        # 1. Masked Self-Attention
        # tgt_mask 保证：位置 i 的 Query 只能与位置 0..i 的 Key 交互
        attn_out = self.self_attn(q=x, k=x, v=x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))  # [batch, tgt_len, d_model]

        # 2. Cross-Attention
        # q=x（解码器当前状态），k=v=memory（编码器输出）
        # 形状说明：
        #   q: [batch, tgt_len, d_model]
        #   k: [batch, src_len, d_model]
        #   输出由 Q 的 seq_len 决定: [batch, tgt_len, d_model]
        cross_out = self.cross_attn(q=x, k=memory, v=memory, mask=src_mask)
        x = self.norm2(x + self.dropout(cross_out))  # [batch, tgt_len, d_model]

        # 3. FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))    # [batch, tgt_len, d_model]

        return x

    def forward_cached(self, x, layer_cache):
        """
        【带 KV Cache 的解码器层】推理专用，每次只处理 1 个新 token。

        layer_cache 结构：
        {
            'self_k':  [batch, n_head, seq_so_far, d_k]  自注意力历史 K（每步追加）
            'self_v':  [batch, n_head, seq_so_far, d_k]  自注意力历史 V（每步追加）
            'cross_k': [batch, n_head, src_len,    d_k]  交叉注意力 K（固定）
            'cross_v': [batch, n_head, src_len,    d_k]  交叉注意力 V（固定）
        }

        x:           [batch, 1, d_model]  新 token 的嵌入
        返回:         output [batch, 1, d_model], updated layer_cache
        """
        # 1. Masked Self-Attention（update_cache=True，新 token 的 K/V 追加进去）
        attn_out, new_self_k, new_self_v = self.self_attn.forward_cached(
            q=x,
            cache_k=layer_cache['self_k'],
            cache_v=layer_cache['self_v'],
            update_cache=True,
        )
        x = self.norm1(x + attn_out)  # [batch, 1, d_model]

        # 2. Cross-Attention（update_cache=False，直接用 prefill 时算好的 K/V）
        cross_out, _, _ = self.cross_attn.forward_cached(
            q=x,
            cache_k=layer_cache['cross_k'],
            cache_v=layer_cache['cross_v'],
            update_cache=False,
        )
        x = self.norm2(x + cross_out)  # [batch, 1, d_model]

        # 3. FFN（每个位置独立处理，不涉及 cache）
        x = self.norm3(x + self.ffn(x))  # [batch, 1, d_model]

        # 返回新 token 的输出，以及更新后的 cache（self_k/v 增长了1步）
        new_cache = {
            'self_k':  new_self_k,          # [batch, n_head, seq_so_far+1, d_k]
            'self_v':  new_self_v,
            'cross_k': layer_cache['cross_k'],  # 不变
            'cross_v': layer_cache['cross_v'],  # 不变
        }
        return x, new_cache


# ======================================================================
# 6. 完整 Transformer 模型
# ======================================================================
class Transformer(nn.Module):
    """
    【完整数据流】
    训练时（Teacher Forcing）：

    src: [batch, src_len]  ──→  Embedding + PosEnc  ──→  [batch, src_len, d_model]
                                        ↓
                              N × EncoderLayer
                                        ↓
                            memory: [batch, src_len, d_model]
                                        ↓ (传入每个 DecoderLayer)

    tgt: [batch, tgt_len]  ──→  Embedding + PosEnc  ──→  [batch, tgt_len, d_model]
                                        ↓
                              N × DecoderLayer（接收 memory）
                                        ↓
                            [batch, tgt_len, d_model]
                                        ↓
                              Linear: [d_model → vocab_size]
                                        ↓
                            logits: [batch, tgt_len, vocab_size]
    """
    def __init__(self, vocab_size, d_model=128, n_head=4, num_layers=2,
                 d_ff=256, dropout=0.1, max_seq_len=200):
        super().__init__()
        self.d_model = d_model

        # 词嵌入查找表: [vocab_size, d_model]
        # padding_idx=0：PAD token 的嵌入向量固定为 0，不参与梯度更新
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)

        # 位置编码（不是参数，是固定的 sin/cos 值）
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # 编码器：num_layers 个 EncoderLayer 堆叠
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout) for _ in range(num_layers)
        ])

        # 解码器：num_layers 个 DecoderLayer 堆叠
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ff, dropout) for _ in range(num_layers)
        ])

        # 输出分类头: [d_model, vocab_size]
        # 把每个位置的 d_model 维特征映射成 vocab_size 个 logit 分数
        self.fc_out = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def encode(self, src, src_mask):
        """
        编码阶段（推理时只执行一次）

        src:      [batch, src_len]         整数 token id
        src_mask: [batch, 1, 1, src_len]   PAD 位置为 0
        返回:     [batch, src_len, d_model]
        """
        # 词嵌入 + 缩放（乘 sqrt(d_model) 让嵌入幅度与位置编码匹配）
        # [batch, src_len] -> [batch, src_len, d_model]
        x = self.embedding(src) * math.sqrt(self.d_model)
        # 叠加位置编码
        x = self.dropout(self.pos_encoding(x))  # [batch, src_len, d_model]

        # 依次通过每个编码器层，形状始终是 [batch, src_len, d_model]
        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x  # memory: [batch, src_len, d_model]

    def decode(self, tgt, memory, src_mask, tgt_mask):
        """
        解码阶段

        tgt:      [batch, tgt_len]          整数 token id
        memory:   [batch, src_len, d_model] 编码器输出（固定）
        返回:     [batch, tgt_len, d_model]
        """
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.dropout(self.pos_encoding(x))  # [batch, tgt_len, d_model]

        # 依次通过每个解码器层，形状始终是 [batch, tgt_len, d_model]
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return x  # [batch, tgt_len, d_model]




    def forward(self, src, tgt, src_mask=None, tgt_mask=None):  
        """
        完整前向传播（训练时使用 Teacher Forcing）

        src:    [batch, src_len]
        tgt:    [batch, tgt_len]   训练时传入完整的目标序列（不含最后一个 EOS）
        src_mask: [batch, 1, 1, src_len]  屏蔽 src 中的 PAD
        tgt_mask: [batch, 1, tgt_len, tgt_len]  因果 mask + 屏蔽 tgt PAD.一个倒三角的矩阵
        返回:   [batch, tgt_len, vocab_size]
        """
        # 编码：处理源序列，得到全局语义表示
        memory = self.encode(src, src_mask)          # [batch, src_len, d_model]

        # 解码：结合 memory 和目标序列，预测每个位置的下一个 token
        dec_out = self.decode(tgt, memory, src_mask, tgt_mask)  # [batch, tgt_len, d_model]

        # 输出层：映射到词表空间
        # [batch, tgt_len, d_model] -> [batch, tgt_len, vocab_size]
        logits = self.fc_out(dec_out)

        return logits

    # ------------------------------------------------------------------
    # 以下两个方法是 KV Cache 推理专用
    # ------------------------------------------------------------------

    def build_cache(self, memory):
        """
        【Prefill 阶段】根据 encoder 输出，初始化每一层的 KV Cache。

        做两件事：
        1. 预计算所有 DecoderLayer 的 cross-attention K/V（来自 memory，固定不变）
        2. 初始化 self-attention 的空 cache（seq_so_far=0）

        memory: [batch, src_len, d_model]  编码器输出
        返回:   cache，list，长度 = num_layers，每个元素是一个 dict
        """
        batch  = memory.size(0)
        src_len = memory.size(1)
        device = memory.device
        cache  = []

        for layer in self.decoder_layers:
            attn = layer.cross_attn

            # 预计算 cross-attention 的 K/V
            # memory: [batch, src_len, d_model]
            # W_k, W_v 权重: [d_model, d_model]
            cross_k = attn.W_k(memory)  # [batch, src_len, d_model]
            cross_v = attn.W_v(memory)

            # 切分多头: [batch, src_len, d_model] -> [batch, n_head, src_len, d_k]
            cross_k = cross_k.view(batch, src_len, attn.n_head, attn.d_k).transpose(1, 2)
            cross_v = cross_v.view(batch, src_len, attn.n_head, attn.d_k).transpose(1, 2)

            # self-attention cache 初始为空（seq_so_far=0）
            sa = layer.self_attn
            empty_k = torch.zeros(batch, sa.n_head, 0, sa.d_k, device=device)
            empty_v = torch.zeros(batch, sa.n_head, 0, sa.d_k, device=device)

            cache.append({
                'self_k':  empty_k,   # 每 decode 步后追加
                'self_v':  empty_v,
                'cross_k': cross_k,   # 固定，整个推理过程不变
                'cross_v': cross_v,
            })

        return cache

    def decode_one_step(self, token_id, pos, cache):
        """
        【Decode 阶段】自回归推理的单步：只处理 1 个新 token。

        token_id: [batch]   新 token 的 id（上一步的预测结果）
        pos:      int        当前位置索引（用于取正确的位置编码）
        cache:    list       各层的 KV Cache（会被更新后返回）

        返回：
            logits:    [batch, vocab_size]  下一个 token 的预测分布
            new_cache: 更新后的 cache（self_k/v 每层都增加了 1 步）

        数据流：
            token_id [batch]
              ↓ unsqueeze + embedding + pos_enc
            x: [batch, 1, d_model]
              ↓ 依次通过每层 DecoderLayer.forward_cached
            x: [batch, 1, d_model]
              ↓ fc_out
            logits: [batch, vocab_size]
        """
        # 词嵌入：[batch] -> [batch, 1] -> [batch, 1, d_model]
        x = self.embedding(token_id.unsqueeze(1)) * math.sqrt(self.d_model)

        # 叠加位置编码（只取第 pos 个位置）
        # self.pos_encoding.pe: [1, max_seq_len, d_model]
        x = x + self.pos_encoding.pe[:, pos:pos + 1, :]  # [batch, 1, d_model]

        # 依次通过每个 DecoderLayer（带 cache）
        new_cache = []
        for i, layer in enumerate(self.decoder_layers):
            x, updated_layer_cache = layer.forward_cached(x, cache[i])
            new_cache.append(updated_layer_cache)

        # 取出这个 token 的特征，映射到词表
        # x: [batch, 1, d_model] -> x[:, 0, :]: [batch, d_model]
        # logits: [batch, vocab_size]
        logits = self.fc_out(x[:, 0, :])

        return logits, new_cache


# ======================================================================
# 7. Mask 工具函数
# ======================================================================
def make_src_mask(src, pad_idx=0):
    """
    源序列 PAD Mask

    目的：忽略填充（PAD）位置，不让模型在这些无意义位置上浪费注意力。

    src:    [batch, src_len]
    返回:   [batch, 1, 1, src_len]
            （1,1 两个维度是为了广播到注意力分数矩阵 [batch, n_head, seq_q, seq_k]）
    """
    # src != pad_idx: 非 PAD 位置为 True(1)，PAD 位置为 False(0)
    mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    # 形状: [batch, 1, 1, src_len]
    return mask


def make_tgt_mask(tgt, pad_idx=0):
    """
    目标序列 Mask = PAD Mask & Causal Mask

    两层保护：
    1. PAD Mask：屏蔽填充位置
    2. Causal Mask（因果掩码/下三角矩阵）：
       位置 i 只能与位置 0..i 做注意力，不能"偷看"未来的 token

    示例（tgt_len=4 的 Causal Mask）：
        位置0: [1, 0, 0, 0]  只能看自己
        位置1: [1, 1, 0, 0]  能看前2个
        位置2: [1, 1, 1, 0]  能看前3个
        位置3: [1, 1, 1, 1]  能看全部

    tgt:    [batch, tgt_len]
    返回:   [batch, 1, tgt_len, tgt_len]
    """
    tgt_len = tgt.size(1)
    device  = tgt.device

    # PAD Mask: [batch, 1, 1, tgt_len]
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)

    # Causal Mask（下三角矩阵）: [1, 1, tgt_len, tgt_len]
    causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).unsqueeze(0).unsqueeze(0)

    # 两个 mask 取交集（AND）：只有两个都为1的位置才保留
    # broadcast: [batch, 1, tgt_len, tgt_len]
    return pad_mask & causal_mask.bool()
