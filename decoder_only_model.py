"""
Decoder-Only Transformer（GPT 风格）
=====================================

【与 Encoder-Decoder 的核心区别】

Encoder-Decoder（model.py）：
    源序列  → Encoder → memory
    目标序列 → Decoder（自注意力 + 交叉注意力）→ 输出
    两个序列分开处理，用 Cross-Attention 桥接

Decoder-Only（本文件）：
    prompt + 已生成的token → 拼成一个序列
    → 一个注意力层搞定（只有自注意力，没有 Cross-Attention）
    → 输出

【结构对比】

EncoderLayer:   Self-Attn → Add&Norm → FFN → Add&Norm
DecoderLayer:   Self-Attn → Add&Norm → Cross-Attn → Add&Norm → FFN → Add&Norm
GPTLayer:       Self-Attn → Add&Norm → FFN → Add&Norm          ← 和 EncoderLayer 一样！
                                                                   但用了 Causal Mask

唯一的区别：
    EncoderLayer 没有 Causal Mask（双向，能看所有位置）
    GPTLayer     有    Causal Mask（单向，只能看过去）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ======================================================================
# 1. 位置编码（与 model.py 完全相同）
# ======================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=200):
        super().__init__()
        pe       = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_seq_len, d_model]

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]


# ======================================================================
# 2. 多头注意力（与 model.py 完全相同）
# ======================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=128, n_head=4):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head  = n_head
        self.d_k     = d_model // n_head

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        batch = q.size(0)
        seq_q = q.size(1)
        seq_k = k.size(1)

        Q = self.W_q(q).view(batch, seq_q, self.n_head, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch, seq_k, self.n_head, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch, seq_k, self.n_head, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch, seq_q, self.d_model)
        return self.W_o(output)

    def forward_prefill(self, q, k, v, mask=None):
        """
        Prefill 专用：和 forward 一样计算注意力，但额外返回 K、V 用于填充 cache。

        与逐 token 的 forward_cached 不同，这里一次性处理整个 prompt，
        但必须加 causal mask 保证因果性（和训练时一致）。

        返回: (output, K, V)
            output: [batch, seq, d_model]
            K, V:   [batch, n_head, seq, d_k]  直接作为后续 decode 的初始 cache
        """
        batch = q.size(0)
        seq_q = q.size(1)
        seq_k = k.size(1)

        Q = self.W_q(q).view(batch, seq_q, self.n_head, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(batch, seq_k, self.n_head, self.d_k).transpose(1, 2)
        V = self.W_v(v).view(batch, seq_k, self.n_head, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch, seq_q, self.d_model)
        return self.W_o(output), K, V

    def forward_cached(self, q, cache_k, cache_v):
        """
        KV Cache 版本（推理专用）
        q:       [batch, 1, d_model]
        cache_k: [batch, n_head, seq_so_far, d_k]
        cache_v: [batch, n_head, seq_so_far, d_k]
        """
        batch = q.size(0)

        Q     = self.W_q(q).view(batch, 1, self.n_head, self.d_k).transpose(1, 2)
        K_new = self.W_k(q).view(batch, 1, self.n_head, self.d_k).transpose(1, 2)
        V_new = self.W_v(q).view(batch, 1, self.n_head, self.d_k).transpose(1, 2)

        # 追加新 token 的 K/V 到 cache
        K = torch.cat([cache_k, K_new], dim=2)  # [batch, n_head, seq_so_far+1, d_k]
        V = torch.cat([cache_v, V_new], dim=2)

        scores       = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        output       = torch.matmul(attn_weights, V)
        output       = output.transpose(1, 2).contiguous().view(batch, 1, self.d_model)

        return self.W_o(output), K, V  # 返回更新后的 K/V cache


# ======================================================================
# 3. 前馈网络（与 model.py 完全相同）
# ======================================================================
class FeedForward(nn.Module):
    def __init__(self, d_model=128, d_ff=256):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


# ======================================================================
# 4. GPT 层（Decoder-Only 的核心积木）
# ======================================================================
class GPTLayer(nn.Module):
    """
    【GPTLayer 和 EncoderLayer 的唯一区别就是有没有 Causal Mask】

    EncoderLayer（双向）：
        Self-Attn（无 causal mask，每个位置看全部）→ Add&Norm → FFN → Add&Norm

    GPTLayer（单向）：
        Self-Attn（有 causal mask，每个位置只看过去）→ Add&Norm → FFN → Add&Norm

    没有 Cross-Attention，因为没有独立的"源序列"，
    prompt 和生成内容都在同一个序列里。
    """
    def __init__(self, d_model=128, n_head=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.ffn       = FeedForward(d_model, d_ff)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        x:    [batch, seq_len, d_model]
        mask: [batch, 1, seq_len, seq_len]  Causal Mask（下三角）

        Causal Mask 保证位置 i 只能看 0..i，
        这使得每个位置都能用来预测"下一个 token"。
        """
        # Self-Attention + 残差 + 归一化
        attn_out = self.self_attn(q=x, k=x, v=x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))  # [batch, seq_len, d_model]

        # FFN + 残差 + 归一化
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))   # [batch, seq_len, d_model]

        return x

    def forward_prefill(self, x, mask):
        """
        Prefill 专用：一次性处理整个 prompt，返回输出和 KV cache。

        x:    [batch, seq_len, d_model]
        mask: [1, 1, seq_len, seq_len]  Causal Mask

        返回: (output, K, V)
            output: [batch, seq_len, d_model]
            K, V:   [batch, n_head, seq_len, d_k]  作为后续 decode 的初始 cache
        """
        attn_out, K, V = self.self_attn.forward_prefill(q=x, k=x, v=x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x, K, V

    def forward_cached(self, x, cache_k, cache_v):
        """
        KV Cache 版本（推理专用，每次只处理 1 个新 token）

        x:       [batch, 1, d_model]
        cache_k: [batch, n_head, seq_so_far, d_k]
        cache_v: [batch, n_head, seq_so_far, d_k]
        """
        # 推理时不需要 mask（cache 里只有历史，没有未来）
        attn_out, new_k, new_v = self.self_attn.forward_cached(x, cache_k, cache_v)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, new_k, new_v


# ======================================================================
# 5. Decoder-Only Transformer（GPT 风格）
# ======================================================================
class DecoderOnlyTransformer(nn.Module):
    """
    【完整数据流 - 训练时】

    输入序列（prompt + 生成内容拼在一起）:
    [5, 3, 7, SEP, 1, 3, 5, 7]

    训练目标（每个位置预测下一个 token）:
    输入:  [5,   3,   7,   SEP, 1,   3,   5,   7  ]
    标签:  [3,   7,   SEP, 1,   3,   5,   7,   EOS]
                                         ↑
                               每个位置往后错一格

    Causal Mask 保证位置 i 预测时只能看 0..i，
    所以所有位置可以并行训练，但信息流是单向的。

    【与 Encoder-Decoder 的训练差异】
    Encoder-Decoder：src 和 tgt 是两个独立输入，分别处理
    Decoder-Only：  src 和 tgt 拼成一个序列，统一处理
    """
    def __init__(self, vocab_size, d_model=128, n_head=4, num_layers=4,
                 d_ff=256, dropout=0.1, max_seq_len=200):
        super().__init__()
        self.d_model    = d_model
        self.num_layers = num_layers

        # 词嵌入
        self.embedding    = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # N 个 GPTLayer 堆叠（没有 EncoderLayer，没有 Cross-Attention）
        self.layers = nn.ModuleList([
            GPTLayer(d_model, n_head, d_ff, dropout) for _ in range(num_layers)
        ])

        # 输出层
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        训练时的前向传播

        x:    [batch, seq_len]  整个序列（prompt + 生成内容）的 token id
        mask: [batch, 1, seq_len, seq_len]  Causal Mask

        返回: logits [batch, seq_len, vocab_size]
              每个位置对"下一个 token"的预测分布
        """
        # 词嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)  # [batch, seq_len, d_model]
        x = self.dropout(self.pos_encoding(x))

        # 依次通过每个 GPTLayer
        for layer in self.layers:
            x = layer(x, mask)

        # 输出层：每个位置预测下一个 token
        return self.fc_out(x)  # [batch, seq_len, vocab_size]

    def prefill(self, prompt_tokens):
        """
        【Prefill 阶段】一次性并行处理整个 prompt，填充 KV Cache。

        与旧版逐 token 调用 decode_one_step 的区别：
            旧版：prompt 有 N 个 token → 串行调用 N 次 decode_one_step → 慢
            新版：一次性送入 N 个 token，用 causal mask 保证因果性 → 快

        必须加 causal mask 的原因：
            模型训练时用了 causal mask（位置 i 只能看 0..i），
            推理时必须保持一致，否则计算出的 KV 和训练时不同，生成就会出错。

        prompt_tokens: [batch, prompt_len]  prompt 的 token id
        返回:
            logits: [batch, vocab_size]  最后一个位置的预测（用于生成第一个新 token）
            cache:  list，每层一个 (K, V) 元组
                    K, V: [batch, n_head, prompt_len, d_k]
        """
        seq_len = prompt_tokens.size(1)
        device  = prompt_tokens.device

        # 词嵌入 + 位置编码
        x = self.embedding(prompt_tokens) * math.sqrt(self.d_model)
        x = self.dropout(self.pos_encoding(x))  # [batch, prompt_len, d_model]

        # Causal Mask：和训练时一样的下三角矩阵
        # [1, 1, prompt_len, prompt_len]
        mask = make_causal_mask(seq_len, device)

        # 依次通过每层，收集 KV Cache
        cache = []
        for layer in self.layers:
            x, K, V = layer.forward_prefill(x, mask)
            cache.append((K, V))
            # K, V: [batch, n_head, prompt_len, d_k]
            # 直接作为后续 decode_one_step 的初始 cache

        # 只取最后一个位置的 logits（预测 prompt 之后的第一个 token）
        logits = self.fc_out(x[:, -1, :])  # [batch, vocab_size]

        return logits, cache

    def build_cache(self, device, batch_size=1):
        """
        初始化空的 KV Cache（推理前调用）

        Decoder-Only 的 cache 比 Encoder-Decoder 简单：
        只有自注意力的 K/V，没有交叉注意力的 K/V。

        返回: cache，list，每层一个 (k, v) 元组，初始为空
        """
        cache = []
        for layer in self.layers:
            n_head = layer.self_attn.n_head
            d_k    = layer.self_attn.d_k
            # 空的 cache：seq_so_far=0
            empty_k = torch.zeros(batch_size, n_head, 0, d_k, device=device)
            empty_v = torch.zeros(batch_size, n_head, 0, d_k, device=device)
            cache.append((empty_k, empty_v))
        return cache

    def decode_one_step(self, token_id, pos, cache):
        """
        推理时单步解码（带 KV Cache）

        token_id: [batch]  当前 token 的 id
        pos:      int       当前位置（用于取位置编码）
        cache:    list      各层的 (cache_k, cache_v)

        返回:
            logits:    [batch, vocab_size]
            new_cache: 更新后的 cache
        """
        # 词嵌入 + 位置编码（只处理 1 个 token）
        x = self.embedding(token_id.unsqueeze(1)) * math.sqrt(self.d_model)
        x = x + self.pos_encoding.pe[:, pos:pos + 1, :]  # [batch, 1, d_model]

        new_cache = []
        for i, layer in enumerate(self.layers):
            cache_k, cache_v = cache[i]
            x, new_k, new_v  = layer.forward_cached(x, cache_k, cache_v)
            new_cache.append((new_k, new_v))

        # 取最后一个 token 的输出，预测下一个
        logits = self.fc_out(x[:, 0, :])  # [batch, vocab_size]

        return logits, new_cache


# ======================================================================
# Mask 工具函数
# ======================================================================
def make_causal_mask(seq_len, device):
    """
    纯 Causal Mask（推理时使用，无 PAD）

    返回: [1, 1, seq_len, seq_len]
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def make_mask(seq, pad_idx=0):
    """
    训练时使用的完整 Mask = PAD Mask & Causal Mask

    变长训练时，批次内序列长度不同，需要补 PAD。
    这个函数同时处理两件事：
      1. PAD Mask：屏蔽 PAD 位置的 Query（那一行全为0）
      2. Causal Mask：屏蔽未来位置（上三角全为0）

    例子（batch=2，seq_len=5，样本2后3个是PAD）：

        样本2的 mask：
                  t0  t1  t2  PAD PAD
            t0  [  1,  0,  0,  0,  0 ]   只看自己（causal）
            t1  [  1,  1,  0,  0,  0 ]   看前两个（causal）
            t2  [  1,  1,  1,  0,  0 ]   看前三个（causal）
            PAD [  0,  0,  0,  0,  0 ]   PAD行全屏蔽（pad mask）
            PAD [  0,  0,  0,  0,  0 ]   PAD行全屏蔽（pad mask）

    seq:  [batch, seq_len]
    返回: [batch, 1, seq_len, seq_len]
    """
    seq_len = seq.size(1)
    device  = seq.device

    # PAD Mask: PAD 位置为 0，其他为 1
    # [batch, 1, 1, seq_len] 广播到 [batch, 1, seq_len, seq_len]（屏蔽整行）
    pad_mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    # Causal Mask: 下三角矩阵
    # [1, 1, seq_len, seq_len]
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)

    # 两者取交集
    return pad_mask & causal_mask.bool()  # [batch, 1, seq_len, seq_len]
