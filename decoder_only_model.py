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

【形状缩写约定（全文通用）】
    B  = batch_size          （批大小，训练时=128）
    S  = seq_len             （序列长度，含PAD）
    D  = d_model             （嵌入维度，训练时=64）
    H  = n_head              （注意力头数，训练时=4）
    dk = d_k = D // H        （每头维度，训练时=16）
    Ff = d_ff                （FFN中间维度，训练时=128）
    V  = vocab_size          （词表大小=13）
    P  = prompt_len          （prefill时prompt的长度）
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
        pe       = torch.zeros(max_seq_len, d_model)          # [max_seq_len, D]
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()  # [max_seq_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                      # [D/2]
        pe[:, 0::2] = torch.sin(position * div_term)          # [max_seq_len, D/2]
        pe[:, 1::2] = torch.cos(position * div_term)          # [max_seq_len, D/2]
        self.register_buffer('pe', pe.unsqueeze(0))            # [1, max_seq_len, D]

    def forward(self, x):
        # x:              [B, S, D]
        # pe[:, :S, :]:   [1, S, D]  → 广播加到每个样本
        # 返回:           [B, S, D]
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
        self.d_k     = d_model // n_head   # dk = D/H

        # 线性投影：D → D（内部等价于 H 个 dk 头并联）
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        """
        训练时全序列注意力（无 KV Cache）

        输入:
            q: [B, Sq, D]   Query 序列（自注意力时 Sq = S）
            k: [B, Sk, D]   Key   序列（自注意力时 Sk = S）
            v: [B, Sk, D]   Value 序列
        """
        batch = q.size(0)   # B
        seq_q = q.size(1)   # Sq
        seq_k = k.size(1)   # Sk

        # ── 线性投影 + 拆分多头 ──────────────────────────────────
        # W_q(q):  [B, Sq, D]
        # .view:   [B, Sq, H, dk]   把 D 维拆成 H 个 dk 头
        # .T(1,2): [B, H, Sq, dk]   把头维移到第2位，方便后续批矩阵乘
        Q = self.W_q(q).view(batch, seq_q, self.n_head, self.d_k).transpose(1, 2)
        # Q: [B, H, Sq, dk]

        K = self.W_k(k).view(batch, seq_k, self.n_head, self.d_k).transpose(1, 2)
        # K: [B, H, Sk, dk]

        V = self.W_v(v).view(batch, seq_k, self.n_head, self.d_k).transpose(1, 2)
        # V: [B, H, Sk, dk]

        # ── 注意力分数 ────────────────────────────────────────────
        # K.T(-2,-1): [B, H, dk, Sk]
        # Q @ K.T:    [B, H, Sq, dk] × [B, H, dk, Sk] = [B, H, Sq, Sk]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: [B, H, Sq, Sk]   每个 query 位置对所有 key 位置的相关性得分

        if mask is not None:
            # mask: [B, 1, Sq, Sk] 或 [1, 1, Sq, Sk]，广播到 [B, H, Sq, Sk]
            scores = scores.masked_fill(mask == 0, -1e9)
            # 被 mask 的位置变成 -∞，softmax 后趋近 0

        # ── Softmax 归一化 ────────────────────────────────────────
        attn_weights = F.softmax(scores, dim=-1)
        # attn_weights: [B, H, Sq, Sk]   每行和为1，表示对 key 位置的注意力权重

        # ── 加权求和 Value ─────────────────────────────────────────
        # attn_weights @ V: [B, H, Sq, Sk] × [B, H, Sk, dk] = [B, H, Sq, dk]
        output = torch.matmul(attn_weights, V)
        # output: [B, H, Sq, dk]

        # ── 合并多头 + 输出投影 ────────────────────────────────────
        # .T(1,2): [B, Sq, H, dk]
        # .view:   [B, Sq, D]      H 个头拼回 D 维
        output = output.transpose(1, 2).contiguous().view(batch, seq_q, self.d_model)
        # output: [B, Sq, D]

        return self.W_o(output)
        # 返回: [B, Sq, D]

    def forward_prefill(self, q, k, v, mask=None):
        """
        Prefill 专用：和 forward 一样计算注意力，但额外返回 K、V 用于填充 cache。

        与逐 token 的 forward_cached 不同，这里一次性处理整个 prompt，
        但必须加 causal mask 保证因果性（和训练时一致）。

        输入:
            q, k, v: [B, P, D]   P = prompt_len
            mask:    [1, 1, P, P]

        返回: (output, K, V)
            output: [B, P, D]
            K, V:   [B, H, P, dk]   直接作为后续 decode 的初始 cache
        """
        batch = q.size(0)   # B
        seq_q = q.size(1)   # P（prompt 长度）
        seq_k = k.size(1)   # P

        # 投影 + 拆头（过程同 forward，形状参考上面）
        Q = self.W_q(q).view(batch, seq_q, self.n_head, self.d_k).transpose(1, 2)
        # Q: [B, H, P, dk]
        K = self.W_k(k).view(batch, seq_k, self.n_head, self.d_k).transpose(1, 2)
        # K: [B, H, P, dk]   ← 这个 K 会被缓存
        V = self.W_v(v).view(batch, seq_k, self.n_head, self.d_k).transpose(1, 2)
        # V: [B, H, P, dk]   ← 这个 V 会被缓存

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: [B, H, P, P]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            # scores: [B, H, P, P]   上三角被置 -∞

        attn_weights = F.softmax(scores, dim=-1)
        # attn_weights: [B, H, P, P]

        output = torch.matmul(attn_weights, V)
        # output: [B, H, P, dk]

        output = output.transpose(1, 2).contiguous().view(batch, seq_q, self.d_model)
        # output: [B, P, D]

        return self.W_o(output), K, V
        # 返回: ([B, P, D], [B, H, P, dk], [B, H, P, dk])

    def forward_cached(self, q, cache_k, cache_v):
        """
        KV Cache 版本（推理专用，每次只处理 1 个新 token）

        输入:
            q:       [B, 1, D]                  当前新 token 的嵌入
            cache_k: [B, H, seq_so_far, dk]     历史所有 token 的 K
            cache_v: [B, H, seq_so_far, dk]     历史所有 token 的 V
        """
        batch = q.size(0)   # B

        # 只对新 token 做投影（只算 1 个位置，不重算历史）
        Q     = self.W_q(q).view(batch, 1, self.n_head, self.d_k).transpose(1, 2)
        # Q: [B, H, 1, dk]

        K_new = self.W_k(q).view(batch, 1, self.n_head, self.d_k).transpose(1, 2)
        # K_new: [B, H, 1, dk]   新 token 的 K

        V_new = self.W_v(q).view(batch, 1, self.n_head, self.d_k).transpose(1, 2)
        # V_new: [B, H, 1, dk]   新 token 的 V

        # 追加新 token 的 K/V 到 cache（沿 seq 维度拼接）
        K = torch.cat([cache_k, K_new], dim=2)
        # K: [B, H, seq_so_far+1, dk]   历史 K + 新 K
        V = torch.cat([cache_v, V_new], dim=2)
        # V: [B, H, seq_so_far+1, dk]

        # Q（1个位置）与全部历史 K 做注意力
        # K.T(-2,-1): [B, H, dk, seq_so_far+1]
        # Q @ K.T:    [B, H, 1, dk] × [B, H, dk, seq_so_far+1] = [B, H, 1, seq_so_far+1]
        scores       = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: [B, H, 1, seq_so_far+1]   1个query 对所有历史位置的得分

        attn_weights = F.softmax(scores, dim=-1)
        # attn_weights: [B, H, 1, seq_so_far+1]

        # attn_weights @ V: [B, H, 1, seq_so_far+1] × [B, H, seq_so_far+1, dk]
        output       = torch.matmul(attn_weights, V)
        # output: [B, H, 1, dk]

        output       = output.transpose(1, 2).contiguous().view(batch, 1, self.d_model)
        # output: [B, 1, D]

        return self.W_o(output), K, V
        # 返回: ([B, 1, D], [B, H, seq_so_far+1, dk], [B, H, seq_so_far+1, dk])


# ======================================================================
# 3. 前馈网络（与 model.py 完全相同）
# ======================================================================
class FeedForward(nn.Module):
    def __init__(self, d_model=128, d_ff=256):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x:          [B, S, D]
        # linear1(x): [B, S, D] → [B, S, Ff]   升维（Ff=128，D=64的2倍）
        # relu:       [B, S, Ff]                非线性激活
        # linear2:    [B, S, Ff] → [B, S, D]   降回 D 维
        return self.linear2(F.relu(self.linear1(x)))
        # 返回: [B, S, D]


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
        训练时全序列前向（带 Causal Mask）

        x:    [B, S, D]
        mask: [B, 1, S, S]  Causal Mask（下三角，含PAD屏蔽）

        Causal Mask 保证位置 i 只能看 0..i，
        这使得每个位置都能用来预测"下一个 token"。
        """
        # ── Self-Attention ────────────────────────────────────────
        attn_out = self.self_attn(q=x, k=x, v=x, mask=mask)
        # attn_out: [B, S, D]

        # ── 残差连接 + 层归一化 ───────────────────────────────────
        x = self.norm1(x + self.dropout(attn_out))
        # x: [B, S, D]

        # ── FFN ──────────────────────────────────────────────────
        ffn_out = self.ffn(x)
        # ffn_out: [B, S, D]

        x = self.norm2(x + self.dropout(ffn_out))
        # x: [B, S, D]

        return x
        # 返回: [B, S, D]

    def forward_prefill(self, x, mask):
        """
        Prefill 专用：一次性处理整个 prompt，返回输出和 KV cache。

        x:    [B, P, D]
        mask: [1, 1, P, P]  Causal Mask

        返回: (output, K, V)
            output: [B, P, D]
            K, V:   [B, H, P, dk]  作为后续 decode 的初始 cache
        """
        attn_out, K, V = self.self_attn.forward_prefill(q=x, k=x, v=x, mask=mask)
        # attn_out: [B, P, D]
        # K: [B, H, P, dk]
        # V: [B, H, P, dk]

        x = self.norm1(x + self.dropout(attn_out))
        # x: [B, P, D]

        ffn_out = self.ffn(x)
        # ffn_out: [B, P, D]

        x = self.norm2(x + self.dropout(ffn_out))
        # x: [B, P, D]

        return x, K, V
        # 返回: ([B, P, D], [B, H, P, dk], [B, H, P, dk])

    def forward_cached(self, x, cache_k, cache_v):
        """
        KV Cache 版本（推理专用，每次只处理 1 个新 token）

        x:       [B, 1, D]
        cache_k: [B, H, seq_so_far, dk]
        cache_v: [B, H, seq_so_far, dk]
        """
        attn_out, new_k, new_v = self.self_attn.forward_cached(x, cache_k, cache_v)
        # attn_out: [B, 1, D]
        # new_k:    [B, H, seq_so_far+1, dk]   已包含新 token 的 K
        # new_v:    [B, H, seq_so_far+1, dk]

        x = self.norm1(x + attn_out)
        # x: [B, 1, D]

        x = self.norm2(x + self.ffn(x))
        # ffn(x): [B, 1, D]
        # x:      [B, 1, D]

        return x, new_k, new_v
        # 返回: ([B, 1, D], [B, H, seq_so_far+1, dk], [B, H, seq_so_far+1, dk])


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

        self.embedding    = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # Embedding 表: [V, D]，把每个 token id 映射到 D 维向量
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        self.layers = nn.ModuleList([
            GPTLayer(d_model, n_head, d_ff, dropout) for _ in range(num_layers)
        ])
        # num_layers 个 GPTLayer 串联，每层输入输出均为 [B, S, D]

        self.fc_out = nn.Linear(d_model, vocab_size)
        # 输出投影: D → V，用于预测下一个 token 的概率分布
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        训练时的前向传播

        x:    [B, S]   整个序列的 token id（prompt + 答案，已去掉最后一个 EOS）
        mask: [B, 1, S, S]  Causal Mask + PAD Mask

        返回: logits [B, S, V]   每个位置对"下一个 token"的预测分布
        """
        # ── 词嵌入 ────────────────────────────────────────────────
        x = self.embedding(x)
        # embedding(x): [B, S] → [B, S, D]   查表，每个 token id 换成 D 维向量

        x = x * math.sqrt(self.d_model)
        # x: [B, S, D]   乘以 √D 防止嵌入值过小，稳定梯度

        x = self.dropout(self.pos_encoding(x))
        # pos_encoding: [B, S, D] + [1, S, D] = [B, S, D]   加入位置信息
        # dropout 后:   [B, S, D]

        # ── 依次通过每个 GPTLayer ─────────────────────────────────
        for layer in self.layers:
            x = layer(x, mask)
            # 每层输入输出均为 [B, S, D]
            # 共 num_layers 层（训练时=2层）

        # ── 输出投影 ──────────────────────────────────────────────
        return self.fc_out(x)
        # fc_out: [B, S, D] → [B, S, V]
        # 每个序列位置输出 V=13 维的 logits，表示"下一个 token"的未归一化概率

    def prefill(self, prompt_tokens):
        """
        【Prefill 阶段】一次性并行处理整个 prompt，填充 KV Cache。

        与旧版逐 token 调用 decode_one_step 的区别：
            旧版：prompt 有 N 个 token → 串行调用 N 次 decode_one_step → 慢
            新版：一次性送入 N 个 token，用 causal mask 保证因果性 → 快

        必须加 causal mask 的原因：
            模型训练时用了 causal mask（位置 i 只能看 0..i），
            推理时必须保持一致，否则计算出的 KV 和训练时不同，生成就会出错。

        prompt_tokens: [B, P]  prompt 的 token id（B 通常=1，推理时单样本）
        返回:
            logits: [B, V]    最后一个位置的预测（用于生成第一个新 token）
            cache:  list，每层一个 (K, V) 元组
                    K, V: [B, H, P, dk]
        """
        seq_len = prompt_tokens.size(1)   # P
        device  = prompt_tokens.device

        # ── 词嵌入 + 位置编码 ─────────────────────────────────────
        x = self.embedding(prompt_tokens) * math.sqrt(self.d_model)
        # x: [B, P, D]

        x = self.dropout(self.pos_encoding(x))
        # x: [B, P, D]

        # ── Causal Mask（和训练完全一致的下三角矩阵）────────────────
        mask = make_causal_mask(seq_len, device)
        # mask: [1, 1, P, P]   下三角为1，上三角为0

        # ── 依次通过每层，收集各层的 KV Cache ─────────────────────
        cache = []
        for layer in self.layers:
            x, K, V = layer.forward_prefill(x, mask)
            # x: [B, P, D]
            # K: [B, H, P, dk]   ← 该层对 prompt 所有位置计算的 K，缓存备用
            # V: [B, H, P, dk]   ← 该层对 prompt 所有位置计算的 V，缓存备用
            cache.append((K, V))

        # ── 取最后一个 prompt token 的 logits ─────────────────────
        # x[:, -1, :]: [B, D]   只要最后一个位置（SEP 位置）的输出
        logits = self.fc_out(x[:, -1, :])
        # logits: [B, V]   预测 prompt 后第一个 token 的分布

        return logits, cache
        # 返回: ([B, V], list of (K[B,H,P,dk], V[B,H,P,dk]) × num_layers)

    def build_cache(self, device, batch_size=1):
        """
        初始化空的 KV Cache（推理前调用，如不使用 prefill）

        Decoder-Only 的 cache 比 Encoder-Decoder 简单：
        只有自注意力的 K/V，没有交叉注意力的 K/V。

        返回: cache，list，每层一个 (k, v) 元组，初始为空（seq_so_far=0）
        """
        cache = []
        for layer in self.layers:
            n_head = layer.self_attn.n_head
            d_k    = layer.self_attn.d_k
            empty_k = torch.zeros(batch_size, n_head, 0, d_k, device=device)
            # empty_k: [B, H, 0, dk]   seq 维为 0，尚无历史
            empty_v = torch.zeros(batch_size, n_head, 0, d_k, device=device)
            # empty_v: [B, H, 0, dk]
            cache.append((empty_k, empty_v))
        return cache

    def decode_one_step(self, token_id, pos, cache):
        """
        推理时单步解码（带 KV Cache）

        token_id: [B]    当前 token 的 id（上一步预测出的 token）
        pos:      int    当前位置（用于取正确的位置编码）
        cache:    list   各层的 (cache_k, cache_v)

        返回:
            logits:    [B, V]
            new_cache: 更新后的 cache（每层 seq 维度+1）
        """
        # ── 词嵌入 + 位置编码（只处理 1 个新 token）──────────────
        x = self.embedding(token_id.unsqueeze(1)) * math.sqrt(self.d_model)
        # token_id.unsqueeze(1): [B] → [B, 1]
        # embedding:             [B, 1] → [B, 1, D]
        # × √D:                  [B, 1, D]

        x = x + self.pos_encoding.pe[:, pos:pos + 1, :]
        # pe[:, pos:pos+1, :]: [1, 1, D]   取第 pos 个位置的编码
        # x: [B, 1, D]

        # ── 依次通过每层，更新 KV Cache ──────────────────────────
        new_cache = []
        for i, layer in enumerate(self.layers):
            cache_k, cache_v = cache[i]
            # cache_k: [B, H, seq_so_far, dk]
            # cache_v: [B, H, seq_so_far, dk]

            x, new_k, new_v  = layer.forward_cached(x, cache_k, cache_v)
            # x:     [B, 1, D]
            # new_k: [B, H, seq_so_far+1, dk]   追加了当前 token 的 K
            # new_v: [B, H, seq_so_far+1, dk]
            new_cache.append((new_k, new_v))

        # ── 输出投影（只取 seq=0 这个唯一位置）──────────────────
        logits = self.fc_out(x[:, 0, :])
        # x[:, 0, :]: [B, D]   squeeze 掉 seq=1 这一维
        # logits:     [B, V]   预测下一个 token 的分布

        return logits, new_cache
        # 返回: ([B, V], list of (K[B,H,seq_so_far+1,dk], V[B,H,seq_so_far+1,dk]) × num_layers)


# ======================================================================
# Mask 工具函数
# ======================================================================
def make_causal_mask(seq_len, device):
    """
    纯 Causal Mask（推理 prefill 时使用，无 PAD）

    torch.tril: 保留下三角（含对角线），上三角置0
    例如 seq_len=4：
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]

    返回: [1, 1, seq_len, seq_len]   前两个1是 batch 和 head 的广播维
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    # mask: [seq_len, seq_len]

    return mask.unsqueeze(0).unsqueeze(0)
    # 返回: [1, 1, seq_len, seq_len]


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

    seq:  [B, S]
    返回: [B, 1, S, S]
    """
    seq_len = seq.size(1)   # S
    device  = seq.device

    # ── PAD Mask ──────────────────────────────────────────────────
    # (seq != pad_idx):          [B, S]      非PAD位置为True
    # .unsqueeze(1).unsqueeze(2):[B, 1, 1, S]
    # 广播到 [B, 1, S, S]：屏蔽整行（某行的 query 是PAD时全行置0）
    pad_mask = (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    # pad_mask: [B, 1, 1, S]

    # ── Causal Mask ───────────────────────────────────────────────
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).unsqueeze(0).unsqueeze(0)
    # tril:       [S, S]
    # unsqueeze×2:[1, 1, S, S]

    # ── 两者取交集 ────────────────────────────────────────────────
    # pad_mask:   [B, 1, 1, S]   广播
    # causal_mask:[1, 1, S, S]   广播
    # &:          [B, 1, S, S]
    return pad_mask & causal_mask.bool()
    # 返回: [B, 1, S, S]   每个样本独立的 mask
