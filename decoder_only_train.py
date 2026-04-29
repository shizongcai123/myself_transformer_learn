"""
Decoder-Only 模型训练与推理
============================

【同样是排序任务，但数据组织方式完全不同】

Encoder-Decoder（train.py）：
    src 和 tgt 是两个独立的输入
    src:    [BOS, 5, 3, 7, 1, EOS]          → 喂给编码器
    tgt_in: [BOS, 1, 3, 5, 7]              → 喂给解码器
    tgt_out:[1, 3, 5, 7, EOS]              → 解码器预测目标

    训练时用 Teacher Forcing 加速
    如果训练时也像推理那样一步步跑，太慢了。

    所以我们直接把正确答案作为输入喂给解码器，让它并行学习每个位置：

    tgt_in:  [BOS, 1, 3, 5, 7]     ← 把正确答案直接告诉它
    tgt_out: [1,   3, 5, 7, EOS]   ← 让它在每个位置预测下一个

Decoder-Only（本文件）：
    prompt 和答案拼成一个序列
    全序列: [5, 3, 7, 1, SEP, 1, 3, 5, 7, EOS]
              ↑ 乱序数字    ↑ 分隔符  ↑ 排序结果
    训练输入:  [5, 3, 7, 1, SEP, 1, 3, 5, 7]       ← 去掉最后的 EOS
    训练标签:  [3, 7, 1, SEP, 1, 3, 5, 7, EOS]     ← 去掉第一个token，整体错一格

    每个位置预测下一个 token，用 Causal Mask 保证只看过去。

【推理时】
    只给 prompt 部分: [5, 3, 7, 1, SEP]
    然后自回归生成直到 EOS: [1, 3, 5, 7, EOS]

【形状缩写约定（全文通用）】
    B  = batch_size       （训练时=128）
    S  = seq_len          （批内最长序列，含PAD）
    D  = d_model          （训练时=64）
    H  = n_head           （训练时=4）
    dk = D // H           （训练时=16）
    V  = vocab_size       （=13）
    P  = prompt_len       （推理时 prompt 的长度）
"""

import torch
import torch.nn as nn
import random
from decoder_only_model import DecoderOnlyTransformer, make_causal_mask, make_mask

# ======================================================================
# 词表
# ======================================================================
PAD  = 0   # 填充（本任务基本不用，保留规范）
EOS  = 1   # 结束符
SEP  = 2   # 分隔符：分隔乱序输入和排序输出（相当于 Encoder-Decoder 里的 BOS）
# 数字 0-9 → token id 3-12
VOCAB_SIZE = 13

def num_to_tok(n: int) -> int: return n + 3
def tok_to_num(t: int) -> int: return t - 3


# ======================================================================
# 数据生成
# ======================================================================
def generate_batch(batch_size: int, min_len: int = 3, max_len: int = 8):
    """
    生成一批【变长】数据，批次内用 PAD 对齐。

    全序列 = [乱序数字..., SEP, 排序数字..., EOS, PAD, PAD, ...]
               ↑ 长度随机                          ↑ 补齐到批次最长

    例子（batch=2，min_len=3, max_len=5）：
        样本1: 数字=[5,3,7,1]（长度4）
            full: [5,3,7,1, SEP, 1,3,5,7, EOS]       长度10
            x:    [5,3,7,1, SEP, 1,3,5,7    ]         长度9   ← 去掉最后EOS
            y:    [3,7,1,SEP, 1,3,5,7, EOS  ]         长度9   ← 整体左移一格

        样本2: 数字=[9,2]（长度2）
            full: [9,2, SEP, 2,9, EOS]                长度6
            x:    [9,2, SEP, 2,9     ]                长度5
            y:    [2,SEP, 2,9, EOS   ]                长度5

        补 PAD 对齐到最长（长度9）：
            x: [[5,3,7,1,SEP,1,3,5,7],
                [9,2,SEP,2,9,PAD,PAD,PAD,PAD]]        → [B, S=9]
            y: [[3,7,1,SEP,1,3,5,7,EOS],
                [2,SEP,2,9,EOS,PAD,PAD,PAD,PAD]]      → [B, S=9]

    损失计算时忽略：
        1. prompt 部分（乱序数字，无规律不需要学）
        2. PAD 部分（ignore_index=PAD）

    返回:
        x:           [B, S]   输入 token id（长序列含PAD）
        y:           [B, S]   标签 token id（x 整体右移一格）
        prompt_lens: list[int] 各样本的 prompt 长度（不含SEP）
    """
    xs, ys, prompt_lens = [], [], []

    for _ in range(batch_size):
        seq_len     = random.randint(min_len, max_len)   # 每条样本随机长度（3~8）
        nums        = [random.randint(0, 9) for _ in range(seq_len)]
        sorted_nums = sorted(nums)

        # full: [乱序token×seq_len, SEP, 排序token×seq_len, EOS]
        # 长度 = 2*seq_len + 2
        full = [num_to_tok(n) for n in nums] + [SEP] + \
               [num_to_tok(n) for n in sorted_nums] + [EOS]

        xs.append(full[:-1])     # 去掉最后的 EOS，长度 = 2*seq_len+1
        ys.append(full[1:])      # 去掉第一个 token，长度 = 2*seq_len+1
        prompt_lens.append(seq_len)   # 记录 prompt 长度（只含数字，不含SEP）

    # 补 PAD 到批次内最长（变长训练必须对齐）
    def pad_batch(seqs):
        max_l = max(len(s) for s in seqs)
        # 短序列末尾补 PAD（id=0）
        return [s + [PAD] * (max_l - len(s)) for s in seqs]

    return (
        torch.tensor(pad_batch(xs), dtype=torch.long),   # [B, S]
        torch.tensor(pad_batch(ys), dtype=torch.long),   # [B, S]
        prompt_lens,                                       # list[int] 长度=B
    )


# ======================================================================
# 训练
# ======================================================================
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")

    # ---------- 超参数 ----------
    MIN_LEN    = 3      # 序列最短长度
    MAX_LEN    = 8      # 序列最长长度（同一批次内长度随机，用 PAD 补齐）
    D_MODEL    = 64     # D
    N_HEAD     = 4      # H，dk = D/H = 16
    NUM_LAYERS = 2
    D_FF       = 128    # FFN 中间维度（D 的 2 倍）
    DROPOUT    = 0.1
    BATCH_SIZE = 128    # B
    NUM_STEPS  = 3000
    LR         = 1e-3

    model = DecoderOnlyTransformer(
        vocab_size  = VOCAB_SIZE,   # V = 13
        d_model     = D_MODEL,      # D = 64
        n_head      = N_HEAD,       # H = 4
        num_layers  = NUM_LAYERS,   # 2 层 GPTLayer
        d_ff        = D_FF,         # Ff = 128
        dropout     = DROPOUT,
    ).to(device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}\n")

    # 损失函数：只在 SEP 之后的位置计算（忽略 prompt 部分的预测）
    # 原因：prompt 是无规律的乱序数字，强迫模型预测没有意义
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("=" * 50)
    print("开始训练（Decoder-Only，排序任务）")
    print("=" * 50)

    for step in range(1, NUM_STEPS + 1):
        model.train()

        # ── 1. 生成数据 ───────────────────────────────────────────
        x, y, prompt_lens = generate_batch(BATCH_SIZE, MIN_LEN, MAX_LEN)
        x = x.to(device)   # x: [B, S]   输入序列（prompt + 答案去掉EOS）
        y = y.to(device)   # y: [B, S]   标签序列（x 整体左移一格）

        # ── 2. 生成 Mask（Causal Mask + PAD Mask）────────────────
        # 变长训练必须同时屏蔽 PAD 位置，否则 PAD 行的注意力计算是无意义的
        mask = make_mask(x)
        # mask: [B, 1, S, S]   下三角=可见（1），上三角/PAD行=屏蔽（0）

        # ── 3. 前向传播 ───────────────────────────────────────────
        logits = model(x, mask)
        # 数据流:
        #   x: [B, S]
        #   → embedding + √D: [B, S, D]
        #   → pos_encoding:   [B, S, D]
        #   → GPTLayer ×2:    [B, S, D]（每层内部经过 MHA + FFN + 残差）
        #   → fc_out:         [B, S, V]
        # logits: [B, S, V]   每个位置输出对"下一个 token"的预测 logits

        # ── 4. 计算损失（只算 SEP 之后的部分）───────────────────
        # 把 prompt 部分的标签替换为 PAD（PAD 被 ignore_index 忽略）
        # y 的前 prompt_len 个位置对应的是"乱序数字"的预测，不需要学习
        masked_y = y.clone()
        # masked_y: [B, S]
        for i, plen in enumerate(prompt_lens):
            masked_y[i, :plen] = PAD   # 忽略前 plen 个位置（prompt 数字部分）
        # masked_y: [B, S]   prompt 部分变为 PAD，SEP 之后保留真实标签

        # 展平后计算交叉熵：
        #   logits.view(-1, V):   [B*S, V]   每个位置的预测分布
        #   masked_y.view(-1):    [B*S]      每个位置的真实 token id
        #   ignore_index=PAD：   PAD 位置（prompt+填充）不计入损失
        loss = criterion(logits.view(-1, VOCAB_SIZE), masked_y.view(-1))
        # loss: scalar   只在 SEP 之后的有效位置上计算平均交叉熵

        # ── 5. 反向传播 ───────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()   # 反向传播，计算所有参数的梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # 梯度裁剪：防止梯度爆炸，将全局梯度范数限制在 1.0
        optimizer.step()  # 更新参数

        if step % 300 == 0:
            acc = evaluate_accuracy(model, device, MIN_LEN, MAX_LEN)
            print(f"Step {step:5d} | Loss: {loss.item():.4f} | 排序准确率: {acc:.1%}")

    print("\n训练完成！\n")
    return model, MIN_LEN, MAX_LEN


# ======================================================================
# 准确率评估
# ======================================================================
def evaluate_accuracy(model, device, min_len=3, max_len=8, n_samples=200):
    correct = 0
    for _ in range(n_samples):
        seq_len = random.randint(min_len, max_len)
        nums    = [random.randint(0, 9) for _ in range(seq_len)]
        result  = inference(model, nums, device)
        if result == sorted(nums):
            correct += 1
    return correct / n_samples


# ======================================================================
# 推理（自回归生成）
# ======================================================================
def inference(model, src_nums: list, device, max_new_tokens: int = 20) -> list:
    """
    【Decoder-Only 的推理方式 — Prefill + Decode】

    prompt:  [5, 3, 7, 1, SEP]   ← 给定（长度 P = seq_len + 1）
    生成:    [1, 3, 5, 7, EOS]   ← 模型自回归输出

    【两阶段流程】

    Prefill 阶段（并行）：
        一次性把整个 prompt 送入模型（带 causal mask），并行计算
        得到所有层的 KV Cache + 最后一个位置的 logits
        → 比逐 token 处理快得多（生产环境 vLLM/TGI 都是这样做的）

    Decode 阶段（串行）：
        每次用 decode_one_step 处理 1 个新 token
        从 cache 读历史 KV，只算新 token 的 Q

    【为什么 prefill 必须加 causal mask？】
        模型训练时用了 causal mask（位置 i 只能看 0..i），
        推理时必须保持一致，否则每个位置能看到"未来"的 token，
        算出的 KV 和训练时不同，后续生成就会出错。

    【与旧版逐 token prefill 的对比】
        旧版：for tok in prompt: decode_one_step(tok)  → N 次串行调用
        新版：model.prefill(prompt_tensor)             → 1 次并行计算
        数学结果完全等价，但新版快得多。
    """
    model.eval()

    # prompt token id 列表，例: [8, 6, 10, 4, SEP]
    prompt_tokens = [num_to_tok(n) for n in src_nums] + [SEP]
    # 长度 P = len(src_nums) + 1

    with torch.no_grad():
        # ── Prefill 阶段：一次性处理整个 prompt ──────────────────
        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
        # prompt_tensor: [1, P]   批大小=1（单样本推理）

        logits, cache = model.prefill(prompt_tensor)
        # 内部数据流：
        #   [1, P] → embedding+√D → [1, P, D]
        #           → pos_encoding  → [1, P, D]
        #           → GPTLayer ×2（forward_prefill）→ [1, P, D]
        #           → fc_out(x[:,-1,:]) → [1, V]
        # logits: [1, V]   SEP 位置的输出，即"排序结果第1个token"的预测分布
        # cache:  list of (K[1,H,P,dk], V[1,H,P,dk]) × num_layers

        next_token = logits.argmax(dim=-1)
        # logits.argmax: [1, V] → [1]   取概率最大的 token id

        # ── Decode 阶段：逐 token 自回归生成 ─────────────────────
        generated = []
        pos = len(prompt_tokens)   # 从 prompt 末尾的下一个位置开始（pos=P）

        while pos < len(prompt_tokens) + max_new_tokens:
            if next_token.item() == EOS:
                break   # 遇到 EOS 停止生成

            generated.append(next_token.item())

            logits, cache = model.decode_one_step(next_token, pos, cache)
            # 内部数据流（每步只处理 1 个 token）：
            #   next_token: [1]
            #   → unsqueeze(1):         [1, 1]
            #   → embedding+√D:         [1, 1, D]
            #   → + pe[:, pos:pos+1, :] [1, 1, D]   加第 pos 个位置编码
            #   → GPTLayer ×2（forward_cached）:
            #       Q: [1, H, 1, dk]
            #       K: [1, H, seq_so_far+1, dk]（拼入新K后）
            #       V: [1, H, seq_so_far+1, dk]
            #       scores: [1, H, 1, seq_so_far+1]
            #       output: [1, 1, D]
            #   → fc_out(x[:,0,:]): [1, V]
            # logits: [1, V]   下一个 token 的预测分布
            # cache 中每层 seq 维度 +1

            next_token = logits.argmax(dim=-1)
            # [1, V] → [1]
            pos += 1

    # 过滤掉特殊 token（只保留数字 token，id≥3），转回数字
    return [tok_to_num(t) for t in generated if t >= 3]


# ======================================================================
# 主程序
# ======================================================================
if __name__ == '__main__':
    # ---------- 训练 ----------
    model, min_len, max_len = train()
    device = next(model.parameters()).device

    # ---------- 推理测试 ----------
    print("=" * 50)
    print("推理测试")
    print("=" * 50)

    test_cases = [
        [5, 3, 7, 1, 2],
        [9, 0, 4, 6, 8],
        [2, 2, 5, 1, 1],
        [0, 0, 0, 0, 1],
        [7, 6, 5, 4, 3],
    ]

    all_correct = 0
    for nums in test_cases:
        result   = inference(model, nums, device)
        expected = sorted(nums)
        ok       = result == expected
        all_correct += int(ok)
        print(f"输入:  {nums}")
        print(f"期望:  {expected}")
        print(f"预测:  {result}  {'✓ 正确' if ok else '✗ 错误'}")
        print()

    print(f"最终准确率: {all_correct}/{len(test_cases)}")

    # ---------- 对比两种架构 ----------
    print()
    print("=" * 50)
    print("架构对比（同一个排序任务）")
    print("=" * 50)
    print("""
  Encoder-Decoder（train.py）           Decoder-Only（本文件）
  ─────────────────────────             ──────────────────────
  src → Encoder → memory                全序列 = prompt + SEP + 答案
  tgt → Decoder → 生成                  一个模型处理所有内容

  有 Cross-Attention                     没有 Cross-Attention
  src 和 tgt 物理分离                    拼在一起，用 Causal Mask 区分

  适合：翻译、摘要等                     适合：通用生成（GPT、LLaMA）
  """)
