"""
任务：序列排序（Sequence Sorting）
================================================
选择这个任务的理由：
  - 零外部数据依赖，随机生成即可
  - 任务本身极简单，便于验证模型是否工作正常
  - 完美展示编码器-解码器架构的数据流

数据示例：
  输入 src: [BOS, 5, 3, 7, 1, 2, EOS]
  输出 tgt: [BOS, 1, 2, 3, 5, 7, EOS]

训练时（Teacher Forcing）：
  解码器输入 tgt_in:  [BOS, 1, 2, 3, 5, 7]     ← 去掉最后的 EOS
  解码器标签 tgt_out: [1, 2, 3, 5, 7, EOS]     ← 去掉第一个 BOS
  模型在位置 i 看到 tgt_in[0..i]，预测 tgt_out[i]

推理时（自回归）：
  step 1: 输入 [BOS]           → 预测 1
  step 2: 输入 [BOS, 1]        → 预测 2
  step 3: 输入 [BOS, 1, 2]     → 预测 3
  ...直到预测出 EOS
"""

import torch
import torch.nn as nn
import random
from model import Transformer, make_src_mask, make_tgt_mask

# ======================================================================
# 词表设计
# ======================================================================
PAD = 0   # 填充符：对齐不同长度的序列（本任务序列等长，主要演示用）
BOS = 1   # Begin Of Sequence：句子开始符
EOS = 2   # End Of Sequence：句子结束符
# 数字 0-9 → token id 3-12（偏移 3 避开特殊符号）
VOCAB_SIZE = 13  # 0(PAD) + 1(BOS) + 2(EOS) + 10个数字

def num_to_tok(n: int) -> int:
    """数字 0-9 → token id 3-12"""
    return n + 3

def tok_to_num(t: int) -> int:
    """token id 3-12 → 数字 0-9"""
    return t - 3


# ======================================================================
# 数据生成
# ======================================================================
def generate_batch(batch_size: int, min_len: int = 3, max_len: int = 8):
    """
    随机生成一批【变长】排序任务数据，批次内用 PAD 对齐到最长序列。

    每条样本的序列长度在 [min_len, max_len] 之间随机选取，
    因此同一批次内各样本长度不同，需要补 PAD 对齐。

    例子（batch 内两个样本，min_len=3, max_len=5）：
        样本1: seq_len=4, 数字=[5,3,7,1]
            src:     [BOS, 5, 3, 7, 1, EOS]          长度6
            tgt_in:  [BOS, 1, 3, 5, 7]               长度5
            tgt_out: [1, 3, 5, 7, EOS]               长度5

        样本2: seq_len=2, 数字=[9,2]
            src:     [BOS, 9, 2, EOS]                 长度4
            tgt_in:  [BOS, 2, 9]                      长度3
            tgt_out: [2, 9, EOS]                      长度3

        补 PAD 对齐后：
            src:     [[BOS,5,3,7,1,EOS],              [batch=2, src_len=6]
                      [BOS,9,2,EOS,PAD,PAD]]
            tgt_in:  [[BOS,1,3,5,7],                  [batch=2, tgt_len=5]
                      [BOS,2,9,PAD,PAD]]
            tgt_out: [[1,3,5,7,EOS],                  [batch=2, tgt_len=5]
                      [2,9,EOS,PAD,PAD]]

    PAD 位置由 src_mask / tgt_mask 自动屏蔽，loss 也通过 ignore_index=PAD 忽略。
    """
    srcs, tgt_ins, tgt_outs = [], [], []

    for _ in range(batch_size):
        seq_len     = random.randint(min_len, max_len)   # ← 每条样本随机长度
        nums        = [random.randint(0, 9) for _ in range(seq_len)]
        sorted_nums = sorted(nums)

        src      = [BOS] + [num_to_tok(n) for n in nums]        + [EOS]
        tgt_full = [BOS] + [num_to_tok(n) for n in sorted_nums] + [EOS]

        tgt_in  = tgt_full[:-1]
        tgt_out = tgt_full[1:]

        srcs.append(src)
        tgt_ins.append(tgt_in)
        tgt_outs.append(tgt_out)

    # 批次内补 PAD 到最长序列
    def pad_batch(seqs: list) -> list:
        max_l = max(len(s) for s in seqs)
        return [s + [PAD] * (max_l - len(s)) for s in seqs]

    return (
        torch.tensor(pad_batch(srcs),     dtype=torch.long),  # [batch, max_src_len]
        torch.tensor(pad_batch(tgt_ins),  dtype=torch.long),  # [batch, max_tgt_len]
        torch.tensor(pad_batch(tgt_outs), dtype=torch.long),  # [batch, max_tgt_len]
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
    D_MODEL    = 64
    N_HEAD     = 4
    NUM_LAYERS = 2
    D_FF       = 128
    DROPOUT    = 0.1
    BATCH_SIZE = 128
    NUM_STEPS  = 2000
    LR         = 1e-3

    # ---------- 模型初始化 ----------
    model = Transformer(
        vocab_size  = VOCAB_SIZE,
        d_model     = D_MODEL,
        n_head      = N_HEAD,
        num_layers  = NUM_LAYERS,
        d_ff        = D_FF,
        dropout     = DROPOUT,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}\n")

    # 损失函数：多分类交叉熵
    # ignore_index=PAD：PAD 位置不计入损失（本任务无PAD，但规范写法）
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    # Adam 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("=" * 50)
    print("开始训练（任务：对随机数字序列排序）")
    print("=" * 50)

    for step in range(1, NUM_STEPS + 1):
        model.train()

        # ---- 1. 生成数据 ----
        src, tgt_in, tgt_out = generate_batch(BATCH_SIZE, MIN_LEN, MAX_LEN)
        src     = src.to(device)      # [batch, src_len]
        tgt_in  = tgt_in.to(device)   # [batch, tgt_len]
        tgt_out = tgt_out.to(device)  # [batch, tgt_len]

        # ---- 2. 生成 Mask ----
        # src_mask: [batch, 1, 1, src_len]  屏蔽 src 中的 PAD
        # tgt_mask: [batch, 1, tgt_len, tgt_len]  因果 mask + 屏蔽 tgt PAD.一个倒三角的矩阵
        src_mask = make_src_mask(src).to(device)
        tgt_mask = make_tgt_mask(tgt_in).to(device)

        # ---- 3. 前向传播 ----
        # logits: [batch, tgt_len, vocab_size]
        # 含义：对于目标序列的每个位置，模型预测下一个 token 的概率分布
        logits = model(src, tgt_in, src_mask, tgt_mask)

        # ---- 4. 计算损失 ----
        # 交叉熵期望输入:
        #   predictions: [N, vocab_size]
        #   labels:      [N]
        # 所以把 [batch, tgt_len, vocab_size] 展平成 [batch*tgt_len, vocab_size]
        #        [batch, tgt_len] 展平成 [batch*tgt_len]
        loss = criterion(
            logits.view(-1, VOCAB_SIZE),   # [batch*tgt_len, vocab_size]
            tgt_out.view(-1)               # [batch*tgt_len]
        )

        # ---- 5. 反向传播 + 参数更新 ----
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪：防止梯度爆炸（训练 Transformer 时的常见技巧）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 200 == 0:
            acc = evaluate_accuracy(model, device, MIN_LEN, MAX_LEN)
            print(f"Step {step:5d} | Loss: {loss.item():.4f} | 排序准确率: {acc:.1%}")

    print("\n训练完成！\n")
    return model, MIN_LEN, MAX_LEN


# ======================================================================
# 准确率评估（贪心解码）
# ======================================================================
def evaluate_accuracy(model, device, min_len=3, max_len=8, n_samples=200):
    """
    随机生成不同长度的样本，验证模型在变长输入上的准确率
    """
    correct = 0
    for _ in range(n_samples):
        seq_len = random.randint(min_len, max_len)
        nums    = [random.randint(0, 9) for _ in range(seq_len)]
        result  = inference(model, nums, device)
        if result == sorted(nums):
            correct += 1
    return correct / n_samples


# ======================================================================
# 推理（真正的自回归解码）
# ======================================================================
def inference(model, src_nums: list, device, max_len: int = 30) -> list:
    """
    自回归推理：每次只生成一个 token，把它加入输入再生成下一个

    【与训练的本质区别】
    训练：一次性喂入完整 tgt，用 causal mask 模拟自回归（并行计算，高效）
    推理：没有"答案"，必须真正逐步生成（串行，每步依赖上一步结果）

    数据流（src=[5,3,1]，期望输出[1,3,5]）：

      Step 0: 编码器处理 src=[BOS,5,3,1,EOS] → memory (只做一次！)

      Step 1: 解码器输入=[BOS]
              → 预测 logits[0] → argmax → token: 1
              → 已生成: [BOS, 1]

      Step 2: 解码器输入=[BOS, 1]
              → 预测 logits[1] → argmax → token: 3
              → 已生成: [BOS, 1, 3]

      Step 3: 解码器输入=[BOS, 1, 3]
              → 预测 logits[2] → argmax → token: 5
              → 已生成: [BOS, 1, 3, 5]

      Step 4: 解码器输入=[BOS, 1, 3, 5]
              → 预测 logits[3] → argmax → token: EOS
              → 停止！返回 [1, 3, 5]

    注意：每一步用的都是截至当前的完整序列（不是只用最新的 token）
    这就是为什么 KV Cache 能优化：历史 token 的 K/V 不需要重复计算。

    src_nums: 输入数字列表，例如 [5, 3, 7, 1, 2]
    返回:     预测的输出数字列表，例如 [1, 2, 3, 5, 7]
    """
    model.eval()

    # 构建 src tensor: [1, src_len]
    src_tokens = [BOS] + [num_to_tok(n) for n in src_nums] + [EOS]
    src        = torch.tensor([src_tokens], dtype=torch.long).to(device)
    src_mask   = make_src_mask(src).to(device)

    with torch.no_grad():
        # ---- 编码阶段（只做一次）----
        # memory: [1, src_len, d_model]
        memory = model.encode(src, src_mask)

        # ---- 自回归解码阶段 ----
        # 初始化：解码器只有 BOS
        generated = [BOS]

        for step in range(max_len):
            # 当前已生成序列 → tensor: [1, current_len]
            tgt_tensor = torch.tensor([generated], dtype=torch.long).to(device)
            tgt_mask   = make_tgt_mask(tgt_tensor).to(device)

            # 解码：得到所有已生成位置的特征
            # dec_out: [1, current_len, d_model]
            dec_out = model.decode(tgt_tensor, memory, src_mask, tgt_mask)

            # 只取最后一个位置的特征来预测"下一个 token"
            # dec_out[:, -1, :]: [1, d_model]
            # logits: [1, vocab_size]
            logits = model.fc_out(dec_out[:, -1, :])

            # 贪心解码：取概率最大的 token
            # argmax(-1) 在 vocab_size 维度取最大值
            next_token = logits.argmax(dim=-1).item()

            generated.append(next_token)

            # 遇到 EOS 停止
            if next_token == EOS:
                break

        # 去掉首尾的 BOS/EOS，把 token id 转回数字
        result = []
        for t in generated[1:]:     # 跳过 BOS
            if t == EOS:
                break
            if t >= 3:              # 合法数字 token
                result.append(tok_to_num(t))

    return result


# ======================================================================
# 带 KV Cache 的推理（对比版）
# ======================================================================
def inference_with_cache(model, src_nums: list, device, max_len: int = 30) -> list:
    """
    带 KV Cache 的自回归推理。

    【与原版 inference 的对比】

    原版 inference（无 cache）：
        每步把完整的已生成序列重新喂给解码器
        step1: decode([BOS])              → 算1个token的K/V
        step2: decode([BOS, t1])          → 重新算2个token的K/V（t0重复算了！）
        step3: decode([BOS, t1, t2])      → 重新算3个token的K/V（t0,t1重复算了！）
        计算量：1 + 2 + 3 + ... + n = O(n²)

    本版 inference_with_cache（有 cache）：
        每步只处理1个新token，历史K/V从cache读取
        step1: decode_one_step(BOS, pos=0) → 算1个token，存入cache
        step2: decode_one_step(t1,  pos=1) → 只算新token，读cache
        step3: decode_one_step(t2,  pos=2) → 只算新token，读cache
        计算量：1 + 1 + 1 + ... = O(n)

    src_nums: 输入数字列表，例如 [5, 3, 7, 1, 2]
    """
    model.eval()

    # 构建 src tensor: [1, src_len]
    src_tokens = [BOS] + [num_to_tok(n) for n in src_nums] + [EOS]
    
    src        = torch.tensor([src_tokens], dtype=torch.long).to(device)
    src_mask   = make_src_mask(src).to(device)

    with torch.no_grad():
        # ---- Prefill 阶段 ----
        # 1. 编码器处理源序列（和无 cache 版本相同，只做一次）
        memory = model.encode(src, src_mask)  # [1, src_len, d_model]   输入prompt得到输出表示，就结束了。后续不管 decoder 生成多少个 token，encoder 都不需要再跑第二次。所以有KV计算，但是不用cache住

        # 2. 根据 memory 初始化 KV Cache
        #    - 预计算所有层的 cross-attention K/V（来自 memory）
        #    - 初始化空的 self-attention cache
        cache = model.build_cache(memory)  # 这里使用encoder的输出来重新计算交叉注意力，不是复用encoder那边的kv权重

        # ---- Decode 阶段 ----
        # 当前 token 从 BOS 开始
        current_token = torch.tensor([BOS], dtype=torch.long).to(device)  # [1]
        generated     = []

        for pos in range(max_len):
            # 只处理 1 个 token，读/写 cache
            # logits: [1, vocab_size]
            logits, cache = model.decode_one_step(current_token, pos, cache)

            # 贪心：取概率最大的 token
            next_token = logits.argmax(dim=-1)  # [1]

            if next_token.item() == EOS:
                break

            generated.append(next_token.item())
            current_token = next_token  # 作为下一步的输入

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
    print("推理测试（自回归逐步生成）")
    print("=" * 50)

    test_cases = [
        [5, 3, 7, 1, 2, 9],
        [9, 0, 4, 6, 8, 2],
        [2, 2, 5, 1, 1, 3],
        [0, 0, 0, 0, 0, 1],
        [7, 6, 5, 4, 3, 2],
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

    # ---------- 对比有无 KV Cache ----------
    print()
    print("=" * 50)
    print("有无 KV Cache 结果对比")
    print("=" * 50)
    print(f"{'输入':<25} {'无Cache':<15} {'有Cache':<15} {'一致'}")
    print("-" * 60)
    for nums in test_cases:
        r1 = inference(model, nums, device)
        r2 = inference_with_cache(model, nums, device)
        match = "✓" if r1 == r2 else "✗"
        print(f"{str(nums):<25} {str(r1):<15} {str(r2):<15} {match}")

    # ---------- 自回归过程可视化 ----------
    print()
    print("=" * 50)
    print("自回归解码过程逐步展示")
    print("=" * 50)

    demo_input = [4, 1, 7, 2, 5, 3]
    print(f"输入序列: {demo_input}")
    print(f"期望输出: {sorted(demo_input)}\n")
    print("逐步生成过程：")

    model.eval()
    src_tokens = [BOS] + [num_to_tok(n) for n in demo_input] + [EOS]
    src        = torch.tensor([src_tokens], dtype=torch.long).to(device)
    src_mask   = make_src_mask(src).to(device)

    with torch.no_grad():
        memory    = model.encode(src, src_mask)
        generated = [BOS]

        for step in range(20):
            tgt_tensor = torch.tensor([generated], dtype=torch.long).to(device)
            tgt_mask   = make_tgt_mask(tgt_tensor).to(device)
            dec_out    = model.decode(tgt_tensor, memory, src_mask, tgt_mask)
            logits     = model.fc_out(dec_out[:, -1, :])
            next_token = logits.argmax(dim=-1).item()

            # 可读化展示
            def tok_name(t):
                if t == BOS: return "BOS"
                if t == EOS: return "EOS"
                if t == PAD: return "PAD"
                return str(tok_to_num(t))

            input_str = "[" + ", ".join(tok_name(t) for t in generated) + "]"
            pred_str  = tok_name(next_token)
            print(f"  Step {step+1}: 解码器输入={input_str:30s} → 预测={pred_str}")

            generated.append(next_token)
            if next_token == EOS:
                break
