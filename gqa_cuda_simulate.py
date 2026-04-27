"""
GQA CUDA Kernel 的 Python 模拟
================================

本文件用纯 Python 逐步模拟 gqa_cuda_kernel.cu 的逻辑，
可以直接运行对照理解，不需要编译 CUDA。

最后和 model.py 中的 GroupedQueryAttention 做结果对比验证。
"""

import torch
import math


def gqa_cuda_kernel_simulate(Q, K_cache, V_cache, n_head, n_kv_head):
    """
    完全模拟 CUDA kernel 的逻辑（Python 版）

    参数（和 CUDA kernel 一一对应）：
        Q:       [batch, n_head, 1, d_k]      当前 token 的 Query
        K_cache: [batch, n_kv_head, seq_k, d_k] KV Cache 中的 Key
        V_cache: [batch, n_kv_head, seq_k, d_k] KV Cache 中的 Value

    返回:
        output:  [batch, n_head, 1, d_k]
    """
    batch_size = Q.size(0)
    d_k = Q.size(-1)
    seq_k = K_cache.size(2)
    n_rep = n_head // n_kv_head
    scale = 1.0 / math.sqrt(d_k)

    output = torch.zeros_like(Q)  # [batch, n_head, 1, d_k]

    # CUDA: 每个 Block 对应一个 (batch_idx, q_head_idx)
    for batch_idx in range(batch_size):
        for q_head_idx in range(n_head):

            # ★ GQA 核心：一行除法完成头映射 ★
            # CUDA: int kv_head_idx = q_head_idx / n_rep;
            kv_head_idx = q_head_idx // n_rep

            # CUDA: 从共享内存读 Q 向量
            q_vec = Q[batch_idx, q_head_idx, 0, :]          # [d_k]
            # CUDA: 从全局内存读对应 KV 头的数据
            k_mat = K_cache[batch_idx, kv_head_idx, :, :]   # [seq_k, d_k]
            v_mat = V_cache[batch_idx, kv_head_idx, :, :]   # [seq_k, d_k]

            # ============================================
            # Pass 1: 计算 score，找 max
            # ============================================
            # CUDA: 每个线程处理部分 pos，最后 block_reduce_max
            # Python: 我们直接向量化算
            scores = torch.zeros(seq_k)
            for pos in range(seq_k):
                # CUDA: 手动点积
                # for (int d = 0; d < d_k; d++)
                #     score += s_q[d] * k_pos[d];
                score = 0.0
                for d in range(d_k):
                    score += q_vec[d].item() * k_mat[pos, d].item()
                scores[pos] = score * scale

            global_max = scores.max().item()

            # ============================================
            # Pass 2: exp + 加权 V 求和
            # ============================================
            # CUDA: 每个线程累加 local_exp_sum 和 local_v_acc
            #        最后 block_reduce_sum
            exp_sum = 0.0
            v_acc = torch.zeros(d_k)

            for pos in range(seq_k):
                # CUDA: 重新计算 score（用计算换内存）
                score = 0.0
                for d in range(d_k):
                    score += q_vec[d].item() * k_mat[pos, d].item()
                score *= scale

                # exp(score - max)
                exp_val = math.exp(score - global_max)
                exp_sum += exp_val

                # 累加加权 V
                for d in range(d_k):
                    v_acc[d] += exp_val * v_mat[pos, d].item()

            # output = v_acc / exp_sum
            for d in range(d_k):
                output[batch_idx, q_head_idx, 0, d] = v_acc[d] / exp_sum

    return output


def gqa_pytorch_reference(Q, K_cache, V_cache, n_head, n_kv_head):
    """
    用 PyTorch 标准操作实现 GQA（model.py 中 GroupedQueryAttention 的注意力计算部分）
    作为正确性参考

    和 CUDA 模拟的唯一区别：这里用 _repeat_kv + matmul，而非手动循环
    """
    d_k = Q.size(-1)
    n_rep = n_head // n_kv_head
    scale = 1.0 / math.sqrt(d_k)

    # _repeat_kv: [batch, n_kv_head, seq_k, d_k] → [batch, n_head, seq_k, d_k]
    batch, n_kv, seq_k, dk = K_cache.shape
    K_expanded = K_cache.unsqueeze(2).expand(batch, n_kv, n_rep, seq_k, dk).reshape(batch, n_head, seq_k, dk)
    V_expanded = V_cache.unsqueeze(2).expand(batch, n_kv, n_rep, seq_k, dk).reshape(batch, n_head, seq_k, dk)

    # 标准 attention
    scores = torch.matmul(Q, K_expanded.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V_expanded)

    return output


# ======================================================================
# 主程序：运行对比测试
# ======================================================================
if __name__ == '__main__':
    torch.manual_seed(42)

    # 参数（和 model.py / CUDA kernel 一致）
    batch_size = 2
    n_head     = 8
    n_kv_head  = 4
    seq_k      = 10
    d_k        = 16

    # 随机数据
    Q       = torch.randn(batch_size, n_head, 1, d_k)
    K_cache = torch.randn(batch_size, n_kv_head, seq_k, d_k)
    V_cache = torch.randn(batch_size, n_kv_head, seq_k, d_k)

    print("=" * 60)
    print("GQA CUDA Kernel Python 模拟 vs PyTorch 参考实现")
    print("=" * 60)
    print(f"参数: batch={batch_size}, n_head={n_head}, n_kv_head={n_kv_head}, "
          f"seq_k={seq_k}, d_k={d_k}")
    print(f"n_rep={n_head // n_kv_head} (每 {n_head // n_kv_head} 个 Q 头共享 1 组 KV)")
    print()

    # --- 方法1: CUDA kernel 模拟 ---
    out_cuda = gqa_cuda_kernel_simulate(Q, K_cache, V_cache, n_head, n_kv_head)

    # --- 方法2: PyTorch 参考实现 ---
    out_pytorch = gqa_pytorch_reference(Q, K_cache, V_cache, n_head, n_kv_head)

    # --- 对比结果 ---
    max_diff = (out_cuda - out_pytorch).abs().max().item()
    print(f"两种实现的最大误差: {max_diff:.2e}")
    print(f"结果一致: {'YES' if max_diff < 1e-5 else 'NO'}")
    print()

    # --- 展示 GQA 头映射关系 ---
    print("=" * 60)
    print("GQA 头映射验证")
    print("=" * 60)
    print()
    print("Q头  → KV头  (CUDA kernel 中: kv_head_idx = q_head_idx / n_rep)")
    print("-" * 40)
    n_rep = n_head // n_kv_head
    for q_h in range(n_head):
        kv_h = q_h // n_rep
        print(f"  Q头{q_h}  →  KV头{kv_h}")
    print()

    # --- 验证共享同一 KV 头的 Q 头确实读到了相同的 K/V ---
    print("=" * 60)
    print("共享验证：Q头0 和 Q头1 使用的 K 数据是否相同")
    print("=" * 60)
    print()
    # Q头0 和 Q头1 都映射到 KV头0
    # 所以它们的 K 数据相同，但 Q 不同 → 输出不同
    print(f"  batch=0, Q头0 输出前4维: {out_cuda[0, 0, 0, :4].tolist()}")
    print(f"  batch=0, Q头1 输出前4维: {out_cuda[0, 1, 0, :4].tolist()}")
    print(f"  → Q 不同所以输出不同，但它们查询的是同一组 K/V (KV头0)")
    print()
    print(f"  batch=0, Q头0 输出前4维: {out_cuda[0, 0, 0, :4].tolist()}")
    print(f"  batch=0, Q头2 输出前4维: {out_cuda[0, 2, 0, :4].tolist()}")
    print(f"  → Q头0 用 KV头0，Q头2 用 KV头1，K/V 数据不同")
