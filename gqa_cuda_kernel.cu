/**
 * GQA (Grouped Query Attention) 完整 CUDA Kernel
 * ================================================
 *
 * 场景：Decode 阶段（seq_q=1，即每次只推理 1 个新 token）
 *       这是 LLM 推理中最核心的 kernel
 *
 * 【与 Python 版的对应关系】
 *
 *   Python (model.py):
 *       K = self._repeat_kv(K)                    ← 先 expand KV 到 n_head 组
 *       scores = torch.matmul(Q, K.transpose())   ← 用通用 matmul
 *       attn = softmax(scores)
 *       output = attn @ V
 *
 *   CUDA (本文件):
 *       kv_head_idx = q_head_idx / n_rep          ← 不 expand，直接算索引
 *       手动算 dot product、softmax、加权求和       ← 全部融合在一个 kernel 里
 *
 * 【Grid / Block 设计】
 *
 *   Grid:  (batch_size, n_head)
 *          每个 Block 负责一个 (样本, Q头) 的组合
 *
 *   Block: BLOCK_SIZE 个线程（如 256）
 *          线程合作处理 seq_k 个 K 位置
 *          线程 tid 负责位置 tid, tid+256, tid+512, ...
 *
 * 【算法：两遍扫描】
 *
 *   为什么不能一遍搞定？因为 softmax 需要全局信息：
 *       softmax(s_i) = exp(s_i - max) / sum(exp(s_j - max))
 *       必须先知道 max 和 sum，才能算每个位置的权重
 *
 *   但可以用一个数学技巧把 3 遍优化到 2 遍：
 *       output = sum(weight_i * V_i)
 *              = sum(exp(s_i - max) / sum_exp * V_i)
 *              = sum(exp(s_i - max) * V_i) / sum_exp
 *                ↑ 分子和分母可以同时算！
 *
 *   所以：
 *     第一遍：算 score，求 max          （需要 block 级 reduce）
 *     第二遍：算 exp(s-max)，同时累加：
 *              - 分母 sum_exp
 *              - 分子 sum(exp(s-max) * V)
 *             最后 output = 分子 / 分母
 */

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define WARP_SIZE  32
#define MAX_D_K    128   // 支持的最大 d_k（寄存器预分配用）


// ================================================================
// 1. Warp 级规约原语
// ================================================================
//
// 【什么是 Warp？】
// GPU 的最小调度单位，32 个线程为一组，它们物理上同步执行
// Warp 内的线程可以通过 __shfl 指令直接交换寄存器数据，无需共享内存
//
// 【__shfl_xor_sync 蝴蝶规约】
// 每次让 "距离" 为 offset 的两个线程交换数据:
//
//   offset=16: 线程0↔16, 1↔17, 2↔18, ... 15↔31
//   offset=8:  线程0↔8,  1↔9,  2↔10, ...
//   offset=4:  线程0↔4,  1↔5,  ...
//   offset=2:  线程0↔2,  1↔3,  ...
//   offset=1:  线程0↔1,  2↔3,  ...
//
//   5 轮后，所有 32 个线程都持有相同的规约结果
//   (xor 的特点：每个线程都参与交换，所以结果广播给所有线程)

__device__ __forceinline__ float warp_reduce_max(float val) {
    // 0xffffffff = 所有 32 个 lane 都参与
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;  // 所有线程都拿到了 warp 内的最大值
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;  // 所有线程都拿到了 warp 内的总和
}


// ================================================================
// 2. Block 级规约
// ================================================================
//
// Block 有多个 Warp（如 256 线程 = 8 个 Warp）
// 需要两级规约：先 Warp 内，再 Warp 间
//
// 方法：
//   Step 1: 每个 Warp 内部规约 → 每个 Warp 得到一个局部结果
//   Step 2: 每个 Warp 的结果写入共享内存
//   Step 3: 第一个 Warp 读取所有 Warp 的结果，再规约一次
//   Step 4: 通过共享内存广播给所有线程

__device__ float block_reduce_max(float val, float* smem_scratch) {
    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;       // 第几个 Warp（0~7）
    int lane_id = tid % WARP_SIZE;       // Warp 内第几个线程（0~31）
    int num_warps = blockDim.x / WARP_SIZE;

    // Step 1: Warp 内规约
    val = warp_reduce_max(val);

    // Step 2: 每个 Warp 的 lane 0 写入共享内存
    if (lane_id == 0) {
        smem_scratch[warp_id] = val;
    }
    __syncthreads();  // 等所有 Warp 写完

    // Step 3: 第一个 Warp 读取所有 Warp 的结果，再做一次规约
    // 只有前 num_warps 个线程有有效数据
    val = (tid < num_warps) ? smem_scratch[tid] : -FLT_MAX;
    if (warp_id == 0) {
        val = warp_reduce_max(val);
    }

    // Step 4: 线程 0 写回共享内存，所有线程读取
    if (tid == 0) smem_scratch[0] = val;
    __syncthreads();
    return smem_scratch[0];  // 所有线程都拿到 Block 内的最大值
}

__device__ float block_reduce_sum(float val, float* smem_scratch) {
    int tid     = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int num_warps = blockDim.x / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane_id == 0) smem_scratch[warp_id] = val;
    __syncthreads();

    val = (tid < num_warps) ? smem_scratch[tid] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);

    if (tid == 0) smem_scratch[0] = val;
    __syncthreads();
    return smem_scratch[0];
}


// ================================================================
// 3. GQA Decode Attention Kernel（主体）
// ================================================================

__global__ void gqa_decode_attention_kernel(
    const float* __restrict__ Q,         // [batch, n_head, d_k]
    const float* __restrict__ K_cache,   // [batch, n_kv_head, seq_k, d_k]
    const float* __restrict__ V_cache,   // [batch, n_kv_head, seq_k, d_k]
    float*       __restrict__ output,    // [batch, n_head, d_k]
    const int n_head,       // Q 头数，如 8
    const int n_kv_head,    // KV 头数，如 4
    const int seq_k,        // 缓存中的 token 数
    const int d_k           // 每头维度，如 16
) {
    // ---- 线程身份 ----
    const int batch_idx  = blockIdx.x;   // 第几个样本
    const int q_head_idx = blockIdx.y;   // 第几个 Q 头
    const int tid        = threadIdx.x;  // Block 内线程编号

    // ================================================================
    // ★ GQA 核心：一行除法完成头映射 ★
    // ================================================================
    //
    //   n_head=8, n_kv_head=4 → n_rep=2
    //
    //   q_head_idx:  0  1  2  3  4  5  6  7
    //   kv_head_idx: 0  0  1  1  2  2  3  3
    //                ↑  ↑
    //        Q头0和Q头1 → 都去读 KV头0 的数据
    //
    //   vLLM 不需要像 Python 那样 expand 出 8 组 K/V，
    //   而是每个线程自己算出该去读哪一组。
    //
    const int n_rep       = n_head / n_kv_head;
    const int kv_head_idx = q_head_idx / n_rep;


    // ---- 指针计算 ----
    //
    // 一维内存中定位多维数组元素:
    //   Q[batch_idx][q_head_idx][dk_i]
    //     = Q + (batch_idx * n_head + q_head_idx) * d_k + dk_i
    //
    //   K_cache[batch_idx][kv_head_idx][pos][dk_i]
    //     = K_cache + ((batch_idx * n_kv_head + kv_head_idx) * seq_k + pos) * d_k + dk_i
    //
    const float* q_ptr  = Q       + (batch_idx * n_head    + q_head_idx)  * d_k;
    const float* k_base = K_cache + (batch_idx * n_kv_head + kv_head_idx) * seq_k * d_k;
    const float* v_base = V_cache + (batch_idx * n_kv_head + kv_head_idx) * seq_k * d_k;
    float*       o_ptr  = output  + (batch_idx * n_head    + q_head_idx)  * d_k;

    const float scale = 1.0f / sqrtf((float)d_k);


    // ---- 共享内存布局 ----
    //
    // 总大小: d_k + (BLOCK_SIZE/WARP_SIZE) 个 float
    //
    // |<------- d_k ------->|<-- num_warps -->|
    // [   Q 向量的副本        | 规约暂存区      ]
    //
    extern __shared__ float smem[];
    float* s_q      = smem;           // [d_k]      Q 向量，所有线程共享读取
    float* s_reduce = smem + d_k;     // [num_warps] Block 规约暂存


    // ---- 把 Q 向量从全局内存加载到共享内存 ----
    //
    // 原因：Q 在 Pass 1 和 Pass 2 都要用，每次都从全局内存读太慢
    // 多个线程合作加载（如 d_k=16，线程 0~15 各加载一个元素）
    //
    for (int i = tid; i < d_k; i += blockDim.x) {
        s_q[i] = q_ptr[i];
    }
    __syncthreads();


    // ================================================================
    // Pass 1：计算所有 score，找全局 max
    // ================================================================
    //
    // 每个线程负责一部分 K 位置（跨步访问）：
    //   线程 0:   pos = 0, 256, 512, ...
    //   线程 1:   pos = 1, 257, 513, ...
    //   线程 255: pos = 255, 511, 767, ...
    //
    // 每个线程维护自己负责位置中的局部最大值
    // 最后通过 block_reduce_max 合并得到全局最大值
    //
    float local_max = -FLT_MAX;

    for (int pos = tid; pos < seq_k; pos += blockDim.x) {
        // 计算 score = Q · K[pos] / sqrt(d_k)
        // 手动点积（不用 matmul，因为只有 1 个 Q 和 1 个 K 向量）
        float score = 0.0f;
        const float* k_pos = k_base + pos * d_k;  // K_cache 中第 pos 个 token
        for (int d = 0; d < d_k; d++) {
            score += s_q[d] * k_pos[d];
        }
        score *= scale;

        local_max = fmaxf(local_max, score);
    }

    // Block 内所有线程合作，找到全局 max
    //
    // 例: 256个线程各自有一个 local_max
    //     → warp 内规约（32个→1个）得到 8 个值
    //     → warp 间规约（8个→1个）得到全局 max
    //     → 广播回所有 256 个线程
    //
    float global_max = block_reduce_max(local_max, s_reduce);


    // ================================================================
    // Pass 2：计算 softmax 权重 + 加权 V 求和（合并两步！）
    // ================================================================
    //
    // 数学原理：
    //   output = Σ_pos [ softmax(score_pos) * V_pos ]
    //          = Σ_pos [ exp(s_pos - max) / Σ_j exp(s_j - max)  *  V_pos ]
    //          = Σ_pos [ exp(s_pos - max) * V_pos ]  /  Σ_j exp(s_j - max)
    //            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^       ^^^^^^^^^^^^^^^^^^^^
    //                     分子（向量）                        分母（标量）
    //
    //   分子和分母可以在同一遍扫描中同时累加！不需要第三遍
    //
    float local_exp_sum = 0.0f;           // 分母的局部累加
    float local_v_acc[MAX_D_K];           // 分子的局部累加（d_k 维向量）
    for (int d = 0; d < d_k; d++) {
        local_v_acc[d] = 0.0f;
    }

    for (int pos = tid; pos < seq_k; pos += blockDim.x) {
        // 重新计算 score（用计算换内存，不存中间结果）
        //
        // 为什么要重算？
        //   如果 seq_k=4096, 要存 4096 个 float 的 score → 16KB per thread
        //   256 个线程 → 4MB 共享内存，远超 GPU 限制（通常 48~100KB）
        //   所以选择"重算 score"而非"存 score"，这也是 FlashAttention 的核心思想
        //
        float score = 0.0f;
        const float* k_pos = k_base + pos * d_k;
        for (int d = 0; d < d_k; d++) {
            score += s_q[d] * k_pos[d];
        }
        score *= scale;

        // exp(score - max)：减去 max 防止溢出
        //   如果不减 max：exp(100) → inf（溢出）
        //   减了 max：    exp(100 - 100) = exp(0) = 1（安全）
        //
        float exp_val = expf(score - global_max);

        // 累加分母
        local_exp_sum += exp_val;

        // 累加分子：exp(s-max) * V[pos]（逐维度）
        const float* v_pos = v_base + pos * d_k;
        for (int d = 0; d < d_k; d++) {
            local_v_acc[d] += exp_val * v_pos[d];
        }
    }

    // ---- Block 规约：合并所有线程的局部结果 ----

    // 分母：所有线程的 local_exp_sum 求和
    float global_exp_sum = block_reduce_sum(local_exp_sum, s_reduce);

    // 分子：对 d_k 的每个维度，所有线程的 local_v_acc[d] 求和
    // 然后除以分母，得到最终输出
    //
    // 例如 d_k=16，需要做 16 次 block 规约
    // 对于 d_k=128 的大模型，这里是性能瓶颈
    // 生产代码会用更精巧的规约方式（如 warp shuffle + 寄存器重排）
    //
    for (int d = 0; d < d_k; d++) {
        float global_v = block_reduce_sum(local_v_acc[d], s_reduce);

        // 只有线程 0 写入最终结果
        if (tid == 0) {
            o_ptr[d] = global_v / global_exp_sum;
        }
        // 必须同步，因为下一轮 block_reduce_sum 会复用同一块共享内存
        __syncthreads();
    }
}


// ================================================================
// 4. Host 端启动函数
// ================================================================

void launch_gqa_decode_attention(
    const float* Q,
    const float* K_cache,
    const float* V_cache,
    float* output,
    int batch_size,
    int n_head,
    int n_kv_head,
    int seq_k,
    int d_k
) {
    // Grid: 每个 Block 处理一个 (batch, q_head) 组合
    dim3 grid(batch_size, n_head);
    dim3 block(BLOCK_SIZE);

    // 共享内存大小 = Q 向量 + 规约暂存
    int num_warps = BLOCK_SIZE / WARP_SIZE;
    int smem_size = (d_k + num_warps) * sizeof(float);

    gqa_decode_attention_kernel<<<grid, block, smem_size>>>(
        Q, K_cache, V_cache, output,
        n_head, n_kv_head, seq_k, d_k
    );
}


// ================================================================
// 5. 测试主函数
// ================================================================

int main() {
    // 参数设置（和 model.py 中的 GQA 对齐）
    const int batch_size = 2;
    const int n_head     = 8;
    const int n_kv_head  = 4;
    const int seq_k      = 10;   // KV cache 中有 10 个历史 token
    const int d_k        = 16;   // 每头维度

    // 分配 Host 内存
    int q_size = batch_size * n_head * d_k;
    int k_size = batch_size * n_kv_head * seq_k * d_k;
    int v_size = k_size;
    int o_size = q_size;

    float* h_Q      = (float*)malloc(q_size * sizeof(float));
    float* h_K      = (float*)malloc(k_size * sizeof(float));
    float* h_V      = (float*)malloc(v_size * sizeof(float));
    float* h_output = (float*)malloc(o_size * sizeof(float));

    // 初始化随机数据
    srand(42);
    for (int i = 0; i < q_size; i++) h_Q[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < k_size; i++) h_K[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < v_size; i++) h_V[i] = (float)rand() / RAND_MAX - 0.5f;

    // 分配 Device 内存
    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q,      q_size * sizeof(float));
    cudaMalloc(&d_K,      k_size * sizeof(float));
    cudaMalloc(&d_V,      v_size * sizeof(float));
    cudaMalloc(&d_output, o_size * sizeof(float));

    // Host → Device
    cudaMemcpy(d_Q, h_Q, q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, k_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, v_size * sizeof(float), cudaMemcpyHostToDevice);

    // 启动 kernel
    launch_gqa_decode_attention(d_Q, d_K, d_V, d_output,
                                batch_size, n_head, n_kv_head, seq_k, d_k);

    // Device → Host
    cudaMemcpy(h_output, d_output, o_size * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印部分结果
    printf("=== GQA CUDA Kernel Output ===\n");
    printf("batch=0, q_head=0, first 4 dims: ");
    for (int d = 0; d < 4; d++) {
        printf("%.4f ", h_output[d]);
    }
    printf("\n");

    printf("batch=0, q_head=1, first 4 dims: ");
    for (int d = 0; d < 4; d++) {
        printf("%.4f ", h_output[d_k + d]);
    }
    printf("\n");

    printf("\n(Q头0 和 Q头1 读的是同一组 KV头0，但 Q 不同所以结果不同)\n");

    // 清理
    free(h_Q); free(h_K); free(h_V); free(h_output);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_output);

    return 0;
}
