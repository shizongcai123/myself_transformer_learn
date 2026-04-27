import torch
import numpy as np

class SimpleAttentionWithCache:
    def __init__(self, d_model=512, n_head=8):
        self.d_model = 512  # 模型维度，每个token用512维向量表示.隐藏层肯定是一样的
        self.n_head = 8     # 注意力头数，把512维分成8组
        self.d_k = 64       # 每个头的维度 = 512 / 8
        
        # 投影矩阵：把输入映射到Q、K、V空间
        self.W_q = torch.randn(512, 512)  # Q的权重矩阵
        self.W_k = torch.randn(512, 512)  # K的权重矩阵  
        self.W_v = torch.randn(512, 512)  # V的权重矩阵
        
        # KV Cache存储
        self.k_cache = None  # 缓存所有历史token的K
        self.v_cache = None  # 缓存所有历史token的V

    def prefill(self, x):
        # 假设输入 x: [10, 512]
        # 意思是：10个token，每个token是512维向量
        seq_len = x.shape[0]  # seq_len = 10
        
        # === 步骤1: 计算Q、K、V ===
        Q = x @ self.W_q  # [10, 512] @ [512, 512] = [10, 512]
        K = x @ self.W_k  # [10, 512] -> [10, 512]
        V = x @ self.W_v  # [10, 512] -> [10, 512]
        
        # === 步骤2: 重塑为多头格式 ===
        # 原理解释：
        # [10, 512] 要变成 [10, 8, 64] 意思是：
        #   10个token，每个token有8个头，每个头64维
        Q = Q.view(seq_len, self.n_head, self.d_k)  # [10, 8, 64]
        K = K.view(seq_len, self.n_head, self.d_k)  # [10, 8, 64]
        V = V.view(seq_len, self.n_head, self.d_k)  # [10, 8, 64]
        
        # 转置：把头和序列维度交换
        # 原因：方便批量计算所有头的attention
        Q = Q.transpose(0, 1)  # [8, 10, 64]
        K = K.transpose(0, 1)  # [8, 10, 64]  
        V = V.transpose(0, 1)  # [8, 10, 64]
        
        # === 步骤3: 初始化KV Cache ===
        # 现在K是 [8, 10, 64]，意思是：
        #   8个头，每个头有10个token的K向量，每个向量64维
        self.k_cache = K  # shape: [8, 10, 64]
        self.v_cache = V  # shape: [8, 10, 64]
        
        # === 步骤4: 计算Attention ===
        # Q: [8, 10, 64] 
        # K.transpose(-2, -1): [8, 64, 10]
        # scores: [8, 10, 10] 
        #   含义：对于每个头，10个token之间的注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        # 例如：scores[0, i, j] 表示第0个头中，第i个token对第j个token的注意力
        
        attn_weights = F.softmax(scores, dim=-1)  # [8, 10, 10]
        output = torch.matmul(attn_weights, V)    # [8, 10, 64]
        
        return output


    def decode(self, x):
        # 假设输入 x: [1, 512]
        # 意思是：1个新token，512维向量
        
        # === 步骤1: 只计算新token的Q、K、V ===
        Q = x @ self.W_q  # [1, 512]
        K = x @ self.W_k  # [1, 512]
        V = x @ self.W_v  # [1, 512]
        
        # === 步骤2: 重塑为多头格式 ===
        Q = Q.view(1, self.n_head, self.d_k)      # [1, 8, 64]
        K = K.view(1, self.n_head, self.d_k)      # [1, 8, 64]
        V = V.view(1, self.n_head, self.d_k)      # [1, 8, 64]
        
        Q = Q.transpose(0, 1)  # [8, 1, 64] - 8个头，每个头有1个新token
        K = K.transpose(0, 1)  # [8, 1, 64]
        V = V.transpose(0, 1)  # [8, 1, 64]
        
        # === 步骤3: 更新KV Cache ===
        # 假设之前cache是 [8, 10, 64] (10个历史token)
        # 新K是 [8, 1, 64] (1个新token)
        # cat后变成 [8, 11, 64] (总共11个token)
        self.k_cache = torch.cat([self.k_cache, K], dim=1)  # [8, 11, 64]
        self.v_cache = torch.cat([self.v_cache, V], dim=1)  # [8, 11, 64]
        
        # === 步骤4: 计算Attention（只用新token的Q）===
        # Q: [8, 1, 64] - 只关注新token的查询
        # self.k_cache: [8, 11, 64] - 所有历史token的键
        # scores: [8, 1, 11] - 新token对每个历史token的注意力
        scores = torch.matmul(Q, self.k_cache.transpose(-2, -1)) / (self.d_k ** 0.5)
        # scores[0, 0, j] 表示第0个头中，新token对第j个历史token的注意力
        
        attn_weights = F.softmax(scores, dim=-1)  # [8, 1, 11]
        output = torch.matmul(attn_weights, self.v_cache)  # [8, 1, 64]
        
        return output
    


# 创建模型
model = SimpleAttentionWithCache(d_model=512, n_head=8)

# === Prefill 阶段 ===
print("=== Prefill 阶段 ===")
prompt = torch.randn(10, 512)  # 10个token的prompt embed ing
print(f"输入形状: {prompt.shape}")  # [10, 512]


# === Decode 阶段 ===
output_prefill = model.prefill(prompt)
print(f"Prefill输出形状: {output_prefill.shape}")  # [8, 10, 64]
print(f"KV Cache形状: {model.k_cache.shape}")  # [8, 10, 64]
print()
print("=== Decode 阶段 - 生成第11个token ===")
new_token = torch.randn(1, 512)
print(f"新token形状: {new_token.shape}")  # [1, 512]

output_decode = model.decode(new_token)
print(f"Decode输出形状: {output_decode.shape}")  # [8, 1, 64]
print(f"更新后的KV Cache形状: {model.k_cache.shape}")  # [8, 11, 64]
print()

# === 继续生成 ===
print("=== 生成第12个token ===")
another_token = torch.randn(1, 512)
output_decode2 = model.decode(another_token)
print(f"再次Decode后KV Cache形状: {model.k_cache.shape}")  # [8, 12, 64]