import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. 基础组件：多头注意力 (自注意力 / 交叉注意力共用)
# ==========================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_head=8):
        super().__init__()
        self.d_model = d_model  # 模型基础维度
        self.n_head = n_head    # 注意力头数
        self.d_k = d_model // n_head # 每个头的维度: 512 / 8 = 64
        
        # 权重矩阵大小: [输入维度, 输出维度] -> [512, 512]
        # 意义：将输入的 512 维特征，线性映射到一个新的 512 维空间，用于后续拆分成多头
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出映射矩阵大小: [512, 512]
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        # 假设输入 q, k, v 的形状: [batch_size, seq_len, d_model]
        # 参数意义: batch_size 是批次大小，seq_len 是序列长度 (句子里的词数)，d_model 是每个词的向量维度
        batch_size = q.size(0)
        seq_q = q.size(1) # Query 的序列长度
        seq_k = k.size(1) # Key/Value 的序列长度 (Cross-Attention 时，seq_q 和 seq_k 可能不同)
        
        # 1. 线性映射
        # [batch_size, seq_q, 512] @ [512, 512]^T -> [batch_size, seq_q, 512]
        Q = self.W_q(q) 
        # [batch_size, seq_k, 512] @ [512, 512]^T -> [batch_size, seq_k, 512]
        K = self.W_k(k) 
        # [batch_size, seq_k, 512] @ [512, 512]^T -> [batch_size, seq_k, 512]
        V = self.W_v(v) 
        
        # 2. 切分多头并转置
        # 动作分解:
        #   a. view: [batch_size, seq_q, 512] -> [batch_size, seq_q, 8, 64]
        #      (将 512 维拆成 8 个头，每个头 64 维)
        #   b. transpose: [batch_size, seq_q, 8, 64] -> [batch_size, 8, seq_q, 64]
        #      (把头数提到前面，方便后续对每个头独立进行矩阵乘法)
        Q = Q.view(batch_size, seq_q, self.n_head, self.d_k).transpose(1, 2) # [batch_size, 8, seq_q, 64]
        K = K.view(batch_size, seq_k, self.n_head, self.d_k).transpose(1, 2) # [batch_size, 8, seq_k, 64]
        V = V.view(batch_size, seq_k, self.n_head, self.d_k).transpose(1, 2) # [batch_size, 8, seq_k, 64]
        
        # 3. 计算注意力分数: Q * K^T / sqrt(d_k)
        # Q: [batch_size, 8, seq_q, 64]
        # K.transpose(-2, -1): [batch_size, 8, 64, seq_k] (交换最后两个维度)
        # 矩阵乘法: [seq_q, 64] @ [64, seq_k] -> [seq_q, seq_k]
        # scores 结果形状: [batch_size, 8, seq_q, seq_k]
        # 意义: 8个头中，每个头的 seq_q 个 token 对 seq_k 个 token 的注意力打分矩阵
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # mask 形状通常为 [1, 1, seq_q, seq_k] 以支持广播机制
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # 4. Softmax 与 V 相乘
        # 在最后一个维度 (seq_k) 上做归一化
        attn_weights = F.softmax(scores, dim=-1)  # 形状不变: [batch_size, 8, seq_q, seq_k]
        
        # attn_weights: [batch_size, 8, seq_q, seq_k]
        # V: [batch_size, 8, seq_k, 64]
        # 矩阵乘法: [seq_q, seq_k] @ [seq_k, 64] -> [seq_q, 64]
        # output 结果形状: [batch_size, 8, seq_q, 64]
        output = torch.matmul(attn_weights, V)    
        
        # 5. 拼接多头并做最后映射
        # transpose: [batch_size, 8, seq_q, 64] -> [batch_size, seq_q, 8, 64]
        # contiguous().view(): 把内存整理连续后，合并最后两维 [8, 64] -> [512]
        # 结果形状: [batch_size, seq_q, 512]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_q, self.d_model)
        
        # 最终经过全连接层映射
        # [batch_size, seq_q, 512] @ [512, 512]^T -> [batch_size, seq_q, 512]
        return self.out_proj(output)

# ==========================================
# 2. 基础组件：前馈神经网络 (FFN)
# ==========================================
class FeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        super().__init__()
        # 参数意义: 
        # linear1 权重矩阵: [d_model, d_ff] -> [512, 2048] (将特征升维，提取更丰富的特征)
        self.linear1 = nn.Linear(d_model, d_ff) 
        # linear2 权重矩阵: [d_ff, d_model] -> [2048, 512] (将特征降维回原大小，保证输入输出维度一致)
        self.linear2 = nn.Linear(d_ff, d_model) 
        
    def forward(self, x):
        # x 输入形状: [batch_size, seq_len, 512]
        # 经过 linear1: [batch_size, seq_len, 512] @ [512, 2048]^T -> [batch_size, seq_len, 2048]
        # 经过 relu 激活函数不改变形状: [batch_size, seq_len, 2048]
        # 经过 linear2: [batch_size, seq_len, 2048] @ [2048, 512]^T -> [batch_size, seq_len, 512]
        return self.linear2(F.relu(self.linear1(x)))

# ==========================================
# 3. 核心积木：编码器层 (Encoder Layer)
# ==========================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, n_head=8, d_ff=2048):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = FeedForward(d_model, d_ff)
        # LayerNorm 内置参数: gamma(缩放) 和 beta(平移)，形状皆为 [d_model] -> [512]
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, src_mask):
        # x 输入形状: [batch_size, src_len, 512]
        
        # 1. 自注意力计算
        # q=x, k=x, v=x
        # attn_out 输出形状: [batch_size, src_len, 512]
        attn_out = self.self_attn(q=x, k=x, v=x, mask=src_mask) 
        
        # 2. 残差连接 & 层归一化
        # x + attn_out 形状: [batch_size, src_len, 512]
        # norm1 输出形状: [batch_size, src_len, 512]
        x = self.norm1(x + attn_out) 
        
        # 3. 前馈网络
        # ffn_out 输出形状: [batch_size, src_len, 512]
        ffn_out = self.ffn(x)
        
        # 4. 第二次残差连接 & 层归一化
        # x 输出形状依然是: [batch_size, src_len, 512]
        x = self.norm2(x + ffn_out)
        
        return x

# ==========================================
# 4. 核心积木：解码器层 (Decoder Layer)
# ==========================================
class DecoderLayer(nn.Module):
    def __init__(self, d_model=512, n_head=8, d_ff=2048):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.cross_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = FeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        # x 输入形状 (目标语言): [batch_size, tgt_len, 512]
        # memory 输入形状 (源语言特征): [batch_size, src_len, 512]
        
        # 1. 带掩码的自注意力
        # attn_out 形状: [batch_size, tgt_len, 512]
        attn_out = self.self_attn(q=x, k=x, v=x, mask=tgt_mask)
        x = self.norm1(x + attn_out) # [batch_size, tgt_len, 512]
        
        # 2. 交叉注意力
        # 注意这里的参数对应关系: Q 来自解码器当前的特征 x，K 和 V 来自编码器的全局特征 memory
        # q=x 形状: [batch_size, tgt_len, 512]
        # k=memory, v=memory 形状: [batch_size, src_len, 512]
        # cross_out 形状由 Q 决定: [batch_size, tgt_len, 512]
        cross_out = self.cross_attn(q=x, k=memory, v=memory, mask=src_mask)
        x = self.norm2(x + cross_out) # [batch_size, tgt_len, 512]
        
        # 3. 前馈网络
        # ffn_out 形状: [batch_size, tgt_len, 512]
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out) # 输出形状: [batch_size, tgt_len, 512]
        
        return x

# ==========================================
# 5. 组装：完整的 Transformer 
# ==========================================
class CustomTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_head=8, num_layers=6):
        super().__init__()
        
        # 词嵌入层权重矩阵 (查找表)
        # src_embed 权重形状: [src_vocab_size, 512]
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        # tgt_embed 权重形状: [tgt_vocab_size, 512]
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码参数矩阵
        # 形状: [1, max_seq_len, 512] -> [1, 1000, 512] (假设支持最大序列长度为1000)
        self.src_pos_embed = nn.Parameter(torch.zeros(1, 1000, d_model)) 
        self.tgt_pos_embed = nn.Parameter(torch.zeros(1, 1000, d_model))
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_head) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_head) for _ in range(num_layers)])
        
        # 最后的线性分类层
        # 权重矩阵: [d_model, tgt_vocab_size] -> [512, tgt_vocab_size]
        # 意义: 将 512 维特征映射成目标词表上每一个词的概率打分
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # src 形状: [batch_size, src_len] (例如 [2, 10] 即2个句子，每句10个词的索引)
        # tgt 形状: [batch_size, tgt_len] (例如 [2, 8] 即2个句子，每句8个词的索引)
        batch, src_len = src.size()
        batch, tgt_len = tgt.size()
        
        # 1. 嵌入计算
        # src_embed(src) 将索引转换为向量，形状变为: [batch_size, src_len, 512]
        # src_pos_embed 切片提取前 src_len 个位置向量: [1, src_len, 512]
        # 相加后利用了广播机制，enc_out 最终形状: [batch_size, src_len, 512]
        enc_out = self.src_embed(src) + self.src_pos_embed[:, :src_len, :]
        
        # 同理，dec_out 最终形状: [batch_size, tgt_len, 512]
        dec_out = self.tgt_embed(tgt) + self.tgt_pos_embed[:, :tgt_len, :]
        
        # 2. 编码阶段
        for layer in self.encoder_layers:
            # 每次输入和输出的形状都在这层内保持不变: [batch_size, src_len, 512]
            enc_out = layer(enc_out, src_mask)
            
        # 3. 解码阶段
        for layer in self.decoder_layers:
            # 每次输入 dec_out 是 [batch_size, tgt_len, 512]
            # memory 是 enc_out，形状是 [batch_size, src_len, 512]
            # 输出 dec_out 的形状依然保持为: [batch_size, tgt_len, 512]
            dec_out = layer(dec_out, memory=enc_out, src_mask=src_mask, tgt_mask=tgt_mask)
            
        # 4. 映射到词表
        # [batch_size, tgt_len, 512] @ [512, tgt_vocab_size]^T 
        # 结果形状: [batch_size, tgt_len, tgt_vocab_size]
        logits = self.fc_out(dec_out)
        
        return logits
    


# === 测试参数设置 ===
batch_size = 2
src_len = 10     # 例如："我 爱 学 习 编 程"
tgt_len = 8      # 例如："I love learning coding"
src_vocab = 5000
tgt_vocab = 6000
d_model = 512

# 初始化我们自己写的模型
model = CustomTransformer(src_vocab, tgt_vocab, d_model=d_model, n_head=8, num_layers=2)

# 生成模拟输入 (整数索引)
src_input = torch.randint(0, src_vocab, (batch_size, src_len)) # [2, 10]
tgt_input = torch.randint(0, tgt_vocab, (batch_size, tgt_len)) # [2, 8]

# 生成 Decoder 的 causal mask (下三角矩阵，防止看未来的词)
# [1, 1, 8, 8] (为了适应多头注意力的广播机制，前面加了维度)
tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)).view(1, 1, tgt_len, tgt_len)

# 运行数据流
print(f"1. 编码器输入形状 (源语言): {src_input.shape}")
print(f"2. 解码器输入形状 (目标语言): {tgt_input.shape}")

# Forward pass
output_logits = model(src_input, tgt_input, tgt_mask=tgt_mask)

print(f"3. 最终输出的预测概率分布形状: {output_logits.shape}")
# 期望输出: [2, 8, 6000] 
# 解释：2个句子，每个句子8个token的位置，每个位置都给出了字典中6000个词的可能性打分。