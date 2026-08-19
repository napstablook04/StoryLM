import torch
import math
import numpy as np


def run_linear(
    d_in: int,
    d_out: int,
    weights: torch.Tensor,   # shape: [d_out, d_in]
    in_features: torch.Tensor,  # shape: [..., d_in]
) -> torch.Tensor:
    # 1. 用 weights 和 in_features 做矩阵乘法（torch.matmul 或 @）结果 shape 应该是 [..., d_out]
    return in_features@weights.T


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: torch.Tensor,   # shape: [vocab_size, d_model]  embedding矩阵（tokenid-对应的词向量）
    token_ids: torch.Tensor,   # shape: [...]，任意形状的整数
) -> torch.Tensor:
    # 1. 用 token_ids 作为索引，从 weights 中取出对应的行 PyTorch tensor 支持用整数 tensor 直接索引
    return weights[token_ids]


def run_softmax( # 归一化 把权重转化为概率分布
    in_features: torch.Tensor,
    dim: int,
) -> torch.Tensor:
     # 1. 沿 dim 维度取最大值，keepdim=True，保持维度以便广播 用 torch.max，它返回 (values, indices)，你只需要 values
    max_values = torch.max(in_features,dim=dim,keepdim=True).values
     # 2. in_features 减去最大值（数值稳定化),对减完的结果调用 torch.exp
    exp_values = torch.exp(in_features - max_values)
     # 3. 沿 dim 维度对 exp 结果求和，keepdim=True
    sum_exp = torch.sum(exp_values,dim=dim,keepdim=True)

    return exp_values/sum_exp  #e^(x_i)/sum(e^(x_j)) softmax公式


def run_silu(  #SiLU：让 FFN 具有非线性
    in_features: torch.Tensor,
) -> torch.Tensor:
    # SiLU(x) = x * sigmoid(x) 用 torch.sigmoid 计算 sigmoid(in_features)
    return in_features * torch.sigmoid(in_features)


def run_rmsnorm(   #RMSNorm 通过计算输入向量的均方根，对输入进行缩放，使其整体数值尺度保持稳定，并通过可学习权重进行进一步缩放。
    d_model: int,  #Softmax 是“你应该关注谁？RMSNorm 是“你的数值规模是不是合适？”
    eps: float,
    weights: torch.Tensor,   # shape: [d_model]
    in_features: torch.Tensor,  # shape: [..., d_model]
) -> torch.Tensor:
    # RMSNorm(x) = x / RMS(x) * weights  其中 RMS(x) = sqrt(mean(x^2) + eps) eps 是一个小的常数，防止除以零
    rms_x = torch.sqrt(torch.mean(in_features ** 2, dim=-1, keepdim=True) + eps) #算mean的时候要选最后一个维度
    return in_features/rms_x * weights

def run_swiglu(  #Swiglu：让 FFN 具有非线性
    d_model: int,
    d_ff: int,
    w1: torch.Tensor,  # shape: [d_ff, d_model]
    w2: torch.Tensor,  # shape: [d_model, d_ff]
    w3: torch.Tensor,  # shape: [d_ff, d_model] : [d_out, d_in]
    in_features: torch.Tensor,  # shape: [..., d_model]
) -> torch.Tensor:
    #FFN(𝑥) = SwiGLU(𝑥,𝑊1,𝑊2,𝑊3) = 𝑊2(SiLU(𝑊1𝑥)⊙𝑊3𝑥)
    #W1x
    x1 = run_linear(d_model, d_ff, w1, in_features)  # shape: [..., d_ff]

    #W3x
    x3 = run_linear(d_model, d_ff, w3, in_features) # shape: [..., d_ff]

    #GLU = SiLU(W1x)⊙W3x
    GLU = run_silu(x1) * x3  # shape: [..., d_ff]  
  
    SwiGLU = run_linear(d_ff, d_model , w2, GLU) #shape: [..., d_model]
    return SwiGLU
    



def run_get_batch(
    dataset: np.ndarray,    # 1D numpy array of integers 
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 1. 随机采样 batch_size 个起始索引，起始索引的有效范围是 0 到 len(dataset) - context_length（不包含）
    #    用 np.random.randint 或 torch.randint 
    # start+context_length 不能达到dataset的长度，这样y才能取到start+1+context_length 这里randint前闭后开
    start_ids = np.random.randint(0, len(dataset) - context_length, size=batch_size)
     # 2. 用这些起始索引，从 dataset 中各取 context_length 个连续元素 → x 即 x 取 [start : start+context_length]
    x = np.array([dataset[start:start + context_length] for start in start_ids])
     # 3. y 是 x 中每个元素+1 ，y 取 [start+1 : start+1+context_length]
    y = np.array([dataset[start + 1:start + 1 + context_length] for start in start_ids])
     # 4. 把 x 和 y 转成 torch tensor，放到指定 device 上return (x, y)
    x_tensor = torch.tensor(x, dtype=torch.long, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    return (x_tensor, y_tensor)


def run_scaled_dot_product_attention(
    Q: torch.Tensor,   # shape: [..., queries, d_k]
    K: torch.Tensor,   # shape: [..., keys, d_k]
    V: torch.Tensor,   # shape: [..., keys, d_v]
    mask: torch.Tensor | None = None,  # shape: [..., queries, keys]，True=保留，False=屏蔽
) -> torch.Tensor:
    # SDPA(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    # 1. 算出 d_k：K 的最后一个维度大小
    d_k = K.shape[-1]
    # 2. scores = Q @ K^T（交换 K 的最后两维）→ shape: [..., queries, keys]
    #    用 torch.transpose 或 .transpose(-2, -1)
    # scores 除以 sqrt(d_k)（用 math.sqrt 或 torch.sqrt） torch.sqrt要输入tensor
    scores = Q @ K.transpose(-2, -1)/math.sqrt(d_k)
    # 4. 如果 mask 不为 None：
    if mask is not None:
        #把 mask 中 False 的位置对应的 scores 设为 -inf（用 torch.where 或 masked_fill）
        #scores = torch.where(mask, scores, float('-inf')) 
        scores = scores.masked_fill(~mask, float('-inf'))  # ~mask 取反，False变True，True变False
    # 5. 对 scores 的最后一个维度做 softmax → attn_weights
    attn_weights = run_softmax(scores, dim=-1)
    # 6. output = attn_weights @ V → shape: [..., queries, d_v]
    output = attn_weights @ V
    # 7. return output
    return output


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: torch.Tensor,  # shape: [..., seq_len, d_k]
    token_positions: torch.Tensor,  # shape: [..., seq_len]，整数，表示每个 token 的位置
) -> torch.Tensor:
    # RoPE 把位置信息编码到 Q/K 中，通过旋转相邻维度对实现
    # 对于维度 (2i, 2i+1)，角度 = pos / theta^(2i/d_k)

    # 1. 算出每个维度对的角度 freq_i = 1 / theta^(2i/d_k)，i 从 0 到 d_k//2 - 1
    #    用 torch.arange 构造 i，用 torch.pow 算 theta^(-2i/d_k)
    #    结果 shape: [d_k//2]
    i = torch.arange(d_k//2)
    freq_i = 1 / theta ** (2 * i / d_k)
    # 2. 把 freq 和 token_positions 做外积（广播）→ angles，shape: [..., seq_len, d_k//2]
    #    提示：token_positions 的 shape 是 [..., seq_len]，freq 是 [d_k//2]
    #    用 unsqueeze/expand 或直接利用广播
    angles = token_positions.unsqueeze(-1)*freq_i
    # 3. 算 cos 和 sin：torch.cos(angles) 和 torch.sin(angles)
    #    shape 都是 [..., seq_len, d_k//2]
    cos = torch.cos(angles) 
    sin = torch.sin(angles)
    # 4. 把 in_query_or_key 按奇偶维度拆成两半：
    #    x_even = 取偶数索引维度 (0,2,4,...)，x_odd = 取奇数索引维度 (1,3,5,...)
    #    提示：用切片，步长为2 Python 的步长切片.语法是 [start : stop : step]。
    #    两半的 shape 都是 [..., seq_len, d_k//2]
    x_even = in_query_or_key[..., ::2]
    x_odd = in_query_or_key[..., 1::2]
    # 5. 应用旋转：
    #    out_even = x_even * cos - x_odd * sin
    #    out_odd  = x_even * sin + x_odd * cos
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    # 6. 把 out_even 和 out_odd 交错拼回去（偶数位放 even，奇数位放 odd）
    #    提示：用 stack(dim=-1) 再 flatten 最后两个维度，
    #    或者用 torch.zeros 预分配再赋值
    # stacked = torch.stack((out_even, out_odd), dim=-1)  # shape: [..., seq_len, d_k//2, 2]
    # out = stacked.flatten(-2)  # shape: [..., seq_len, d
    out = torch.zeros_like(in_query_or_key)
    out[...,::2] = out_even
    out[...,1::2] = out_odd
    # 7. return 拼好的结果
    return out


def run_multihead_self_attention(  # MultiHead(𝑄,𝐾,𝑉) = Concat(head1,…,headℎ)
                                   # for head𝑖 = Attention(𝑄𝑖,𝐾𝑖,𝑉𝑖)
                                   # MultiHeadSelfAttention(𝑥) = 𝑊𝑂MultiHead(𝑊𝑄𝑥,𝑊𝐾𝑥,𝑊𝑉𝑥)
    d_model: int,
    num_heads: int,
    q_proj_weight: torch.Tensor,   # [d_model, d_model]
    k_proj_weight: torch.Tensor,   # [d_model, d_model]
    v_proj_weight: torch.Tensor,   # [d_model, d_model]
    o_proj_weight: torch.Tensor,   # [d_model, d_model] 
    in_features: torch.Tensor,     # [..., seq_len, d_model]
) -> torch.Tensor:
    # 1. 用 run_linear 分别对 in_features 做四次线性投影，得到 Q, K, V
    #    q_proj_weight 的 shape 是 [d_model, d_model]，
    #    所以 d_in=d_model, d_out=d_model
    Q = run_linear(d_model, d_model, q_proj_weight, in_features)
    K = run_linear(d_model, d_model, k_proj_weight, in_features)
    V = run_linear(d_model, d_model, v_proj_weight, in_features)
    # 2. 算出 d_k = d_model // num_heads
    d_k = d_model // num_heads
    # 3. 把 Q, K, V 从 [..., seq_len, d_model] reshape 成多头形式 [..., seq_len, num_heads, d_k]
    #    然后转置成 [..., num_heads, seq_len, d_k]
    seq_len = in_features.shape[-2]
    Q = Q.reshape(*Q.shape[:-2], seq_len, num_heads, d_k).transpose(-3, -2)
    K = K.reshape(*K.shape[:-2], seq_len, num_heads, d_k).transpose(-3, -2)
    V = V.reshape(*V.shape[:-2], seq_len, num_heads, d_k).transpose(-3, -2)   
    # 4. 调用 run_scaled_dot_product_attention(Q, K, V) 输出 shape: [..., num_heads, seq_len, d_k]
    mask = torch.tril(
    torch.ones(seq_len, seq_len,
               dtype=torch.bool,
               device=in_features.device)
)
    heads = run_scaled_dot_product_attention(Q, K, V, mask)

    # 5. 把多头输出转置回来 [..., num_heads, seq_len, d_k]->[..., seq_len, num_heads, d_k]，再 flatten 回 [..., seq_len, d_model]
    multi_head = heads.transpose(-3, -2).reshape(*heads.shape[:-3], seq_len, d_model)
    # 6. 用 run_linear 做输出投影（o_proj_weight），return
    output = run_linear(d_model, d_model, o_proj_weight, multi_head)
    return output


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    in_features: torch.Tensor,
    token_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    # 和 run_multihead_self_attention 几乎一样，但多两步：
    #
    # 1. 做 Q, K, V 投影（同上）
    Q = run_linear(d_model, d_model, q_proj_weight, in_features)
    K = run_linear(d_model, d_model, k_proj_weight, in_features)
    V = run_linear(d_model, d_model, v_proj_weight, in_features)
    # 2. reshape 成多头（同上）
    d_k = d_model // num_heads
    seq_len = in_features.shape[-2]
    Q = Q.reshape(*Q.shape[:-2], seq_len, num_heads, d_k).transpose(-3, -2)
    K = K.reshape(*K.shape[:-2], seq_len, num_heads, d_k).transpose(-3, -2)
    V = V.reshape(*V.shape[:-2], seq_len, num_heads, d_k).transpose(-3, -2)
    # 3. 【新增】对 Q 和 K 应用 run_rope（不是对 V！）
    #    d_k = d_model // num_heads，注意 RoPE 操作的是最后一个维度 d_k
    #    token_positions：如果为 None，默认用 arange(0, seq_len)
    if token_positions is None:
        token_positions = torch.arange(0, seq_len,device=in_features.device)
    roped_Q = run_rope(d_k, theta, max_seq_len, Q, token_positions)
    roped_K = run_rope(d_k, theta, max_seq_len, K, token_positions)
    # 4. 调用 SDPA
    mask = torch.tril(
    torch.ones(seq_len, seq_len,
               dtype=torch.bool,
               device=in_features.device))
    MHA = run_scaled_dot_product_attention(roped_Q, roped_K, V, mask)
    # 5. 转置回来、flatten、输出投影（同上）
    MHA = MHA.transpose(-3, -2).reshape(*MHA.shape[:-3], seq_len, d_model)
    # 6. return
    output = run_linear(d_model, d_model, o_proj_weight, MHA)
    return output


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, torch.Tensor], #权重文件在adapters里
    in_features: torch.Tensor,  # shape: [batch, seq_len, d_model]
) -> torch.Tensor:
    # Pre-Norm Transformer Block:
    # x' = x + MHA_with_RoPE(RMSNorm(x))          ← 第一个子层
    # y = x' + SwiGLU(RMSNorm(x'))                  ← 第二个子层
    x = in_features
    x_norm = run_rmsnorm(d_model, 1e-5, weights["ln1.weight"], x)
    attn_out = run_multihead_self_attention_with_rope(d_model, num_heads, max_seq_len, theta,
                                                      weights['attn.q_proj.weight'],
                                                      weights['attn.k_proj.weight'],
                                                      weights['attn.v_proj.weight'],
                                                      weights['attn.output_proj.weight'],
                                                      x_norm,
                                                      token_positions=None)
    x = x + attn_out

    x_norm = run_rmsnorm(d_model, 1e-5, weights["ln2.weight"], x)

    ffn_out = run_swiglu(d_model, d_ff,
                           weights['ffn.w1.weight'],
                           weights['ffn.w2.weight'],
                           weights['ffn.w3.weight'],
                           x_norm)
    y = x + ffn_out

    return y
    # ---- 第一个子层：注意力 ----
    # 1. 对 in_features 做 RMSNorm 调用 run_rmsnorm
    #    用 weights["ln1.weight"]，shape [d_model]，eps 固定 1e-5
    # 2. 把 norm 后的结果送进 MHA with RoPE 调用 run_multihead_self_attention_with_rope
    #    从 weights 里取四个投影权重：
    #    "attn.q_proj.weight", "attn.k_proj.weight", "attn.v_proj.weight", "attn.output_proj.weight"
    #    token_positions=None（让它自动用 arange）
    # 3. 残差连接：把第 1 步的 norm 输出替换成原始 in_features，
    #    第 2 步的 MHA 输出加到 in_features 上（不是加到 norm 后的输出上）
    #    命名：x = in_features + mha_output

    # ---- 第二个子层：FFN ----
    # 4. 对 x 做 RMSNorm, weights["ln2.weight"]
    # 5. 把 norm 后的结果送进 SwiGLU
    #    从 weights 里取三个 FFN 权重： "ffn.w1.weight", "ffn.w2.weight", "ffn.w3.weight"
    #    调用你的 run_SwiGLU（注意你函数名是大写 S，和 adapters 里的名字对不上要统一）
    # 6. 残差连接： = x + swiglu_output

    # 7. return y

def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, torch.Tensor],
    in_indices: torch.Tensor,  # shape: [batch, seq_len]
) -> torch.Tensor:
    # 完整的语言模型：Embedding → N × TransformerBlock → RMSNorm → LM Head

    in_features = run_embedding(vocab_size, d_model, weights["token_embeddings.weight"], in_indices)
    #shape [batch, seq_len, d_model]

    for i in range(num_layers):
        # 构造该层的 weights 子字典，key 的前缀是 "layers.{i}." 例如第 0 层的权重 key 是 "layers.0.attn.q_proj.weight" 
        layer_weights = {k[len(f"layers.{i}."):]: v  #以layers.{i}.开头的key去掉前缀后作为新字典的key，value不变
                         for k, v in weights.items() 
                         if k.startswith(f"layers.{i}.")}
        
        in_features = run_transformer_block(
            d_model, num_heads, d_ff, context_length, rope_theta, layer_weights, in_features)
        # shape: [batch, seq_len, d_model]

    #post_norm
    x = run_rmsnorm(d_model, 1e-5, weights["ln_final.weight"], in_features) #shape: [batch, seq_len, d_model]

    logits = run_linear(d_model, vocab_size, weights["lm_head.weight"], x) #shape: [batch, seq_len, vocab_size]
    return logits


    # 1. Token Embedding
    #    用 weights["token_embeddings.weight"]，shape [vocab_size, d_model]
    #    调用 run_embedding(vocab_size, d_model, weights["token_embeddings.weight"], in_indices)
    #    得到 hidden states，shape: [batch, seq_len, d_model]

    # 2. 逐层通过 Transformer Block（num_layers 层）
    #    for i in range(num_layers):
    #        构造该层的 weights 子字典，key 的前缀是 "layers.{i}."
    #        例如第 0 层的权重 key 是 "layers.0.attn.q_proj.weight" 等
    #        提示：遍历 weights，筛出 key 以 "layers.{i}." 开头的条目，
    #              去掉前缀 "layers.{i}." 后传入 run_transformer_block
    #    每层的输出作为下一层的输入

    # 3. 最终 RMSNorm
    #    用 weights["ln_final.weight"]
    #    调用 run_rmsnorm

    # 4. LM Head
    #    用 weights["lm_head.weight"]，shape [vocab_size, d_model]
    #    run_linear(d_model, vocab_size, weights["lm_head.weight"], x)
    #    输出 shape: [batch, seq_len, vocab_size]

    # 5. return
    
