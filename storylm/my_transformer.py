import regex as re
from collections import Counter
from collections import defaultdict
from typing import Iterable, Iterator
import json


#---BPE and Tokenizer
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pre_tokenizer(text):
    token_counts = Counter()

    for match in re.finditer(PAT, text):
        tokens = match.group()

        token_bytes =tuple ([bytes([t]) for t in tokens.encode('utf-8')])

        token_counts[token_bytes] += 1

    return token_counts

def count_pair_and_occ(token_counts):

    pair_counts = Counter()

    pair_occ = defaultdict(set)

    for token, count in token_counts.items():
        for i in range(len(token)-1):
            pair = (token[i], token[i+1])
            pair_counts[pair] += count
            pair_occ[pair].add(token)

    return pair_counts,pair_occ

def merge_pair(token_counts, pair_counts, pair_occ, best_pair):

    affected_tokens = list(pair_occ[best_pair])


    for token in affected_tokens:
        
        i = 0
        new_token = []

        while i < len(token):
            if (i < len(token) - 1 and (token[i], token[i+1]) == best_pair ):
                new_token.append(token[i] + token[i+1])
                i += 2
            else:
                new_token.append(token[i])
                i += 1

        new_token = tuple(new_token)
        count = token_counts.pop(token)

        token_counts[new_token] += count #更新token_counts:老的删掉，新的继承原来的次数，若已存在，累加。

        for i in range(len(token) - 1): 
             
            pair = (token[i],token[i+1]) 
            pair_occ[pair].discard(token) #更新pair_occ:和老token关联的所有pair全部删掉老token，并补入新token
            
            pair_counts[pair] -= count #更新pair_counts:和老token关联的所有pair全部剪掉老token的影响,新token产生的所有pair继承老token的次数

            if pair_counts[pair] == 0:
                del pair_counts[pair]

        for i in range(len(new_token) - 1):
            pair = (new_token[i],new_token[i+1])

            pair_occ[pair].add(new_token)

            pair_counts[pair] += count

    return token_counts,pair_counts,pair_occ


def build_vocab(special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}

    for i, token in enumerate(special_tokens, start=256):
        vocab[i] = token.encode("utf-8")

    return vocab


def train_bpe(input_path, vocab_size, special_tokens):
    with open(input_path, "r", encoding="utf-8") as f:
     text = f.read()

    #1.初始化  token_counts
    token_counts = Counter()

    if special_tokens:
        #特殊字符硬分割
        p = "|".join(re.escape(token) for token in special_tokens)
        parts = re.split(p,text)
        for part in parts:
            token_counts.update(pre_tokenizer(part))
    else:
        token_counts.update(pre_tokenizer(text))

    #2.初始化vocab
    vocab = build_vocab(special_tokens)
    next_id = len(vocab)
    
    #3.初始化 pair_counts,pair_occ
    pair_counts, pair_occ = count_pair_and_occ(token_counts)

    merges = [] 

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best_pair = max(pair_counts,key = lambda p :(pair_counts[p],p))

        token_counts, pair_counts, pair_occ = merge_pair(token_counts,pair_counts,pair_occ,best_pair)

        new_byte = best_pair[0] + best_pair[1]

        vocab[next_id] = new_byte
        next_id += 1 

        merges.append(best_pair)

    return vocab, merges
    

def apply_merges(token, merge_rank):
    token = list(token)

    if len(token) < 2:
        return token

    while True:
        best_pos = -1
        best_rank = float("inf")

        for i in range(len(token) - 1):
            rank = merge_rank.get((token[i], token[i + 1]),float("inf"),)

            if rank < best_rank:
                best_rank = rank
                best_pos = i

        if best_pos == -1:
            break

        token[best_pos] += token[best_pos + 1]
        del token[best_pos + 1]

    return token

class Tokenizer:

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(special_tokens or [],key=len,reverse=True,)

        self.token_to_id = {token: i for i, token in vocab.items()} # 反向字典：能从token找到对应的index
        self.merge_rank = {pair: i for i, pair in enumerate(merges)}

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")

            if token_bytes not in self.token_to_id:
                new_id = len(vocab)
                self.vocab[new_id] = token_bytes
                self.token_to_id[token_bytes] = new_id

  
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,):
        # 读取 vocab
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        vocab = {
            int(k): bytes(v)
            for k, v in vocab_data.items()
        }

        # 读取 merges
        merges = []

        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                pair = json.loads(line)

                merges.append((bytes(pair[0]),bytes(pair[1])))

        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]: 
        ''' 把输入文本按照训练好的 BPE 规则进行切分和 merge
            然后把最终得到的 BPE token 转换成词表vocab中对应的整数 ID
        '''
        encode_list = []

        special_tokens = self.special_tokens
        token_to_id = self.token_to_id
        merge_rank = self.merge_rank

        if special_tokens:
            p = "(" + "|".join( re.escape(token)for token in special_tokens) + ")" #和之前不一样，加了括号，special_token也会保存下来
            parts = re.split(p, text) # PARTS: DOC1,<|endoftext|>,DOC2

        else:
            parts = [text]

        for part in parts:
            # special token
            if part in special_tokens:
                encode_list.append(token_to_id[part.encode("utf-8")])
                continue

            # 普通文本
            for match in re.finditer(PAT, part):
                token = match.group()
                token_bytes = tuple(bytes([b])for b in token.encode("utf-8"))

                token_bytes = apply_merges(token_bytes,merge_rank)

                for byte_token in token_bytes:
                    encode_list.append(token_to_id[byte_token])

        return encode_list

    def encode_iterable(self, iterable: Iterable[str],) -> Iterator[int]:  
     for text in iterable:
        yield from self.encode(text)
        

    def decode(self, ids: list[int]) -> str:
        byte_data =b''.join(self.vocab[i] for i in ids)
        return byte_data.decode("utf-8", errors="replace")


#----- model -----
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
    i = torch.arange(
    d_k // 2,
    device=in_query_or_key.device,
    dtype=torch.float32,
)
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

    
#------train optimizer----
import torch
import math
from typing import Iterable

def run_cross_entropy(
    inputs: torch.Tensor,   # shape: [batch_size, vocab_size]，logits（未归一化）
    targets: torch.Tensor,  # shape: [batch_size]，正确类别的整数索引
) -> torch.Tensor:
    
    # 交叉熵 = -log(softmax(logits)_target) = -log(exp(logits_target) / sum(exp(logits)))
    #  = -log(exp(logits_target)) + log(sum(exp(logits))) = -logit_target + log_sum_exp
    max_logits = torch.max(inputs, dim = -1, keepdim=True).values
    inputs = inputs - max_logits
    log_sum_exp = inputs.exp().sum(dim= -1).log()
    logits_target = inputs[torch.arange(inputs.shape[0]), targets]
    loss = -logits_target + log_sum_exp
    return loss.mean()




def run_gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    # 把所有参数的梯度整体裁剪到不超过 max_l2_norm
    
    parameters = list(parameters)
    all_grad = torch.cat([p.grad.flatten() 
                          for p in parameters if p.grad is not None])
    L2 = torch.norm(all_grad, 2)
    if L2 > max_l2_norm:
        clip_ratio = max_l2_norm / L2
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(clip_ratio) # = p.grad * clip_ratio 区别在于原地修改



def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:

    if it < warmup_iters:
        lr = max_learning_rate * it / warmup_iters

    elif warmup_iters <= it < cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        lr = min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(progress * math.pi))

    else:
        lr = min_learning_rate
    return lr


def get_adamw_cls() -> type[torch.optim.Optimizer]:
    # 返回一个自定义的 AdamW 优化器类，继承 torch.optim.Optimizer
    #
    class MyAdamW(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8):
            defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
            super().__init__(params, defaults)

        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    grad = p.grad.data
                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p.data)
                        state['exp_avg_sq'] = torch.zeros_like(p.data)

                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    beta1, beta2 = group['betas']

                    state['step'] += 1

                    # Update biased first moment estimate
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    # Update biased second raw moment estimate
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # Compute bias-corrected first and second moment estimates
                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                    step_size = group['lr'] / bias_correction1

                    # Update parameters
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

                    # Apply weight decay directly to the parameters
                    if group['weight_decay'] != 0:
                        p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

    return MyAdamW
    


def run_save_checkpoint(model, optimizer, iteration, out):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }, out)


def run_load_checkpoint(src, model, optimizer):
    torch_load = torch.load(src)
    model.load_state_dict(torch_load['model_state_dict'])
    optimizer.load_state_dict(torch_load['optimizer_state_dict'])
    return torch_load['iteration']

# ============ nn.Module 版本 ============
# 以下把上面所有 run_xxx 函数改写为 PyTorch nn.Module 子类，
# 使得权重被自动管理（梯度追踪、state_dict 保存/加载等），
# 可以直接用 optimizer 训练。
import torch.nn as nn


def init_trunc_normal_(param: nn.Parameter) -> None:
    """权重初始化：截断正态分布，mean=0, std=0.02, 截断在 ±0.04（2σ）。"""
    nn.init.trunc_normal_(param, mean=0.0, std=0.02, a=-0.04, b=0.04)


class MyRMSNorm(nn.Module):
    """RMS 归一化层 x / RMS(x) * weight 稳定数值尺度。"""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))  # 可学习的缩放因子

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms_x = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms_x * self.weight


class MyRoPE:
    """旋转位置编码（非 nn.Module 无参数 纯计算）。
    通过旋转 Q/K 的相邻维度对来注入位置信息。"""
    def __init__(self, d_k: int, theta: float, max_seq_len: int):
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

    def __call__(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """x: [..., seq_len, d_k], token_positions: [..., seq_len]"""
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(0, seq_len, device=x.device)

        i = torch.arange(self.d_k // 2, device=x.device)
        freq_i = 1.0 / self.theta ** (2 * i / self.d_k)       # [d_k//2]
        angles = token_positions.unsqueeze(-1) * freq_i        # [..., seq_len, d_k//2]
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos

        out = torch.zeros_like(x)
        out[..., ::2] = out_even
        out[..., 1::2] = out_odd
        return out


class MySwiGLU(nn.Module):
    """SwiGLU 前馈网络:W2(SiLU(W1·x) ⊙ W3·x)"""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff))
        self.w3 = nn.Parameter(torch.empty(d_ff, d_model))
        init_trunc_normal_(self.w1)
        init_trunc_normal_(self.w2)
        init_trunc_normal_(self.w3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d_model]
        x1 = x @ self.w1.T          # [..., d_ff]
        x3 = x @ self.w3.T          # [..., d_ff]
        glu = run_silu(x1) * x3     # [..., d_ff]
        return glu @ self.w2.T       # [..., d_model]


class MyMultiHeadSelfAttention(nn.Module):
    """多头自注意力 + RoPE + 因果掩码。
    W_q·x, W_k·x, W_v·x → 拆多头 → RoPE(Q,K) → SDPA → 拼回 → W_o"""
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rope = MyRoPE(self.d_k, theta, max_seq_len)

        self.q_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.k_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.v_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.o_proj = nn.Parameter(torch.empty(d_model, d_model))
        init_trunc_normal_(self.q_proj)
        init_trunc_normal_(self.k_proj)
        init_trunc_normal_(self.v_proj)
        init_trunc_normal_(self.o_proj)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # x: [..., seq_len, d_model]
        seq_len = x.shape[-2]

        Q = x @ self.q_proj.T   # [..., seq_len, d_model]
        K = x @ self.k_proj.T
        V = x @ self.v_proj.T

        # 拆多头: [..., seq_len, num_heads, d_k] → [..., num_heads, seq_len, d_k]
        Q = Q.reshape(*Q.shape[:-2], seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        K = K.reshape(*K.shape[:-2], seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V = V.reshape(*V.shape[:-2], seq_len, self.num_heads, self.d_k).transpose(-3, -2)

        # RoPE 只作用于 Q 和 K
        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)

        # 因果掩码（下三角）
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        attn_out = run_scaled_dot_product_attention(Q, K, V, mask)

        # 拼回: [..., num_heads, seq_len, d_k] → [..., seq_len, d_model]
        attn_out = attn_out.transpose(-3, -2).reshape(*attn_out.shape[:-3], seq_len, self.d_model)
        return attn_out @ self.o_proj.T


class MyTransformerBlock(nn.Module):
    """Pre-Norm Transformer Block:
    x' = x + MHA(RMSNorm(x))        ← 注意力子层 + 残差
    y  = x' + FFN(RMSNorm(x'))       ← FFN 子层 + 残差"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.ln1 = MyRMSNorm(d_model)
        self.attn = MyMultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta)
        self.ln2 = MyRMSNorm(d_model)
        self.ffn = MySwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, d_model]
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MyTransformerLM(nn.Module):
    """完整的 GPT 风格 Decoder-only Transformer 语言模型。

    数据流：
    token_ids → Embedding → [TransformerBlock × num_layers] → RMSNorm → Linear → logits

    作用：
    - 输入一段 token ID 序列，输出每个位置对词表中每个 token 的预测分数（logits）
    - 训练时配合交叉熵损失，学习"给定上文预测下一个 token"
    - 推理时可以从 logits 采样得到下一个 token，实现自回归文本生成
    """
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model

        # Token Embedding: token_id → 词向量
        self.token_embeddings = nn.Parameter(torch.empty(vocab_size, d_model))
        init_trunc_normal_(self.token_embeddings)

        # N 层 Transformer Block
        self.layers = nn.ModuleList([
            MyTransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])

        # 最终 RMSNorm
        self.ln_final = MyRMSNorm(d_model)

        # LM Head: d_model → vocab_size，输出每个 token 的 logits
        self.lm_head = nn.Parameter(torch.empty(vocab_size, d_model))
        init_trunc_normal_(self.lm_head)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [batch, seq_len]，整数 token ID
        Returns:
            logits: [batch, seq_len, vocab_size]，每个位置的未归一化预测分数
        """
        # Embedding
        x = self.token_embeddings[token_ids]   # [batch, seq_len, d_model]

        # 逐层通过 Transformer Block
        for layer in self.layers:
            x = layer(x)

        # 最终归一化 + LM Head
        x = self.ln_final(x)                   # [batch, seq_len, d_model]
        logits = x @ self.lm_head.T             # [batch, seq_len, vocab_size]
        return logits