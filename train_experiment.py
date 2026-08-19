"""
完整的训练实验脚本，涵盖以下实验：
1. 基线训练 + 学习率调优
2. Batch size 实验
3. 文本生成
4. 消融实验：去 RMSNorm、Post-norm、NoPE、SiLU vs SwiGLU

使用方法（子命令形式，没有 --run）：
    # 基线训练（找最佳学习率）
    uv run train_experiment.py baseline --lr 3e-4 --compile

    # 从 checkpoint 继续训练（默认 NEVER 自动 resume，必须显式指定）
    uv run train_experiment.py baseline --resume checkpoints/baseline_best_latest.pt

    # 学习率扫描
    uv run train_experiment.py lr_sweep --lr 1e-4 3e-4 1e-3 3e-3

    # Batch size 实验（默认固定 LR，只变 batch size；--scale_lr 切换线性 scaling）
    uv run train_experiment.py batch_sweep --batch_sizes 8 32 64 128

    # 生成文本
    uv run train_experiment.py generate --checkpoint checkpoints/baseline_best.pt

    # 消融：去 RMSNorm / Post-norm / NoPE / SiLU 替代 SwiGLU
    uv run train_experiment.py ablate_no_norm --lr 3e-4
    uv run train_experiment.py ablate_post_norm --lr 3e-4
    uv run train_experiment.py ablate_nope --lr 3e-4
    uv run train_experiment.py ablate_silu --lr 3e-4

    # 验证实现正确性（单 batch overfit）
    uv run train_experiment.py overfit_test
"""

import argparse
import csv
import json
import sys
import time
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
torch.set_float32_matmul_precision("high")

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from storylm.my_transformer import (
    MyTransformerLM,
    MyRMSNorm,
    MySwiGLU,
    MyMultiHeadSelfAttention,
    MyTransformerBlock,
    MyRoPE,
    init_trunc_normal_,
    run_silu,
    run_scaled_dot_product_attention,
    run_softmax,
)
from storylm.model import (
    run_get_batch,
)
from storylm.train import (
    run_cross_entropy,
    run_gradient_clipping,
    run_get_lr_cosine_schedule,
    get_adamw_cls,
    run_save_checkpoint,
    run_load_checkpoint,
)
from storylm.tokenizer import Tokenizer


# ============================================================
# 数据路径
# ============================================================
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_NPY = DATA_DIR / "tinystories_train.npy"
VALID_NPY = DATA_DIR / "tinystories_valid.npy"
VOCAB_PATH = DATA_DIR / "vocab.json"
MERGES_PATH = DATA_DIR / "merges.txt"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ============================================================
# 实验日志（CSV 格式，记录 step / train_loss / val_loss / wall_time）
# ============================================================
class ExperimentLogger:
    """CSV 日志：保持文件句柄常开，避免每步 open/close。"""
    def __init__(self, run_name: str, log_dir: Path = LOG_DIR):
        self.run_name = run_name
        self.log_dir = log_dir
        self.log_file = log_dir / f"{run_name}.csv"

        # 打开句柄（训练结束调用 close()）
        self.file = open(self.log_file, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["step", "train_loss", "val_loss", "lr", "wall_time"])
        self.file.flush()

    def log(self, step: int, train_loss: float, val_loss: Optional[float],
            lr: float, wall_time: float):
        self.writer.writerow([
            step,
            f"{train_loss:.6f}",
            f"{val_loss:.6f}" if val_loss is not None else "",
            f"{lr:.2e}",
            f"{wall_time:.1f}",
        ])
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.close()

    def save_config(self, config: dict):
        config_file = self.log_dir / f"{self.run_name}_config.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)


# ============================================================
# 验证损失评估
# ============================================================
@torch.no_grad()
def evaluate_val_loss(
    model: nn.Module,
    dataset: np.ndarray,
    vocab_size: int,
    context_length: int,
    device: str,
    num_batches: int = 50,
    batch_size: int = 32,
    seed: int = 1234,
) -> float:
    """在验证集上评估 per-token 交叉熵损失。

    用一个独立的随机源（np.random.default_rng(seed)）自己采样验证 batch，
    完全不触碰 numpy 全局随机状态——因此既不会污染训练的随机采样流，
    每次评估、以及不同实验（lr / batch / 消融）之间又都在同一批验证数据上
    比较，val 曲线可比。注意：不能复用 run_get_batch，因为它内部用的是全局
    np.random，独立实例对它不生效，故这里自行构造 batch。
    """
    model.eval()
    rng = np.random.default_rng(seed)
    n = len(dataset)
    total_loss = 0.0
    total_tokens = 0

    for _ in range(num_batches):
        starts = rng.integers(0, n - context_length, size=batch_size)
        x = np.stack([dataset[s:s + context_length] for s in starts])
        y = np.stack([dataset[s + 1:s + 1 + context_length] for s in starts])
        x = torch.tensor(x, dtype=torch.long, device=device)
        y = torch.tensor(y, dtype=torch.long, device=device)

        logits = model(x)  # [batch, seq_len, vocab_size]
        B, S, V = logits.shape
        loss = run_cross_entropy(logits.reshape(B * S, V), y.reshape(B * S))
        total_loss += loss.item() * B * S
        total_tokens += B * S

    model.train()
    return total_loss / total_tokens


# ============================================================
# 核心训练循环
# ============================================================
def train_loop(
    model: nn.Module,
    train_dataset: np.ndarray,
    valid_dataset: np.ndarray,
    config: dict,
    logger: ExperimentLogger,
    checkpoint_path: Optional[str] = None,
    resume: Optional[str] = None,
):
    """
    通用训练循环。

    config 字典应包含：
        vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta,
        batch_size, total_tokens, lr, min_lr, warmup_tokens, weight_decay,
        beta1, beta2, eps, max_grad_norm,
        val_interval (每隔多少 tokens 评估一次), save_interval, compile_model
    """
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    AdamW = get_adamw_cls()
    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=(config["beta1"], config["beta2"]),
        eps=config["eps"],
    )

    start_step = 0
    start_time = time.time()

    # Resume from checkpoint（必须在 torch.compile 之前，否则 state_dict key 带 _orig_mod. 前缀对不上）
    if resume is not None:
        start_step = run_load_checkpoint(resume, model, optimizer)
        print(f"Resumed from step {start_step}")

    # 可选：torch.compile 加速（放在 resume 之后，保存 checkpoint 时保存未编译的 module）
    if config.get("compile_model", False):
        model = torch.compile(model)

    # torch.compile 包装后，state_dict 的 key 会带 _orig_mod. 前缀；
    # 统一从原始 module 保存，保证 checkpoint 可以直接被裸模型加载
    base_model = getattr(model, "_orig_mod", model)

    # 计算 total_steps
    tokens_per_step = config["batch_size"] * config["context_length"]
    total_steps = config["total_tokens"] // tokens_per_step
    warmup_steps = config["warmup_tokens"] // tokens_per_step
    cosine_cycle_steps = total_steps  # cosine decay 在训练结束时刚好到达 min_lr
    max_loss = config.get("max_loss", float("inf"))  # 超过该值视为发散

    print(f"{'='*60}")
    print(f"Training config:")
    for k, v in config.items():
        if k != "device":
            print(f"  {k}: {v}")
    print(f"  device: {device}")
    print(f"  tokens_per_step: {tokens_per_step}")
    print(f"  total_steps: {total_steps}")
    print(f"  warmup_steps: {warmup_steps}")
    print(f"{'='*60}")

    best_val_loss = float("inf")

    def save_checkpoint(path: Optional[str], step: int):
        if path:
            run_save_checkpoint(base_model, optimizer, step, path)

    try:
        for step in range(start_step, total_steps):
            # Cosine 学习率调度
            lr = run_get_lr_cosine_schedule(
                it=step,
                max_learning_rate=config["lr"],
                min_learning_rate=config["min_lr"],
                warmup_iters=warmup_steps,
                cosine_cycle_iters=cosine_cycle_steps,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # 获取 batch
            model.train()
            x, y = run_get_batch(
                train_dataset, config["batch_size"], config["context_length"], device
            )

            # 前向
            logits = model(x)  # [batch, seq_len, vocab_size]
            B, S, V = logits.shape
            logits_flat = logits.reshape(B * S, V)
            targets_flat = y.reshape(B * S)

            # 计算损失
            loss = run_cross_entropy(logits_flat, targets_flat)

            # 发散检测：NaN/Inf 或 loss 爆炸（很多发散不会抛异常，必须主动检查）
            if not torch.isfinite(loss):
                raise RuntimeError(f"Diverged: non-finite loss at step {step}")
            if loss.item() > max_loss:
                raise RuntimeError(
                    f"Diverged: train_loss={loss.item():.2f} > {max_loss} at step {step}"
                )

            # 反向
            loss.backward()

            # 梯度裁剪
            run_gradient_clipping(model.parameters(), config["max_grad_norm"])

            # 更新参数
            optimizer.step()
            optimizer.zero_grad()

            # 记录
            tokens_processed = (step + 1) * tokens_per_step
            wall_time = time.time() - start_time

            # 周期性 checkpoint（每 save_interval tokens 保存一次 latest）
            if tokens_processed % config["save_interval"] < tokens_per_step:
                if checkpoint_path:
                    save_checkpoint(checkpoint_path.replace(".pt", "_latest.pt"), step)
                print(f"[Step {step:5d}] saved periodic checkpoint")

            # 验证
            val_loss = None
            if tokens_processed % config["val_interval"] < tokens_per_step:
                val_loss = evaluate_val_loss(
                    model, valid_dataset, config["vocab_size"],
                    config["context_length"], device,
                )
                if not math.isfinite(val_loss):
                    raise RuntimeError(f"Diverged: non-finite val loss at step {step}")
                print(f"[Step {step:5d}] train_loss={loss.item():.4f}  val_loss={val_loss:.4f}  lr={lr:.2e}  time={wall_time:.0f}s")

                # 保存最佳 checkpoint
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(checkpoint_path, step)
            elif (step + 1) % max(1, total_steps // 20) == 0:
                # 定期打印进度
                print(f"[Step {step:5d}] train_loss={loss.item():.4f}  lr={lr:.2e}  time={wall_time:.0f}s")

            logger.log(step, loss.item(), val_loss, lr, wall_time)

        # 训练结束，保存最终 checkpoint
        if checkpoint_path:
            save_checkpoint(checkpoint_path.replace(".pt", "_final.pt"), total_steps)

        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
        print(f"Total wall time: {time.time() - start_time:.0f}s")
    finally:
        logger.close()

    return model, best_val_loss


# ============================================================
# 文本生成（自回归）
# ============================================================
@torch.no_grad()
def generate_text(
    model: nn.Module,
    tokenizer: Tokenizer,
    prompt: str = "",
    max_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.9,
    device: str = "cuda",
) -> str:
    """
    自回归生成文本。

    Args:
        model: 训练好的语言模型
        tokenizer: BPE tokenizer
        prompt: 起始文本（空字符串则从 <BOS> 开始）
        max_tokens: 最多生成多少 token
        temperature: softmax 温度，越高越随机
        top_p: nucleus sampling 的概率阈值
        device: 推理设备
    """
    model.eval()

    # 从 tokenizer 查 <|endoftext|> 的真实 ID（本 tokenizer 中它是 256，
    # 但不要硬编码——special token 的 ID 取决于词表构造方式）
    eos_id = tokenizer.token_to_id["<|endoftext|>".encode("utf-8")]

    # 编码 prompt
    if prompt:
        token_ids = tokenizer.encode(prompt)
    else:
        # 没有 BOS token；用 <|endoftext|> 作为文档边界，表示"开始一个新故事"
        token_ids = [eos_id]

    # 转成 tensor
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    generated = list(token_ids)

    for _ in range(max_tokens):
        # 截断到 context_length
        ctx = model.context_length
        input_ids = input_ids[:, -ctx:]

        # 前向
        logits = model(input_ids)  # [1, seq_len, vocab_size]
        next_logits = logits[0, -1, :]  # [vocab_size]

        # Temperature 缩放
        next_logits = next_logits / temperature

        # Top-p (nucleus) sampling
        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        # 找到 cumulative_probs > top_p 的位置
        mask = cumulative_probs - probs > top_p
        sorted_logits[mask] = float("-inf")

        # 重新算概率
        probs = torch.softmax(sorted_logits, dim=-1)
        next_token = sorted_indices[torch.multinomial(probs, 1)]
        next_token_id = next_token.item()

        generated.append(next_token_id)

        # 生成到 <|endoftext|>（文档结束）则停止
        if next_token_id == eos_id:
            break

        # 更新 input_ids
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_token_id]], device=device)], dim=1
        )

    # 解码
    text = tokenizer.decode(generated)
    return text


# ============================================================
# 模型构建函数（支持各种消融变体）
# ============================================================
def build_model(config: dict, ablation: str = "none") -> nn.Module:
    """
    根据 config 和 ablation 类型构建模型。

    ablation 选项：
        "none"           - 标准模型 (pre-norm + RoPE + SwiGLU)
        "no_norm"        - 去掉所有 RMSNorm
        "post_norm"      - 把 pre-norm 改成 post-norm
        "nope"           - 去掉 RoPE 位置编码
        "silu"           - 用 SiLU FFN 替代 SwiGLU（参数量大致匹配）
    """
    vocab_size = config["vocab_size"]
    context_length = config["context_length"]
    d_model = config["d_model"]
    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    d_ff = config["d_ff"]
    rope_theta = config["rope_theta"]

    if ablation == "silu":
        # SiLU FFN 消融：用 d_ff = 4 * d_model，
        # 以近似匹配 SwiGLU 的参数量（SwiGLU 有 3 个权重矩阵，SiLU 只有 2 个）。
        # 校验：SwiGLU 3*512*1344 ≈ 2.06M vs SiLU 2*512*2048 ≈ 2.10M，匹配良好。
        d_ff_actual = 4 * d_model
    else:
        d_ff_actual = d_ff

    if ablation == "none" or ablation == "nope":
        # 标准模型，可能去掉 RoPE
        model = _build_standard_model(
            vocab_size, context_length, d_model, num_layers,
            num_heads, d_ff_actual, rope_theta, use_rope=(ablation != "nope"),
        )
    elif ablation == "no_norm":
        model = _build_no_norm_model(
            vocab_size, context_length, d_model, num_layers,
            num_heads, d_ff_actual, rope_theta,
        )
    elif ablation == "post_norm":
        model = _build_post_norm_model(
            vocab_size, context_length, d_model, num_layers,
            num_heads, d_ff_actual, rope_theta,
        )
    elif ablation == "silu":
        model = _build_silu_model(
            vocab_size, context_length, d_model, num_layers,
            num_heads, d_ff_actual, rope_theta,
        )
    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    return model


class NoNormAttention(nn.Module):
    """支持可选 RoPE 的多头自注意力（与 MyMultiHeadSelfAttention 相同，但供消融模型复用）。"""
    def __init__(self, d_model, num_heads, max_seq_len, theta, use_rope=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope
        if use_rope:
            self.rope = MyRoPE(self.d_k, theta, max_seq_len)
        self.q_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.k_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.v_proj = nn.Parameter(torch.empty(d_model, d_model))
        self.o_proj = nn.Parameter(torch.empty(d_model, d_model))
        init_trunc_normal_(self.q_proj)
        init_trunc_normal_(self.k_proj)
        init_trunc_normal_(self.v_proj)
        init_trunc_normal_(self.o_proj)

    def forward(self, x, token_positions=None):
        seq_len = x.shape[-2]
        Q = x @ self.q_proj.T
        K = x @ self.k_proj.T
        V = x @ self.v_proj.T
        Q = Q.reshape(*Q.shape[:-2], seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        K = K.reshape(*K.shape[:-2], seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        V = V.reshape(*V.shape[:-2], seq_len, self.num_heads, self.d_k).transpose(-3, -2)
        if self.use_rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        attn_out = run_scaled_dot_product_attention(Q, K, V, mask)
        attn_out = attn_out.transpose(-3, -2).reshape(*attn_out.shape[:-3], seq_len, self.d_model)
        return attn_out @ self.o_proj.T


class NoNormBlock(nn.Module):
    """去掉 RMSNorm 的 Transformer Block"""
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        self.attn = NoNormAttention(d_model, num_heads, max_seq_len, theta)
        self.ffn = MySwiGLU(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.ffn(x)
        return x


class NoNormTransformer(nn.Module):
    """去掉所有 RMSNorm 的 Transformer LM"""
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = nn.Parameter(torch.empty(vocab_size, d_model))
        init_trunc_normal_(self.token_embeddings)
        self.layers = nn.ModuleList([
            NoNormBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Parameter(torch.empty(vocab_size, d_model))
        init_trunc_normal_(self.lm_head)

    def forward(self, token_ids):
        x = self.token_embeddings[token_ids]
        for layer in self.layers:
            x = layer(x)
        logits = x @ self.lm_head.T
        return logits


class PostNormBlock(nn.Module):
    """Post-Norm Transformer Block:
    z = RMSNorm(x + MHA(x))
    y = RMSNorm(z + FFN(z))
    """
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        self.attn = NoNormAttention(d_model, num_heads, max_seq_len, theta)
        self.ffn = MySwiGLU(d_model, d_ff)
        self.ln1 = MyRMSNorm(d_model)
        self.ln2 = MyRMSNorm(d_model)

    def forward(self, x):
        z = self.ln1(x + self.attn(x))
        y = self.ln2(z + self.ffn(z))
        return y


class PostNormTransformer(nn.Module):
    """Post-Norm Transformer LM"""
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = nn.Parameter(torch.empty(vocab_size, d_model))
        init_trunc_normal_(self.token_embeddings)
        self.layers = nn.ModuleList([
            PostNormBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.ln_final = MyRMSNorm(d_model)
        self.lm_head = nn.Parameter(torch.empty(vocab_size, d_model))
        init_trunc_normal_(self.lm_head)

    def forward(self, token_ids):
        x = self.token_embeddings[token_ids]
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = x @ self.lm_head.T
        return logits


class SiLUFeedForward(nn.Module):
    """SiLU FFN: W2(SiLU(W1·x))（无门控）"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(d_ff, d_model))
        self.w2 = nn.Parameter(torch.empty(d_model, d_ff))
        init_trunc_normal_(self.w1)
        init_trunc_normal_(self.w2)

    def forward(self, x):
        return run_silu(x @ self.w1.T) @ self.w2.T


class SiLUTransformerBlock(nn.Module):
    """使用 SiLU FFN 的 Transformer Block"""
    def __init__(self, d_model, num_heads, d_ff, max_seq_len, theta):
        super().__init__()
        self.ln1 = MyRMSNorm(d_model)
        self.attn = NoNormAttention(d_model, num_heads, max_seq_len, theta)
        self.ln2 = MyRMSNorm(d_model)
        self.ffn = SiLUFeedForward(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SiLUTransformer(nn.Module):
    """使用 SiLU FFN 的 Transformer LM"""
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = nn.Parameter(torch.empty(vocab_size, d_model))
        init_trunc_normal_(self.token_embeddings)
        self.layers = nn.ModuleList([
            SiLUTransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.ln_final = MyRMSNorm(d_model)
        self.lm_head = nn.Parameter(torch.empty(vocab_size, d_model))
        init_trunc_normal_(self.lm_head)

    def forward(self, token_ids):
        x = self.token_embeddings[token_ids]
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = x @ self.lm_head.T
        return logits


class NoPeTransformer(nn.Module):
    """去掉 RoPE 位置编码的 Transformer LM"""
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = nn.Parameter(torch.empty(vocab_size, d_model))
        init_trunc_normal_(self.token_embeddings)
        self.layers = nn.ModuleList([
            MyTransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
            for _ in range(num_layers)
        ])
        self.ln_final = MyRMSNorm(d_model)
        self.lm_head = nn.Parameter(torch.empty(vocab_size, d_model))
        init_trunc_normal_(self.lm_head)
        # 覆盖每层的 attention，关闭 RoPE
        for layer in self.layers:
            layer.attn.use_rope = False

    def forward(self, token_ids):
        x = self.token_embeddings[token_ids]
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = x @ self.lm_head.T
        return logits


def _build_standard_model(vocab_size, context_length, d_model, num_layers,
                           num_heads, d_ff, rope_theta, use_rope=True):
    if use_rope:
        return MyTransformerLM(vocab_size, context_length, d_model, num_layers,
                                num_heads, d_ff, rope_theta)
    else:
        return NoPeTransformer(vocab_size, context_length, d_model, num_layers,
                               num_heads, d_ff, rope_theta)


def _build_no_norm_model(vocab_size, context_length, d_model, num_layers,
                          num_heads, d_ff, rope_theta):
    return NoNormTransformer(vocab_size, context_length, d_model, num_layers,
                               num_heads, d_ff, rope_theta)


def _build_post_norm_model(vocab_size, context_length, d_model, num_layers,
                           num_heads, d_ff, rope_theta):
    return PostNormTransformer(vocab_size, context_length, d_model, num_layers,
                                 num_heads, d_ff, rope_theta)


def _build_silu_model(vocab_size, context_length, d_model, num_layers,
                       num_heads, d_ff, rope_theta):
    return SiLUTransformer(vocab_size, context_length, d_model, num_layers,
                             num_heads, d_ff, rope_theta)


# ============================================================
# 默认超参数配置
# ============================================================
def get_default_config():
    return {
        "vocab_size": 10000,
        "context_length": 256,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,
        "rope_theta": 10000.0,
        # 训练超参数
        "batch_size": 64,
        "total_tokens": 327_680_000,
        "lr": 3e-4,
        "min_lr": 3e-5,
        "warmup_tokens": 10_000_000,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "max_grad_norm": 1.0,
        # 日志 / 保存
        "val_interval": 10_000_000,    # 每 10M tokens 评估一次
        "save_interval": 50_000_000,   # 每 50M tokens 保存一次 latest checkpoint
        "max_loss": 25.0,              # train loss 超过该值视为发散（初始 loss ≈ ln(10000) ≈ 9.2）
        "compile_model": False,        # 默认关闭，调通后再用 --compile 开启加速
    }


# ============================================================
# 各实验入口
# ============================================================
def run_single_experiment(run_name: str, lr: float, ablation: str = "none",
                          batch_size: int = 64, extra_config: dict = None,
                          resume: Optional[str] = None, compile_model: bool = False):
    """运行单个训练实验。默认从零训练；只有显式传 resume 才会恢复 checkpoint。"""
    config = get_default_config()
    config["lr"] = lr
    config["batch_size"] = batch_size
    config["min_lr"] = lr * 0.1  # min_lr 通常是 max_lr 的 1/10
    config["compile_model"] = compile_model

    if extra_config:
        config.update(extra_config)

    # warmup 不超过总预算的 1/10；放在 extra_config 之后，
    # 兼容自定义 total_tokens（如小 batch 的短预算），避免 warmup > 总步数
    config["warmup_tokens"] = min(config["warmup_tokens"], config["total_tokens"] // 10)

    # 构建 logger
    logger = ExperimentLogger(run_name)
    logger.save_config(config)

    # 构建模型
    model = build_model(config, ablation)

    # Checkpoint 路径（注意：默认不再自动 resume——否则同名旧 checkpoint 会让新实验
    # 从旧状态继续，污染 lr sweep / ablation 的结果）
    ckpt_path = str(CHECKPOINT_DIR / f"{run_name}_best.pt")

    # 训练（mmap 按需读页，避免把 1GB 训练集整个塞进内存）
    print(f"\n{'#'*60}")
    print(f"# Experiment: {run_name}")
    print(f"# lr={lr}, batch_size={batch_size}, ablation={ablation}")
    print(f"# resume={resume}, compile={compile_model}")
    print(f"{'#'*60}\n")

    model, best_val = train_loop(
        model=model,
        train_dataset=np.load(TRAIN_NPY, mmap_mode="r"),
        valid_dataset=np.load(VALID_NPY, mmap_mode="r"),
        config=config,
        logger=logger,
        checkpoint_path=ckpt_path,
        resume=resume,
    )
    return model, best_val


def run_lr_sweep(lrs: list[float], compile_model: bool = False):
    """学习率扫描实验。发散（NaN/Inf/loss 爆炸）会以 RuntimeError 抛出并记录。"""
    results = {}
    for lr in lrs:
        run_name = f"lr_{lr:.0e}"
        try:
            _, best_val = run_single_experiment(run_name, lr, compile_model=compile_model)
            results[lr] = best_val
            print(f"  lr={lr:.0e} → best_val_loss={best_val:.4f}")
        except RuntimeError as e:
            results[lr] = f"DIVERGED: {e}"
            print(f"  lr={lr:.0e} → DIVERGED: {e}")

    # 汇总
    print(f"\n{'='*40}")
    print("Learning Rate Sweep Results:")
    for lr, val in results.items():
        print(f"  lr={lr:.0e}: {val}")
    return results


def run_batch_sweep(batch_sizes: list[int], lr: float = 3e-4,
                    scale_lr: bool = False, compile_model: bool = False,
                    small_batch_tokens: Optional[int] = None,
                    small_batch_threshold: int = 8):
    """Batch size 扫描实验。

    默认固定 LR（用 baseline 调出的最佳值），只变 batch size——
    这样第一轮看到的是 batch size 本身的影响；
    如果某些 batch size 明显需要不同的 LR，再针对它们重新调。
    传 scale_lr=True 可切换到线性 scaling（lr ∝ batch_size）做对比。

    small_batch_tokens：极小 batch（< small_batch_threshold）在固定 327M token 预算下
    步数爆炸、GPU 吃不饱，跑满要数小时~十几小时。传入该值后，这些 run 改用较小的
    token 预算（如 20M），只跑出学习曲线的形状/稳定性趋势即可，避免拖垮整晚挂机。
    传 None 则所有 batch 一律跑满 total_tokens。
    """
    results = {}
    base_bs = 64
    for bs in batch_sizes:
        run_name = f"batch_{bs}"
        if scale_lr:
            lr_bs = lr * (bs / base_bs)
        else:
            lr_bs = lr

        extra = None
        if small_batch_tokens and bs < small_batch_threshold:
            extra = {"total_tokens": small_batch_tokens}
            print(f"  [small batch] bs={bs} < {small_batch_threshold}，改用 "
                  f"{small_batch_tokens:,} tokens 短预算（只看趋势，不跑满）")

        try:
            _, best_val = run_single_experiment(
                run_name, lr_bs, batch_size=bs, compile_model=compile_model,
                extra_config=extra,
            )
            results[bs] = (lr_bs, best_val)
            print(f"  batch_size={bs}, lr={lr_bs:.2e} → best_val_loss={best_val:.4f}")
        except RuntimeError as e:
            results[bs] = (lr_bs, f"DIVERGED: {e}")
            print(f"  batch_size={bs} → DIVERGED: {e}")

    print(f"\n{'='*40}")
    print("Batch Size Sweep Results:")
    for bs, (lr_used, val) in results.items():
        print(f"  batch_size={bs}, lr={lr_used:.2e}: {val}")
    return results


def run_generation(checkpoint_path: str, prompt: str = "Once upon a time",
                    max_tokens: int = 256, temperature: float = 0.8, top_p: float = 0.9):
    """加载 checkpoint 并生成文本。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载 tokenizer
    tokenizer = Tokenizer.from_files(
        str(VOCAB_PATH), str(MERGES_PATH), special_tokens=["<|endoftext|>"]
    )

    # 构建模型
    config = get_default_config()
    model = MyTransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
    )

    # 加载权重
    AdamW = get_adamw_cls()
    optimizer = AdamW(model.parameters())
    run_load_checkpoint(checkpoint_path, model, optimizer)
    model = model.to(device)

    # 生成
    text = generate_text(model, tokenizer, prompt, max_tokens, temperature, top_p, device)

    print(f"\n{'='*60}")
    print(f"Generated text ({len(text)} chars):")
    print(f"{'='*60}")
    print(text)
    print(f"{'='*60}")

    # 保存到文件
    output_path = LOG_DIR / "generated_text.txt"
    with open(output_path, "w") as f:
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Temperature: {temperature}, Top-p: {top_p}\n")
        f.write(f"Max tokens: {max_tokens}\n")
        f.write(f"{'='*60}\n")
        f.write(text)
    print(f"\nSaved to: {output_path}")

    return text


# ============================================================
# 快速验证：overfit 单个 batch
# ============================================================
def overfit_single_batch(ablation: str = "none", steps: int = 200):
    """验证实现正确性：模型应该能在单个 batch 上快速过拟合到 loss ≈ 0。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = get_default_config()
    config["batch_size"] = 4
    config["context_length"] = 64

    model = build_model(config, ablation).to(device)
    AdamW = get_adamw_cls()
    optimizer = AdamW(model.parameters(), lr=1e-3)

    # 固定的单个 batch（先设 seed 再取样，保证可复现；mmap 避免整个数据集进内存）
    train_data = np.load(TRAIN_NPY, mmap_mode="r")
    np.random.seed(42)
    x, y = run_get_batch(train_data, 4, 64, device)

    print(f"Overfitting single batch ({ablation=}, {steps} steps)...")
    for step in range(steps):
        logits = model(x)
        B, S, V = logits.shape
        loss = run_cross_entropy(logits.reshape(B*S, V), y.reshape(B*S))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0 or step == steps - 1:
            print(f"  step {step:4d}: loss = {loss.item():.6f}")

    if loss.item() < 1.0:
        print(f"✓ Loss dropped to {loss.item():.4f} — implementation looks correct!")
    else:
        print(f"✗ Loss is {loss.item():.4f} — something might be wrong (expected < 1.0)")


# ============================================================
# 参数计数
# ============================================================
def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    embedding = sum(p.numel() for n, p in model.named_parameters() if "embedding" in n or "lm_head" in n)
    non_embedding = total - embedding
    return {"total": total, "embedding": embedding, "non_embedding": non_embedding}


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Training experiments")
    subparsers = parser.add_subparsers(dest="run", help="Experiment to run")

    def add_train_args(p):
        p.add_argument("--lr", type=float, default=3e-4)
        p.add_argument("--batch_size", type=int, default=64)
        p.add_argument("--resume", type=str, default=None,
                       help="显式指定 checkpoint 才会 resume；默认从零训练")
        p.add_argument("--compile", action="store_true",
                       help="开启 torch.compile 加速（建议调通后再用）")
        p.add_argument("--max_grad_norm", type=float, default=None,
                       help="覆盖梯度裁剪阈值；设很大值(如 1e9)等于关闭裁剪，用于复现发散")
        p.add_argument("--tag", type=str, default=None,
                       help="run 名后缀，避免覆盖已有结果（如 baseline_noclip）")

    # 基线训练
    add_train_args(subparsers.add_parser("baseline", help="Baseline training"))

    # 学习率扫描
    lr_parser = subparsers.add_parser("lr_sweep", help="Learning rate sweep")
    lr_parser.add_argument("--lr", type=float, nargs="+",
                            default=[1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3])
    lr_parser.add_argument("--compile", action="store_true")

    # Batch size 扫描
    bs_parser = subparsers.add_parser("batch_sweep", help="Batch size sweep")
    bs_parser.add_argument("--batch_sizes", type=int, nargs="+",
                            default=[1, 16, 64, 128, 256],
                            help="覆盖 1 到显存上限；注意 bs=1 跑满 327M tokens 会很慢")
    bs_parser.add_argument("--lr", type=float, default=3e-4,
                           help="所有 batch size 共用的学习率（默认固定，只变 batch size）")
    bs_parser.add_argument("--scale_lr", action="store_true",
                           help="改用线性 scaling：lr ∝ batch_size")
    bs_parser.add_argument("--small_batch_tokens", type=int, default=20_000_000,
                           help="batch < 阈值的 run 改用此 token 预算（默认 20M），"
                                "避免极小 batch 跑满 327M 耗时过长；设 0 关闭（全部跑满）")
    bs_parser.add_argument("--small_batch_threshold", type=int, default=8,
                           help="batch 小于该值时启用短预算（默认 8，即 bs<8）")
    bs_parser.add_argument("--compile", action="store_true")

    # 生成文本
    gen_parser = subparsers.add_parser("generate", help="Generate text")
    gen_parser.add_argument("--checkpoint", type=str, required=True)
    gen_parser.add_argument("--prompt", type=str, default="Once upon a time")
    gen_parser.add_argument("--max_tokens", type=int, default=256)
    gen_parser.add_argument("--temperature", type=float, default=0.8)
    gen_parser.add_argument("--top_p", type=float, default=0.9)

    # 消融实验
    for ablation_name in ["ablate_no_norm", "ablate_post_norm", "ablate_nope", "ablate_silu"]:
        add_train_args(subparsers.add_parser(ablation_name, help=f"Ablation: {ablation_name}"))

    # Overfit 测试
    subparsers.add_parser("overfit_test", help="Overfit single batch to verify implementation")

    args = parser.parse_args()

    if args.run == "baseline":
        extra = {"max_grad_norm": args.max_grad_norm} if args.max_grad_norm is not None else None
        name = "baseline" + (f"_{args.tag}" if args.tag else "")
        run_single_experiment(name, args.lr, batch_size=args.batch_size,
                              resume=args.resume, compile_model=args.compile,
                              extra_config=extra)

    elif args.run == "lr_sweep":
        run_lr_sweep(args.lr, compile_model=args.compile)

    elif args.run == "batch_sweep":
        sbt = args.small_batch_tokens if args.small_batch_tokens and args.small_batch_tokens > 0 else None
        run_batch_sweep(args.batch_sizes, lr=args.lr, scale_lr=args.scale_lr,
                        compile_model=args.compile,
                        small_batch_tokens=sbt,
                        small_batch_threshold=args.small_batch_threshold)

    elif args.run == "generate":
        run_generation(args.checkpoint, args.prompt, args.max_tokens,
                       args.temperature, args.top_p)

    elif args.run == "ablate_no_norm":
        run_single_experiment("ablate_no_norm", args.lr,
                              ablation="no_norm", batch_size=args.batch_size,
                              resume=args.resume, compile_model=args.compile)

    elif args.run == "ablate_post_norm":
        run_single_experiment("ablate_post_norm", args.lr,
                              ablation="post_norm", batch_size=args.batch_size,
                              resume=args.resume, compile_model=args.compile)

    elif args.run == "ablate_nope":
        run_single_experiment("ablate_nope", args.lr,
                              ablation="nope", batch_size=args.batch_size,
                              resume=args.resume, compile_model=args.compile)

    elif args.run == "ablate_silu":
        run_single_experiment("ablate_silu", args.lr,
                              ablation="silu", batch_size=args.batch_size,
                              resume=args.resume, compile_model=args.compile)

    elif args.run == "overfit_test":
        overfit_single_batch()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
