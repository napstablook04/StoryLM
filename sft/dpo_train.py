"""DPO（Direct Preference Optimization）训练器：在 SFT 终点模型上修复
t=0.8 采样的模式坍缩（掉进罐头吸引子故事、主题词零出现）。

损失：
    L = -log σ( β · [ (logπ(y_w|x) - logπ_ref(y_w|x))
                    - (logπ(y_l|x) - logπ_ref(y_l|x)) ] )
与 SFT（只有正梯度）不同，负样本拿到显式下压梯度，正好对应
"吸引子罐头故事压不下去"的残留问题。

偏好对来自 _mine_dpo_pairs.py（同一指令多次采样，命中主题 vs 坍缩）。
tokenization 与 SFT 完全同构：chat 模板 + prompt_len 做 response-only
掩码；两端同结尾状态的对子才补 <|endoftext|>（终止信号对称，
见挖掘脚本的配对规则）。

用法（Windows cmd）：
    set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128&& uv run python -X utf8 dpo_train.py
    冒烟：uv run python -X utf8 dpo_train.py --smoke
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from storylm.my_transformer import MyTransformerLM
from storylm.tokenizer import Tokenizer
from storylm.train import (
    run_get_lr_cosine_schedule,
    run_gradient_clipping,
    get_adamw_cls,
    run_save_checkpoint,
)
from sft_train import (
    EOS_ID, SPECIAL_TOKENS, SFT_VOCAB_SIZE, load_jsonl, load_sft_checkpoint,
)
from train_experiment import CHECKPOINT_DIR, ExperimentLogger, get_default_config

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
VOCAB_PATH = DATA_DIR / "vocab.json"
MERGES_PATH = DATA_DIR / "merges.txt"
SFT_FINAL_CKPT = PROJECT_ROOT / "checkpoints" / "sft_expanded_v4_best_final.pt"

CONTEXT_LENGTH = 256


def load_pairs(path):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pairs.append({
                "instruction": rec["instruction"],
                "chosen": rec["chosen"],
                "rejected": rec["rejected"],
                "both_ended": rec.get("both_ended", True),
            })
    return pairs


def fake_pairs_from_sft(path, n, seed):
    """冒烟用假对：R4 B 类样本两两拼成 (chosen, rejected)。
    只为验证训练循环 / 日志 / 无 NaN，语义无意义。"""
    recs = [r for r in load_jsonl(path) if r["task"] == "B"]
    rng = random.Random(seed)
    rng.shuffle(recs)
    return [
        {"instruction": a["instruction"], "chosen": a["response"],
         "rejected": b["response"], "both_ended": True}
        for a, b in zip(recs[:n], recs[n:2 * n])
    ]


def encode_pair_seqs(pair, tokenizer, max_len=CONTEXT_LENGTH):
    """chosen / rejected 各编码成完整序列，返回 [(ids, prompt_len), ...]。

    与 prepare_sft_data.encode_pair 同 trick：prefix 单独编码取 prompt_len，
    两次编码中特殊 token 切段一致，token 边界必然相同。
    挖掘时两端都自然结尾的对补 <|endoftext|>（同 SFT 数据）；
    截断对不补——配对规则保证两端结尾状态相同，比较只落在续写内容上。
    """
    prefix = f"<|user|> {pair['instruction']} <|assistant|>"
    prompt_len = len(tokenizer.encode(prefix))
    suffix = " <|endoftext|>" if pair["both_ended"] else ""
    out = []
    for resp in (pair["chosen"], pair["rejected"]):
        ids = tokenizer.encode(f"{prefix} {resp}{suffix}")[:max_len]
        out.append((ids, min(prompt_len, len(ids))))
    return out


def build_batch(pair_seqs, device):
    """4 条 (ids, prompt_len) pad 成 (x, mask)：mask 标出 response 段的
    预测位置（t 处预测 t+1，t 从 prompt_len-1 到 len-2）。"""
    seq_len = max(len(ids) for ids, _ in pair_seqs)
    x = torch.full((len(pair_seqs), seq_len), EOS_ID, dtype=torch.long)
    mask = torch.zeros(len(pair_seqs), seq_len, dtype=torch.bool)
    for row, (ids, prompt_len) in enumerate(pair_seqs):
        n = len(ids)
        x[row, :n] = torch.tensor(ids, dtype=torch.long)
        mask[row, prompt_len - 1:n - 1] = True  # 预测 prompt 末位之后的所有 token
    return x.to(device), mask.to(device)


def seq_logps(logits, x, mask):
    """每条序列 response 段的 logprob 总和与 token 数。

    logits: (B, T, V)；mask[t]=True 表示 logits[t] 用于预测 x[t+1]。
    gather 出每个位置目标 token 的 logprob 后按行（乘 mask）求和，
    返回 (sums, counts)，counts 供 per-token 均值诊断。
    """
    logprobs = F.log_softmax(logits[:, :-1].float(), dim=-1)
    token_logps = logprobs.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)
    pos = mask[:, :-1]
    sums = (token_logps * pos).sum(dim=-1)
    counts = pos.sum(dim=-1).clamp(min=1)
    return sums, counts


class DPOModel:
    """policy + 冻结 ref 的组合：一次前向拿到两模型的两端 logprob。"""

    def __init__(self, policy, ref, device, beta):
        self.policy = policy
        self.ref = ref
        self.device = device
        self.beta = beta

    def forward_pairs(self, x, mask):
        """x 形如 [c0, r0, c1, r1, ...]。返回 DPO loss 与诊断量。"""
        pol_sums, cnts = seq_logps(self.policy(x), x, mask)
        with torch.no_grad():
            ref_sums, _ = seq_logps(self.ref(x), x, mask)
        n = pol_sums.shape[0] // 2
        pi_c, pi_r = pol_sums[0::2], pol_sums[1::2]
        rf_c, rf_r = ref_sums[0::2], ref_sums[1::2]
        logits = self.beta * ((pi_c - rf_c) - (pi_r - rf_r))
        loss = -F.logsigmoid(logits).mean()
        return {
            "loss": loss,
            "accuracy": (logits > 0).float().mean().item(),
            "margin": logits.mean().item(),
            "chosen_logp": pi_c.mean().item(),
            "rejected_logp": pi_r.mean().item(),
            "chosen_logp_per_tok": (pi_c / cnts[0::2]).mean().item(),
            "rejected_logp_per_tok": (pi_r / cnts[1::2]).mean().item(),
        }


@torch.no_grad()
def evaluate(model, pairs, tokenizer, device, batch_pairs):
    model.policy.eval()
    sums = {"loss": 0.0, "accuracy": 0.0, "margin": 0.0,
            "chosen_logp_per_tok": 0.0}
    n_batches = 0
    for start in range(0, len(pairs), batch_pairs):
        chunk = pairs[start:start + batch_pairs]
        seqs = [s for p in chunk for s in encode_pair_seqs(p, tokenizer)]
        x, mask = build_batch(seqs, device)
        stats = model.forward_pairs(x, mask)
        for k in sums:
            sums[k] += stats[k]
        n_batches += 1
    model.policy.train()
    return {k: v / n_batches for k, v in sums.items()}


def main():
    parser = argparse.ArgumentParser(description="StoryLM DPO 训练")
    parser.add_argument("--pairs", type=Path, default=DATA_DIR / "dpo_pairs.jsonl")
    parser.add_argument("--init-checkpoint", type=Path, default=SFT_FINAL_CKPT)
    parser.add_argument("--run-name", type=str, default="sft_dpo_v1")
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-pairs", type=int, default=8,
                        help="每步的对数（序列数 = 2×，显存与 SFT bs16 同级）")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-pairs", type=int, default=100)
    parser.add_argument("--val-interval", type=int, default=25)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="假对冒烟：R4 数据拼 3 步，产物带 _smoke 后缀")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    base_cfg = get_default_config()

    tokenizer = Tokenizer.from_files(
        VOCAB_PATH, MERGES_PATH, special_tokens=SPECIAL_TOKENS)
    common = dict(vocab_size=SFT_VOCAB_SIZE, context_length=base_cfg["context_length"],
                  d_model=base_cfg["d_model"], num_layers=base_cfg["num_layers"],
                  num_heads=base_cfg["num_heads"], d_ff=base_cfg["d_ff"],
                  rope_theta=base_cfg["rope_theta"])
    policy = MyTransformerLM(**common)
    load_sft_checkpoint(policy, str(args.init_checkpoint))
    ref = MyTransformerLM(**common)
    load_sft_checkpoint(ref, str(args.init_checkpoint))
    for p in ref.parameters():
        p.requires_grad_(False)
    policy.to(device).train()
    ref.to(device).eval()

    if args.smoke:
        pairs = fake_pairs_from_sft(DATA_DIR / "sft_train_r4.jsonl", 44, args.seed)
        args.epochs = 1
        args.run_name += "_smoke"
    else:
        pairs = load_pairs(args.pairs)

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_val = min(args.val_pairs, len(pairs) // 5) if not args.smoke else 8
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]

    steps_per_epoch = math.ceil(len(train_pairs) / args.batch_pairs)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    min_lr = args.lr * args.min_lr_ratio
    if args.smoke:
        total_steps = min(total_steps, 3)

    AdamW = get_adamw_cls()
    optimizer = AdamW(policy.parameters(), lr=args.lr,
                      weight_decay=args.weight_decay,
                      betas=(base_cfg["beta1"], base_cfg["beta2"]),
                      eps=base_cfg["eps"])

    config = {
        "init_checkpoint": str(args.init_checkpoint),
        "pairs_file": str(args.pairs),
        "train_pairs": len(train_pairs), "val_pairs": len(val_pairs),
        "beta": args.beta, "epochs": args.epochs,
        "batch_pairs": args.batch_pairs, "lr": args.lr, "min_lr": min_lr,
        "warmup_steps": warmup_steps, "total_steps": total_steps,
        "weight_decay": args.weight_decay, "seed": args.seed,
        "smoke": args.smoke,
    }
    print("=" * 60)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"  device: {device}")
    print("=" * 60)

    logger = ExperimentLogger(args.run_name)
    logger.save_config(config)
    model = DPOModel(policy, ref, device, args.beta)
    ckpt_path = str(CHECKPOINT_DIR / f"{args.run_name}_final.pt")

    start_time = time.time()
    step = 0
    baseline_tok_logp = None
    order = list(range(len(train_pairs)))
    stop = False
    try:
        for epoch in range(args.epochs):
            if stop:
                break
            rng.shuffle(order)
            for bstart in range(0, len(order), args.batch_pairs):
                indices = order[bstart:bstart + args.batch_pairs]
                lr = run_get_lr_cosine_schedule(
                    it=step, max_learning_rate=args.lr, min_learning_rate=min_lr,
                    warmup_iters=warmup_steps, cosine_cycle_iters=total_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                chunk = [train_pairs[i] for i in indices]
                seqs = [s for p in chunk for s in encode_pair_seqs(p, tokenizer)]
                x, mask = build_batch(seqs, device)
                stats = model.forward_pairs(x, mask)

                if baseline_tok_logp is None:
                    baseline_tok_logp = stats["chosen_logp_per_tok"]
                if not torch.isfinite(stats["loss"]):
                    raise RuntimeError(f"Diverged: non-finite loss at step {step}")

                stats["loss"].backward()
                run_gradient_clipping(policy.parameters(), base_cfg["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                val_loss = None
                if (step % args.val_interval == 0 or step == total_steps) and val_pairs:
                    val = evaluate(model, val_pairs, tokenizer, device,
                                   args.batch_pairs)
                    val_loss = val["loss"]
                    print(f"          [val] loss={val['loss']:.4f} "
                          f"acc={val['accuracy']:.2f} margin={val['margin']:.2f} "
                          f"chosen logp/tok={val['chosen_logp_per_tok']:.3f}",
                          flush=True)

                if step % args.log_interval == 0 or step == total_steps:
                    print(f"[Step {step:4d}/{total_steps}] loss={stats['loss']:.4f} "
                          f"acc={stats['accuracy']:.2f} margin={stats['margin']:.2f} "
                          f"logp/tok chosen={stats['chosen_logp_per_tok']:.3f} "
                          f"rejected={stats['rejected_logp_per_tok']:.3f} "
                          f"(chosen 起点 {baseline_tok_logp:.3f}) lr={lr:.2e} "
                          f"time={time.time() - start_time:.0f}s", flush=True)
                logger.log(step, stats["loss"], val_loss, lr, time.time() - start_time)

                if step >= total_steps:
                    stop = True
                    break
    finally:
        logger.close()

    run_save_checkpoint(policy, optimizer, step, ckpt_path)
    print(f"\nDPO complete ({step} steps). Saved: {ckpt_path}")
    print(f"Total wall time: {time.time() - start_time:.0f}s")


if __name__ == "__main__":
    main()
