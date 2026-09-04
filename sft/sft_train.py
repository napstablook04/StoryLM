"""
SFT（指令微调）训练脚本：在 baseline_best.pt 之上微调 StoryLM。

与 train_experiment.py（预训练）的关系：
    - 复用同一套模型结构 / AdamW / 梯度裁剪 / cosine 调度（storylm.train）
    - 新增三件 SFT 专属的东西：
        1. checkpoint 手术：词表 10000 -> 10002（<|assistant|>/<|user|> 两行新嵌入
           用旧词表均值初始化，避免随机噪声造成离谱 logits）
        2. response-only loss：label 加 -100 mask，prompt / padding 不计损失
           （--loss-mode full 为消融对照组：全序列计损失）
        3. 按 epoch 遍历 jsonl 样本（SFT 每条数据只看一遍/epoch，
           不像预训练那样在 token 流上随机开窗）

用法：
    # 正式 SFT（response-only loss）
    uv run python sft_train.py train --run-name sft_response

    # 消融：full loss 对照组
    uv run python sft_train.py train --run-name sft_full --loss-mode full

    # 生成对比：SFT vs 预训练 baseline
    uv run python sft_train.py generate --checkpoint checkpoints/sft_response_best.pt --compare baseline

显存注意：1050 Ti 4GB 实测 bs=16 可跑（峰值 2.14GB），bs>=32 OOM，默认即 16。
"""

import argparse
import json
import math
import random
import re
import time
from pathlib import Path

import numpy as np
import torch

from storylm.my_transformer import MyTransformerLM
from storylm.tokenizer import Tokenizer
from storylm.train import (
    run_cross_entropy_ignore_index,
    run_gradient_clipping,
    run_get_lr_cosine_schedule,
    get_adamw_cls,
    run_save_checkpoint,
)
from train_experiment import ExperimentLogger, get_default_config, CHECKPOINT_DIR

PROJECT_ROOT = Path(__file__).parent

DATA_DIR = PROJECT_ROOT / "data"
VOCAB_PATH = DATA_DIR / "vocab.json"
MERGES_PATH = DATA_DIR / "merges.txt"
TRAIN_JSONL = DATA_DIR / "sft_train.jsonl"
VALID_JSONL = DATA_DIR / "sft_valid.jsonl"
BASELINE_CKPT = PROJECT_ROOT / "checkpoint" / "baseline_best.pt"

SPECIAL_TOKENS = ["<|endoftext|>", "<|user|>", "<|assistant|>"]
# 词表布局：0-255 是 256 个单字节 token，<|endoftext|>=256，BPE 合并 token 从 257 起，
# SFT 新增 <|assistant|>=10000、<|user|>=10001（由 prepare_sft_data.py 固化进 vocab.json）
EOS_ID = 256
IGNORE_INDEX = -100

# 模型结构必须与预训练一致，只有 vocab_size 从 10000 扩到 10002
SFT_VOCAB_SIZE = 10002


# ============================================================
# 数据
# ============================================================
def load_jsonl(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def run_get_sft_batch(samples, indices, context_length, device, loss_mode):
    """把一组 jsonl 样本 pad 成一个 batch，返回 (x, labels)。

    - 右填充 + 模型内置因果 mask：pad 位置在真实 token 之后，不会向前泄漏，
      模型结构零改动。
    - label[t] = x[t+1]（预测下一个 token）：
        response 模式：t 从 prompt_len-1 开始（即 <|assistant|> 位置预测
        第一个回答 token），prompt 段与 padding 段全部 -100；
        full 模式：t 从 0 开始，prompt 也计损失（消融对照）。
    - 末尾 <|endoftext|> 计入损失，模型由此学会"回答完就停"。
    """
    seqs = [samples[i]["input_ids"] for i in indices]
    seq_len = max(len(s) for s in seqs)
    x = torch.full((len(seqs), seq_len), EOS_ID, dtype=torch.long)
    labels = torch.full((len(seqs), seq_len), IGNORE_INDEX, dtype=torch.long)

    for row, i in enumerate(indices):
        rec = samples[i]
        seq = rec["input_ids"]
        length = min(len(seq), context_length)  # 数据侧已过滤 >256，防御性截断
        x[row, :length] = torch.tensor(seq[:length], dtype=torch.long)
        first_pred = 0 if loss_mode == "full" else rec["prompt_len"] - 1
        if first_pred < length - 1:
            labels[row, first_pred:length - 1] = x[row, first_pred + 1:length]

    return x.to(device), labels.to(device)


@torch.no_grad()
def evaluate_sft_loss(model, samples, context_length, device, batch_size=16):
    """验证集 response-only 交叉熵（按有效 token 数加权平均）。

    无论训练用哪种 loss_mode，验证指标固定为 response-only——
    这是两种模式真正要比较的目标："给定指令，回答写得好不好"。
    """
    was_training = model.training
    model.eval()
    loss_sum, token_sum = 0.0, 0
    for start in range(0, len(samples), batch_size):
        indices = list(range(start, min(start + batch_size, len(samples))))
        x, labels = run_get_sft_batch(samples, indices, context_length, device, "response")
        logits = model(x)
        vocab = logits.shape[-1]
        n_valid = int((labels.reshape(-1) != IGNORE_INDEX).sum())
        loss = run_cross_entropy_ignore_index(logits.reshape(-1, vocab), labels.reshape(-1))
        loss_sum += loss.item() * n_valid
        token_sum += n_valid
    if was_training:
        model.train()
    return loss_sum / max(token_sum, 1)


# ============================================================
# Checkpoint 手术：加载预训练权重 + 扩词表
# ============================================================
def load_pretrained_weights(model, checkpoint_path):
    """把 baseline checkpoint 装进 vocab_size=10002 的模型。

    token_embeddings / lm_head 各扩 2 行（<|assistant|>=10000、<|user|>=10001），
    新行用旧 10000 行的均值初始化：新 token 从"平均语义"出发而非随机噪声，
    初始 logits 不离谱，SFT 起步稳定。其余参数 strict 加载，保证真的加载成功。
    """
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = dict(ck["model_state_dict"])
    old_vocab = state["token_embeddings"].shape[0]
    n_add = model.vocab_size - old_vocab
    if n_add > 0:
        for key in ("token_embeddings", "lm_head"):
            mean_row = state[key].mean(dim=0, keepdim=True)
            state[key] = torch.cat([state[key], mean_row.expand(n_add, -1).clone()], dim=0)
    model.load_state_dict(state)  # strict=True：形状对不上立刻报错
    print(f"Loaded pretrained weights from {checkpoint_path} (iteration {ck.get('iteration')}); "
          f"vocab {old_vocab} -> {model.vocab_size}, new rows = mean of old rows")


def load_sft_checkpoint(model, checkpoint_path):
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    print(f"Loaded SFT checkpoint {checkpoint_path} (iteration {ck.get('iteration')})")


# ============================================================
# 训练
# ============================================================
def sft_train(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    base_cfg = get_default_config()
    context_length = base_cfg["context_length"]

    train_samples = load_jsonl(args.train_data)
    valid_samples = load_jsonl(args.valid_data)
    print(f"Train samples: {len(train_samples)}   Valid samples: {len(valid_samples)}")

    model = MyTransformerLM(
        vocab_size=SFT_VOCAB_SIZE,
        context_length=context_length,
        d_model=base_cfg["d_model"],
        num_layers=base_cfg["num_layers"],
        num_heads=base_cfg["num_heads"],
        d_ff=base_cfg["d_ff"],
        rope_theta=base_cfg["rope_theta"],
    )
    load_pretrained_weights(model, args.init_checkpoint)
    model.to(device)

    # SFT 用全新优化器状态：预训练的 AdamW 动量对新任务（尤其新扩的 2 行）无意义
    AdamW = get_adamw_cls()
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(base_cfg["beta1"], base_cfg["beta2"]),
        eps=base_cfg["eps"],
    )

    steps_per_epoch = math.ceil(len(train_samples) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    min_lr = args.lr * args.min_lr_ratio

    config = {
        "init_checkpoint": str(args.init_checkpoint),
        "loss_mode": args.loss_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "min_lr": min_lr,
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
        "weight_decay": args.weight_decay,
        "max_grad_norm": base_cfg["max_grad_norm"],
        "seed": args.seed,
        "train_samples": len(train_samples),
        "valid_samples": len(valid_samples),
        "note": "val metric is always response-only CE",
    }
    print(f"{'=' * 60}")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"  device: {device}")
    print(f"{'=' * 60}")

    logger = ExperimentLogger(args.run_name)
    logger.save_config(config)

    ckpt_path = str(CHECKPOINT_DIR / f"{args.run_name}_best.pt")
    order = list(range(len(train_samples)))
    rng = random.Random(args.seed)
    best_val = float("inf")
    start_time = time.time()
    step = 0

    try:
        for epoch in range(args.epochs):
            rng.shuffle(order)
            for bstart in range(0, len(order), args.batch_size):
                indices = order[bstart:bstart + args.batch_size]

                lr = run_get_lr_cosine_schedule(
                    it=step,
                    max_learning_rate=args.lr,
                    min_learning_rate=min_lr,
                    warmup_iters=warmup_steps,
                    cosine_cycle_iters=total_steps,
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                model.train()
                x, labels = run_get_sft_batch(
                    train_samples, indices, context_length, device, args.loss_mode
                )
                logits = model(x)
                vocab = logits.shape[-1]
                loss = run_cross_entropy_ignore_index(
                    logits.reshape(-1, vocab), labels.reshape(-1)
                )

                if not torch.isfinite(loss):
                    raise RuntimeError(f"Diverged: non-finite loss at step {step}")

                loss.backward()
                run_gradient_clipping(model.parameters(), base_cfg["max_grad_norm"])
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                val_loss = None
                if step % args.val_interval == 0 or step == total_steps:
                    val_loss = evaluate_sft_loss(
                        model, valid_samples, context_length, device, args.batch_size
                    )
                    if not math.isfinite(val_loss):
                        raise RuntimeError(f"Diverged: non-finite val loss at step {step}")
                    if val_loss < best_val:
                        best_val = val_loss
                        run_save_checkpoint(model, optimizer, step, ckpt_path)
                    wall = time.time() - start_time
                    print(f"[Step {step:4d}/{total_steps}] ep{epoch} "
                          f"train_loss={loss.item():.4f}  val_loss(resp)={val_loss:.4f}  "
                          f"lr={lr:.2e}  time={wall:.0f}s")
                elif step % 50 == 0:
                    wall = time.time() - start_time
                    print(f"[Step {step:4d}/{total_steps}] ep{epoch} "
                          f"train_loss={loss.item():.4f}  lr={lr:.2e}  time={wall:.0f}s")

                logger.log(step, loss.item(), val_loss, lr, time.time() - start_time)

        run_save_checkpoint(
            model, optimizer, total_steps, ckpt_path.replace(".pt", "_final.pt")
        )
        print(f"\nSFT complete. Best response-only val loss: {best_val:.4f}")
        print(f"Total wall time: {time.time() - start_time:.0f}s")
    finally:
        logger.close()


# ============================================================
# 生成对比
# ============================================================
DEMO_PROMPTS = [
    "Continue this story: One day, Mia found a small blue box in the garden. "
    "The box had a little golden key on it.",
    "Continue this story: Tom and his dad went to the park. Tom saw a big red "
    "ball under a tree.",
    "Write a short story about a bird and a cat.",
    "Write a short story about a birthday cake.",
    "Write a short story about rain.",
]


@torch.no_grad()
def generate_one(model, tokenizer, prompt_ids, device, max_new_tokens=200,
                 temperature=0.8, top_k=50, top_p=None,
                 bias_ids=None, bias_value=0.0, bias_until=0):
    """从 prompt_ids 自回归采样到 EOS / 新特殊 token / 上限为止。

    temperature + top-k/top-p 手写实现（与项目 from-scratch 风格一致）。
    受控生成（推理端主题引导）：主题词首子词出现 bias_until 次之前，
    每步给 bias_ids 这些 token 的 logits 加 bias_value。
    """
    ids = list(prompt_ids)
    context_length = model.context_length
    bias_set = set(bias_ids) if bias_ids else set()
    bias_count = 0
    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([ids[-context_length:]], dtype=torch.long, device=device)
            logits = model(x)[0, -1]
            if bias_set and bias_count < bias_until:
                for tid in bias_set:
                    logits[tid] += bias_value
            if temperature <= 0:
                next_id = int(torch.argmax(logits))
            else:
                logits = logits / temperature
                if top_k and top_k < logits.shape[-1]:
                    kth = torch.topk(logits, top_k).values[-1]
                    logits = torch.where(logits < kth, torch.tensor(float("-inf"), device=device), logits)
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    probs_sorted = torch.softmax(sorted_logits, dim=-1)
                    cum = torch.cumsum(probs_sorted, dim=-1)
                    keep = (cum - probs_sorted) < top_p  # 自身之前累计质量 < p 的保留
                    keep[0] = True
                    sorted_logits = torch.where(keep, sorted_logits,
                                                torch.tensor(float("-inf"), device=device))
                    logits = torch.full_like(logits, float("-inf")).scatter_(-1, sorted_idx, sorted_logits)
                probs = torch.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, 1))
            # 停在 <|endoftext|>；若模型误生成 <|user|>/<|assistant|>（id>=10000）也视为结束
            if next_id == EOS_ID or next_id >= 10000:
                break
            ids.append(next_id)
            if next_id in bias_set:
                bias_count += 1
    return ids


def sft_generate(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    base_cfg = get_default_config()
    context_length = base_cfg["context_length"]
    tokenizer = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, special_tokens=SPECIAL_TOKENS)

    common = dict(vocab_size=SFT_VOCAB_SIZE, context_length=context_length,
                  d_model=base_cfg["d_model"], num_layers=base_cfg["num_layers"],
                  num_heads=base_cfg["num_heads"], d_ff=base_cfg["d_ff"],
                  rope_theta=base_cfg["rope_theta"])

    sft_model = MyTransformerLM(**common).to(device)
    load_sft_checkpoint(sft_model, args.checkpoint)
    sft_model.eval()

    baseline_model = None
    if args.compare == "baseline":
        baseline_model = MyTransformerLM(**common | {"vocab_size": 10000}).to(device)
        load_pretrained_weights(baseline_model, BASELINE_CKPT)
        baseline_model.eval()

    for instruction in DEMO_PROMPTS:
        print(f"\n{'=' * 70}\nINSTRUCTION: {instruction}\n{'-' * 70}")

        # SFT：chat 模板 prompt
        prompt = tokenizer.encode(f"<|user|> {instruction} <|assistant|>")
        ids = generate_one(sft_model, tokenizer, prompt, device,
                           max_new_tokens=args.num_tokens)
        print(f"SFT:\n{tokenizer.decode(ids[len(prompt):]).strip()}")

        # baseline：没学过特殊 token，直接喂原始指令文本（公平对照）
        if baseline_model is not None:
            raw_ids = tokenizer.encode(instruction)
            ids = generate_one(baseline_model, tokenizer, raw_ids, device,
                               max_new_tokens=args.num_tokens)
            print(f"\nBASELINE (no stop training, runs to cap):\n{tokenizer.decode(ids[len(raw_ids):]).strip()}")


def extract_topic(instruction):
    """从 B 类指令抽主题词："Write a short story about X." → X 的最后一个词。"""
    m = re.search(r"about\s+(.+?)[.!]?\s*$", instruction, re.IGNORECASE)
    if not m:
        return None
    words = m.group(1).strip().strip("\"'!?.").split()
    return words[-1].lower() if words else None


def sft_chat(args):
    """交互式对话：输入一条指令，模型现场生成一个回答。"""
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    base_cfg = get_default_config()
    tokenizer = Tokenizer.from_files(VOCAB_PATH, MERGES_PATH, special_tokens=SPECIAL_TOKENS)

    model = MyTransformerLM(
        vocab_size=SFT_VOCAB_SIZE,
        context_length=base_cfg["context_length"],
        d_model=base_cfg["d_model"],
        num_layers=base_cfg["num_layers"],
        num_heads=base_cfg["num_heads"],
        d_ff=base_cfg["d_ff"],
        rope_theta=base_cfg["rope_theta"],
    )
    load_sft_checkpoint(model, args.checkpoint)
    model.to(device)
    model.eval()

    print("\nStoryLM SFT 交互模式。输入指令回车生成；temperature=0 贪心，0.8 带随机性。")
    print("任务 A：Continue this story: <开头两句>")
    print("任务 B：Write a short story about <主题>.")
    if args.bias_value > 0:
        print(f"主题引导已开启：从 about <主题> 指令自动抽词，bias={args.bias_value}，"
              f"出现 {args.bias_until} 次后停止引导")
    print("输入 quit 退出。\n")
    while True:
        try:
            instruction = input("user> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not instruction:
            continue
        if instruction.lower() in {"quit", "exit", "q"}:
            break
        prompt = tokenizer.encode(f"<|user|> {instruction} <|assistant|>")
        b_ids = None
        if args.bias_value > 0:
            topic = extract_topic(instruction)
            if topic:
                b_ids = sorted({tokenizer.encode(f" {topic}")[0],
                                tokenizer.encode(topic)[0]})
        ids = generate_one(model, tokenizer, prompt, device,
                           max_new_tokens=args.num_tokens,
                           temperature=args.temperature, top_k=args.top_k,
                           top_p=args.top_p,
                           bias_ids=b_ids, bias_value=args.bias_value,
                           bias_until=args.bias_until)
        print("assistant> " + tokenizer.decode(ids[len(prompt):]).strip() + "\n")


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="StoryLM SFT: train / generate")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="SFT 训练")
    p_train.add_argument("--run-name", type=str, default="sft_response")
    p_train.add_argument("--init-checkpoint", type=Path, default=BASELINE_CKPT)
    p_train.add_argument("--train-data", type=Path, default=TRAIN_JSONL)
    p_train.add_argument("--valid-data", type=Path, default=VALID_JSONL)
    p_train.add_argument("--loss-mode", choices=["response", "full"], default="response")
    p_train.add_argument("--epochs", type=int, default=3)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--min-lr-ratio", type=float, default=0.1)
    p_train.add_argument("--warmup-ratio", type=float, default=0.03)
    p_train.add_argument("--weight-decay", type=float, default=0.1)
    p_train.add_argument("--val-interval", type=int, default=200, help="每多少步验证一次")
    p_train.add_argument("--device", type=str, default=None)
    p_train.add_argument("--seed", type=int, default=42)

    p_gen = sub.add_parser("generate", help="SFT vs baseline 生成对比")
    p_gen.add_argument("--checkpoint", type=Path,
                       default=CHECKPOINT_DIR / "sft_response_best.pt")
    p_gen.add_argument("--compare", choices=["none", "baseline"], default="baseline")
    p_gen.add_argument("--num-tokens", type=int, default=200)
    p_gen.add_argument("--device", type=str, default=None)
    p_gen.add_argument("--seed", type=int, default=42)

    p_chat = sub.add_parser("chat", help="交互式对话")
    p_chat.add_argument("--checkpoint", type=Path,
                        default=CHECKPOINT_DIR / "sft_response_best.pt")
    p_chat.add_argument("--temperature", type=float, default=0.8)
    p_chat.add_argument("--top-k", type=int, default=50)
    p_chat.add_argument("--top-p", type=float, default=None)
    p_chat.add_argument("--num-tokens", type=int, default=200)
    p_chat.add_argument("--bias-value", type=float, default=0.0,
                        help="主题词 logit 引导强度（0 关闭；2.0 为推荐值）")
    p_chat.add_argument("--bias-until", type=int, default=2,
                        help="主题词出现多少次后停止引导")
    p_chat.add_argument("--device", type=str, default=None)

    args = parser.parse_args()
    if args.command == "train":
        sft_train(args)
    elif args.command == "generate":
        sft_generate(args)
    elif args.command == "chat":
        sft_chat(args)


if __name__ == "__main__":
    main()
