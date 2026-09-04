"""DPO 偏好对挖掘：用 R4 final 检查点对主题指令多次采样，命中主题的 vs
掉进吸引子的天然构成 (chosen, rejected)。

流程（DPO_HANDOFF.md §3.1）：
    1. 从 759 词白名单采主题词构造 B 型指令（探针词及词干变形全部剔除，
       单/双主题 60%/40%——单主题产出率约为双主题 2 倍且评测探针全单主题）
    2. 每条指令生成 5 个：1 贪心 + 4 个 t=0.8 采样（seed 1-4，top_k 50，200 tokens）
    3. 奖励 = Σ_topics min(词干出现次数, 5) + 1·(EOS 正常结尾)
    4. chosen / rejected 必须同结尾状态（都 EOS 或都跑满上限）——
       终止信号两端对称，避免教出"别截断"而非"别跑题"（冒烟实测
       t=0.8 约 75% 样本跑满 150 tokens 不结尾，硬性双 EOS 产出率仅 10%）
    5. chosen：优先每个主题 ≥2 次，退而求其次每个 ≥1 且总 ≥4；
       EOS 结尾、≥30 tokens、奖励最高者
       rejected：t=0.8 样本里同组奖励最低、≥30 tokens，与 chosen 分差 ≥2
    6. 输出 data/dpo_pairs.jsonl

用法：
    冒烟：uv run python -X utf8 _mine_dpo_pairs.py --n-instructions 20 --output data/dpo_pairs_smoke.jsonl --preview
    正式：uv run python -X utf8 _mine_dpo_pairs.py --n-instructions 1500
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch

from prepare_sft_data import WORD_RE, _stem
from sft_probe_eval import _norm_stem
from storylm.my_transformer import MyTransformerLM
from storylm.tokenizer import Tokenizer
from sft_train import (
    SPECIAL_TOKENS, SFT_VOCAB_SIZE, generate_one, load_sft_checkpoint,
)
from train_experiment import get_default_config

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CKPT = PROJECT_ROOT / "checkpoints" / "sft_expanded_v4_best_final.pt"

# 探针词（含 control 组——DPO 若在探针词上训练，t=0.8 评测即泄漏作废）
PROBE_WORDS = [
    "bird", "cake", "park", "sharing",
    "moon", "school", "king", "baby", "farm", "beach",
    "dragon", "robot", "pirate", "train", "rainbow", "zorp",
]
PROBE_STEMS = {_norm_stem(w) for w in PROBE_WORDS}

MAX_NEW_TOKENS = 200
TOP_K = 50
TEMP = 0.8
MIN_STORY_TOKENS = 30   # 两端都要求：太短的样本不是"正常故事"
MIN_REWARD_GAP = 2      # chosen 与 rejected 的最低奖励分差


def build_instructions(n, seed):
    """从白名单采 n 条互不重复的 B 型指令。单/双 60%/40%：
    双主题 chosen 要求两个词都命中（模型常整个忽略第二个词，产出率减半），
    而要修的模式坍缩与评测探针都是单主题指令，以单为主。"""
    words = json.loads(
        (DATA_DIR / "topic_allowlist_expanded.json").read_text(encoding="utf-8")
    )["words"]
    banned = [w for w in words if _norm_stem(w) in PROBE_STEMS]
    pool = [w for w in words if _norm_stem(w) not in PROBE_STEMS]
    print(f"allowlist {len(words)} words, dropped {len(banned)} probe collisions: {banned}")

    rng = random.Random(seed)
    instructions = []
    seen = set()

    def make_one():
        if rng.random() < 0.60:
            topic = rng.choice(pool)
            return f"Write a short story about {topic}.", (topic,)
        a, b = rng.sample(pool, 2)
        if _stem(a) == _stem(b):  # 同词根不组对（"cat and cats" 没有额外信息）
            return None
        return f"Write a short story about {a} and {b}.", (a, b)

    while len(instructions) < n:
        made = make_one()
        if made is None or made[0] in seen:
            continue
        seen.add(made[0])
        instructions.append({"instruction": made[0], "topics": list(made[1])})
    return instructions


def count_mentions(story, topics):
    """与评测同口径：词干归并（_norm_stem）后统计每个主题词的出现次数。"""
    stems = [_norm_stem(w) for w in WORD_RE.findall(story.lower())]
    return [sum(1 for s in stems if s == _norm_stem(t)) for t in topics]


def gen_five(model, tokenizer, device, instruction):
    """1 贪心 + 4 个 t=0.8 采样（seed 1-4）。返回 [(text, n_tokens, ended)]。"""
    prompt = tokenizer.encode(f"<|user|> {instruction} <|assistant|>")
    outs = []
    for i in range(5):
        if i == 0:
            ids = generate_one(model, tokenizer, prompt, device,
                               max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)
        else:
            torch.manual_seed(i)
            ids = generate_one(model, tokenizer, prompt, device,
                               max_new_tokens=MAX_NEW_TOKENS,
                               temperature=TEMP, top_k=TOP_K)
        n_tok = len(ids) - len(prompt)
        # generate_one 在 EOS / 特殊 token 处停且不追加——没到上限即"正常结尾"
        ended = n_tok < MAX_NEW_TOKENS
        outs.append((tokenizer.decode(ids[len(prompt):]), n_tok, ended))
    return outs


def pick_pair(samples, topics):
    """从 5 个样本里选 (chosen, rejected)，返回 (pair 或 None, 失败原因)。

    按结尾状态分组（EOS 组 / 跑满上限组），只在组内配对：终止信号两端
    对称，DPO 学到的差异只剩主题跟随。优先 EOS 组（结尾完整更贴近
    SFT 数据形态），EOS 组配不出再试上限组。
    chosen：组内奖励最高者，主题命中严格档（每个 ≥2）优先、
            宽松档（每个 ≥1 且总 ≥4）兜底；≥30 tokens。
    rejected：仅从 t=0.8 样本选（贪心是模型的众数行为，不适合当反例），
            组内奖励最低、≥30 tokens，与 chosen 分差 ≥2。
    """

    def reward(mentions, ended):
        return sum(min(m, 5) for m in mentions) + (1 if ended else 0)

    scored = []
    for i, (text, n_tok, ended) in enumerate(samples):
        mentions = count_mentions(text, topics)
        scored.append({"i": i, "greedy": i == 0, "text": text, "n_tok": n_tok,
                       "ended": ended, "mentions": mentions,
                       "reward": reward(mentions, ended),
                       "strict": all(m >= 2 for m in mentions)})

    def chosen_ok(s):
        if s["n_tok"] < MIN_STORY_TOKENS:
            return False
        if s["strict"]:
            return True
        return all(m >= 1 for m in s["mentions"]) and sum(s["mentions"]) >= 4

    def rank_key(s):  # 严格档排前（True > False），再按奖励、长度
        return (s["strict"], s["reward"], s["n_tok"])

    fail = "no_chosen"
    for want_ended in (True, False):  # 先 EOS 组，后上限组
        members = [s for s in scored if s["ended"] == want_ended]
        chosen = max(
            (s for s in members if chosen_ok(s)), key=rank_key, default=None
        )
        if chosen is None:
            continue
        rejected = min(
            (s for s in members
             if not s["greedy"] and s is not chosen and s["n_tok"] >= MIN_STORY_TOKENS),
            key=lambda s: s["reward"], default=None,
        )
        if rejected is None or chosen["reward"] - rejected["reward"] < MIN_REWARD_GAP:
            fail = "no_gap"
            continue
        return {
            "instruction": None,  # 由调用方填
            "topics": list(topics),
            "chosen": chosen["text"].strip(),
            "rejected": rejected["text"].strip(),
            "chosen_mentions": chosen["mentions"],
            "rejected_mentions": rejected["mentions"],
            "chosen_reward": chosen["reward"],
            "rejected_reward": rejected["reward"],
            "chosen_tokens": chosen["n_tok"],
            "rejected_tokens": rejected["n_tok"],
            "both_ended": want_ended,
        }, None
    return None, fail


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    parser.add_argument("--n-instructions", type=int, default=1500)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "dpo_pairs.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", action="store_true", help="冒烟时抽读前 3 对")
    args = parser.parse_args()

    instructions = build_instructions(args.n_instructions, args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_default_config()
    tokenizer = Tokenizer.from_files(
        DATA_DIR / "vocab.json", DATA_DIR / "merges.txt", special_tokens=SPECIAL_TOKENS)
    model = MyTransformerLM(
        vocab_size=SFT_VOCAB_SIZE, context_length=cfg["context_length"],
        d_model=cfg["d_model"], num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"], rope_theta=cfg["rope_theta"])
    load_sft_checkpoint(model, str(args.checkpoint))
    model.to(device).eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_pairs = 0
    stats = {"no_chosen": 0, "no_gap": 0}
    with args.output.open("w", encoding="utf-8") as f:
        for idx, rec in enumerate(instructions):
            samples = gen_five(model, tokenizer, device, rec["instruction"])
            pair, why = pick_pair(samples, rec["topics"])
            if pair is None:
                stats[why] += 1
                continue
            pair["instruction"] = rec["instruction"]
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            f.flush()
            n_pairs += 1

            if (idx + 1) % 25 == 0:
                el = time.time() - t0
                eta = el / (idx + 1) * (len(instructions) - idx - 1)
                print(f"[{idx + 1}/{len(instructions)}] pairs={n_pairs} "
                      f"({n_pairs / (idx + 1):.0%})  elapsed={el / 60:.1f}m  "
                      f"eta={eta / 60:.1f}m", flush=True)

    el = time.time() - t0
    print(f"\nDone: {n_pairs}/{len(instructions)} instructions -> {args.output}")
    print(f"failures: {stats}  wall={el / 60:.1f}m  "
          f"({el / max(len(instructions), 1):.1f}s/instruction)")

    if args.preview:
        with args.output.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                p = json.loads(line)
                print(f"\n{'=' * 70}\n[{i}] {p['instruction']}")
                print(f"-- chosen (mentions={p['chosen_mentions']} r={p['chosen_reward']}):")
                print(p["chosen"][:400])
                print(f"-- rejected (mentions={p['rejected_mentions']} r={p['rejected_reward']}):")
                print(p["rejected"][:400])


if __name__ == "__main__":
    main()
