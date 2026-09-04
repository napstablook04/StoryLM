"""解码配置对比实验：温度 / top-p / 主题词 logit 引导 vs 基线（t=0.8 + top-k 50）。

对 R4 交付 checkpoint 在 16 个探针主题上逐一评测各解码配置：
  - 指标与 sft_probe_eval 完全一致（TMR@2 / any_mention / mention / SSS / TAS）
  - 同种子配对采样（seed 0-4，0 为贪心），降低跨配置方差
  - 额外记录 spam 率（主题词出现 >6 次，防引导把故事带沟里）与生成长度

用法：
    uv run python -X utf8 _decode_compare.py --checkpoint checkpoints/sft_expanded_v4_best_final.pt
"""
import argparse
import json
import time
from pathlib import Path

import torch

from sft_probe_eval import (
    MAX_TOKENS, N_SAMPLES, PROBES, get_cooccurrence, score_story,
)
from storylm.my_transformer import MyTransformerLM
from storylm.tokenizer import Tokenizer
from sft_train import (
    SPECIAL_TOKENS, SFT_VOCAB_SIZE, generate_one, load_sft_checkpoint,
)
from train_experiment import get_default_config

PROJECT_ROOT = Path(__file__).parent
OUT_PATH = PROJECT_ROOT / "experiments" / "decode_compare.json"

# (名字, generate_one 参数; guided=keyword 表示主题引导强度)
CONFIGS = [
    ("base_t0.8_k50",      dict(temperature=0.8, top_k=50)),
    ("t0.7_k50",           dict(temperature=0.7, top_k=50)),
    ("t0.6_k50",           dict(temperature=0.6, top_k=50)),
    ("t0.8_topp0.9",       dict(temperature=0.8, top_k=None, top_p=0.9)),
    ("guide_b1.0",         dict(temperature=0.8, top_k=50, guided=1.0)),
    ("guide_b2.0",         dict(temperature=0.8, top_k=50, guided=2.0)),
    ("guide_b4.0",         dict(temperature=0.8, top_k=50, guided=4.0)),
    ("topp0.9_guide_b2.0", dict(temperature=0.8, top_k=None, top_p=0.9, guided=2.0)),
]


def bias_ids_for(tokenizer, topic):
    """主题词（带前导空格 + 裸形式）的首子词 token id。"""
    ids = set()
    for form in (f" {topic}", topic):
        toks = tokenizer.encode(form)
        if toks:
            ids.add(toks[0])
    return sorted(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/sft_expanded_v4_best_final.pt")
    parser.add_argument("--configs", type=str, default=None,
                        help="逗号分隔的配置名子集（冒烟测试用）")
    parser.add_argument("--topics", type=str, default=None,
                        help="逗号分隔的主题子集（冒烟测试用）")
    args = parser.parse_args()

    configs = CONFIGS
    if args.configs:
        keep = set(args.configs.split(","))
        configs = [(n, c) for n, c in CONFIGS if n in keep]
    active_topics = set(args.topics.split(",")) if args.topics else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cooc = get_cooccurrence()
    cfg = get_default_config()
    tokenizer = Tokenizer.from_files(
        PROJECT_ROOT / "data" / "vocab.json", PROJECT_ROOT / "data" / "merges.txt",
        special_tokens=SPECIAL_TOKENS)
    model = MyTransformerLM(
        vocab_size=SFT_VOCAB_SIZE, context_length=cfg["context_length"],
        d_model=cfg["d_model"], num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"], rope_theta=cfg["rope_theta"])
    load_sft_checkpoint(model, args.checkpoint)
    model.to(device).eval()

    all_topics = [t for ts in PROBES.values() for t in ts]
    all_results = {}

    for name, conf in configs:
        kw = dict(conf)
        guided = kw.pop("guided", 0.0)
        t0 = time.time()
        per_topic = {}
        for band, topics in PROBES.items():
            for topic in topics:
                if active_topics is not None and topic not in active_topics:
                    continue
                instruction = f"Write a short story about {topic}."
                prompt = tokenizer.encode(f"<|user|> {instruction} <|assistant|>")
                b_ids = bias_ids_for(tokenizer, topic) if guided else None
                stories, scores = [], []
                for i in range(N_SAMPLES):
                    torch.manual_seed(i)  # 与 probe 评测一致：0 贪心，1-4 采样
                    temp = 0.0 if i == 0 else kw["temperature"]
                    gen_kw = {k: v for k, v in kw.items() if k != "temperature"}
                    if temp <= 0:
                        gen_kw.pop("top_p", None)
                        gen_kw.pop("top_k", None)
                    ids = generate_one(
                        model, tokenizer, prompt, device, max_new_tokens=MAX_TOKENS,
                        temperature=temp, bias_ids=b_ids,
                        bias_value=guided, bias_until=2, **gen_kw)
                    stories.append(tokenizer.decode(ids[len(prompt):]))
                    scores.append(score_story(stories[-1], topic, cooc[topic]["related"]))
                per_topic[topic] = {
                    "band": band,
                    "tmr2": sum(s["mentions"] >= 2 for s in scores) / len(scores),
                    "any_mention": sum(s["mentions"] >= 1 for s in scores) / len(scores),
                    "mention": sum(s["mention"] for s in scores) / len(scores),
                    "sss": sum(s["sss"] for s in scores) / len(scores),
                    "tas": sum(s["tas"] for s in scores) / len(scores),
                    "spam": sum(s["mentions"] > 6 for s in scores) / len(scores),
                    "stories": stories,
                }

        bands = {}
        for band in PROBES:
            ts = [per_topic[t] for t in PROBES[band] if t in per_topic]
            if not ts:
                continue
            bands[band] = {k: sum(t[k] for t in ts) / len(ts)
                           for k in ("tmr2", "any_mention", "mention", "sss", "tas", "spam")}
        all_results[name] = {"bands": bands, "topics": per_topic,
                             "checkpoint": args.checkpoint, "guided": guided}

        c = bands.get("control(in-list)", {})
        h = bands.get("unseen-high-freq", {})
        m = bands.get("unseen-mid-freq", {})

        def fmt(b, key):
            return f"{b[key]:.0%}" if b else "  -  "

        print(f"[{name}] ({time.time() - t0:.0f}s)  TMR@2: "
              f"control {fmt(c, 'tmr2')} / high {fmt(h, 'tmr2')} / mid {fmt(m, 'tmr2')}  "
              f"any: {fmt(c, 'any_mention')}/{fmt(h, 'any_mention')}/{fmt(m, 'any_mention')}  "
              f"spam: {fmt(c, 'spam')}/{fmt(h, 'spam')}/{fmt(m, 'spam')}")

    OUT_PATH.write_text(json.dumps(all_results, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nwritten: {OUT_PATH}")

    # 定性对照：最弱控制词 sharing 在各配置下的贪心故事开头
    print("\n===== sharing 故事开头（贪心 sample 0）各配置对照 =====")
    for name, _ in configs:
        if "sharing" not in all_results[name]["topics"]:
            continue
        s = all_results[name]["topics"]["sharing"]["stories"][0]
        print(f"\n--- {name} ---\n{s[:200]}")


if __name__ == "__main__":
    main()
