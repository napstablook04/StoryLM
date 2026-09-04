"""SFT 主题跟随探针评测：固定主题集 + 自动指标，跨模型版本可比。

指标（"强相关"的操作化定义）：
    TMR@2   主题词（词干归并）在生成故事中出现 >=2 次的比例（排除顺嘴一提）
    mention min(提及次数, 3) / 3
    SSS     生成故事对主题语料共现词 top-20 的覆盖（满 10 个计 1.0）
    TAS     0.5 * mention + 0.5 * SSS

用法：
    uv run python sft_probe_eval.py --checkpoint checkpoints/sft_response_best.pt --name round0_baseline
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch

from prepare_sft_data import STOPWORDS, WORD_RE, _stem, iter_stories
from storylm.my_transformer import MyTransformerLM
from storylm.tokenizer import Tokenizer
from sft_train import (
    SPECIAL_TOKENS, SFT_VOCAB_SIZE, generate_one, load_sft_checkpoint,
)
from train_experiment import get_default_config

PROJECT_ROOT = Path(__file__).parent
CORPUS = PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt"
COOC_CACHE = PROJECT_ROOT / "data" / "probe_cooccurrence.json"
OUT_DIR = PROJECT_ROOT / "experiments"

# 探针主题集：所有词在任何轮次都不得进入白名单（held-out 测试集）
PROBES = {
    "control(in-list)": ["bird", "cake", "park", "sharing"],
    "unseen-high-freq": ["moon", "school", "king", "baby", "farm", "beach"],
    "unseen-mid-freq": ["dragon", "robot", "pirate", "train", "rainbow"],
    "unseen-oov": ["zorp"],
}
N_SAMPLES = 5          # 1 次贪心 + 4 次采样（t=0.8, seed 1-4）
MAX_TOKENS = 150
COOC_CACHE_VERSION = 2  # 词干 e-归一后共现表需重建


def _norm_stem(word: str) -> str:
    """评测用词干：在数据管线 _stem 基础上再做去尾 e 归一。

    修复 share/sharing、cake/cakes 这类 e-连接词的词干不一致
    （_stem("sharing")="shar" 而 _stem("share")="share"，直接比较会漏配）。
    查询词与正文词两侧做同一变换，保持对称。
    """
    s = _stem(word)
    return s[:-1] if s.endswith("e") else s


def build_cooccurrence():
    """一遍扫描语料：全局词频 df + 每个探针主题的共现词计数。"""
    topics = [t for ts in PROBES.values() for t in ts]
    topic_stems = {t: _norm_stem(t) for t in topics}
    df = Counter()
    cooc = {t: Counter() for t in topics}
    hit = {t: 0 for t in topics}

    for story in iter_stories(CORPUS, limit=100_000):
        words = set(WORD_RE.findall(story.lower()))
        df.update(words)
        stems = {_norm_stem(w) for w in words}
        for t in topics:
            if topic_stems[t] in stems:
                hit[t] += 1
                cooc[t].update(words)

    out = {}
    for t in topics:
        scored = [
            (w, c / max(df[w], 1) * math.log(1 + c))  # 提升 × 频次，压掉全局高频词
            for w, c in cooc[t].items()
            if c >= 5 and w not in STOPWORDS and len(w) >= 3 and _norm_stem(w) != topic_stems[t]
        ]
        scored.sort(key=lambda x: -x[1])
        out[t] = {"corpus_hits": hit[t], "related": [w for w, _ in scored[:20]]}
    return out


def get_cooccurrence():
    cache = COOC_CACHE.with_name(
        COOC_CACHE.name.replace(".json", f"_v{COOC_CACHE_VERSION}.json"))
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    data = build_cooccurrence()
    cache.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def sample_stories(model, tokenizer, device, instruction, num_tokens):
    """同一指令生成 N_SAMPLES 个故事：1 贪心 + 其余 t=0.8 采样。"""
    prompt = tokenizer.encode(f"<|user|> {instruction} <|assistant|>")
    outs = []
    for i in range(N_SAMPLES):
        torch.manual_seed(i)  # i=0 用贪心（temperature=0），采样种子固定可复现
        temp = 0.0 if i == 0 else 0.8
        ids = generate_one(model, tokenizer, prompt, device,
                           max_new_tokens=num_tokens, temperature=temp, top_k=50)
        outs.append(tokenizer.decode(ids[len(prompt):]))
    return outs


def score_story(story: str, topic: str, related: list[str]) -> dict:
    words = [w.lower() for w in WORD_RE.findall(story)]
    topic_stem = _norm_stem(topic)
    mentions = sum(1 for w in words if _norm_stem(w) == topic_stem)
    support = len({w for w in words if w in set(related)})
    return {
        "mentions": mentions,
        "mention": min(mentions, 3) / 3,
        "sss": min(support, 10) / 10,
        "tas": 0.5 * min(mentions, 3) / 3 + 0.5 * min(support, 10) / 10,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--name", type=str, required=True, help="报告名，如 round0_baseline")
    args = parser.parse_args()

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

    results = {}
    for band, topics in PROBES.items():
        for topic in topics:
            instruction = f"Write a short story about {topic}."
            stories = sample_stories(model, tokenizer, device, instruction, MAX_TOKENS)
            scores = [score_story(s, topic, cooc[topic]["related"]) for s in stories]
            results[topic] = {
                "band": band, "corpus_hits": cooc[topic]["corpus_hits"],
                "related": cooc[topic]["related"],
                "tmr2": sum(s["mentions"] >= 2 for s in scores) / len(scores),
                "any_mention": sum(s["mentions"] >= 1 for s in scores) / len(scores),
                "mention": sum(s["mention"] for s in scores) / len(scores),
                "sss": sum(s["sss"] for s in scores) / len(scores),
                "tas": sum(s["tas"] for s in scores) / len(scores),
                "stories": stories,
            }
            print(f"{topic:10} [{band:16}] tmr2={results[topic]['tmr2']:.2f} "
                  f"tas={results[topic]['tas']:.2f}")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"probe_{args.name}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    bands = {}
    for band in PROBES:
        rs = [r for r in results.values() if r["band"] == band]
        bands[band] = {
            "tmr2": sum(r["tmr2"] for r in rs) / len(rs),
            "tas": sum(r["tas"] for r in rs) / len(rs),
            "sss": sum(r["sss"] for r in rs) / len(rs),
        }
    lines = [f"# Probe eval: {args.name}", "", f"checkpoint: `{args.checkpoint}`", ""]
    lines += ["| band | TMR@2 | SSS | TAS |", "| --- | --- | --- | --- |"]
    for band, m in bands.items():
        lines.append(f"| {band} | {m['tmr2']:.0%} | {m['sss']:.2f} | {m['tas']:.2f} |")
    lines += ["", "## Per-topic", "", "| topic | corpus_hits | TMR@2 | mention | SSS | TAS |",
              "| --- | --- | --- | --- | --- | --- |"]
    for topic, r in results.items():
        lines.append(f"| {topic} | {r['corpus_hits']} | {r['tmr2']:.0%} | "
                     f"{r['mention']:.2f} | {r['sss']:.2f} | {r['tas']:.2f} |")
    lines += ["", "## Generated stories", ""]
    for topic, r in results.items():
        lines += [f"### {topic} ({r['band']}, corpus_hits={r['corpus_hits']})", ""]
        for i, s in enumerate(r["stories"]):
            lines += [f"**sample {i}**: {s.strip()}", ""]
    (OUT_DIR / f"probe_{args.name}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwritten: {OUT_DIR / f'probe_{args.name}.md'}")


if __name__ == "__main__":
    main()
