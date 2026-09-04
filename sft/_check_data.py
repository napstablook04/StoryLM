"""抽检重建后的 SFT 数据质量：同词根重复 / 探针泄漏 / 主题覆盖 / 样例。"""
import json
import random
import re
from collections import Counter
from pathlib import Path

from prepare_sft_data import _stem

PROBES = {"moon", "school", "king", "baby", "farm", "beach",
          "dragon", "robot", "pirate", "train", "rainbow", "zorp"}

recs = [json.loads(l) for l in open("data/sft_train.jsonl", encoding="utf-8")]
b = [r for r in recs if r["task"] == "B"]

# B 指令里的主题词
topic_re = re.compile(r"about (.+)\.$")
dup_stem, probe_hits, single, pairs = [], [], 0, 0
all_words = Counter()
for r in b:
    m = topic_re.search(r["instruction"])
    words = m.group(1).split(" and ") if m else []
    all_words.update(words)
    if len(words) == 1:
        single += 1
    else:
        pairs += 1
    stems = [_stem(w) for w in words]
    if len(set(stems)) != len(stems):
        dup_stem.append(r["instruction"])
    probe_hits += [w for w in words if w in PROBES]

print(f"B samples: {len(b)}  single-topic: {single}  two-topic: {pairs}")
print(f"distinct topic words used: {len(all_words)}")
print(f"same-stem duplicate prompts: {len(dup_stem)}  {dup_stem[:3]}")
print(f"probe leakage: {len(probe_hits)}  {probe_hits[:5]}")
print(f"topic usage: max={max(all_words.values())} median={sorted(all_words.values())[len(all_words)//2]}")

rng = random.Random(0)
rare = [w for w, c in all_words.items() if c <= 3]
print(f"\nrandom B instructions (incl. rare topics):")
samples = rng.sample(b, 6) + rng.sample([r for r in b if any(w in rare for w in topic_re.search(r['instruction']).group(1).split(' and '))], 4)
for r in samples:
    print(" ", r["instruction"])
