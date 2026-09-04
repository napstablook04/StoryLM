"""验证：control 四词在 B 任务训练数据里的绑定信号强度对比。"""
import json

from prepare_sft_data import WORD_RE
from sft_probe_eval import _norm_stem

rows = [json.loads(l) for l in open("data/sft_train_r4.jsonl", encoding="utf-8")]
b_rows = [r for r in rows if r["task"] == "B"]
print(f"B 任务样本数: {len(b_rows)}")

for topic in ("bird", "cake", "park", "sharing"):
    stem = _norm_stem(topic)
    hits = []
    for r in b_rows:
        ins = r["instruction"].lower()
        if f"about {topic}" in ins or f"about {topic} and" in ins or ins.rstrip(".").endswith(f"about {topic}"):
            words = [w.lower() for w in WORD_RE.findall(r["response"])]
            hits.append(sum(1 for w in words if _norm_stem(w) == stem))
    if not hits:
        print(f"{topic:8s} 无 B 指令样本")
        continue
    hits.sort()
    n = len(hits)
    med = hits[n // 2]
    lo = sum(h < 2 for h in hits)
    print(f"{topic:8s} 指令样本 {n:4d} 个 | 响应内词干提及: 中位 {med}, "
          f"均值 {sum(hits)/n:.1f}, 范围 {hits[0]}-{hits[-1]}, <2 次占 {lo/n:.0%}")
