"""从候选表里找语料高频但不在白名单中的词（冒烟测试用 held-out 主题）。"""
import json

allow = set(json.loads(
    open("data/topic_allowlist_expanded.json", encoding="utf-8").read())["words"])

rows = []
for line in open("_candidates.txt", encoding="utf-8"):
    parts = line.split()
    if len(parts) == 2 and parts[0].isdigit():
        df, w = int(parts[0]), parts[1]
        if w not in allow and df >= 400:
            rows.append((df, w))

rows.sort(reverse=True)
print(len(rows), "个 df>=400 的白名单外词，第 40-160 名：")
print(", ".join(f"{w}({df})" for df, w in rows[40:160]))
