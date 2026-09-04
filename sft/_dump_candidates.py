"""导出扩容候选池：无白名单门槛，df>=30 + 停用词 + 大写占比剔专名，按 df 排序。"""
from prepare_sft_data import STOPWORDS, WORD_RE, iter_stories
from collections import Counter

df = Counter()
cap = Counter()
total = Counter()
for story in iter_stories("data/TinyStoriesV2-GPT4-train.txt", limit=100_000):
    words = WORD_RE.findall(story)
    df.update(set(w.lower() for w in words))
    for w in words:
        k = w.lower()
        total[k] += 1
        if w[0].isupper():
            cap[k] += 1

cands = []
for w, c in df.items():
    if w in STOPWORDS or len(w) < 3 or c < 30:
        continue
    if total[w] > 0 and cap[w] / total[w] > 0.4:
        continue  # 专有名词
    cands.append((c, w))

cands.sort(reverse=True)
print("total candidates:", len(cands))
with open("_candidates.txt", "w", encoding="utf-8") as f:
    for c, w in cands:
        f.write(f"{c:7d} {w}\n")
