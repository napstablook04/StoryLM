import json
import re

from sft_probe_eval import PROBES, _norm_stem

d = json.load(open("experiments/probe_v5_round4_final.json", encoding="utf-8"))
hit1 = hit2 = n = 0
all_topics = [t for ts in PROBES.values() for t in ts]
for topic in all_topics:
    story = d[topic]["stories"][0]  # sample 0 = greedy
    stem = _norm_stem(topic)
    cnt = len(re.findall(stem, story.lower()))
    n += 1
    hit1 += cnt >= 1
    hit2 += cnt >= 2
    print(f"{topic:10s} greedy_mentions={cnt}")
print(f"\ngreedy on-topic: >=1 mention {hit1}/{n}, >=2 mentions {hit2}/{n}")
