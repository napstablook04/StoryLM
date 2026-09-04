"""对比 b2.0 与 b4.0 的故事质量（同主题同样本下标）。"""
import json

d = json.load(open("experiments/decode_compare.json", encoding="utf-8"))

for cfg in ("guide_b2.0", "guide_b4.0"):
    print(f"\n{'='*70}\n[{cfg}]")
    for t, i in (("moon", 4), ("school", 2), ("baby", 3), ("sharing", 3)):
        print(f"\n--- {t} sample {i} ---")
        print(d[cfg]["topics"][t]["stories"][i][:330])
