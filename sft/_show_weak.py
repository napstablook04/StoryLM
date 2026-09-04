import json

d = json.load(open("experiments/probe_v5_round2_final.json", encoding="utf-8"))
for t in ("baby", "farm", "sharing", "rainbow"):
    print("=" * 30, t)
    for i, s in enumerate(d[t]["stories"]):
        print(f"--- sample {i}:")
        print(s.strip()[:550])
        print()
