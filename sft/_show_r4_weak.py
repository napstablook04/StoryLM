import json

d = json.load(open("experiments/probe_v5_round4_final.json", encoding="utf-8"))
for t in ("sharing", "moon", "baby"):
    e = d[t]
    print(f"=== {t} (band={e['band']}, tmr2={e['tmr2']}, any_mention={e['any_mention']}) ===")
    for i, s in enumerate(e["stories"]):
        print(f"--- sample {i} ---")
        print(s[:300].replace("\n", " "))
    print()
