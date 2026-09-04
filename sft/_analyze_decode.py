"""分析解码对比结果：guide_b4.0 逐主题 + 故事质量抽查 + zorp 检查。"""
import json

d = json.load(open("experiments/decode_compare.json", encoding="utf-8"))

print("=== guide_b4.0 逐主题 ===")
for t, e in d["guide_b4.0"]["topics"].items():
    print(f"{t:10s} [{e['band']:16s}] tmr2={e['tmr2']:.0%} any={e['any_mention']:.0%} "
          f"spam={e['spam']:.0%} mention={e['mention']:.2f}")

print("\n=== 基线 vs b4.0 弱主题 tmr2 对比 ===")
for t in ("sharing", "park", "moon", "school", "baby", "beach", "rainbow", "zorp"):
    b = d["base_t0.8_k50"]["topics"][t]["tmr2"]
    g = d["guide_b4.0"]["topics"][t]["tmr2"]
    print(f"{t:10s}  base={b:.0%}  guide_b4.0={g:.0%}")

print("\n=== guide_b4.0 故事质量抽查（sharing sample 3 / moon sample 4 / school sample 2）===")
for t, i in (("sharing", 3), ("moon", 4), ("school", 2)):
    print(f"\n--- {t} sample {i} ---")
    print(d["guide_b4.0"]["topics"][t]["stories"][i][:400])

print("\n=== zorp 在引导下的故事（sample 1）===")
print(d["guide_b4.0"]["topics"]["zorp"]["stories"][1][:300])
