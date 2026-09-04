# sft/ — SFT 与 DPO 代码包

StoryLM 的 SFT（指令微调）与 DPO（偏好优化）全部代码与总结文档。
所有脚本从**仓库根目录**运行（数据在 `data/`、检查点在 `checkpoints/`，脚本间的 import 同目录可直接解析）。

## 核心管线

| 文件 | 用途 |
| --- | --- |
| `prepare_sft_data.py` | SFT 数据管线：任务 A（续写）/ 任务 B（主题写作，TF-IDF 抽主题词）、chat 模板、response-only loss mask |
| `sft_train.py` | SFT 训练 + 引导解码 + 交互 chat（`--bias-value` 主题词 logit 引导） |
| `sft_probe_eval.py` | 探针评测：16 主题 × 4 band，TMR@2 / mention / SSS / TAS |
| `sft_review.py` | 将 SFT 审计样本渲染为 markdown 供人工审核 |

## 白名单与数据检查

| 文件 | 用途 |
| --- | --- |
| `_build_allowlist.py` | 主题白名单构建（163 → 759 词，断言探针零泄漏；候选输入 `_candidates.txt`） |
| `_check_data.py` | 抽检 SFT 数据质量：词根重复 / 探针泄漏 / 主题覆盖 |
| `_check_binding.py` | 查证指令-正文绑定（如 sharing 的训练曝光分析） |
| `_dump_candidates.py` | 导出扩容候选池（df≥30，剔专名） |
| `_find_heldout.py` | 从候选表找白名单外高频词（冒烟测试 held-out 主题） |

## 解码分析与失败定性

| 文件 | 用途 |
| --- | --- |
| `_decode_compare.py` | 8 种解码配置对比（temperature / top-p / 主题引导 b） |
| `_analyze_decode.py` | 分析解码对比结果（b=4.0 逐主题 + 质量抽查 + zorp） |
| `_quality_compare.py` | b=2.0 vs b=4.0 同主题质量对比 |
| `_greedy_rate.py` | 贪心解码命中率统计（16 主题 15/16 命中的证据来源） |
| `_show_weak.py` / `_show_r4_weak.py` | 展示 miss 样本（模式坍缩定性） |
| `_smoke_topics.py` / `_smoke_guided.py` | 探针集外冒烟验证（引导前后对比） |

## DPO（进行中）

| 文件 | 用途 |
| --- | --- |
| `_mine_dpo_pairs.py` | 从模型自身采样挖掘偏好对（命中 vs 坍缩） |
| `dpo_train.py` | DPO 训练器（policy/ref 双模型，response-only mask） |

## 文档（`docs/`）

- [`SFT工作全记录.md`](docs/SFT工作全记录.md) —— 13 节完整迭代记录（问题 → 思路 → 做法 → 结果）
- [`SFT_final_report.md`](docs/SFT_final_report.md) —— 终版总结报告
- [`decode_compare.json`](docs/decode_compare.json) —— 解码对比原始数据

逐轮日志与探针评测报告见仓库根目录 `experiments/`（`overnight_report.md`、`probe_v5_round*_final.md`）。
