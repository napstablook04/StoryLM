# StoryLM

> **从零实现一个小语言模型的完整生命周期** —— 预训练（手写 BPE 分词器、Transformer、优化器）→ SFT 指令微调（对齐 + 评测体系 + 五轮单变量迭代）→ DPO 偏好优化，全程不依赖高层网络模块。

<p>
<img alt="Python" src="https://img.shields.io/badge/Python-3.12+-blue">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.11-ee4c2c">
<img alt="from-scratch" src="https://img.shields.io/badge/built-from%20scratch-success">
<img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

一个约 **22.7M 参数**的 decoder-only Transformer(RoPE · RMSNorm · SwiGLU),完全从零实现——**不使用 `nn.Linear` / `nn.MultiheadAttention` / `nn.LayerNorm` 等高层模块**,所有权重都是 `nn.Parameter` + 手写前向。项目分三个阶段：

1. **预训练**：TinyStories 上约 24 分钟训练到验证损失 **1.30**，能生成通顺连贯的短故事；
2. **SFT 指令微调**：让模型从"只会续写"到"听懂指令、围绕任意主题写作"——五轮单变量实验迭代 + 自建评测体系，unseen 主题 TMR@2 从 33% 提升到 **96%**（含推理期引导）；
3. **DPO 偏好优化**（进行中）：从模型自身采样挖掘偏好对，针对性修复 t=0.8 采样的模式坍缩。

---

## ✨ 特性

- **完全从零实现**:BPE 分词器、RoPE 旋转位置编码、RMSNorm、SwiGLU、多头因果自注意力、AdamW、cosine 学习率调度、梯度裁剪、checkpoint —— 全部手写,不调高层 API。
- **完整的训练→对齐链路**:预训练超参扫描/架构消融 → SFT 数据自动构造、response-only loss masking、五轮单变量迭代 → DPO 偏好优化。
- **自建指令跟随评测体系**:held-out 探针集(16 主题 × 4 band,防泄漏)+ 可计算指标(TMR@2 / mention / SSS / TAS),每个改动的贡献可归因、可复现。
- **推理期零成本优化**:主题词 logit 引导解码(top-k/top-p 可选),单变量归因表 + 8 种解码配置对比。
- **开箱即用的可视化**:一个脚本把训练日志自动画成学习曲线(支持按步数 / 按墙钟时间、深浅双色、发散标注)。
- **工程细节**:`torch.compile` 加速、mmap 按需读数据、发散保护、确定性验证评估。

---

## 🎬 生成示例

### 预训练模型（验证损失 1.30，只会续写）

prompt `"Once upon a time"`，temperature=0.8, top-p=0.9：

```text
Once upon a time, there was a big bear. The bear was very independent. He liked to do
things by himself. One day, the bear was walking in the woods. He saw a big tree and
wanted to climb it. The bear climbed the tree. He saw a bird and said, "Hi, bird!" ...
The big bear and the little bear became friends. They played together in the woods.
The big bear was not so independent anymore. He was happy to have a new friend.
And they both lived happily ever after.
```

### SFT 模型（听懂指令、围绕主题写）

指令 `Write a short story about dragon.`（t=0.8 + 主题引导 b=2.0）：

- 探针集外冒烟验证：白名单内 snow 5/5、doctor 5/5；白名单外 chair 引导后 5/5、hugged 5/5
- 贪婪解码下 16 个探针主题 **15/16 语义命中**——确定性解码几乎从不跑题

---

## 📊 预训练实验结果

> 完整实验记录见 [`exp/experiment_log.md`](exp/experiment_log.md)。

### Baseline

| 指标 | 数值 |
| ---- | ---- |
| 最佳验证损失 | **1.2994**(per-token 交叉熵) |
| 学习率 | 3e-3(经扫描得出的最优) |
| 训练时间 | ~24 min @ RTX 4090(≈23 万 tokens/s) |

![训练曲线](figures/light/baseline_curve.png)

### 学习率:U 型曲线与稳定性边界

学习率扫描呈现典型的 U 型:谷底在 **3e-3**(1.296),两侧对称升高。

| lr | 1e-4 | 3e-4 | 1e-3 | **3e-3** | 5e-3 | 1e-2 | 3e-2 |
| -- | ---- | ---- | ---- | -------- | ---- | ---- | ---- |
| val | 1.472 | 1.347 | 1.301 | **1.296** | 1.318 | 1.407 | 1.743 |

![学习率扫描](figures/light/lr_sweep.png)

**一个更深的发现**:单纯加大学习率**很难让模型发散**——lr 加到 1e-1 都稳如泰山。追查后确认:稳定性由**梯度裁剪 + AdamW 自适应更新**两道防线共同维持,必须**同时**关闭梯度裁剪且把 lr 拉到 1.0,才在 step 174 触发发散(loss 爆炸)。这比"调大 lr 就发散"的朴素预期深刻得多。

![发散实验:关裁剪 + lr=1.0](figures/light/baseline_noclip_lr1_curve.png)

### Batch Size:甜点区与显存上限

固定总 token 预算下,batch size 与最终性能**非单调**——存在甜点区(16),且并非越大越好;192 是 RTX 4090 (24GB) 能稳定训练的最大 batch,224 直接 OOM。

| batch | 1 | **16** | 64 | 128 | 192 | 224 |
| ----- | -- | ----- | -- | --- | --- | --- |
| val | 1.741 | **1.322** | 1.342 | 1.374 | 1.404 | OOM |

![Batch size 扫描](figures/light/batch_sweep.png)

### 架构消融(对照:标准模型 lr=3e-4,val=1.347)

| 消融 | 结果 | 结论 |
| ---- | ---- | ---- |
| 去掉 RMSNorm | **step 1842 发散** | 归一化对训练稳定性是决定性的 |
| Post-Norm(替代 Pre-Norm) | 1.398 | Pre-Norm 更优 |
| 去掉位置编码(NoPE) | 1.345 | ≈ RoPE:短序列 + 因果掩码下位置编码增益有限 |
| SiLU FFN(替代 SwiGLU) | 1.376 | SwiGLU 的门控机制有效 |

![架构消融对比](figures/light/ablations.png)

**组件重要性排序**:RMSNorm(稳定性) > SwiGLU 门控 ≈ Pre-Norm(性能) > 位置编码(此任务下影响最小)。

---

## 🎯 SFT：从"续写"到"指令跟随"

> 完整迭代记录见 [`sft/docs/SFT工作全记录.md`](sft/docs/SFT工作全记录.md)，
> 终版报告见 [`sft/docs/SFT_final_report.md`](sft/docs/SFT_final_report.md)，
> 代码见 [`sft/`](sft/)（含逐文件说明 [`sft/README.md`](sft/README.md)）。

### 任务与数据

两类指令，数据全部由语料自动构造（`prepare_sft_data.py`）：

- **任务 A（续写）**：`Continue this story: <开头两句>` → 补全故事
- **任务 B（主题写作）**：`Write a short story about <主题词>.` → 围绕指定主题写（TF-IDF 抽取主题词）

统一 chat 模板 `<|user|> {指令} <|assistant|> {响应} <|endoftext|>`，词表扩到 10002（+`<|user|>`/`<|assistant|>`），**response-only loss masking**。

### 评测体系

16 个 held-out 探针主题分 4 个 band（control / unseen-high / unseen-mid / oov，**任何一轮不得进入训练白名单**），每主题 5 个故事（1 贪心 + 4 个 t=0.8 采样），指标：

- **TMR@2**（主指标）：主题词（词干归并）出现 ≥2 次的故事占比
- **mention**：写得多深；**SSS**：覆盖主题语料共现词比例；**TAS** = 两者综合

### 迭代结果（每轮只改一个变量）

| 阶段 | control | unseen-high | unseen-mid | 关键变量 |
| ---- | ------- | ----------- | ---------- | -------- |
| 起点（val-best 早停 + 旧口径） | 33% | 33% | 27% | — |
| R0-final（163 词白名单） | 55% | 67% | 80% | 检查点选择 val-best → final |
| R1（759 词） | 60% | 63% | 88% | 白名单扩建 |
| R2 | 65% | 70% | 88% | 数据质量三连修（含单主题指令格式） |
| R3（B=32k） | 80% | 73% | 88% | 数据加量 |
| R4（B=64k，交付模型） | 75% | 70% | 92% | 剂量饱和检验（证伪"再加量有效"） |
| **R4 + 主题引导 b=2.0（推荐）** | **80%** | **93%** | **96%** | 推理期 logit 引导（零训练成本） |
| R4 + 引导 b=4.0（演示） | **85%** | **100%** | **96%** | 偏置加大（偶发退化） |

**达标判定**：unseen 93%/96% ✓✓，TAS 较起点约翻倍 ✓——核心目标达成。

### 关键发现

1. **检查点选择是最大单一杠杆（+30~50pp）**：val loss 回升反映的是记忆化，而指令跟随能力全程持续增长——"val 最优早停"恰好截断了目标能力。对齐类任务应**对着目标能力选检查点，而不是对着代理指标**。
2. **多样性逼出泛化**：163 词 × 92 次/词可逐词死记，759 词 × 中位 10 次后只能学"指令词进入正文"的通用规则 → 泛化到任意词。
3. **指令格式覆盖是隐形坑**：训练只见过 "about X and Y" 就不会 "about X."，25% 单主题变体精准修复被压制的词。
4. **数据剂量对两个目标不同步**：加量持续改善 val loss，但主题绑定在 B=32k 饱和（64k 统计打平）。
5. **失败要定性到机制**：剩余失分不是绑定失败而是 t=0.8 采样模式坍缩（跨主题复用同一罐头故事 = 铁证；贪心解码 15/16 命中证明条件分布本身是对的）→ 推理端引导零成本拿下。
6. **能力边界**：OOV 词复制（zorp，0%）是多 BPE 片 + 22.7M 无 copying 机制的规模问题，数据与解码均无解。

### 交互入口

```bash
# SFT 模型对话（已内置主题词 logit 引导，自动抽取 about <主题> 加偏置）
uv run python sft_train.py chat --checkpoint checkpoints/sft_expanded_v4_best_final.pt --bias-value 2.0
```

---

## 🧠 模型架构

| 组件 | 设置 |
| ---- | ---- |
| 类型 | Decoder-only Transformer(GPT 风格) |
| 参数量 | 22.7M(非嵌入 12.5M) |
| 层数 / d_model / heads | 4 / 512 / 16 |
| FFN | SwiGLU,d_ff=1344(≈ 8/3·d_model) |
| 位置编码 | RoPE(θ=10000) |
| 归一化 | RMSNorm,Pre-Norm 结构 |
| 词表 / 上下文长度 | 10000(BPE，SFT 后 10002)/ 256 |
| 优化器 | AdamW(β=0.9/0.999)+ cosine 调度 + warmup + 梯度裁剪 |
| 初始化 | 截断正态(std=0.02) |

---

## 🚀 快速开始

```bash
# 0. 安装依赖（使用 uv）
uv sync

# 1. 准备数据（下载 TinyStories，训练 BPE 分词器并编码为 .npy）
#    需在 data/ 下得到 tinystories_train.npy / tinystories_valid.npy / vocab.json / merges.txt
#    数据处理脚本见 storylm/prepare_data.py

# 2. 训练 baseline（约 24 分钟 @ RTX 4090）
uv run train_experiment.py baseline --lr 3e-3 --compile

# 3. 预训练模型生成
uv run train_experiment.py generate --checkpoint checkpoints/baseline_best.pt \
    --prompt "Once upon a time"

# 4. 预训练实验复现
uv run train_experiment.py lr_sweep    --lr 1e-4 3e-4 1e-3 3e-3 --compile
uv run train_experiment.py batch_sweep --batch_sizes 1 16 64 128 192 --compile
uv run train_experiment.py ablate_no_norm   --lr 3e-4 --compile   # 及 post_norm / nope / silu

# 5. 一键出图（深浅两版 → figures/）
uv run plot_experiments.py

# 复现发散实验（关闭梯度裁剪 + 大学习率）
uv run train_experiment.py baseline --lr 1.0 --max_grad_norm 1e9 --tag noclip_lr1
```

### SFT 复现

```bash
# 1. 构造 SFT 数据（任务 A 续写 + 任务 B 主题写作，自动 TF-IDF 抽主题词）
uv run prepare_sft_data.py            # 白名单见 data/topic_allowlist_expanded.json（759 词）

# 2. SFT 训练（response-only loss masking，3 epoch，固定超参保证归因干净）
uv run sft_train.py train --train-data data/sft_train_r4.jsonl --valid-data data/sft_valid_r4.jsonl

# 3. 探针评测（TMR@2 / mention / SSS / TAS，16 主题 × 4 band）
uv run sft_probe_eval.py --checkpoint checkpoints/sft_expanded_v4_best_final.pt

# 4. SFT 模型交互对话（含主题引导解码）
uv run sft_train.py chat --checkpoint checkpoints/sft_expanded_v4_best_final.pt --bias-value 2.0

# 5. DPO 偏好优化（进行中：挖掘偏好对 → DPO 训练）
uv run dpo_train.py --help
```

自检(验证实现正确性,单 batch 应快速过拟合到 ~0):

```bash
uv run train_experiment.py overfit_test
```

---

## 📁 项目结构

```
StoryLM/
├── storylm/
│   ├── my_transformer.py   # 从零实现：BPE 分词器 + Transformer 全部组件
│   ├── model.py            # 张量算子（linear/embedding/attention/rope/...）
│   ├── train.py            # 交叉熵、梯度裁剪、cosine 调度、AdamW、checkpoint
│   └── tokenizer.py        # BPE 训练与编解码
├── train_experiment.py     # 预训练 / 实验 / 生成 统一入口
├── sft/                    # SFT + DPO 代码包（见 sft/README.md 逐文件说明）
│   ├── prepare_sft_data.py # SFT 数据管线（任务构造 + TF-IDF 主题抽取 + loss mask）
│   ├── sft_train.py        # SFT 训练 / 引导解码 / 交互 chat
│   ├── sft_probe_eval.py   # SFT 评测（TMR@2 / mention / SSS / TAS）
│   ├── dpo_train.py        # DPO 偏好优化训练
│   ├── _*.py               # 数据检查 / 白名单构建 / 解码分析 / 冒烟验证等脚本
│   └── docs/               # SFT 工作全记录、终版报告、解码对比数据
├── plot_experiments.py     # 学习曲线可视化
├── exp/experiment_log.md           # 预训练详细实验记录
├── experiments/
│   ├── SFT工作全记录.md            # SFT 完整迭代记录（问题→思路→做法→结果）
│   ├── SFT_final_report.md         # SFT 终版总结报告
│   ├── overnight_report.md         # 逐轮实验日志
│   ├── probe_v5_round*_final.md    # 各轮探针评测报告
│   └── decode_compare.json         # 解码对比原始数据
├── checkpoints/            # 模型检查点（baseline / SFT 各轮）
├── figures/                # 生成的学习曲线图（light/ 与 dark/）
├── logs/                   # 训练日志（CSV）
└── data/                   # 数据集与 SFT 数据快照（sft_*_r{0163..r4}.jsonl）
```

---

## 🔑 关键结论

1. **超参**:最佳学习率 3e-3,baseline 验证损失 1.30;batch size 存在甜点区(16),并非越大越好。
2. **稳定性**:训练稳定性来自梯度裁剪与 AdamW 自适应的叠加,单调加大 lr 不足以发散——这是一个反直觉但可复现的工程发现。
3. **架构**:RMSNorm 去掉即发散(稳定性关键);Pre-Norm > Post-Norm;SwiGLU 门控优于纯 SiLU;短序列任务下 NoPE ≈ RoPE(有意思的负结果)。
4. **检查点选择是被低估的最大杠杆**：val loss 回升 = 记忆化 ≠ 能力退化，指令跟随能力随训练全程增长（+30~50pp）。
5. **多样性逼出泛化**：主题覆盖从 163 词扩到 759 词，逐词死记不可行后模型才学到通用的指令绑定规则。
6. **失败模式定性到机制再动手**：采样坍缩 ≠ 知识缺失（贪心 15/16 命中），推理端 logit 引导零训练成本即可修复大半。

---

## 🛠 技术栈

Python · PyTorch(仅用张量与 autograd,不用高层 nn 层)· NumPy · matplotlib · uv

---

## 📚 参考

- Eldan & Li, *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?* (2023) —— 训练数据集
- Su et al., *RoFormer: Rotary Position Embedding* (2021) —— RoPE
- Shazeer, *GLU Variants Improve Transformer* (2020) —— SwiGLU
- Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019) —— RMSNorm
- Touvron et al., *LLaMA* (2023) —— 整体架构参考
- Rafailov et al., *Direct Preference Optimization* (2023) —— DPO

---

## 📄 License

MIT
