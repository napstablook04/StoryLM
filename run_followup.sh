#!/usr/bin/env bash
# ============================================================
# 补充实验：
#   1) 找发散学习率（3e-3 已收敛到 1.296，从 5e-3 往上找发散边界）
#   2) 找 4090 的精确显存上限（128 可跑 / 256 OOM，探测 160 192 224）
#   3) 用最佳学习率 3e-3 重跑 baseline（覆盖旧的 3e-4 baseline）
# 用法：nohup bash run_followup.sh > run_followup.log 2>&1 &
# ============================================================
cd "$(dirname "$0")"
ts()  { date '+%F %T'; }
run() { echo; echo "==== [$(ts)] uv run $* ===="; uv run "$@"; echo "==== [$(ts)] done: $1 ===="; }

echo "######## 补充实验开始：$(ts) ########"

# 1. 找发散：更大的学习率。发散的 run 会很快抛异常退出（几分钟）
run train_experiment.py lr_sweep --lr 5e-3 1e-2 3e-2 --compile

# 2. 显存精确上限：在 128(OK) 和 256(OOM) 之间探测；OOM 的秒级失败，可跑的顺便拿数据点
#    lr 沿用 3e-4，与已有 batch_1/16/64/128 同一学习率，可画进同一张图
run train_experiment.py batch_sweep --batch_sizes 160 192 224 --lr 3e-4 --compile

# 3. 用最佳学习率 3e-3 重跑 baseline（覆盖 checkpoints/baseline_best.pt 与 logs/baseline.csv）
run train_experiment.py baseline --lr 3e-3 --compile

# 4. 用新 baseline 重新生成文本
run train_experiment.py generate --checkpoint checkpoints/baseline_best.pt

# 5. 重新出图（深浅两版）
run plot_experiments.py

echo; echo "######## 补充实验完成：$(ts) ########"
