#!/usr/bin/env bash
# ============================================================
# TinyStories 全套实验，一次挂机跑完
# ============================================================
# 推荐用法（后台挂机，关掉终端也不会中断）：
#     nohup bash run_all_experiments.sh > run_all.log 2>&1 &
#     tail -f run_all.log        # 随时查看进度
#
# 说明：
#   - 不使用 `set -e`：某个实验发散并非零退出时，不应中断后续实验
#   - LR 统一用 3e-4（常用默认，TinyStories 上通常接近最佳）。若 lr_sweep 结果
#     显示别的学习率更好，第二天用那个 lr 重跑 baseline 即可
#   - batch_sweep 中 bs<8 自动用 20M 短预算，避免 bs=1 跑满 327M（要十几小时）
# ============================================================

cd "$(dirname "$0")"
LR=3e-4

ts()  { date '+%F %T'; }
run() { echo; echo "==== [$(ts)] uv run $* ===="; uv run "$@"; echo "==== [$(ts)] done: $1 ===="; }

echo "######## 全套实验开始：$(ts) ########"

# 0. 实现自检：单 batch 过拟合应快速降到 ~0（几十秒，确认没回归）
run train_experiment.py overfit_test

# 1. 主模型 baseline（generate 要用它的 checkpoint），最先跑，约 25 分钟
run train_experiment.py baseline --lr $LR --compile

# 2. 学习率扫描（含大概率发散的 3e-3，满足“至少一个发散 run”）
run train_experiment.py lr_sweep --lr 1e-4 3e-4 1e-3 3e-3 --compile

# 3. 四个消融实验（各约 25 分钟；no_norm 在 3e-4 可能发散，正是要观察的现象）
run train_experiment.py ablate_no_norm   --lr $LR --compile
run train_experiment.py ablate_post_norm --lr $LR --compile
run train_experiment.py ablate_nope      --lr $LR --compile
run train_experiment.py ablate_silu      --lr $LR --compile

# 4. batch size 扫描（最耗时，放最后；bs=1 走 20M 短预算）
run train_experiment.py batch_sweep --batch_sizes 1 16 64 128 256 --lr $LR --compile

# 5. 文本生成（用 baseline 最佳 checkpoint）
run train_experiment.py generate --checkpoint checkpoints/baseline_best.pt

# 6. 一键出图（深浅两版 -> figures/）
run plot_experiments.py

echo
echo "######## 全部完成：$(ts) ########"
echo "图在 figures/light 和 figures/dark；生成文本在 logs/generated_text.txt"
