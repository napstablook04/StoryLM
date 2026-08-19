"""
实验可视化脚本：把 logs/*.csv 训练日志画成学习曲线图。

训练实验通常需要 learning curve 来分析收敛，且最好能同时从 gradient steps
和 wall-clock time 两个视角观察。本脚本从 CSV 日志
（列：step, train_loss, val_loss, lr, wall_time）自动生成：

    1. baseline 单次训练曲线（train + val，按 step 和按 wall-clock 两个子图）
    2. 学习率扫描对比图（多条 val 曲线，自动标注发散的 run）
    3. batch size 扫描对比图
    4. 消融实验对比图（各变体 vs baseline）

设计遵循可视化规范：色盲安全的固定顺序配色、单一 y 轴、train 曲线淡化作背景、
val 曲线为主体、图例常在、直接标注最优点、深浅色各出一版。

用法：
    uv run plot_experiments.py                 # 扫描 logs/，深浅色都出，写到 figures/
    uv run plot_experiments.py --theme light   # 只出浅色
    uv run plot_experiments.py --logs-dir logs --out-dir figures --smooth 51
"""

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无显示环境也能出图
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).parent

# ============================================================
# 配色（来自已验证的色盲安全分类调色板，固定顺序、绝不循环取色）
# ============================================================
PALETTE = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"],
        "surface": "#ffffff",
        "panel": "#f6f6f4",
        "text_primary": "#1a1a19",
        "text_secondary": "#5c5c57",
        "grid": "#e3e3df",
        "diverged": "#e34948",
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"],
        "surface": "#1a1a19",
        "panel": "#242423",
        "text_primary": "#ffffff",
        "text_secondary": "#c3c2b7",
        "grid": "#3a3a38",
        "diverged": "#e66767",
    },
}


def apply_theme(theme: str) -> dict:
    """设置全局 matplotlib 风格，返回该主题的调色板字典。"""
    c = PALETTE[theme]
    plt.rcParams.update({
        "figure.facecolor": c["surface"],
        "savefig.facecolor": c["surface"],
        "axes.facecolor": c["surface"],
        "axes.edgecolor": c["grid"],
        "axes.labelcolor": c["text_secondary"],
        "axes.titlecolor": c["text_primary"],
        "text.color": c["text_primary"],
        "xtick.color": c["text_secondary"],
        "ytick.color": c["text_secondary"],
        "grid.color": c["grid"],
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.8,
        "grid.alpha": 0.6,
        "lines.linewidth": 2.0,
        "figure.dpi": 130,
    })
    return c


# ============================================================
# 数据读取
# ============================================================
class RunLog:
    """一条训练 run 的日志数据。"""
    # 发散判据阈值：正常训练初始 loss ≈ ln(vocab) ≈ 9.2，收敛后 ~1-2。
    # 结束时仍高于该值，说明发散/爆炸/没收敛。
    DIVERGE_THRESHOLD = 15.0

    def __init__(self, path: Path):
        self.path = path
        self.name = path.stem
        self.step, self.train_loss, self.wall_time = [], [], []
        self.val_step, self.val_loss, self.val_time = [], [], []
        self.diverged = False
        self.diverged_step = None
        self._load()

    def _load(self):
        with open(self.path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    s = int(row["step"])
                    tl = float(row["train_loss"])
                    wt = float(row["wall_time"])
                except (ValueError, KeyError):
                    continue
                self.step.append(s)
                self.train_loss.append(tl)
                self.wall_time.append(wt)
                if row.get("val_loss"):
                    self.val_step.append(s)
                    self.val_loss.append(float(row["val_loss"]))
                    self.val_time.append(wt)
        self._detect_divergence()

    def _detect_divergence(self):
        """发散 = 出现非有限值，或结束时 train_loss 仍高于阈值（未收敛/爆炸）。
        只看结束状态，避免把训练初期的高 loss 误判为发散。"""
        if not self.train_loss:
            return
        for s, tl in zip(self.step, self.train_loss):
            if not math.isfinite(tl):
                self.diverged = True
                self.diverged_step = s
                return
        if self.train_loss[-1] > self.DIVERGE_THRESHOLD:
            self.diverged = True
            self.diverged_step = self.step[-1]

    @property
    def best_val(self):
        return min(self.val_loss) if self.val_loss else None

    @property
    def best_val_step(self):
        if not self.val_loss:
            return None
        return self.val_step[int(np.argmin(self.val_loss))]

    def x(self, xaxis: str):
        return self.wall_time if xaxis == "time" else self.step

    def val_x(self, xaxis: str):
        return self.val_time if xaxis == "time" else self.val_step


def smooth(y, window: int):
    """滑动平均，用于淡化每步 train_loss 的噪声。"""
    if window <= 1 or len(y) < window:
        return np.asarray(y, dtype=float)
    y = np.asarray(y, dtype=float)
    kernel = np.ones(window) / window
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, kernel, mode="same")[pad:-pad]


XLABEL = {"step": "Gradient steps", "time": "Wall-clock time (s)"}


# ============================================================
# 绘图：单次训练曲线（train + val），step 与 time 两个子图
# ============================================================
def plot_single_run(run: RunLog, c: dict, smooth_win: int, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    blue = c["series"][0]

    for ax, xaxis in zip(axes, ("step", "time")):
        x = run.x(xaxis)
        # train_loss：原始很淡 + 平滑主线
        ax.plot(x, run.train_loss, color=blue, alpha=0.18, linewidth=1.0, zorder=1)
        ax.plot(x, smooth(run.train_loss, smooth_win), color=blue, linewidth=2.0,
                label="Train loss (smoothed)", zorder=2)
        # val_loss：明显的标记线
        if run.val_loss:
            vx = run.val_x(xaxis)
            ax.plot(vx, run.val_loss, color=c["series"][1], linewidth=2.0,
                    marker="o", markersize=5, label="Val loss", zorder=3)
            # 直接标注最优 val 点
            bi = int(np.argmin(run.val_loss))
            ax.scatter([vx[bi]], [run.val_loss[bi]], s=90, facecolor="none",
                       edgecolor=c["series"][1], linewidth=2.0, zorder=4)
            ax.annotate(f"best val = {run.val_loss[bi]:.3f}",
                        (vx[bi], run.val_loss[bi]),
                        textcoords="offset points", xytext=(8, 12),
                        color=c["text_primary"], fontsize=10, fontweight="bold")
        ax.set_xlabel(XLABEL[xaxis])
        ax.set_ylabel("Cross-entropy loss (per token)")
        ax.set_title(f"by {'wall-clock' if xaxis == 'time' else 'steps'}",
                     fontsize=11, color=c["text_secondary"])

    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle(f"Training curve — {run.name}", fontsize=14, fontweight="bold",
                 color=c["text_primary"])
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================
# 绘图：多 run 对比（val 曲线；发散 run 用虚线 + 标注）
# ============================================================
def plot_comparison(runs, c: dict, smooth_win: int, title: str, out: Path,
                    xaxis: str = "step", baseline: RunLog = None):
    fig, ax = plt.subplots(figsize=(8.5, 5.2), constrained_layout=True)
    series = c["series"]

    # baseline 作为灰色参考背景（若提供且不在 runs 里）
    ordered = list(runs)
    if baseline is not None and baseline not in ordered:
        ax.plot(baseline.val_x(xaxis) or baseline.x(xaxis),
                baseline.val_loss or smooth(baseline.train_loss, smooth_win),
                color=c["text_secondary"], linewidth=1.8, linestyle=(0, (4, 3)),
                label="baseline", zorder=2)

    # 组内最大步数：识别"提前终止"的 run。训练脚本发散时，爆炸的 loss 在
    # raise 之前未写入 CSV，导致 CSV 最后一行看似正常；只能靠"步数远少于
    # 同组其它 run"来判定它中途发散了。
    group_max_step = max((r.step[-1] for r in ordered if r.step), default=0)
    if baseline is not None and baseline.step:
        group_max_step = max(group_max_step, baseline.step[-1])

    def is_diverged(r):
        early_stop = bool(r.step) and group_max_step > 0 and r.step[-1] < 0.6 * group_max_step
        return r.diverged or (not r.val_loss) or early_stop

    for i, run in enumerate(ordered):
        color = series[i % len(series)]
        label = _pretty_label(run.name)
        if is_diverged(run):
            # 发散/提前终止：画平滑 train_loss 虚线；y 轴聚焦正常范围时，爆炸段会冲出顶部，
            # 直观表达"发散"，同时不压扁正常曲线。图例标注 (diverged)。
            ax.plot(run.x(xaxis), smooth(run.train_loss, smooth_win), color=color,
                    linewidth=1.6, linestyle=(0, (3, 2)), alpha=0.9,
                    label=f"{label} (diverged)", zorder=3)
        else:
            ax.plot(run.val_x(xaxis), run.val_loss, color=color, linewidth=2.0,
                    marker="o", markersize=4, label=label, zorder=3)

    # y 轴聚焦到正常（未发散）run 的 val loss 范围，给顶部留白让发散线可见
    normal = [r for r in ordered if not is_diverged(r)]
    if baseline is not None and baseline.val_loss and not is_diverged(baseline):
        normal.append(baseline)
    if normal:
        allv = [v for r in normal for v in r.val_loss]
        ymin, ymax = min(allv), max(allv)
        pad = (ymax - ymin) * 0.15 + 0.05
        ax.set_ylim(max(0.0, ymin - pad), ymax + pad * 2.5)

    ax.set_xlabel(XLABEL[xaxis])
    ax.set_ylabel("Val loss (per token)")
    ax.set_title(title, fontsize=13, fontweight="bold", color=c["text_primary"])
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def _pretty_label(name: str) -> str:
    """把文件名转成图例标签：lr_1e-04 -> lr=1e-04，batch_8 -> bs=8。"""
    if name.startswith("lr_"):
        return "lr=" + name[3:]
    if name.startswith("batch_"):
        return "bs=" + name[6:]
    if name.startswith("ablate_"):
        return name[7:]
    return name


# ============================================================
# 主入口：扫描 logs/，自动分类并出图
# ============================================================
def collect(logs_dir: Path):
    runs = {}
    for p in sorted(logs_dir.glob("*.csv")):
        runs[p.stem] = RunLog(p)
    return runs


def main():
    parser = argparse.ArgumentParser(description="Plot experiment learning curves")
    parser.add_argument("--logs-dir", type=str, default=str(PROJECT_ROOT / "logs"))
    parser.add_argument("--out-dir", type=str, default=str(PROJECT_ROOT / "figures"))
    parser.add_argument("--theme", choices=["light", "dark", "both"], default="both")
    parser.add_argument("--smooth", type=int, default=51,
                        help="train_loss 滑动平均窗口（步）")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    out_root = Path(args.out_dir)
    runs = collect(logs_dir)
    if not runs:
        print(f"未在 {logs_dir} 找到任何 .csv 日志")
        return

    themes = ["light", "dark"] if args.theme == "both" else [args.theme]

    for theme in themes:
        c = apply_theme(theme)
        out_dir = out_root / theme
        out_dir.mkdir(parents=True, exist_ok=True)
        made = []

        baseline = runs.get("baseline")

        # 1. 单次训练曲线（baseline，以及任何未归类的独立 run）
        classified = set()
        if baseline:
            made.append(plot_single_run(baseline, c, args.smooth,
                                        out_dir / "baseline_curve.png"))
            classified.add("baseline")

        # 2. 学习率扫描
        lr_runs = [r for n, r in runs.items() if n.startswith("lr_")]
        lr_runs.sort(key=lambda r: _lr_value(r.name))
        if len(lr_runs) >= 2:
            made.append(plot_comparison(lr_runs, c, args.smooth,
                        "Learning rate sweep (val loss)", out_dir / "lr_sweep.png"))
            classified.update(r.name for r in lr_runs)

        # 3. batch size 扫描
        bs_runs = [r for n, r in runs.items() if n.startswith("batch_")]
        bs_runs.sort(key=lambda r: _int_suffix(r.name))
        if len(bs_runs) >= 2:
            made.append(plot_comparison(bs_runs, c, args.smooth,
                        "Batch size sweep (val loss)", out_dir / "batch_sweep.png"))
            classified.update(r.name for r in bs_runs)

        # 4. 消融实验（各变体 vs baseline）
        abl_runs = [r for n, r in runs.items() if n.startswith("ablate_")]
        if abl_runs:
            made.append(plot_comparison(abl_runs, c, args.smooth,
                        "Ablations vs baseline (val loss)",
                        out_dir / "ablations.png", baseline=baseline))
            classified.update(r.name for r in abl_runs)

        # 5. 其它未归类的独立 run，各画一张单曲线
        for n, r in runs.items():
            if n not in classified:
                made.append(plot_single_run(r, c, args.smooth,
                            out_dir / f"{n}_curve.png"))

        print(f"[{theme}] 生成 {len(made)} 张图 -> {out_dir}")
        for m in made:
            print(f"    {m.relative_to(out_root)}")


def _lr_value(name: str) -> float:
    try:
        return float(name[3:])
    except ValueError:
        return float("inf")


def _int_suffix(name: str) -> int:
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 10**9


if __name__ == "__main__":
    main()
