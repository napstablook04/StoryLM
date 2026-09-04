"""把 SFT 审计样本渲染成人类可读的 markdown，供人工审核。

用法：
    uv run python sft_review.py            # 渲染 train/valid 两份审计 -> data/sft_review.md
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


def render(out_path: Path):
    lines = ["# SFT 数据人工审核文件", ""]

    # 1. 主题词表
    lex = json.loads((DATA_DIR / "sft_topic_lexicon.json").read_text(encoding="utf-8"))
    lines += [
        "## 1. 主题词表（任务 B 用，共 %d 词）" % len(lex["words"]),
        "",
        " ".join(lex["words"]),
        "",
        "审核要点：这些词都会以 \"Write a short story about X.\" 的形式成为指令。",
        "不该出现的：人名、动词/形容词以外的怪词、读不通的词。",
        "",
    ]

    # 2. 训练/验证抽检
    for split in ["train", "valid"]:
        audit = DATA_DIR / f"sft_{split}_audit.jsonl"
        if not audit.exists():
            continue
        records = [json.loads(l) for l in audit.read_text(encoding="utf-8").splitlines()]
        lines += ["## 2. %s 集抽检（%d 条）" % ("训练" if split == "train" else "验证", len(records)), ""]
        for i, r in enumerate(records, 1):
            lines += [
                "### %d. [%s] prompt %d / %d tokens" % (i, r["task"], r["prompt_len"], len(r["input_ids"])),
                "",
                "**指令**: %s" % r["instruction"],
                "",
                "**回答**: %s" % r["response"],
                "",
            ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"written: {out_path}")


if __name__ == "__main__":
    render(DATA_DIR / "sft_review.md")
