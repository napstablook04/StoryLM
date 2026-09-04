"""冒烟测试：探针集之外的任意主题词，验证交付 checkpoint 的主题跟随能力。"""
import json
from pathlib import Path

import torch

from prepare_sft_data import WORD_RE
from sft_probe_eval import _norm_stem, sample_stories
from storylm.my_transformer import MyTransformerLM
from storylm.tokenizer import Tokenizer
from sft_train import SPECIAL_TOKENS, SFT_VOCAB_SIZE, load_sft_checkpoint
from train_experiment import get_default_config

PROJECT_ROOT = Path(__file__).parent
allow = set(json.loads(
    (PROJECT_ROOT / "data" / "topic_allowlist_expanded.json").read_text(encoding="utf-8")
)["words"])

pool = ["snow", "doctor", "garden", "window", "music", "chair", "apple", "bed",
        "star", "fish", "boat", "party", "dance", "book", "rain", "bear",
        "frog", "sand", "song", "balloon"]
in_list = [w for w in pool if w in allow][:3]
out_list = ["chair", "hugged", "listen", "worry"]  # 白名单外、语料 df>=400、非探针
print("in-list topics:  ", in_list)
print("out-of-list:     ", out_list)
print()

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = get_default_config()
tokenizer = Tokenizer.from_files(
    PROJECT_ROOT / "data" / "vocab.json", PROJECT_ROOT / "data" / "merges.txt",
    special_tokens=SPECIAL_TOKENS)
model = MyTransformerLM(
    vocab_size=SFT_VOCAB_SIZE, context_length=cfg["context_length"],
    d_model=cfg["d_model"], num_layers=cfg["num_layers"],
    num_heads=cfg["num_heads"], d_ff=cfg["d_ff"], rope_theta=cfg["rope_theta"])
load_sft_checkpoint(model, "checkpoints/sft_expanded_v4_best_final.pt")
model.to(device).eval()

for tag, topics in (("in-list", in_list), ("out-of-list", out_list)):
    for t in topics:
        stories = sample_stories(model, tokenizer, device,
                                 f"Write a short story about {t}.", 150)
        counts = []
        for s in stories:
            words = [w.lower() for w in WORD_RE.findall(s)]
            counts.append(sum(1 for w in words if _norm_stem(w) == _norm_stem(t)))
        tmr = sum(c >= 2 for c in counts)
        print(f"=== {t} [{tag}] mentions/sample={counts} tmr2={tmr}/5 ===")
        print(stories[0][:230].replace("\n", " "))
        print()
