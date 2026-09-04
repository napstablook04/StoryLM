"""引导解码在探针集外主题上的泛化验证：基线 vs guide_b2.0 / b4.0。"""
from pathlib import Path

import torch

from prepare_sft_data import WORD_RE
from sft_probe_eval import MAX_TOKENS, _norm_stem
from sft_train import (
    SPECIAL_TOKENS, SFT_VOCAB_SIZE, generate_one, load_sft_checkpoint,
)
from storylm.my_transformer import MyTransformerLM
from storylm.tokenizer import Tokenizer
from train_experiment import get_default_config

PROJECT_ROOT = Path(__file__).parent


def bias_ids_for(tokenizer, topic):
    ids = set()
    for form in (f" {topic}", topic):
        toks = tokenizer.encode(form)
        if toks:
            ids.add(toks[0])
    return sorted(ids)


def mentions(story, topic):
    words = [w.lower() for w in WORD_RE.findall(story)]
    return sum(1 for w in words if _norm_stem(w) == _norm_stem(topic))


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

TOPICS = ["snow", "doctor", "garden", "chair", "hugged", "listen", "worry"]

for tag, guided in (("base  ", 0.0), ("guide2", 2.0), ("guide4", 4.0)):
    print(f"\n===== {tag} =====")
    for t in TOPICS:
        prompt = tokenizer.encode(f"<|user|> Write a short story about {t}. <|assistant|>")
        b_ids = bias_ids_for(tokenizer, t) if guided else None
        counts = []
        for i in range(5):
            torch.manual_seed(i)
            temp = 0.0 if i == 0 else 0.8
            ids = generate_one(model, tokenizer, prompt, device,
                               max_new_tokens=MAX_TOKENS, temperature=temp,
                               top_k=50, bias_ids=b_ids,
                               bias_value=guided, bias_until=2)
            counts.append(mentions(tokenizer.decode(ids[len(prompt):]), t))
        print(f"{t:8s} mentions/sample={counts}  tmr2={sum(c >= 2 for c in counts)}/5")
