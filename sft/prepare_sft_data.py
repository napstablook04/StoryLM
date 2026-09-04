"""从 TinyStories 原始语料构造 SFT（指令微调）数据集。

用法（两次调用，分别产出训练集与验证集）：
    uv run python prepare_sft_data.py --input data/TinyStoriesV2-GPT4-train.txt --output data/sft_train.jsonl
    uv run python prepare_sft_data.py --input data/TinyStoriesV2-GPT4-valid.txt --output data/sft_valid.jsonl --pairs-per-task 250 --reuse-lexicon data/sft_topic_lexicon.json

流程：
    1. （幂等）把 <|user|> / <|assistant|> 固化进 vocab.json
    2. 流式读语料，按 <|endoftext|> 切出单篇故事
    3. 随机抽样建主题词表并落盘（data/sft_topic_lexicon.json，待人工审核）
    4. 从两个互不重叠的故事池构造任务对：
         A 续写：前两句作指令、其余作回答
         B 主题生成：TF-IDF 选 1-2 个主题词作指令、全篇作回答（选不出主题则丢弃）
    5. 编码 + 长度过滤，输出 jsonl、人工抽检样本与统计
"""

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

from storylm.tokenizer import Tokenizer

STORY_SEP = "<|endoftext|>"
SPECIAL_TOKENS = ["<|endoftext|>", "<|user|>", "<|assistant|>"]
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
STOPWORDS = {
    "a", "about", "after", "again", "all", "almost", "also", "am", "an", "and", "any",
    "are", "around", "as", "at", "be", "because", "been", "before", "being", "between",
    "both", "but", "by", "can", "could", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her",
    "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves",
    "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "you", "your", "yours", "yourself", "yourselves"
}

# 主题词白名单：人工审定的主题题库（审定了初版自动词表后固化）。
# 收录标准（人工审核）：
#   1. 具体名词（bird/park/garden）
#   2. 活动词与情感/品格词（play/share/help/scared/brave）——它们本身
#      就是儿童故事的主题（sharing、helping、克服害怕都是经典情节）
# 排除：叙事胶水词（said/went/put/took/get 等语法成分，永远不是主题）、
#       内容空泛的形容词（big/new/pretty/nice）、颜色词。
# 专有名词（人名 Tim/Lily）由 build_lexicon 的大写占比过滤自动剔除。
TOPIC_ALLOWLIST = {
    # 具体名词
    "air", "animals", "adventure", "bag", "ball", "bear", "bed", "bird", "birds",
    "book", "books", "box", "boy", "brother", "bug", "bunny", "cake", "car", "cars",
    "cat", "clothes", "colors", "dad", "day", "dog", "door", "dress", "face", "family",
    "fish", "floor", "flower", "flowers", "food", "forest", "frog", "friend", "friends",
    "game", "games", "garden", "girl", "grass", "ground", "hands", "hat", "head",
    "hill", "hole", "home", "house", "kids", "kitchen", "leaves", "light", "magic",
    "man", "mess", "mom", "morning", "mouth", "mouse", "night", "paper", "park",
    "party", "people", "picture", "pictures", "pond", "rabbit", "race", "rain",
    "rock", "room", "sea", "sky", "song", "sound", "squirrel", "stick", "store",
    "story", "sun", "surprise", "table", "tail", "town", "toy", "toys", "tree",
    "trees", "voice", "water", "window", "wind", "woods", "world", "yard",
    # 活动词（含语料中的动名词形式，动名词本身就是名词）
    "play", "playing", "share", "sharing", "help", "helping", "give", "giving",
    "swim", "swimming", "fly", "flying", "jump", "jumping", "dance", "dancing",
    "sing", "singing", "laugh", "laughing", "smile", "smiling", "sleep", "sleeping",
    "hug", "hugging", "clean", "cleaning", "eat", "eating", "climb", "climbing",
    "read", "reading", "run", "running", "walk", "walking", "learn", "learning",
    "grow", "growing",
    # 情感 / 品格（儿童故事的经典主题）
    "happy", "sad", "scared", "angry", "brave", "kind", "proud", "excited",
    "surprised", "tired", "careful", "silly", "funny", "shy", "curious",
    "friendly", "worried", "lonely", "sorry", "gentle", "wise", "safe",
}

# 扩建版白名单：759 词（初版 163 + 人工通读候选表 rank 1-2600 新增 596）。
# 由 _build_allowlist.py 生成并校验（新增词必须在候选集内、探针词禁入）。
# 假设：白名单过小 → 模型按词死记主题绑定；扩大一个数量级后只能学
# "指令里的内容词要写进故事" 这条通用规则，从而泛化到任意主题。
_EXPANDED = Path(__file__).with_name("data") / "topic_allowlist_expanded.json"
if _EXPANDED.exists():
    TOPIC_ALLOWLIST = set(
        json.loads(_EXPANDED.read_text(encoding="utf-8"))["words"])
    assert len(TOPIC_ALLOWLIST) >= 163, "expanded allowlist must be a superset of the original"


def words_of(story):
    return WORD_RE.findall(story.lower())


def extend_vocab(vocab_path, tokens):
    """把缺失的特殊 token 固化写回 vocab.json（幂等，可重复运行）。

    按 len 降序追加，与 Tokenizer.__init__ 的动态注册顺序一致
    （<|assistant|>=10000、<|user|>=10001），保证两种路径取到相同 id。
    """
    vocab = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    existing = {bytes(v) for v in vocab.values()}
    next_id = len(vocab)
    changed = False
    for token in sorted(tokens, key=len, reverse=True):
        token_bytes = token.encode("utf-8")
        if token_bytes in existing:
            continue
        vocab[str(next_id)] = list(token_bytes)
        existing.add(token_bytes)
        print(f"vocab.json: added special token {token} -> id {next_id}")
        next_id += 1
        changed = True
    if changed:
        Path(vocab_path).write_text(json.dumps(vocab), encoding="utf-8")
    return len(vocab)


def iter_stories(path, limit=None):
    """流式按 <|endoftext|> 切分语料，内存中只保留当前故事。

    跳过含 "<|" 的故事：防止正文与模板标记冲突（正常语料不应出现）。
    """
    buf = ""
    count = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            buf += line
            while STORY_SEP in buf:
                raw, _, buf = buf.partition(STORY_SEP)
                story = raw.strip()
                if story and "<|" not in story:
                    count += 1
                    yield story
                    if limit is not None and count >= limit:
                        return
    tail = buf.strip()
    if tail and "<|" not in tail:
        yield tail


def build_lexicon(stories, max_vocab=100_000, min_doc_freq=50, cap_ratio=0.4):
    """构建主题词表 = 白名单 ∩ 语料高频词，返回 (词表, 被剔除的专有名词)。

    三道过滤：
    - 白名单（TOPIC_ALLOWLIST）：人工审定的具体名词题库，动词/形容词/抽象词不收
    - 停用词 / 短词 / 低频词
    - 专有名词：英文人名（Tim/Lily/Spot）几乎总以大写形式出现，普通名词
      （bird/ball）只在句首偶发大写。大写占比 > cap_ratio 判为人名剔除——
      "Write a short story about Tom" 不是自然指令，主题应是内容词。
    """
    df = Counter()
    cap = Counter()
    total = Counter()
    for story in stories:
        words = WORD_RE.findall(story)
        df.update(set(word.lower() for word in words))
        for word in words:
            key = word.lower()
            total[key] += 1
            if word[0].isupper():
                cap[key] += 1

    candidates = []
    proper_nouns = []
    for word, count in df.items():
        if word not in TOPIC_ALLOWLIST:
            continue
        if word in STOPWORDS or len(word) < 3 or count < min_doc_freq:
            continue
        if total[word] > 0 and cap[word] / total[word] > cap_ratio:
            proper_nouns.append(word)
            continue
        candidates.append(word)

    lexicon = sorted(candidates, key=lambda w: (-df[w], w))[:max_vocab]
    return lexicon, sorted(proper_nouns)


def build_idf(stories, lexicon):
    """平滑 idf（sklearn 同款公式），避免零除。"""
    total_docs = max(len(stories), 1)
    doc_freq = Counter()
    for story in stories:
        story_words = set(words_of(story))
        doc_freq.update(word for word in story_words if word in lexicon)
    return {word: math.log((1 + total_docs) / (1 + doc_freq.get(word, 0))) + 1.0 for word in lexicon}


def _stem(word: str) -> str:
    """极简词干归并（复数/动名词/过去式后缀），仅供主题去重，非完整 stemmer。

    例子：toys->toy, singing->sing, running->run, stories->story。
    """
    for suffix in ("ing", "ies", "es", "ed", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            base = word[: len(word) - len(suffix)]
            if len(base) >= 2 and base[-1] == base[-2] and base[-1] not in "lsz":
                base = base[:-1]  # 双写辅音还原：runn->run
            if suffix == "ies":
                base += "y"
            return base
    return word


def select_topics(story, lexicon, idf, top_k=2):
    """TF-IDF 选主题：本篇高频（tf）× 语料稀有（idf）= 本篇的专属主题词。

    主槽位要求 tf>=2：主题词必须在正文复现，"指令词 -> 正文词"的
    绑定信号才成立（tf=1 的词可能是顺嘴一提）。
    同词根去重：top-k 里不允许出现同一词干两次
    （"fly and flying" / "toy and toys" 这类重复指令没有额外信息，
    第二个槽位应让给词干不同的下一个候选）。
    """
    tf = Counter(word for word in words_of(story) if word in lexicon)
    if not tf:
        return []
    pool = {w for w, c in tf.items() if c >= 2} or set(tf)
    ranked = sorted(pool, key=lambda word: tf[word] * idf.get(word, 1.0), reverse=True)
    chosen = []
    for word in ranked:
        if all(_stem(word) != _stem(c) for c in chosen):
            chosen.append(word)
        if len(chosen) == top_k:
            break
    return chosen


def split_sentences(story):
    cleaned = story.strip()
    if not cleaned:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def pair_A(story):
    """任务 A（续写）：前两句作指令，其余作回答。"""
    sentences = split_sentences(story)
    if len(sentences) < 3:
        return None
    instruction = "Continue this story: " + " ".join(sentences[:2])
    response = " ".join(sentences[2:])
    if len(response) < 30:
        return None
    return instruction, response


def pair_B(story, topics, rng=None, single_topic_prob=0.25):
    """任务 B（主题生成）：主题词作指令，全篇作回答。

    25% 的双主题样本只保留主主题：覆盖 "about X." 单主题指令格式
    （真实用户 prompt 与探针评测都是单主题，训练分布不能只有 "X and Y"）。
    选不出主题的故事直接丢弃——固定指令配随机故事等于教模型
    "这条指令没有含义"，宁可少几条数据。
    """
    if not topics:
        return None
    if rng is not None and len(topics) > 1 and rng.random() < single_topic_prob:
        topics = topics[:1]
    topic_text = " and ".join(topics)
    instruction = f"Write a short story about {topic_text}."
    response = story.strip()
    if len(response) < 25:
        return None
    return instruction, response


def encode_pair(instruction, response, tokenizer, max_seq_len=256,
                min_response_tokens=50, min_prompt_tokens=10):
    """编码为模板序列，返回 ((ids, prompt_len), None) 或 (None, 拒绝原因)。

    prompt_len 的两次编码 trick 依赖 tokenizer 先按特殊 token 切段：
    前缀在两次编码中切出的段字符串逐字相同，token 边界必然一致。
    EOS 计入 response 段（末尾的 <|endoftext|>），保证模型学会停止。
    """
    prompt_prefix = f"<|user|> {instruction} <|assistant|>"
    text = f"{prompt_prefix} {response} <|endoftext|>"
    ids = tokenizer.encode(text)
    prompt_len = len(tokenizer.encode(prompt_prefix))

    if len(ids) > max_seq_len:
        return None, "too_long"
    if prompt_len < min_prompt_tokens:
        return None, "prompt_short"
    if len(ids) - prompt_len < min_response_tokens:
        return None, "response_short"
    return (ids, prompt_len), None


def collect_pairs(task, pool, make_pair, tokenizer, quota, **encode_kwargs):
    """遍历故事池直到配额，返回 (records, 拒绝统计)。"""
    records = []
    rejected = Counter()
    for story in pool:
        if len(records) >= quota:
            break
        pair = make_pair(story)
        if pair is None:
            rejected["construct"] += 1
            continue
        encoded, reason = encode_pair(pair[0], pair[1], tokenizer, **encode_kwargs)
        if encoded is None:
            rejected[reason] += 1
            continue
        instruction, response = pair
        ids, prompt_len = encoded
        records.append({
            "task": task,
            "instruction": instruction,
            "response": response,
            "input_ids": ids,
            "prompt_len": prompt_len,
        })
    return records, rejected


def build_dataset(input_path, vocab_path, merges_path, output_path, *,
                  seed=42, read_limit=500_000, lexicon_sample=100_000,
                  pairs_per_task=7500, pairs_per_task_b=None, max_seq_len=256,
                  min_response_tokens=50, min_prompt_tokens=10, audit_size=100,
                  reuse_lexicon=None):
    output_path = Path(output_path)
    if pairs_per_task_b is None:
        pairs_per_task_b = pairs_per_task

    # 1. 固化特殊 token（幂等），保证 id 稳定为 <|assistant|>=10000、<|user|>=10001
    final_vocab_size = extend_vocab(vocab_path, SPECIAL_TOKENS)
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=SPECIAL_TOKENS)

    # 2. 流式读语料 + 固定种子 shuffle（数据管线可复现）
    stories = list(iter_stories(input_path, limit=read_limit))
    if len(stories) < 100:
        raise ValueError(f"Only {len(stories)} stories parsed from {input_path}; format mismatch?")
    rng = random.Random(seed)
    rng.shuffle(stories)
    print(f"Loaded stories: {len(stories)} (read_limit={read_limit})")

    # 3. 主题词表：复用（构造验证集时应复用训练集词表）或现场构建并落盘
    lexicon_file = output_path.with_name("sft_topic_lexicon.json")
    if reuse_lexicon:
        data = json.loads(Path(reuse_lexicon).read_text(encoding="utf-8"))
        lexicon = data["words"]
        idf = data["idf"]
        rest = stories  # 词表来自训练集，验证集故事可全部用于构造数据对
        print(f"Reused topic lexicon: {len(lexicon)} words from {reuse_lexicon}")
    else:
        lexicon_n = min(lexicon_sample, max(1, len(stories) // 4))
        lexicon_source = stories[:lexicon_n]
        rest = stories[lexicon_n:]
        lexicon, proper_nouns = build_lexicon(lexicon_source)
        idf = build_idf(lexicon_source, lexicon)
        Path(lexicon_file).write_text(
            json.dumps({
                "lexicon_sample_size": lexicon_n,
                "words": lexicon,
                "idf": idf,
                "dropped_proper_nouns": proper_nouns,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"Topic lexicon: {len(lexicon)} words from {lexicon_n} stories "
              f"({len(proper_nouns)} proper nouns dropped by capitalization ratio)")
        print(f"  -> saved to {lexicon_file} (HUMAN REVIEW REQUIRED before final use)")

    # 4. 互不重叠的故事池：A 取前半，B 取后半（同一 response 不以两种指令重复出现）
    half = len(rest) // 2
    pool_A, pool_B = rest[:half], rest[half:]
    topics_used = []

    def make_B(story):
        topics = select_topics(story, lexicon, idf)
        if len(topics) > 1 and rng.random() < 0.25:
            topics = topics[:1]  # 单主题指令变体（与 pair_B 的默认概率一致，掷一次硬币）
        topics_used.extend(topics)
        return pair_B(story, topics)

    encode_kwargs = dict(max_seq_len=max_seq_len, min_response_tokens=min_response_tokens,
                         min_prompt_tokens=min_prompt_tokens)
    records_A, rejected_A = collect_pairs("A", pool_A, pair_A, tokenizer, pairs_per_task, **encode_kwargs)
    records_B, rejected_B = collect_pairs("B", pool_B, make_B, tokenizer, pairs_per_task_b, **encode_kwargs)
    records = records_A + records_B

    # 5. 落盘：数据 jsonl + 人工抽检样本
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    audit = rng.sample(records, min(audit_size, len(records)))
    audit_path = output_path.with_name(output_path.stem + "_audit.jsonl")
    with audit_path.open("w", encoding="utf-8") as f:
        for item in audit:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # 6. 统计
    total_tokens = sum(len(r["input_ids"]) for r in records)
    avg_len = total_tokens / max(len(records), 1)
    print(f"\n=== SFT dataset stats ===")
    print(f"Samples: {len(records)} (A: {len(records_A)}, B: {len(records_B)})")
    print(f"Rejections: A={dict(rejected_A)}  B={dict(rejected_B)}")
    print(f"Total tokens: {total_tokens:,}  avg seq len: {avg_len:.0f}")
    print(f"Top-20 topics in B: {Counter(topics_used).most_common(20)}")
    print(f"Saved: {output_path}")
    print(f"Audit sample ({len(audit)} records, HUMAN REVIEW REQUIRED): {audit_path}")
    print(f"\nNOTE: 模型侧 vocab_size 需为 {final_vocab_size}；加载 baseline_best.pt 时须给")
    print(f"      token_embeddings / lm_head 各扩 2 行（SFT 训练脚本的 checkpoint 手术）。")


def main():
    parser = argparse.ArgumentParser(description="Build the SFT dataset from TinyStories raw text.")
    parser.add_argument("--input", required=True, help="TinyStories raw txt (stories separated by <|endoftext|>)")
    parser.add_argument("--output", type=str, default="data/sft_train.jsonl")
    parser.add_argument("--vocab-path", type=str, default="data/vocab.json")
    parser.add_argument("--merges-path", type=str, default="data/merges.txt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--read-limit", type=int, default=500_000, help="最多读取的故事篇数（流式，控制时长与内存）")
    parser.add_argument("--lexicon-sample", type=int, default=100_000)
    parser.add_argument("--pairs-per-task", type=int, default=7500)
    parser.add_argument("--pairs-b", type=int, default=None,
                        help="B 任务配额（默认同 --pairs-per-task；加大 B 可强化主题绑定）")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--min-response-tokens", type=int, default=50)
    parser.add_argument("--min-prompt-tokens", type=int, default=10)
    parser.add_argument("--audit-size", type=int, default=100)
    parser.add_argument("--reuse-lexicon", type=str, default=None,
                        help="复用已审核的主题词表 json（构造验证集时指定训练集的词表）")
    args = parser.parse_args()

    for path in [args.input, args.vocab_path, args.merges_path]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    build_dataset(
        args.input,
        args.vocab_path,
        args.merges_path,
        args.output,
        seed=args.seed,
        read_limit=args.read_limit,
        lexicon_sample=args.lexicon_sample,
        pairs_per_task=args.pairs_per_task,
        pairs_per_task_b=args.pairs_b,
        max_seq_len=args.max_seq_len,
        min_response_tokens=args.min_response_tokens,
        min_prompt_tokens=args.min_prompt_tokens,
        audit_size=args.audit_size,
        reuse_lexicon=args.reuse_lexicon,
    )


if __name__ == "__main__":
    main()
