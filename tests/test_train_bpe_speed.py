import time

from storylm.tokenizer import train_bpe


input_path = "data/TinyStoriesV2-GPT4-valid.txt"

print("开始训练 BPE...")
print("文件:", input_path)

start = time.time()

vocab, merges = train_bpe(
    input_path=input_path,
    vocab_size=10000,
    special_tokens=["<|endoftext|>"],
)

elapsed = time.time() - start

print("\n========== 结果 ==========")
print(f"vocab size : {len(vocab)}")
print(f"merges     : {len(merges)}")
print(f"time       : {elapsed:.2f} 秒")
print(f"time       : {elapsed / 60:.2f} 分钟")