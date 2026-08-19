import numpy as np

from storylm.tokenizer import Tokenizer


VOCAB_PATH = "data/vocab.json"
MERGES_PATH = "data/merges.txt"

TRAIN_PATH = "data/TinyStoriesV2-GPT4-train.txt"
VALID_PATH = "data/TinyStoriesV2-GPT4-valid.txt"

TRAIN_OUT = "data/tinystories_train.npy"
VALID_OUT = "data/tinystories_valid.npy"


def encode_file(tokenizer, input_path, output_path):
    print(f"Encoding: {input_path}")

    tokens = []

    with open(input_path, "r", encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            tokens.append(token_id)

    tokens = np.asarray(tokens, dtype=np.uint16)

    np.save(output_path, tokens)

    print(f"Saved: {output_path}")
    print(f"Number of tokens: {len(tokens):,}")
    print(f"Size: {tokens.nbytes / 1024**2:.2f} MB")


def main():
    tokenizer = Tokenizer.from_files(
        VOCAB_PATH,
        MERGES_PATH,
        special_tokens=["<|endoftext|>"],
    )

    encode_file(
        tokenizer,
        TRAIN_PATH,
        TRAIN_OUT,
    )

    encode_file(
        tokenizer,
        VALID_PATH,
        VALID_OUT,
    )


if __name__ == "__main__":
    main()