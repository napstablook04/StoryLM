import regex as re
from collections import Counter
from collections import defaultdict
from typing import Iterable, Iterator
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pre_tokenizer(text):
    token_counts = Counter()

    for match in re.finditer(PAT, text):
        tokens = match.group()

        token_bytes =tuple ([bytes([t]) for t in tokens.encode('utf-8')])

        token_counts[token_bytes] += 1

    return token_counts

def count_pair_and_occ(token_counts):

    pair_counts = Counter()

    pair_occ = defaultdict(set)

    for token, count in token_counts.items():
        for i in range(len(token)-1):
            pair = (token[i], token[i+1])
            pair_counts[pair] += count
            pair_occ[pair].add(token)

    return pair_counts,pair_occ

def merge_pair(token_counts, pair_counts, pair_occ, best_pair):

    affected_tokens = list(pair_occ[best_pair])


    for token in affected_tokens:
        
        i = 0
        new_token = []

        while i < len(token):
            if (i < len(token) - 1 and (token[i], token[i+1]) == best_pair ):
                new_token.append(token[i] + token[i+1])
                i += 2
            else:
                new_token.append(token[i])
                i += 1

        new_token = tuple(new_token)
        count = token_counts.pop(token)

        token_counts[new_token] += count #更新token_counts:老的删掉，新的继承原来的次数，若已存在，累加。

        for i in range(len(token) - 1): 
             
            pair = (token[i],token[i+1]) 
            pair_occ[pair].discard(token) #更新pair_occ:和老token关联的所有pair全部删掉老token，并补入新token
            
            pair_counts[pair] -= count #更新pair_counts:和老token关联的所有pair全部剪掉老token的影响,新token产生的所有pair继承老token的次数

            if pair_counts[pair] == 0:
                del pair_counts[pair]

        for i in range(len(new_token) - 1):
            pair = (new_token[i],new_token[i+1])

            pair_occ[pair].add(new_token)

            pair_counts[pair] += count

    return token_counts,pair_counts,pair_occ


def build_vocab(special_tokens):
    vocab = {i: bytes([i]) for i in range(256)}

    for i, token in enumerate(special_tokens, start=256):
        vocab[i] = token.encode("utf-8")

    return vocab


import os
import regex as re
from collections import Counter, defaultdict


def find_chunk_boundaries(
    file,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    将文件按照 special token 划分成若干 chunk。
    每个 chunk 的边界都落在 split_special_token 的位置，
    因此不会把一个特殊 token 切成两半。
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # 获取文件大小
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    # 理想情况下每个 chunk 的大小
    chunk_size = file_size // desired_num_chunks

    # 初始边界
    chunk_boundaries = [
        i * chunk_size
        for i in range(desired_num_chunks + 1)
    ]

    # 最后一个边界必须是文件末尾
    chunk_boundaries[-1] = file_size

    # 每次向后搜索 4KB
    mini_chunk_size = 4096

    # 找中间的边界
    for bi in range(1, len(chunk_boundaries) - 1):

        initial_position = chunk_boundaries[bi]

        file.seek(initial_position)

        while True:

            mini_chunk = file.read(mini_chunk_size)

            # 到文件末尾了
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # 找 special token
            found_at = mini_chunk.find(
                split_special_token
            )

            if found_at != -1:

                chunk_boundaries[bi] = (
                    initial_position + found_at
                )

                break

            initial_position += mini_chunk_size

    # 去掉重复边界并排序
    return sorted(set(chunk_boundaries))


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
):
    # ========================================================
    # 1. 初始化 token_counts
    # ========================================================

    token_counts = Counter()

    # --------------------------------------------------------
    # 用 special token 作为 chunk 的分界点
    # --------------------------------------------------------

    split_special_token = special_tokens[0].encode("utf-8")

    # 先以二进制打开文件，寻找 chunk 边界
    with open(input_path, "rb") as f:

        boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=4,
            split_special_token=split_special_token,
        )

        print("Chunk boundaries:", boundaries)
        print("Number of chunks:", len(boundaries) - 1)

        # ====================================================
        # 2. 逐个 chunk 做 pre-tokenization
        # ====================================================

        for start, end in zip(
            boundaries[:-1],
            boundaries[1:],
        ):

            f.seek(start)

            chunk = f.read(
                end - start
            ).decode(
                "utf-8",
                errors="ignore",
            )

            # --------------------------------------------
            # 特殊 token 硬切分
            # --------------------------------------------

            if special_tokens:

                p = "|".join(
                    re.escape(token)
                    for token in special_tokens
                )

                parts = re.split(
                    p,
                    chunk,
                )

                for part in parts:
                    token_counts.update(
                        pre_tokenizer(part)
                    )

            else:

                token_counts.update(
                    pre_tokenizer(chunk)
                )

    # ========================================================
    # 3. 初始化 vocab
    # ========================================================

    vocab = build_vocab(special_tokens)

    next_id = len(vocab)

    # ========================================================
    # 4. 初始化 pair counts
    # ========================================================

    pair_counts, pair_occ = count_pair_and_occ(
        token_counts
    )

    merges = []

    # ========================================================
    # 5. BPE merge
    # ========================================================

    while len(vocab) < vocab_size:

        if not pair_counts:
            break

        # 找出现次数最多的 pair
        best_pair = max(
            pair_counts,
            key=lambda p: (
                pair_counts[p],
                p,
            ),
        )

        # merge
        token_counts, pair_counts, pair_occ = merge_pair(
            token_counts,
            pair_counts,
            pair_occ,
            best_pair,
        )

        # 新 token
        new_byte = (
            best_pair[0] +
            best_pair[1]
        )

        vocab[next_id] = new_byte

        next_id += 1

        merges.append(best_pair)

    return vocab, merges



'''
def train_bpe(input_path, vocab_size, special_tokens):
    with open(input_path, "r", encoding="utf-8") as f:
     text = f.read()

    #1.初始化  token_counts
    token_counts = Counter()

    if special_tokens:
        #特殊字符硬分割
        p = "|".join(re.escape(token) for token in special_tokens)
        parts = re.split(p,text)
        for part in parts:
            token_counts.update(pre_tokenizer(part))
    else:
        token_counts.update(pre_tokenizer(text))

    #2.初始化vocab
    vocab = build_vocab(special_tokens)
    next_id = len(vocab)
    
    #3.初始化 pair_counts,pair_occ
    pair_counts, pair_occ = count_pair_and_occ(token_counts)

    merges = [] 

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        best_pair = max(pair_counts,key = lambda p :(pair_counts[p],p))

        token_counts, pair_counts, pair_occ = merge_pair(token_counts,pair_counts,pair_occ,best_pair)

        new_byte = best_pair[0] + best_pair[1]

        vocab[next_id] = new_byte
        next_id += 1 

        merges.append(best_pair)

    return vocab, merges
 '''

def apply_merges(token, merge_rank):
    token = list(token)

    if len(token) < 2:
        return token

    while True:
        best_pos = -1
        best_rank = float("inf")

        for i in range(len(token) - 1):
            rank = merge_rank.get((token[i], token[i + 1]),float("inf"),)

            if rank < best_rank:
                best_rank = rank
                best_pos = i

        if best_pos == -1:
            break

        token[best_pos] += token[best_pos + 1]
        del token[best_pos + 1]

    return token

class Tokenizer:

    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(special_tokens or [],key=len,reverse=True,)

        self.token_to_id = {token: i for i, token in vocab.items()} # 反向字典：能从token找到对应的index
        self.merge_rank = {pair: i for i, pair in enumerate(merges)}

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")

            if token_bytes not in self.token_to_id:
                new_id = len(vocab)
                self.vocab[new_id] = token_bytes
                self.token_to_id[token_bytes] = new_id

  
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,):
        # 读取 vocab
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        vocab = {
            int(k): bytes(v)
            for k, v in vocab_data.items()
        }

        # 读取 merges
        merges = []

        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                pair = json.loads(line)

                merges.append((bytes(pair[0]),bytes(pair[1])))

        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]: 
        ''' 把输入文本按照训练好的 BPE 规则进行切分和 merge
            然后把最终得到的 BPE token 转换成词表vocab中对应的整数 ID
        '''
        encode_list = []

        special_tokens = self.special_tokens
        token_to_id = self.token_to_id
        merge_rank = self.merge_rank

        if special_tokens:
            p = "(" + "|".join( re.escape(token)for token in special_tokens) + ")" #和之前不一样，加了括号，special_token也会保存下来
            parts = re.split(p, text) # PARTS: DOC1,<|endoftext|>,DOC2

        else:
            parts = [text]

        for part in parts:
            # special token
            if part in special_tokens:
                encode_list.append(token_to_id[part.encode("utf-8")])
                continue

            # 普通文本
            for match in re.finditer(PAT, part):
                token = match.group()
                token_bytes = tuple(bytes([b])for b in token.encode("utf-8"))

                token_bytes = apply_merges(token_bytes,merge_rank)

                for byte_token in token_bytes:
                    encode_list.append(token_to_id[byte_token])

        return encode_list

    def encode_iterable(self, iterable: Iterable[str],) -> Iterator[int]:  
     for text in iterable:
        yield from self.encode(text)
        

    def decode(self, ids: list[int]) -> str:
        byte_data =b''.join(self.vocab[i] for i in ids)
        return byte_data.decode("utf-8", errors="replace")