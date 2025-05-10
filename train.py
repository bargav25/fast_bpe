
from pretokenize import get_pre_token_counter
from typing import List, Tuple, Dict, Set
import regex as re
from collections import defaultdict, Counter
from tqdm import tqdm
import pickle
import time 
import argparse

def merge(token: Tuple[int], pair: Tuple[int, int], new_id: int) -> Tuple[int]:
    res, i = [], 0

    while i < len(token):

        if i+1 < len(token) and (token[i], token[i+1]) == pair:
            res.append(new_id)
            i += 1
        else:
            res.append(token[i])

        i += 1

    return tuple(res)


def train_bpe_tokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str] = ()
) -> Tuple[dict[int, bytes], dict[tuple[[int, int], int]]]:

    start_time = time.time()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Initialize vocabulary
    vocab = {i: bytes([i]) for i in range(256)}
    next_id = 256

    merges: dict[tuple[int, int], int] = {}

    special_map = {}
    for tok in special_tokens:
        special_map[tok] = next_id
        vocab[next_id] = tok.encode("utf-8")
        next_id += 1
    
    # Validate and calculate merges needed
    num_initial = 256 + len(special_tokens)
    if vocab_size < num_initial:
        raise ValueError(f"vocab_size must be ≥ {num_initial}")
    num_merges = vocab_size - num_initial
    
    token_counts = get_pre_token_counter(input_path, special_map, parallelize = True) 

    # Caching in hashmaps for BPE training
    pair_to_tokens = defaultdict(set)
    token_to_pairs = defaultdict(list)
    pair_counts = Counter()

    for token, count in token_counts.items():
        pairs = list(zip(token, token[1:]))
        token_to_pairs[token] = pairs
        for p in pairs:
            pair_to_tokens[p].add(token)
            pair_counts[p] += count


    # Perform BPE merges
    for _ in tqdm(range(num_merges)):
        if not pair_counts:
            break

        # Select best pair with correct tie-breaking
        best_pair = max(
            pair_counts.items(),
            key=lambda kv: (kv[1], (vocab[kv[0][0]], vocab[kv[0][1]]))
        )[0]

        merges[best_pair] = next_id

        a, b = best_pair
        vocab[next_id] = vocab[a] + vocab[b]

        # Update tokens and pairs affected by the merge
        tokens_to_update = list(pair_to_tokens.get(best_pair, set()))

        for token in tokens_to_update:
            if token not in token_counts:
                continue

            freq = token_counts[token]
            old_pairs = token_to_pairs[token]

            # Decrement old pair counts
            for p in old_pairs:
                pair_counts[p] -= freq
                if pair_counts[p] <= 0:
                    del pair_counts[p]
                pair_to_tokens[p].discard(token)

            # Remove the old token
            del token_counts[token]
            del token_to_pairs[token]

            merged = merge(token, best_pair, next_id)

            merged_tuple = tuple(merged)

            # Update with merged token
            token_counts[merged_tuple] += freq
            new_pairs = list(zip(merged_tuple, merged_tuple[1:]))
            token_to_pairs[merged_tuple] = new_pairs

            for p in new_pairs:
                pair_counts[p] += freq
                pair_to_tokens[p].add(merged_tuple)

        next_id += 1

    print("Total Time taken for BPE: ", time.time() - start_time)

    return vocab, merges

def save_tokenizer_data(out_filepath, vocab, merges, special_tokens):

    with open(out_filepath,"wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges, "special_tokens": special_tokens}, f)
    
    print(f"Succesfully saved to {out_filepath}")

if __name__ == "__main__":

    special_tokens = ["<|endoftext|>"]

    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab_size", help = "vocab_size", default=10000, type=int)
    parser.add_argument("--input", help = "input text file path")
    parser.add_argument("--output", help = "output pkl file path for saving tokenizer")

    args = parser.parse_args()

    vocab, merges = train_bpe_tokenizer(args.input, args.vocab_size, special_tokens)

    save_tokenizer_data(args.output, vocab, merges, special_tokens)