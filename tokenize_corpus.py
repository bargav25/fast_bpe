import argparse
import numpy as np
import multiprocessing as mp
from typing import List
from bpe_tokenizer import BPETokenizer
import os
from tqdm import tqdm


def init_worker(tokenizer_path: str):
    global TOKENIZER
    TOKENIZER = BPETokenizer.from_file(tokenizer_path)


def encode_chunk(text_chunk: str) -> List[int]:
    return TOKENIZER.encode(text_chunk.strip())


def main():
    parser = argparse.ArgumentParser(description="Parallel BPE tokenization into .memmap")
    parser.add_argument("--input", type=str, required=True, help="Path to large text file")
    parser.add_argument("--output", type=str, required=True, help="Path to output .memmap file")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to BPE tokenizer .pkl file")
    parser.add_argument("--dtype", type=str, default="int32", help="NumPy dtype (default: int32)")
    parser.add_argument("--workers", type=int, default=16, help="Number of parallel workers")
    args = parser.parse_args()

    print(f"📖 Reading and splitting corpus from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        corpus = f.read()

    chunks = corpus.split("<|endoftext|>")
    print(f"🧩 Split into {len(chunks)} chunks.")

    print(f"⚙️ Tokenizing with {args.workers} workers...")
    with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(args.tokenizer,)) as pool:
        token_lists = list(tqdm(pool.imap(encode_chunk, chunks), total=len(chunks)))

    token_ids = np.array([tid for sublist in token_lists for tid in sublist], dtype=args.dtype)

    print(f"💾 Writing {len(token_ids)} tokens to {args.output} as memmap...")
    mmap = np.memmap(args.output, dtype=args.dtype, mode="w+", shape=(len(token_ids),))
    mmap[:] = token_ids[:]
    mmap.flush()

    print(f"✅ Done. Config for training:")
    print(f"    dtype: {args.dtype}")
    print(f"    data_shape: [{len(token_ids)}]")


if __name__ == "__main__":
    main()