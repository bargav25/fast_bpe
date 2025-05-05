import pickle
import pytest
from pathlib import Path
from bpe_tokenizer import BPETokenizer, merge

# 1) First, create a tiny “trained” tokenizer pickle in your tmp_path fixture:
@pytest.fixture(scope="module")
def tiny_tokenizer_pickle(tmp_path_factory):
    # Build a trivial vocab/merges/special map
    vocab = {i: bytes([i]) for i in range(256)}
    # add a merge for "ab" -> id 256
    vocab[256] = b"ab"
    merges = {(97, 98): 256}
    special = {"<|endoftext|>": 257}
    vocab[257] = b"<|endoftext|>"

    p = tmp_path_factory.mktemp("tok") / "tiny_tok.pkl"
    with open(p, "wb") as f:
        pickle.dump({"vocab": vocab, "merges": merges, "special_token_map": special}, f)
    return str(p)

# 2) Create a small sample text file
@pytest.fixture(scope="module")
def sample_file(tmp_path_factory):
    contents = [
        "ab\n",                  # tests merge
        "hello<|endoftext|>!",   # tests special token + punctuation
        "12345 799\n",               # numbers
    ]
    d = tmp_path_factory.mktemp("data")
    p = d / "sample.txt"
    p.write_text("".join(contents), encoding="utf-8")
    return str(p)

def test_parallel_vs_serial_encode(tiny_tokenizer_pickle, sample_file):
    # Load tokenizer
    tok = BPETokenizer.from_file(tiny_tokenizer_pickle)

    # 1) Serial encode by iterating file handle
    with open(sample_file, "r", encoding="utf-8") as f:
        serial_ids = list(tok.encode_iterable(f))

    # 2) Parallel encode the same file
    parallel_ids = tok.parallel_encode_file(sample_file, num_workers=4)

    # They must be identical
    assert parallel_ids == serial_ids

def test_parallel_id_counts(tiny_tokenizer_pickle, sample_file):
    # Another sanity check: counts of each ID match serial
    tok = BPETokenizer.from_file(tiny_tokenizer_pickle)

    serial_ids = list(tok.encode_iterable(open(sample_file, "r", encoding="utf-8")))
    from collections import Counter
    cnt_serial = Counter(serial_ids)
    cnt_parallel = Counter(tok.parallel_encode_file(sample_file, num_workers=4))

    assert cnt_serial == cnt_parallel