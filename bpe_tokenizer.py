import regex as re
import pickle
from collections import defaultdict, Counter
from typing import Dict, Tuple, List, Iterable, Iterator
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

# Worker function for parallel encoding
def _encode_lines_worker(args):
    """
    Worker for parallel_encode_file: loads tokenizer from pickle and encodes a chunk of lines.
    """
    lines_chunk, pickle_path = args
    tokenizer = BPETokenizer.from_file(pickle_path)
    ids: List[int] = []
    for line in lines_chunk:
        ids.extend(tokenizer.encode(line))
    return ids

def merge(token: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """
    Merge occurrences of 'pair' in 'token' list with 'new_id'.
    """
    res = []
    i = 0
    while i < len(token):
        if i + 1 < len(token) and (token[i], token[i+1]) == pair:
            res.append(new_id)
            i += 2
        else:
            res.append(token[i])
            i += 1
    return res

class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) Tokenizer class with parallel file encoding.
    """

    def __init__(
        self,
        merges: Dict[Tuple[int, int], int],
        vocab: Dict[int, bytes],
        special_token_map: Dict[str, int]
    ):
        self.merges = merges
        self.vocab = vocab
        self.special_token_map = special_token_map
        # Pre-compile token regex using `regex` for \p unicode categories
        self.PAT = re.compile(
            r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )
        if special_token_map:
            pattern = '(' + '|'.join(map(re.escape, special_token_map.keys())) + ')'
            self.SPECIAL_PAT = re.compile(pattern)
        else:
            self.SPECIAL_PAT = None

    def encode(self, text: str) -> List[int]:
        """
        Encode a single string to a list of token IDs.
        """
        parts = [text]
        if self.SPECIAL_PAT:
            parts = self.SPECIAL_PAT.split(text)
        token_ids: List[int] = []

        # Initial tokenization into raw byte sequences
        tokens: List[List[int]] = []
        for part in parts:
            if self.SPECIAL_PAT and part in self.special_token_map:
                tokens.append([self.special_token_map[part]])
            else:
                for sub in self.PAT.findall(part):
                    tokens.append(list(sub.encode('utf-8')))

        # Apply BPE merges
        for token in tokens:
            for pair, new_id in self.merges.items():
                token = merge(token, pair, new_id)
            token_ids.extend(token)

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode lines from an iterable (e.g., file handle), yielding token IDs one by one.
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: List[int]) -> str:
        """
        Decode a list of token IDs back to a string.
        """
        byte_seq = b''.join(self.vocab[i] for i in ids)
        return byte_seq.decode('utf-8', errors='replace')

    def __call__(self, text: str) -> List[int]:
        return self.encode(text)

    @classmethod
    def from_file(cls, filepath: str) -> 'BPETokenizer':
        """
        Load tokenizer data from a pickle file and return an instance.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        vocab = data['vocab']
        merges = data['merges']
        special = data.get('special_token_map', {})
        inst = cls(merges=merges, vocab=vocab, special_token_map=special)
        inst.tokenizer_file = filepath
        return inst

    def parallel_encode_file(self, file_path: str, num_workers: int = 8) -> List[int]:
        """
        Encode an entire file in parallel, returning a flat list of token IDs.
        """
        lines = Path(file_path).read_text(encoding='utf-8').splitlines(keepends=True)
        chunks = [lines[i::num_workers] for i in range(num_workers)]
        args = [(chunk, self.tokenizer_file) for chunk in chunks]

        all_ids: List[int] = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for ids in executor.map(_encode_lines_worker, args):
                all_ids.extend(ids)
        return all_ids

if __name__ == "__main__":

    tokenizer = BPETokenizer.from_file("output/bpe_tokenizer.pkl")


    ids = tokenizer("Hello, how you are doing ? <|endoftext|> I am sardar gabbarsingh haha!!, <|endoftext|> naam tho suna hoga ")

    print(tokenizer.decode(ids))



    