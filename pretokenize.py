    

import regex as re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
import mmap
import time
from tqdm import tqdm

def find_chunk_boundaries(file, desired_num_chunks, split_special_token):
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))

def process_chunk_mmap_worker(args):
    mmap_path, start, end, special_token_bytes, regex_patterns = args
    special_pat_str, token_pat_str = regex_patterns
    try:
        with open(mmap_path, "rb") as f:
            mmap_obj = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            chunk_bytes = mmap_obj[start:end]
            text = chunk_bytes.decode("utf-8", errors="replace")

        special_token_strings = [st.decode("utf-8") for st in special_token_bytes]
        special_pat = re.compile(special_pat_str)
        parts = special_pat.split(text)

        token_lists = []
        token_pat = re.compile(token_pat_str)
        for part in parts:
            if part in special_token_strings:
                token_lists.append((part.encode("utf-8"),))
            else:
                for sub in token_pat.findall(part):
                    byte_ids = tuple(sub.encode("utf-8"))
                    token_lists.append(byte_ids)

        return Counter(map(tuple, token_lists))
    except Exception as e:
        print(f"Error processing chunk {start}-{end}: {str(e)}")
        return Counter()

def get_pre_token_counter_parallel(input_path, special_map, split_token=b"<|endoftext|>", num_workers=8):
    start_time = time.time()

    special_token_bytes = [tok.encode("utf-8") for tok in special_map.keys()]
    special_token_strings = [tok.decode("utf-8") for tok in special_token_bytes]
    special_pat_str = '(' + '|'.join(map(re.escape, special_token_strings)) + ')'
    token_pat_str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_workers, split_token)

    args = [
        (
            input_path,
            start,
            end,
            special_token_bytes,
            (special_pat_str, token_pat_str)
        ) for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    print("Pretokenization Started")

    token_counts = Counter()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_chunk_mmap_worker, arg) for arg in args]
        for future in as_completed(futures):
            chunk_counts = future.result()
            for token_seq, count in chunk_counts.items():
                if len(token_seq) == 1 and isinstance(token_seq[0], bytes):
                    special_str = token_seq[0].decode("utf-8")
                    token_counts[(special_map[special_str],)] += count
                else:
                    token_counts[tuple(token_seq)] += count

    print("Pretokenization completed")
    print("Time Taken:", time.time() - start_time)
    return token_counts



def get_pre_token_counter_normal(input_path, special_map):

    start_time = time.time()


    special_tokens = list(special_map.keys())

    with open(input_path, 'r', encoding = 'utf-8') as f:
        text = f.read()
    
    if special_tokens:
        special_pat = re.compile('(' + '|'.join(map(re.escape, special_tokens)) + ')')
        parts = special_pat.split(text)
    else:
        parts = [text]

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    token_lists = []
    for part in parts:
        if part in special_tokens:
            token_lists.append(list(part.encode('utf-8')))
        else:
            for sub in re.findall(PAT, part):
                token_lists.append(list(sub.encode('utf-8')))

    pre_token_counter = Counter(tuple(t) for t in token_lists)

    print("Pretokenization completed")
    print("Time Taken: ", time.time() - start_time)

    return pre_token_counter

def get_pre_token_counter(input_path, special_map, parallelize = False):

    if parallelize:
        return get_pre_token_counter_parallel(input_path, special_map)
    else:
        return get_pre_token_counter_normal(input_path, special_map)
    

