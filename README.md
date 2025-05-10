# BPE-from-Scratch

A pure-Python Byte Pair Encoding (BPE) toolkit with parallel preprocessing and efficient merge caching. Train and use a custom BPE tokenizer end-to-end, entirely on CPU, at scale.

---

## 🚀 Project Overview

This repository implements a BPE tokenizer **from scratch** in Python, featuring:

- **Parallel pretokenization** using `mmap` and `multiprocessing` to pretokenize large corpora efficiently.  
- **Merge-level caching**: incremental updates of pair counts to avoid recomputing over the entire vocabulary each merge.  
- **Fast training**: Trained on the TinyStories V2 dataset (~2.12 million documents) in **~2 minutes** on a CPU.  
---

### 1. Train the Tokenizer

```bash
python train.py --vocab_size <vocab_size> --input <input_text_file_path> --output <output_tokenizer_file_in_pkl> 
```

#### 2. Test

```bash
pytest test.py
```

#### 3. To Encode random string

```bash
pytest bpe_tokenizer.py
```


#### 4. To encode a corpus

```bash
python tokenize_corpus.py --input <input_text_file_path> --output <output_memmmap_file> --tokenizer <trained_tokenizer_file_in_pkl>
```