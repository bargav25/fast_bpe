# BPE-from-Scratch

A pure-Python Byte Pair Encoding (BPE) toolkit with parallel preprocessing and efficient merge caching. Train and use a custom BPE tokenizer end-to-end, entirely on CPU, at scale.

---

## 🚀 Project Overview

This repository implements a BPE tokenizer **from scratch** in Python, featuring:

- **Parallel pretokenization** using `mmap` and `multiprocessing` to tokenize large corpora efficiently.  
- **Merge-level caching**: incremental updates of pair counts to avoid recomputing over the entire vocabulary each merge.  
- **Fast training**: Trained on the TinyStories V2 dataset (~2.12 million documents) in **~2 minutes** on a CPU.  
---
