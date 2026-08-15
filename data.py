"""FineWeb -> packed (seq_len + 1) token blocks.

The stream order is deterministic, so val = the first `val_batches` batches and
train = everything after them. Same split in train.py and eval.py, no leakage.
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

DATASET = "HuggingFaceFW/fineweb"
SUBSET = "sample-10BT"


def tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name)
    return tok


def block_stream(tok, seq_len, cache_dir=None):
    ds = load_dataset(DATASET, name=SUBSET, split="train", streaming=True, cache_dir=cache_dir)
    eos = tok.eos_token_id
    buf = []
    for example in ds:
        buf += tok(example["text"], add_special_tokens=False)["input_ids"]
        if eos is not None:
            buf.append(eos)
        while len(buf) >= seq_len + 1:
            yield buf[: seq_len + 1]
            buf = buf[seq_len + 1 :]


def batches(stream, batch_size, device):
    batch = []
    for block in stream:
        batch.append(block)
        if len(batch) == batch_size:
            t = torch.tensor(batch, dtype=torch.long, device=device)
            yield t[:, :-1], t[:, 1:]
            batch = []


def split(tok, seq_len, batch_size, val_batches, device, cache_dir=None):
    """-> (val: list of (x, y), train: iterator of (x, y))."""
    stream = batches(block_stream(tok, seq_len, cache_dir), batch_size, device)
    val = [next(stream) for _ in range(val_batches)]
    return val, stream
