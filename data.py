"""FineWeb -> packed (seq_len + 1) token blocks.

The stream order is deterministic, so val = the first `val_batches` batches and
train = everything after them. Same split in train.py and eval.py, no leakage.

    python data.py tokenizers/fineweb16k     # train the 16k BPE once, then reuse
"""

import argparse
import itertools

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

DATASET = "HuggingFaceFW/fineweb"
SUBSET = "sample-10BT"


def tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name)
    return tok


def train_tokenizer(out, vocab_size=16384, n_docs=200_000):
    """Own ByteLevel BPE on FineWeb.

    Qwen3 ships a 151936-token vocabulary; tied embeddings alone would then be 78M
    parameters at d_model=512, i.e. the whole budget is spent before the first block
    and every step pays for a 152k-wide lm_head. A 16k vocabulary keeps both readings
    of the 10M limit true at once (see NOTES).
    """
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    ds = load_dataset(DATASET, name=SUBSET, split="train", streaming=True)
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator((e["text"] for e in itertools.islice(ds, n_docs)),
                            vocab_size=vocab_size, special_tokens=["<|endoftext|>"])
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<|endoftext|>")
    tok.save_pretrained(out)
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


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--vocab-size", type=int, default=16384)
    p.add_argument("--n-docs", type=int, default=200_000)
    args = p.parse_args()
    tok = train_tokenizer(args.out, args.vocab_size, args.n_docs)
    print(f"{len(tok)} tokens -> {args.out}")
