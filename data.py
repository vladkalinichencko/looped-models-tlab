"""Leak-free FineWeb tokenizer, document split, and reusable token blocks."""

from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

DATASET = "HuggingFaceFW/fineweb"
SUBSET = "sample-10BT"
REVISION = "9bb295ddab0e05d785b879661af7260fed5140fc"
TOKENIZER_DIR = Path("tokenizers/fineweb16k-clean")
SELECTION = range(0, 200)
FINAL = range(200, 2_000)
TRAIN_START = 2_000


@dataclass(frozen=True)
class Config:
    seq_len: int = 512
    batch_size: int = 16
    train_tokens: int = 8_000_000


def documents():
    return load_dataset(DATASET, name=SUBSET, split="train", streaming=True, revision=REVISION)


def train_tokenizer(path: Path = TOKENIZER_DIR, n_docs: int = 200_000):
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    chosen = itertools.islice(documents(), TRAIN_START, TRAIN_START + n_docs)
    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator((row["text"] for row in chosen), vocab_size=16_384,
                            special_tokens=["<|endoftext|>"])
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<|endoftext|>")
    path.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(path)
    (path / "source.json").write_text(json.dumps({
        "dataset": DATASET, "subset": SUBSET, "revision": REVISION,
        "document_indices": [TRAIN_START, TRAIN_START + n_docs],
    }, indent=2) + "\n")
    return tok


def tokenizer(path: Path = TOKENIZER_DIR):
    if not (path / "tokenizer.json").exists():
        return train_tokenizer(path)
    return AutoTokenizer.from_pretrained(path)


def tokenizer_hash(path: Path = TOKENIZER_DIR) -> str:
    return hashlib.sha256((path / "tokenizer.json").read_bytes()).hexdigest()[:12]


def prepare(tok, cfg: Config):
    key = f"fineweb_{tokenizer_hash()}_{cfg.seq_len}_{cfg.train_tokens}"
    cache = Path("datasets") / f"{key}.pt"
    manifest = Path("datasets") / f"{key}.json"
    if cache.exists() and manifest.exists():
        return torch.load(cache, weights_only=False), manifest

    train_blocks = math.ceil(cfg.train_tokens / cfg.seq_len)
    blocks = {name: [] for name in ("selection", "final", "train")}
    ids = {name: [] for name in blocks}
    buffers = {name: [] for name in blocks}

    for index, row in enumerate(documents()):
        name = "selection" if index in SELECTION else "final" if index in FINAL else "train"
        if name == "train" and index < TRAIN_START:
            continue
        if name == "train" and len(blocks[name]) >= train_blocks:
            break
        ids[name].append(row["id"])
        buffers[name].extend(tok(row["text"], add_special_tokens=False)["input_ids"])
        buffers[name].append(tok.eos_token_id)
        while len(buffers[name]) >= cfg.seq_len + 1 and (name != "train" or len(blocks[name]) < train_blocks):
            blocks[name].append(buffers[name][:cfg.seq_len + 1])
            del buffers[name][:cfg.seq_len + 1]

    if len(blocks["train"]) != train_blocks or not blocks["selection"] or not blocks["final"]:
        raise RuntimeError({name: len(rows) for name, rows in blocks.items()})

    payload = {name: torch.tensor(rows, dtype=torch.long) for name, rows in blocks.items()}
    cache.parent.mkdir(exist_ok=True)
    torch.save(payload, cache)
    manifest.write_text(json.dumps({
        "dataset": DATASET, "subset": SUBSET, "revision": REVISION,
        "tokenizer": str(TOKENIZER_DIR), "tokenizer_sha256": tokenizer_hash(),
        "config": asdict(cfg), "documents": ids,
    }, indent=2) + "\n")
    return payload, manifest


def batches(blocks: torch.Tensor, batch_size: int, device: str):
    for start in range(0, len(blocks) - batch_size + 1, batch_size):
        batch = blocks[start:start + batch_size].to(device)
        yield batch[:, :-1], batch[:, 1:]


if __name__ == "__main__":
    tok = tokenizer()
    payload, manifest = prepare(tok, Config())
    print({name: list(value.shape) for name, value in payload.items()}, flush=True)
    print(manifest, flush=True)
    os._exit(0)  # Arrow can hang while shutting down its global thread pool on macOS.
