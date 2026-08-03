"""Tokenize a corpus into flat uint16 .bin shards.

Default is the FineWeb-Edu 10BT sample with a 32k byte-level BPE trained on a
slice of it.  ~6B tokens is what a 2-day RTX 5090 run consumes, so the default
cap is set a little above that.
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def _train_bpe(texts, vocab_size: int, out_path: str):
    try:
        from tokenizers import ByteLevelBPETokenizer
    except ImportError as e:  # pragma: no cover - optional dependency
        raise SystemExit(
            "training a BPE needs `pip install tokenizers`, or pass --tokenizer gpt2"
        ) from e
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(texts, vocab_size=vocab_size, min_frequency=2,
                            special_tokens=["<|endoftext|>"])
    tok.save(out_path)
    return tok


def _load_tokenizer(kind: str, vocab_size: int, data_dir: str, sample_texts=None):
    if kind == "gpt2":
        try:
            import tiktoken
        except ImportError as e:  # pragma: no cover
            raise SystemExit("`pip install tiktoken` for --tokenizer gpt2") from e
        enc = tiktoken.get_encoding("gpt2")
        return (lambda s: enc.encode_ordinary(s) + [enc.eot_token]), enc.n_vocab
    path = os.path.join(data_dir, "tokenizer.json")
    try:
        from tokenizers import Tokenizer
    except ImportError as e:  # pragma: no cover - optional dependency
        raise SystemExit(
            "`pip install tokenizers` for the 32k BPE, or pass --tokenizer gpt2"
        ) from e

    if not os.path.exists(path):
        if sample_texts is None:
            raise SystemExit(f"no tokenizer at {path} and no sample to train on")
        print(f"training {vocab_size}-token BPE (this takes a few minutes)...")
        _train_bpe(sample_texts, vocab_size, path)
    tok = Tokenizer.from_file(path)
    eot = tok.token_to_id("<|endoftext|>") or 0
    return (lambda s: tok.encode(s).ids + [eot]), tok.get_vocab_size()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build train.bin / val.bin")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--name", default="sample-10BT")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--tokenizer", default="bpe32k", choices=["bpe32k", "gpt2"])
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=8_000_000_000)
    ap.add_argument("--val-tokens", type=int, default=10_000_000)
    ap.add_argument("--bpe-train-docs", type=int, default=200_000)
    args = ap.parse_args(argv)

    try:
        from datasets import load_dataset
    except ImportError as e:  # pragma: no cover
        raise SystemExit("`pip install datasets` to prepare data") from e

    os.makedirs(args.out_dir, exist_ok=True)
    ds = load_dataset(args.dataset, name=args.name or None, split=args.split, streaming=True)

    sample = None
    if args.tokenizer == "bpe32k" and not os.path.exists(
        os.path.join(args.out_dir, "tokenizer.json")
    ):
        print(f"collecting {args.bpe_train_docs} docs to train the tokenizer...")
        sample = [
            r[args.text_key]
            for _, r in zip(range(args.bpe_train_docs), ds)
        ]
    encode, vocab = _load_tokenizer(args.tokenizer, args.vocab_size, args.out_dir, sample)
    if vocab > 65535:
        raise SystemExit(f"vocab {vocab} does not fit in uint16")
    print(f"tokenizer ready: vocab={vocab}")

    val_path = os.path.join(args.out_dir, "val.bin")
    train_path = os.path.join(args.out_dir, "train.bin")
    written = {"val": 0, "train": 0}
    buf: list[int] = []
    f = open(val_path, "wb")
    phase = "val"

    for i, rec in enumerate(ds):
        buf.extend(encode(rec[args.text_key]))
        if len(buf) >= 1_000_000:
            np.asarray(buf, np.uint16).tofile(f)
            written[phase] += len(buf)
            buf = []
            if phase == "val" and written["val"] >= args.val_tokens:
                f.close()
                f = open(train_path, "wb")
                phase = "train"
                print(f"val.bin done ({written['val']:,} tokens); writing train.bin")
            elif phase == "train":
                if written["train"] % 100_000_000 < 1_000_000:
                    print(f"  {written['train']:,} train tokens", flush=True)
                if written["train"] >= args.max_tokens:
                    break
    if buf:
        np.asarray(buf, np.uint16).tofile(f)
        written[phase] += len(buf)
    f.close()
    print(f"done: {written['train']:,} train tokens, {written['val']:,} val tokens in {args.out_dir}")


if __name__ == "__main__":
    main()
