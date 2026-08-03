"""Tokenize a corpus into flat uint16 .bin shards.

Default is the FineWeb-Edu 10BT sample with a 32k byte-level BPE trained on a
slice of it.

Sizing note: the `main` run consumes 12000 steps x 524288 tokens = 6.3B, so the
8B default leaves headroom.  Ablations need far less - a `tiny` trial is 800
steps x 16384 = 13M tokens, and trials sample random offsets from the same
file, so ~1B is already generous.  Prepare a small file first with
``--max-tokens 1000000000`` if you want to start tuning while the full corpus
builds; there is no reason to hold an accelerator idle waiting for 8B tokens.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

# Must be set before `tokenizers` is imported for its Rust thread pool to be used.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def preflight(tokenizer: str) -> None:
    """Check every dependency before doing any downloading.

    Collecting the BPE training sample streams hundreds of thousands of
    documents, so discovering a missing import afterwards wastes real time and
    bandwidth.
    """
    missing = []
    try:
        import datasets  # noqa: F401
    except ImportError:
        missing.append("datasets")
    if tokenizer == "gpt2":
        try:
            import tiktoken  # noqa: F401
        except ImportError:
            missing.append("tiktoken")
    else:
        try:
            import tokenizers  # noqa: F401
        except ImportError:
            missing.append("tokenizers")
    if missing:
        raise SystemExit(
            f"missing required packages: {', '.join(missing)}\n"
            f"  pip install {' '.join(missing)}\n"
            '  (or pip install -e ".[data]" to get all of them)'
        )


def _train_bpe(texts, vocab_size: int, out_path: str):
    from tokenizers import ByteLevelBPETokenizer

    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(texts, vocab_size=vocab_size, min_frequency=2,
                            special_tokens=["<|endoftext|>"])
    tok.save(out_path)
    return tok


def _load_tokenizer(kind: str, vocab_size: int, data_dir: str, sample_texts=None, threads: int = 0):
    """Return ``(encode_batch, vocab_size)``.

    Batch encoding matters: both backends release the GIL and parallelize
    internally, so encoding a list of documents is several times faster than
    looping one at a time, and tokenization is the bottleneck here - not the
    network.
    """
    n_threads = threads or (os.cpu_count() or 8)
    if kind == "gpt2":
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        eot = enc.eot_token

        def encode_batch(texts):
            return [ids + [eot] for ids in enc.encode_ordinary_batch(texts, num_threads=n_threads)]

        return encode_batch, enc.n_vocab

    from tokenizers import Tokenizer

    path = os.path.join(data_dir, "tokenizer.json")
    if not os.path.exists(path):
        if sample_texts is None:
            raise SystemExit(f"no tokenizer at {path} and no sample to train on")
        print(f"training {vocab_size}-token BPE (this takes a few minutes)...", flush=True)
        _train_bpe(sample_texts, vocab_size, path)
    tok = Tokenizer.from_file(path)
    eot_id = tok.token_to_id("<|endoftext|>")
    eot = 0 if eot_id is None else eot_id

    def encode_batch(texts):
        return [e.ids + [eot] for e in tok.encode_batch(texts)]

    return encode_batch, tok.get_vocab_size()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build train.bin / val.bin")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--name", default="sample-10BT")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--tokenizer", default="bpe32k", choices=["bpe32k", "gpt2"])
    ap.add_argument("--vocab-size", type=int, default=32768)
    ap.add_argument("--max-tokens", type=int, default=8_000_000_000,
                    help="train tokens to write; ~1e9 is plenty for ablations")
    ap.add_argument("--val-tokens", type=int, default=10_000_000)
    ap.add_argument("--bpe-train-docs", type=int, default=200_000)
    ap.add_argument("--batch-docs", type=int, default=1000,
                    help="documents per encode_batch call")
    ap.add_argument("--threads", type=int, default=0, help="0 = all cores")
    args = ap.parse_args(argv)

    preflight(args.tokenizer)
    from datasets import load_dataset

    os.makedirs(args.out_dir, exist_ok=True)
    ds = load_dataset(args.dataset, name=args.name or None, split=args.split, streaming=True)

    sample = None
    if args.tokenizer == "bpe32k" and not os.path.exists(
        os.path.join(args.out_dir, "tokenizer.json")
    ):
        print(f"collecting {args.bpe_train_docs:,} docs to train the tokenizer...", flush=True)
        sample = [r[args.text_key] for _, r in zip(range(args.bpe_train_docs), ds)]
    encode_batch, vocab = _load_tokenizer(
        args.tokenizer, args.vocab_size, args.out_dir, sample, args.threads
    )
    del sample
    if vocab > 65535:
        raise SystemExit(f"vocab {vocab} does not fit in uint16")
    print(f"tokenizer ready: vocab={vocab}", flush=True)

    val_path = os.path.join(args.out_dir, "val.bin")
    train_path = os.path.join(args.out_dir, "train.bin")
    written = {"val": 0, "train": 0}
    buf: list[int] = []
    docs: list[str] = []
    f = open(val_path, "wb")
    phase = "val"
    t0 = time.time()
    next_report = 100_000_000

    def flush_tokens():
        nonlocal buf
        if buf:
            np.asarray(buf, np.uint16).tofile(f)
            written[phase] += len(buf)
            buf = []

    def report():
        nonlocal next_report
        if written["train"] >= next_report:
            dt = time.time() - t0
            rate = written["train"] / max(dt, 1e-9)
            left = max(args.max_tokens - written["train"], 0) / max(rate, 1e-9)
            print(
                f"  {written['train']/1e9:.2f}B / {args.max_tokens/1e9:.2f}B train tokens "
                f"| {rate/1e6:.1f}M tok/s | ~{left/3600:.1f}h left",
                flush=True,
            )
            next_report += 100_000_000

    stop = False
    for rec in ds:
        docs.append(rec[args.text_key])
        if len(docs) < args.batch_docs:
            continue
        for ids in encode_batch(docs):
            buf.extend(ids)
        docs = []
        if len(buf) >= 4_000_000:
            flush_tokens()
            if phase == "val" and written["val"] >= args.val_tokens:
                f.close()
                f = open(train_path, "wb")
                phase = "train"
                t0 = time.time()
                print(f"val.bin done ({written['val']:,} tokens); writing train.bin", flush=True)
            elif phase == "train":
                report()
                if written["train"] >= args.max_tokens:
                    stop = True
        if stop:
            break

    if docs and not stop:
        for ids in encode_batch(docs):
            buf.extend(ids)
    flush_tokens()
    f.close()
    print(
        f"done in {(time.time() - t0)/60:.1f} min: {written['train']:,} train tokens, "
        f"{written['val']:,} val tokens in {args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
