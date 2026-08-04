"""Tokenize a corpus into flat uint16 .bin shards.

Default is the FineWeb-Edu 10BT sample with a 32k byte-level BPE trained on a
slice of it.  ``--mix`` instead interleaves several sources by weight - the
named ``reason`` mix (educational web + FineMath + Python + synthetic
textbooks) is the one the `reason` preset is meant to train on.  A mix run
also writes one ``val_<tag>.bin`` per source next to the combined ``val.bin``,
so per-domain loss can be measured after (or during) training.

New tokenizers split numbers into individual digits (`Digits` pre-tokenizer),
which is a cheap, well-established win for arithmetic; an existing
``tokenizer.json`` is loaded as-is, so already-tokenized corpora stay valid.

Sizing note: the `main` run consumes 12000 steps x 524288 tokens = 6.3B, so the
8B default leaves headroom.  Ablations need far less - a `tiny` trial is 800
steps x 16384 = 13M tokens, and trials sample random offsets from the same
file, so ~1B is already generous.  Prepare a small file first with
``--max-tokens 1000000000`` if you want to start tuning while the full corpus
builds; there is no reason to hold an accelerator idle waiting for 8B tokens.
"""

from __future__ import annotations

import argparse
import json
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


# -- weighted mixes ----------------------------------------------------
#
# (tag, dataset, config, text_key, weight).  A config of ``dir:x`` selects a
# subdirectory (``data_dir``) instead of a named config.  Every source here
# streams with its text inline and loads without a dataset script - that
# rules out more sets than you would expect: python-edu and stack-edu only
# hold blob ids pointing at S3, and github-code-clean needs a loading script
# that `datasets` >= 3 refuses to run.
MIXES: dict[str, list[tuple[str, str, str | None, str, float]]] = {
    # Aimed at math / code / reasoning: educational web keeps general language
    # ability, FineMath and Python carry the target skills, and a slice of
    # synthetic textbooks adds clean expository structure.  If you have an HF
    # token and have accepted the StarCoder terms, swapping the code entry for
    #   code=bigcode/starcoderdata:dir:python:content:0.20
    # buys stronger license/quality filtering at the same weight.
    "reason": [
        ("web", "HuggingFaceFW/fineweb-edu", "sample-100BT", "text", 0.45),
        ("math", "HuggingFaceTB/finemath", "finemath-3plus", "text", 0.25),
        ("code", "codeparrot/codeparrot-clean", None, "content", 0.20),
        ("synth", "HuggingFaceTB/smollm-corpus", "cosmopedia-v2", "text", 0.10),
    ],
}


def parse_mix(spec: str) -> list[tuple[str, str, str | None, str, float]]:
    """``reason`` or ``tag=dataset:config:text_key:weight,...`` -> normalized specs.

    ``config`` may itself contain a colon (``dir:python``), so the entry is
    parsed from both ends: dataset first, weight and text_key last.
    """
    if spec in MIXES:
        entries = MIXES[spec]
    else:
        entries = []
        for part in spec.split(","):
            try:
                tag, rest = part.split("=", 1)
                fields = rest.split(":")
                ds, key, w = fields[0], fields[-2], float(fields[-1])
                cfg = ":".join(fields[1:-2])
            except (ValueError, IndexError):
                raise SystemExit(
                    f"bad mix entry {part!r}; want tag=dataset:config:text_key:weight "
                    "(empty config allowed, dir:x selects a subdirectory), or a "
                    "named mix: " + ", ".join(MIXES)
                )
            entries.append((tag.strip(), ds, cfg or None, key, w))
    total = sum(w for *_, w in entries)
    return [(t, d, c, k, w / total) for t, d, c, k, w in entries]


def _open_stream(dataset: str, config: str | None, split: str):
    from datasets import load_dataset

    kw: dict = {}
    if config and config.startswith("dir:"):
        kw["data_dir"] = config[4:]
    elif config:
        kw["name"] = config
    return load_dataset(dataset, split=split, streaming=True, **kw)


def _source_stream(dataset: str, config: str | None, split: str, key: str, skip: int = 0):
    """Yield document texts from one streaming source."""
    ds = _open_stream(dataset, config, split)
    if skip:
        ds = ds.skip(skip)
    for rec in ds:
        yield rec[key]


def _mix_stream(specs, split: str, skips: dict[str, int] | None = None, seed: int = 0,
                skip_docs: int = 0):
    """Interleave the sources by weight into one text stream.

    ``skips`` drops each source's val documents; ``skip_docs`` then skips into
    the *mixed* stream, which is deterministic given (specs, skips, seed) - the
    resume mechanism for sharded runs.  Skipping is linear in the distance (the
    stream is iterated and discarded), so a restart deep into a long prep pays
    download time but no tokenization time for the skipped span.
    """
    from datasets import interleave_datasets

    skips = skips or {}
    streams = []
    for tag, dataset, config, key, _w in specs:
        d = _open_stream(dataset, config, split)
        if skips.get(tag):
            d = d.skip(skips[tag])
        d = d.select_columns([key])
        if key != "text":
            d = d.rename_column(key, "text")
        streams.append(d)
    if len(streams) == 1:
        mixed = streams[0]
    else:
        mixed = interleave_datasets(
            streams, probabilities=[w for *_, w in specs], seed=seed
        )
    if skip_docs:
        mixed = mixed.skip(skip_docs)
    for rec in mixed:
        yield rec["text"]


def _train_bpe(texts, vocab_size: int, out_path: str):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    # Individual digits keep arithmetic learnable: "2048" becomes four tokens
    # with stable meanings instead of one opaque one.
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Digits(individual_digits=True),
            pre_tokenizers.ByteLevel(add_prefix_space=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer)
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


def _write_bin(texts, encode_batch, path: str, max_tokens: int, batch_docs: int,
               label: str) -> tuple[int, int]:
    """Tokenize a text stream into ``path`` until ``max_tokens``.

    Returns ``(tokens_written, docs_consumed)``.  ``docs_consumed`` counts every
    document pulled from the iterator (a partial batch pending at the stop
    point is dropped, not written), so a later stream may ``skip`` that many
    documents without ever overlapping this file.
    """
    written = 0
    docs = 0
    buf: list[int] = []
    batch: list[str] = []
    t0 = time.time()
    next_report = 100_000_000

    def flush(f):
        nonlocal written, buf
        np.asarray(buf, np.uint16).tofile(f)
        written += len(buf)
        buf = []

    with open(path, "wb") as f:
        for text in texts:
            batch.append(text)
            docs += 1
            if len(batch) < batch_docs:
                continue
            for ids in encode_batch(batch):
                buf.extend(ids)
            batch = []
            if len(buf) >= min(4_000_000, max_tokens - written):
                # A capped write lands the file on max_tokens exactly instead
                # of overshooting by up to a whole buffer, which matters for
                # the small weighted val bins.
                if written + len(buf) > max_tokens:
                    buf = buf[: max_tokens - written]
                flush(f)
                if written >= next_report:
                    rate = written / max(time.time() - t0, 1e-9)
                    left = max(max_tokens - written, 0) / max(rate, 1e-9)
                    print(
                        f"  [{label}] {written/1e9:.2f}B / {max_tokens/1e9:.2f}B tokens "
                        f"| {rate/1e6:.1f}M tok/s | ~{left/3600:.1f}h left",
                        flush=True,
                    )
                    next_report += 100_000_000
                if written >= max_tokens:
                    break
        else:  # stream exhausted before the budget: keep the tail
            if batch:
                for ids in encode_batch(batch):
                    buf.extend(ids)
        if buf and written < max_tokens:
            buf = buf[: max_tokens - written]
            flush(f)
    print(f"  [{label}] {written:,} tokens, {docs:,} docs -> {path}", flush=True)
    return written, docs


def _save_json_atomic(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _validate_parts(manifest: dict, out_dir: str) -> None:
    """Drop manifest entries whose files are missing or short.

    Parts record *cumulative* document positions, so only a trailing run of
    bad parts can be pruned - the first invalid part invalidates everything
    after it.  (Typically this is one part that a crash or an interrupted
    download left incomplete.)
    """
    good = []
    for p in manifest.get("parts", []):
        path = os.path.join(out_dir, p["file"])
        if os.path.exists(path) and os.path.getsize(path) == p["tokens"] * 2:
            good.append(p)
        else:
            break
    dropped = len(manifest.get("parts", [])) - len(good)
    if dropped:
        print(f"  [train] dropping {dropped} incomplete part(s); resuming before them",
              flush=True)
    manifest["parts"] = good


def _write_train_sharded(make_stream, encode_batch, out_dir: str, max_tokens: int,
                         shard_tokens: int, batch_docs: int, manifest: dict,
                         save_manifest) -> int:
    """Write the train stream as ``train_partNNNN.bin`` shards with exact resume.

    After each completed part the manifest records the part's token count and
    the cumulative *document* position in the mixed stream; a restart rebuilds
    the same deterministic stream, skips to that position, and continues with
    the next part.  A crash loses at most the in-flight token buffer (~4M
    tokens) and never duplicates data.
    """
    parts = manifest.setdefault("parts", [])
    tokens_done = sum(p["tokens"] for p in parts)
    docs_done = parts[-1]["docs"] if parts else 0
    if tokens_done >= max_tokens:
        return tokens_done
    if parts:
        print(f"  [train] resuming after {len(parts)} part(s), "
              f"{tokens_done/1e9:.2f}B tokens (skipping {docs_done:,} docs)", flush=True)

    texts = make_stream(docs_done)
    buf: list[int] = []
    batch: list[str] = []
    docs = docs_done
    exhausted = False
    t0 = time.time()
    done0 = tokens_done
    next_report = tokens_done + 100_000_000

    while tokens_done < max_tokens and not exhausted:
        target = min(shard_tokens, max_tokens - tokens_done)
        name = f"train_part{len(parts):04d}.bin"
        path = os.path.join(out_dir, name)
        # Parts are written under a temp name and renamed on completion, so a
        # concurrent GCS mirror only ever sees finished, immutable part files
        # (uploading a still-growing file aborts with a size-changed error).
        tmp = path + ".writing"
        written = 0
        with open(tmp, "wb") as f:
            while written < target:
                while not exhausted and len(buf) < min(4_000_000, target - written):
                    try:
                        batch.append(next(texts))
                        docs += 1
                    except StopIteration:
                        exhausted = True
                    if len(batch) >= batch_docs or (exhausted and batch):
                        for ids in encode_batch(batch):
                            buf.extend(ids)
                        batch = []
                if not buf:
                    break
                n = min(len(buf), target - written)
                np.asarray(buf[:n], np.uint16).tofile(f)
                del buf[:n]
                written += n
                tokens_done += n
                if tokens_done >= next_report:
                    rate = (tokens_done - done0) / max(time.time() - t0, 1e-9)
                    left = max(max_tokens - tokens_done, 0) / max(rate, 1e-9)
                    print(
                        f"  [train] {tokens_done/1e9:.2f}B / {max_tokens/1e9:.2f}B tokens "
                        f"| {rate/1e6:.1f}M tok/s | ~{left/3600:.1f}h left",
                        flush=True,
                    )
                    next_report += 100_000_000
        if written:
            os.replace(tmp, path)
            parts.append({"file": name, "tokens": written, "docs": docs})
            save_manifest()
            print(f"  [train] {name} done ({written:,} tokens, {docs:,} docs total)",
                  flush=True)
        else:
            os.remove(tmp)
    manifest["train_tokens"] = tokens_done
    manifest["complete"] = True
    save_manifest()
    return tokens_done


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build train.bin / val.bin")
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--name", default="sample-10BT")
    ap.add_argument("--split", default="train")
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--mix", default=None,
                    help="named mix (" + ", ".join(MIXES) + ") or "
                         "tag=dataset:config:text_key:weight,...; overrides --dataset")
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
    ap.add_argument("--seed", type=int, default=0, help="mix interleaving seed")
    ap.add_argument("--shard-tokens", type=int, default=0,
                    help="write train as train_partNNNN.bin shards of this many "
                         "tokens, with an incremental manifest and exact resume; "
                         "re-running the same command continues where it stopped "
                         "(concatenate parts into train.bin when done). 0 = one file.")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    preflight(args.tokenizer)
    os.makedirs(args.out_dir, exist_ok=True)
    specs = parse_mix(args.mix) if args.mix else None
    if args.shard_tokens and not specs:
        # One uniform sharded path: a single source is a one-entry mix.
        specs = [("main", args.dataset, args.name, args.text_key, 1.0)]

    # Sharded runs resume: the manifest must have been produced by the same
    # settings, or the skip arithmetic silently builds a different corpus.
    manifest_path = os.path.join(args.out_dir, "mix_manifest.json")
    args_sig = {
        "mix_specs": [list(s) for s in specs] if specs else None,
        "split": args.split,
        "seed": args.seed,
        "val_tokens": args.val_tokens,
        "shard_tokens": args.shard_tokens,
        "vocab_size": args.vocab_size,
        "tokenizer": args.tokenizer,
    }
    manifest = None
    if args.shard_tokens and os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest.get("args") != args_sig:
            raise SystemExit(
                f"{manifest_path} was written with different settings; delete the "
                "out dir (or pass matching flags) before resuming"
            )
        if not os.path.exists(os.path.join(args.out_dir, "tokenizer.json")):
            raise SystemExit(
                "manifest present but tokenizer.json missing - the corpus would be "
                "retokenized inconsistently.  Delete the out dir to start over."
            )
        if manifest.get("complete") and manifest.get("train_tokens", 0) >= args.max_tokens:
            print(
                f"manifest says this prep is complete at "
                f"{manifest['train_tokens']:,} tokens; nothing to do", flush=True
            )
            return
        # A larger --max-tokens re-opens a finished prep: the writer appends
        # parts from the recorded position and re-derives the flag.
        manifest.pop("complete", None)
    if args.shard_tokens and manifest is None:
        manifest = {"args": args_sig, "val_docs_skipped": {}, "parts": []}

    def fresh_texts():
        if specs:
            return _mix_stream(specs, args.split, seed=args.seed)
        return _source_stream(args.dataset, args.name, args.split, args.text_key)

    sample = None
    if args.tokenizer == "bpe32k" and not os.path.exists(
        os.path.join(args.out_dir, "tokenizer.json")
    ):
        print(f"collecting {args.bpe_train_docs:,} docs to train the tokenizer...", flush=True)
        sample = [t for _, t in zip(range(args.bpe_train_docs), fresh_texts())]
    encode_batch, vocab = _load_tokenizer(
        args.tokenizer, args.vocab_size, args.out_dir, sample, args.threads
    )
    del sample
    if vocab > 65535:
        raise SystemExit(f"vocab {vocab} does not fit in uint16")
    print(f"tokenizer ready: vocab={vocab}", flush=True)

    val_path = os.path.join(args.out_dir, "val.bin")
    train_path = os.path.join(args.out_dir, "train.bin")
    t_start = time.time()

    if args.shard_tokens:
        save_manifest = lambda: _save_json_atomic(manifest, manifest_path)  # noqa: E731
        skips = manifest["val_docs_skipped"]
        val_parts = [os.path.join(args.out_dir, f"val_{t}.bin") for t, *_ in specs]
        if not (skips and all(os.path.exists(p) for p in val_parts)):
            n_val = 0
            for (tag, dataset, config, key, w), part in zip(specs, val_parts):
                tokens, docs = _write_bin(
                    _source_stream(dataset, config, args.split, key),
                    encode_batch, part, max(1, int(args.val_tokens * w)),
                    args.batch_docs, f"val:{tag}",
                )
                skips[tag] = docs
                n_val += tokens
            manifest["val_tokens"] = n_val
            save_manifest()
        with open(val_path, "wb") as out:
            for part in val_parts:
                with open(part, "rb") as g:
                    out.write(g.read())
        _validate_parts(manifest, args.out_dir)
        save_manifest()

        n_train = _write_train_sharded(
            lambda skip: _mix_stream(specs, args.split, skips, args.seed, skip),
            encode_batch, args.out_dir, args.max_tokens, args.shard_tokens,
            args.batch_docs, manifest, save_manifest,
        )
        n_parts = len(manifest["parts"])
        print(
            f"done in {(time.time() - t_start)/60:.1f} min: {n_train:,} train tokens in "
            f"{n_parts} part(s), {manifest.get('val_tokens', 0):,} val tokens in {args.out_dir}\n"
            f"  stitch with: cat {args.out_dir}/train_part*.bin > {args.out_dir}/train.bin\n"
            f"  (or server-side: gsutil compose ...part*.bin .../train.bin - "
            "scripts/prep_to_gcs.sh does this for you)",
            flush=True,
        )
        return

    if not specs:
        # Single source, single pass: val is the head of the stream, train the rest.
        texts = fresh_texts()
        n_val, _ = _write_bin(texts, encode_batch, val_path, args.val_tokens,
                              args.batch_docs, "val")
        n_train, _ = _write_bin(texts, encode_batch, train_path, args.max_tokens,
                                args.batch_docs, "train")
    else:
        # Per-source val bins first (so per-domain loss is measurable later),
        # then the combined val.bin, then the interleaved train stream with
        # each source skipping the documents its val bin consumed.
        skips: dict[str, int] = {}
        n_val = 0
        val_parts = []
        for tag, dataset, config, key, w in specs:
            part = os.path.join(args.out_dir, f"val_{tag}.bin")
            tokens, docs = _write_bin(
                _source_stream(dataset, config, args.split, key),
                encode_batch, part, max(1, int(args.val_tokens * w)),
                args.batch_docs, f"val:{tag}",
            )
            skips[tag] = docs
            n_val += tokens
            val_parts.append(part)
        with open(val_path, "wb") as out:
            for part in val_parts:
                with open(part, "rb") as g:
                    out.write(g.read())
        print(f"  [val] combined {n_val:,} tokens -> {val_path}", flush=True)

        n_train, _ = _write_bin(
            _mix_stream(specs, args.split, skips, seed=args.seed),
            encode_batch, train_path, args.max_tokens, args.batch_docs, "train",
        )
        manifest = {
            "mix": args.mix,
            "specs": [
                {"tag": t, "dataset": d, "config": c, "text_key": k, "weight": w}
                for t, d, c, k, w in specs
            ],
            "val_docs_skipped": skips,
            "train_tokens": n_train,
            "val_tokens": n_val,
            "seed": args.seed,
        }
        with open(os.path.join(args.out_dir, "mix_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    print(
        f"done in {(time.time() - t_start)/60:.1f} min: {n_train:,} train tokens, "
        f"{n_val:,} val tokens in {args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
