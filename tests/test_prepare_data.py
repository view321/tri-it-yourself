"""Offline tests for tri.prepare_data: mix parsing and sharded-resume logic.

Everything network-facing is faked; what matters here is the bookkeeping -
that a resumed prep continues from the recorded document position, never
duplicates data, and prunes incomplete parts.
"""

import json
import os

import numpy as np
import pytest

from tri.prepare_data import (
    MIXES,
    _save_json_atomic,
    _validate_parts,
    _write_train_sharded,
    parse_mix,
)


# -- mix parsing -------------------------------------------------------


def test_named_mix_weights_are_normalized():
    specs = parse_mix("reason")
    assert [t for t, *_ in specs] == ["web", "math", "code", "synth"]
    assert sum(w for *_, w in specs) == pytest.approx(1.0)


def test_custom_mix_with_dir_config_and_empty_config():
    specs = parse_mix(
        "code=bigcode/starcoderdata:dir:python:content:3,"
        "py=codeparrot/codeparrot-clean::content:1"
    )
    (t1, d1, c1, k1, w1), (t2, d2, c2, k2, w2) = specs
    assert (t1, d1, c1, k1) == ("code", "bigcode/starcoderdata", "dir:python", "content")
    assert (t2, d2, c2, k2) == ("py", "codeparrot/codeparrot-clean", None, "content")
    assert w1 == pytest.approx(0.75) and w2 == pytest.approx(0.25)


def test_bad_mix_entry_is_rejected():
    with pytest.raises(SystemExit, match="bad mix entry"):
        parse_mix("nonsense")


# -- sharded writing with resume ---------------------------------------
#
# Fake corpus: document i is the string str(i); the fake tokenizer turns it
# into DOC_TOKENS copies of (i % 250), so every output byte identifies the
# document it came from and concatenated outputs can be compared exactly.

DOC_TOKENS = 10


def fake_stream(skip: int):
    def gen():
        i = skip
        while True:
            yield str(i)
            i += 1

    return gen()


def encode_batch(texts):
    return [[int(t) % 250] * DOC_TOKENS for t in texts]


def run_sharded(tmp_path, max_tokens, shard_tokens, manifest=None):
    manifest = manifest if manifest is not None else {"parts": []}
    path = os.path.join(str(tmp_path), "mix_manifest.json")
    n = _write_train_sharded(
        fake_stream, encode_batch, str(tmp_path), max_tokens, shard_tokens,
        batch_docs=2, manifest=manifest,
        save_manifest=lambda: _save_json_atomic(manifest, path),
    )
    return n, manifest


def concat_parts(tmp_path, manifest):
    out = b""
    for p in manifest["parts"]:
        with open(os.path.join(str(tmp_path), p["file"]), "rb") as f:
            out += f.read()
    return out


def test_sharded_write_sizes_and_cumulative_docs(tmp_path):
    n, manifest = run_sharded(tmp_path, max_tokens=250, shard_tokens=100)
    assert n == 250
    assert [p["tokens"] for p in manifest["parts"]] == [100, 100, 50]
    assert [p["docs"] for p in manifest["parts"]] == [10, 20, 26]
    assert manifest["complete"]
    # parts are renamed into place on completion; nothing in-flight remains
    assert not [f for f in os.listdir(str(tmp_path)) if f.endswith(".writing")]


def test_resume_continues_identically(tmp_path):
    straight = tmp_path / "straight"
    split = tmp_path / "split"
    straight.mkdir()
    split.mkdir()

    _, m_straight = run_sharded(straight, max_tokens=300, shard_tokens=100)

    # First segment stops cleanly at 200 tokens; the "restart" reuses the
    # manifest exactly as a re-run after a preemption would.
    _, m1 = run_sharded(split, max_tokens=200, shard_tokens=100)
    m1.pop("complete")
    m1.pop("train_tokens")
    _, m2 = run_sharded(split, max_tokens=300, shard_tokens=100, manifest=m1)

    assert concat_parts(split, m2) == concat_parts(straight, m_straight)
    assert [p["docs"] for p in m2["parts"]] == [p["docs"] for p in m_straight["parts"]]


def test_validate_parts_prunes_incomplete_tail(tmp_path):
    _, manifest = run_sharded(tmp_path, max_tokens=300, shard_tokens=100)
    # Truncate the middle part: it and everything after must be dropped,
    # because later parts' document offsets build on it.
    victim = os.path.join(str(tmp_path), manifest["parts"][1]["file"])
    with open(victim, "r+b") as f:
        f.truncate(50)
    _validate_parts(manifest, str(tmp_path))
    assert [p["file"] for p in manifest["parts"]] == ["train_part0000.bin"]


def test_pruned_manifest_resumes_and_repairs(tmp_path):
    _, manifest = run_sharded(tmp_path, max_tokens=300, shard_tokens=100)
    reference = concat_parts(tmp_path, manifest)
    victim = os.path.join(str(tmp_path), manifest["parts"][2]["file"])
    with open(victim, "r+b") as f:
        f.truncate(20)
    _validate_parts(manifest, str(tmp_path))
    manifest.pop("complete")
    manifest.pop("train_tokens")
    _, repaired = run_sharded(tmp_path, max_tokens=300, shard_tokens=100,
                              manifest=manifest)
    assert concat_parts(tmp_path, repaired) == reference


def test_exhausted_stream_completes_early(tmp_path):
    def finite_stream(skip):
        return iter([str(i) for i in range(skip, 7)])

    manifest = {"parts": []}
    n = _write_train_sharded(
        finite_stream, encode_batch, str(tmp_path), max_tokens=1000,
        shard_tokens=40, batch_docs=2, manifest=manifest,
        save_manifest=lambda: None,
    )
    assert n == 7 * DOC_TOKENS
    assert manifest["complete"]
    sizes = [os.path.getsize(os.path.join(str(tmp_path), p["file"]))
             for p in manifest["parts"]]
    assert sum(sizes) == 2 * 7 * DOC_TOKENS
