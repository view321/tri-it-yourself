"""Checkpointing, with genuine 2-bit packing for ternary weights on disk.

In memory ternary weights are int8 (every matmul widens them anyway - there is
no ternary tensor core).  On disk they are packed 4-per-byte, which is where
"2 bits per weight" stops being an aspiration and becomes a file size.
"""

from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from .quant import pack2, unpack2


def _keys(tree) -> list[str]:
    return [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(tree)[0]]


def save(path: str, tree, pack: bool = True, extra: dict | None = None) -> str:
    """Save a pytree by leaf path.  int8 leaves are bit-packed when ``pack``."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    leaves = jax.tree_util.tree_leaves(tree)
    keys = _keys(tree)
    arrays, meta = {}, {}
    for k, v in zip(keys, leaves):
        v = np.asarray(v)
        if pack and v.dtype == np.int8:
            arrays[k] = np.asarray(pack2(jnp.asarray(v)))
            meta[k] = {"shape": list(v.shape), "dtype": "int8", "packed": True}
        else:
            arrays[k] = v
            meta[k] = {"shape": list(v.shape), "dtype": str(v.dtype), "packed": False}
    payload = {"__meta__": json.dumps({"leaves": meta, "extra": extra or {}})}
    np.savez(path, **payload, **arrays)
    return path


def load(path: str, template):
    """Load into the structure of ``template`` (a freshly built pytree)."""
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["__meta__"]))["leaves"]
        keys = _keys(template)
        treedef = jax.tree_util.tree_structure(template)
        out = []
        for k in keys:
            if k not in f:
                raise KeyError(f"checkpoint {path} is missing leaf {k}")
            info = meta[k]
            arr = f[k]
            if info["packed"]:
                arr = np.asarray(unpack2(jnp.asarray(arr), tuple(info["shape"])))
            out.append(jnp.asarray(arr, dtype=jnp.dtype(info["dtype"])))
        return jax.tree_util.tree_unflatten(treedef, out)


def read_extra(path: str) -> dict:
    with np.load(path, allow_pickle=False) as f:
        return json.loads(str(f["__meta__"]))["extra"]


def checkpoint_bytes(path: str) -> int:
    return os.path.getsize(path)
