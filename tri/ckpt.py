"""Checkpointing, with genuine 2-bit packing for ternary weights on disk.

In memory ternary weights are int8 (every matmul widens them anyway - there is
no ternary tensor core).  On disk they are packed 4-per-byte, which is where
"2 bits per weight" stops being an aspiration and becomes a file size.
"""

from __future__ import annotations

import glob
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from .quant import pack2, unpack2


def _keys(tree) -> list[str]:
    return [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(tree)[0]]


def _is_ternary(v: np.ndarray) -> bool:
    """Only pack arrays that actually live on the lattice.

    Being int8 is not enough: with ``--sign-momentum-dtype int8`` the momentum
    buffer is also int8 but spans [-127, 127], and ``pack2`` keeps two bits per
    value.  Packing it would silently truncate every buffer entry.
    """
    return v.dtype == np.int8 and v.size > 0 and int(np.abs(v).max()) <= 1


def save(path: str, tree, pack: bool = True, extra: dict | None = None) -> str:
    """Save a pytree by leaf path.  Ternary leaves are bit-packed when ``pack``.

    The write is atomic (tmp file + rename): a preemption mid-save must leave
    either the previous checkpoint or the new one, never a truncated file that
    ``--resume auto`` would then try to load.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    leaves = jax.tree_util.tree_leaves(tree)
    keys = _keys(tree)
    arrays, meta = {}, {}
    for k, v in zip(keys, leaves):
        v = np.asarray(v)
        if pack and _is_ternary(v):
            arrays[k] = np.asarray(pack2(jnp.asarray(v)))
            meta[k] = {"shape": list(v.shape), "dtype": "int8", "packed": True}
        else:
            arrays[k] = v
            meta[k] = {"shape": list(v.shape), "dtype": str(v.dtype), "packed": False}
    payload = {"__meta__": json.dumps({"leaves": meta, "extra": extra or {}})}
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **payload, **arrays)
    os.replace(tmp, path)
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


# -- resumable training state ------------------------------------------
#
# A params-only checkpoint is enough to sample from but NOT to resume: the
# optimizer state carries Muon/Adam moments, the sign optimizer's momentum,
# its PRNG key, and the step counters that drive every learning-rate schedule.
# Dropping it would silently restart all schedules from warmup.


def save_train_state(path: str, params, opt_state, meta: dict, pack: bool = True) -> str:
    return save(path, {"params": params, "opt": opt_state}, pack=pack, extra=meta)


def load_train_state(path: str, params_template, opt_state_template):
    """Return ``(params, opt_state, meta)`` restored from a full checkpoint."""
    tree = load(path, {"params": params_template, "opt": opt_state_template})
    return tree["params"], tree["opt"], read_extra(path)


def latest_checkpoint(run_dir: str, prefix: str = "ckpt_") -> str | None:
    """Newest step-numbered checkpoint in a run directory, or None."""
    found = sorted(glob.glob(os.path.join(run_dir, f"{prefix}*.npz")))
    return found[-1] if found else None


def rotate_checkpoints(run_dir: str, keep_last: int, prefix: str = "ckpt_") -> list[str]:
    """Delete all but the newest ``keep_last`` checkpoints.  Returns removed paths."""
    if keep_last <= 0:
        return []
    found = sorted(glob.glob(os.path.join(run_dir, f"{prefix}*.npz")))
    removed = []
    for old in found[:-keep_last]:
        os.remove(old)
        removed.append(old)
    return removed
