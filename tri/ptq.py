"""Post-training quantization of a float checkpoint into the ternary layout.

This exists to make the project's central claim falsifiable.  Comparing a
`bf16` run against a `sign` run at matched tokens answers "does latent-free
ternary training reach the same loss as float training", which is interesting
but not the question anyone deploying a model asks.  A float model that has to
fit in 2 bits per weight does not get deployed in float - it gets quantized,
and it loses something in the process.

So the comparison that matters is between things that occupy the same memory
at inference:

  bf16, no quantization    reference ceiling, but 16-32 bits/weight
  bf16 -> PTQ ternary      the naive low-memory baseline (this module)
  ste                      QAT: latent float weights, ternary at inference
  sign                     latent-free ternary

The claim "training natively in ternary is worth it" needs `sign` to beat the
PTQ number, and to be competitive with `ste` while using far less training
state (18 bits/weight against 64).  Beating an unquantized bf16 model was never
the goal and would be a surprising thing to expect.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from . import ckpt as ckpt_io
from .config import ModelConfig, TrainConfig
from .data import build_data
from .model import init_params
from .quant import ternarize


def load_run(run_dir: str, ckpt: str = "final.npz") -> tuple[ModelConfig, dict]:
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = ModelConfig(**json.load(f)["model"])
    template = init_params(jax.random.PRNGKey(0), cfg)
    return cfg, ckpt_io.load(os.path.join(run_dir, ckpt), template)


def to_ternary(params: dict, cfg: ModelConfig) -> tuple[dict, ModelConfig]:
    """Quantize every block linear with BitNet's absmean rule.

    Embeddings, the tied head and the norm gains stay in float, matching how the
    `sign` and `ste` arms are trained - only the block linears are ternary in
    any of the three, so quantizing more here would compare different things.
    """
    if cfg.quant != "bf16":
        raise ValueError(f"expected a float checkpoint, got quant={cfg.quant!r}")

    def convert(node):
        if isinstance(node, dict):
            if "w" in node and jnp.ndim(node["w"]) == 2 and "s" not in node:
                q, scale = ternarize(node["w"], per_row=True)
                return {"w": q.astype(jnp.int8), "s": scale[0].astype(jnp.float32)}
            return {k: convert(v) for k, v in node.items()}
        return node

    return convert(params), dataclasses.replace(cfg, quant="sign")


def sparsity(params: dict) -> float:
    """Fraction of ternary weights sitting at zero."""
    leaves = [x for x in jax.tree_util.tree_leaves(params) if x.dtype == jnp.int8]
    if not leaves:
        return 0.0
    return float(sum(jnp.sum(x == 0) for x in leaves) / sum(x.size for x in leaves))


def evaluate_both(run_dir: str, tc: TrainConfig, ckpt: str = "final.npz") -> dict:
    """Val CE of a float checkpoint before and after ternarization."""
    from .train import evaluate, make_eval_step

    cfg, params = load_run(run_dir, ckpt)
    _, val_src = build_data(tc, cfg)

    fp = evaluate(make_eval_step(cfg), params, val_src, tc, cfg)
    qparams, qcfg = to_ternary(params, cfg)
    q = evaluate(make_eval_step(qcfg), qparams, val_src, tc, qcfg)

    return {
        "run_dir": run_dir,
        "fp_val_ce": fp["val_ce"],
        "ptq_val_ce": q["val_ce"],
        "degradation": q["val_ce"] - fp["val_ce"],
        "zero_frac": sparsity(qparams),
        "per_loops_fp": {k: v for k, v in fp.items() if k.startswith("val_ce_L")},
        "per_loops_ptq": {k: v for k, v in q.items() if k.startswith("val_ce_L")},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Post-training-quantize a bf16 run to ternary and evaluate both"
    )
    ap.add_argument("run_dir", help="a run directory trained with --quant bf16")
    ap.add_argument("--ckpt", default="final.npz")
    ap.add_argument("--dataset", default="bin", choices=["synthetic", "induction", "bin"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--uniform", type=float, default=None)
    args = ap.parse_args(argv)

    tc = TrainConfig(dataset=args.dataset, data_dir=args.data_dir,
                     eval_batches=args.eval_batches, batch_size=args.batch_size)
    r = evaluate_both(args.run_dir, tc, args.ckpt)

    print(f"{args.run_dir}")
    print(f"  bf16, unquantized   val_ce {r['fp_val_ce']:.4f}   (reference ceiling)")
    print(f"  bf16 -> PTQ ternary val_ce {r['ptq_val_ce']:.4f}   "
          f"(+{r['degradation']:.4f} from quantizing)")
    print(f"  zeros after PTQ     {r['zero_frac']:.1%}")
    if args.uniform:
        print(f"  uniform baseline    {args.uniform:.4f}")
    print("\n  A `sign` or `ste` run must beat the PTQ number to justify itself;")
    print("  the unquantized bf16 number is a ceiling, not a competitor.")
    return r


if __name__ == "__main__":
    main()
