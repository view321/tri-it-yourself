#!/usr/bin/env bash
# End-to-end CPU check: tests, then a short run in each of the three modes.
# Needs no GPU and no downloaded data; finishes in a couple of minutes.
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"

echo "== unit tests =="
$PY -m pytest -q

echo
echo "== short training runs (synthetic induction task) =="
for quant in bf16 ste sign; do
  $PY -m tri.train --preset smoke --quant "$quant" \
      --run-name "smoke-$quant" --out-dir runs/smoke --steps 400
done

echo
echo "== sampling from the ternary checkpoint =="
$PY -m tri.sample runs/smoke/smoke-sign --n-new 16

echo
echo "== checkpoint sizes (ternary weights are packed 4-per-byte) =="
ls -la runs/smoke/*/final.npz
