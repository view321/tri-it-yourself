#!/usr/bin/env bash
# Runs ON the TPU VM inside tmux: train with exact resume, mirroring
# checkpoints to GCS every 5 minutes.  Checkpoint writes are atomic
# (tmp+rename), so a sync can only ever copy a complete file, and the GCS
# mirror (-d) tracks local rotation instead of accumulating every checkpoint.
set -uo pipefail
cd "$(dirname "$0")/.."
: "${GCS_BUCKET:?set GCS_BUCKET}"
RUN_NAME="${RUN_NAME:-reason-ef}"

sync_runs() { gsutil -m -q rsync -d -r runs/ "$GCS_BUCKET/runs/"; }

( while true; do sleep 300; sync_runs; done ) &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null' EXIT

while true; do
  python -m tri.train --preset reason --quant sign --sign-rule ef \
      --dataset bin --data-dir data \
      --run-name "$RUN_NAME" --resume auto \
      ${TRAIN_ARGS:-}
  code=$?
  sync_runs
  if [ "$code" -eq 0 ]; then
    echo "training finished; final state mirrored to $GCS_BUCKET/runs/$RUN_NAME"
    break
  fi
  echo "train exited with $code; resuming in 15s"
  sleep 15
done
