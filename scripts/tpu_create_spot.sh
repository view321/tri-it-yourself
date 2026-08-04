#!/usr/bin/env bash
# Create the spot TPU VM (idempotent: exits quietly if it already exists).
# Spot capacity comes and goes; if creation fails with a capacity error the
# babysitter just retries on its next cycle.
set -euo pipefail
cd "$(dirname "$0")"
. ./tpu_env.sh

if gcloud compute tpus tpu-vm describe "$TPU_NAME" \
    --zone="$TPU_ZONE" --project="$TPU_PROJECT" >/dev/null 2>&1; then
  echo "$TPU_NAME already exists in $TPU_ZONE"
  exit 0
fi

gcloud compute tpus tpu-vm create "$TPU_NAME" \
  --zone="$TPU_ZONE" --project="$TPU_PROJECT" \
  --accelerator-type="$TPU_TYPE" --version="$TPU_VERSION" \
  --spot
echo "created $TPU_NAME ($TPU_TYPE, spot) in $TPU_ZONE"
