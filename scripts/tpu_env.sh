#!/usr/bin/env bash
# Shared settings for the spot-TPU pipeline.  Override any of these in the
# environment; the defaults target the cheapest v6e spot region (us-east1 at
# $0.6534/chip-hour vs $1.40 in us-east5 and $1.78 in europe-west4 - check
# the Billing Catalog for current numbers, spot prices move).
export TPU_NAME="${TPU_NAME:-tri-spot}"
export TPU_ZONE="${TPU_ZONE:-us-east1-d}"
export TPU_TYPE="${TPU_TYPE:-v6e-1}"
export TPU_VERSION="${TPU_VERSION:-v2-alpha-tpuv6e}"
export TPU_PROJECT="${TPU_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
# Bucket that holds data/ (tokens + tokenizer) and runs/ (checkpoints).
# Create it once, in the same region as the TPU so transfers stay free:
#   gsutil mb -l us-east1 "$GCS_BUCKET"
export GCS_BUCKET="${GCS_BUCKET:-gs://${TPU_PROJECT}-tri}"
export REPO_URL="${REPO_URL:-https://github.com/view321/tri-it-yourself}"
export RUN_NAME="${RUN_NAME:-reason-ef}"
# Extra flags for tri.train (e.g. tuned knobs from the ef ablation).
export TRAIN_ARGS="${TRAIN_ARGS:-}"
