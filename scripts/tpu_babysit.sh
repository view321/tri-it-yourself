#!/usr/bin/env bash
# Runs OUTSIDE the TPU (Cloud Shell, a Linux box, anything with gcloud that
# stays up): recreate the spot TPU whenever it is preempted and re-run the
# bootstrap, which resumes training from the newest checkpoint in GCS.
#
# Windows note: `gcloud ... ssh` can hang on PuTTY prompts; run this from
# Cloud Shell or a Linux machine instead.
set -uo pipefail
cd "$(dirname "$0")"
. ./tpu_env.sh

while true; do
  STATE=$(gcloud compute tpus tpu-vm describe "$TPU_NAME" \
      --zone="$TPU_ZONE" --project="$TPU_PROJECT" \
      --format="value(state)" 2>/dev/null || echo "MISSING")
  case "$STATE" in
    READY)
      ;;
    CREATING|RESTARTING|STARTING|REPAIRING)
      echo "$(date -u +%H:%M:%S) state=$STATE; waiting"
      ;;
    *)
      echo "$(date -u +%H:%M:%S) state=$STATE; recreating"
      gcloud compute tpus tpu-vm delete "$TPU_NAME" \
          --zone="$TPU_ZONE" --project="$TPU_PROJECT" --quiet 2>/dev/null || true
      if bash ./tpu_create_spot.sh; then
        sleep 30
        gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
            --zone="$TPU_ZONE" --project="$TPU_PROJECT" --quiet \
            --command="git clone '$REPO_URL' ~/tri-it-yourself 2>/dev/null; \
GCS_BUCKET='$GCS_BUCKET' REPO_URL='$REPO_URL' RUN_NAME='$RUN_NAME' TRAIN_ARGS='$TRAIN_ARGS' \
bash ~/tri-it-yourself/scripts/tpu_bootstrap.sh" \
          && echo "$(date -u +%H:%M:%S) bootstrap ok" \
          || echo "$(date -u +%H:%M:%S) bootstrap failed; will retry next cycle"
      else
        echo "$(date -u +%H:%M:%S) create failed (likely no spot capacity); retrying later"
      fi
      ;;
  esac
  sleep 120
done
