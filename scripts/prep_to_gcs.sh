#!/usr/bin/env bash
# Preemption-safe data prep: sharded tri.prepare_data with a continuous GCS
# mirror, resumable by simply re-running this script.  On completion the
# shards are stitched server-side (gsutil compose) into the single train.bin
# the loader expects, and the part objects are removed.
#
# Run it on any VM in the bucket's region - a cheap on-demand e2 CPU VM is
# immune to preemption and costs a few dollars total; a spot TPU VM works too,
# it just needs a re-run after each preemption and pays a stream-skip to get
# back to where it was.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GCS_BUCKET:?set GCS_BUCKET, e.g. gs://my-project-tri}"
MIX="${MIX:-reason}"
MAX_TOKENS="${MAX_TOKENS:-34000000000}"
# 1.25B tokens/part keeps a 34B-token prep under gsutil compose's 32-object cap.
SHARD_TOKENS="${SHARD_TOKENS:-1250000000}"

mkdir -p data
echo "pulling any previous progress from $GCS_BUCKET/data/ ..."
gsutil -m -q rsync -r "$GCS_BUCKET/data/" data/ 2>/dev/null || true

( while true; do sleep 300; gsutil -m -q rsync -x '.*\.(writing|tmp)$' -r data/ "$GCS_BUCKET/data/" || true; done ) &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null' EXIT

python -m tri.prepare_data --mix "$MIX" --out-dir data \
    --max-tokens "$MAX_TOKENS" --shard-tokens "$SHARD_TOKENS" "$@"

kill "$SYNC_PID" 2>/dev/null || true
trap - EXIT
gsutil -m -q rsync -r data/ "$GCS_BUCKET/data/"

parts=$(gsutil ls "$GCS_BUCKET/data/train_part"*.bin 2>/dev/null || true)
if [ -n "$parts" ]; then
  n=$(echo "$parts" | wc -l)
  if [ "$n" -gt 32 ]; then
    echo "$n parts exceed gsutil compose's 32-object limit; raise SHARD_TOKENS" >&2
    exit 1
  fi
  echo "composing $n parts into train.bin ..."
  # shellcheck disable=SC2086  # word-splitting is the point
  gsutil compose $parts "$GCS_BUCKET/data/train.bin"
  gsutil -m -q rm $parts
fi
echo "data ready in $GCS_BUCKET/data"
