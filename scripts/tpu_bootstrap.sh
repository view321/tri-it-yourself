#!/usr/bin/env bash
# Runs ON the TPU VM (freshly created or recreated after preemption): install,
# pull data and checkpoints down from GCS, and start/attach the training loop
# in tmux.  Safe to run repeatedly.
set -euo pipefail
. "$(dirname "$0")/tpu_env.sh" 2>/dev/null || true

REPO_DIR="$HOME/tri-it-yourself"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone "${REPO_URL:?set REPO_URL}" "$REPO_DIR"
fi
cd "$REPO_DIR"
git pull --ff-only

if ! python -c "import jax" 2>/dev/null || ! python -c "import tri" 2>/dev/null; then
  pip install -q -U "jax[tpu]"
  pip install -q -e ".[data]"
fi
python -c "import jax; assert jax.devices()[0].platform == 'tpu', jax.devices()"

# Tokens + tokenizer down (no-op when already present and unchanged).
mkdir -p data runs
gsutil -m rsync -r "${GCS_BUCKET:?set GCS_BUCKET}/data/" data/
# Checkpoints down so --resume auto finds the newest surviving state.
gsutil -m rsync -r "$GCS_BUCKET/runs/" runs/ || true

if tmux has-session -t tri 2>/dev/null; then
  echo "training session already running; attach with: tmux attach -t tri"
else
  tmux new-session -d -s tri \
    "GCS_BUCKET='$GCS_BUCKET' RUN_NAME='${RUN_NAME:-reason-ef}' TRAIN_ARGS='${TRAIN_ARGS:-}' bash scripts/tpu_train_loop.sh 2>&1 | tee -a train_console.log"
  echo "started tmux session 'tri' (attach with: tmux attach -t tri)"
fi
