#!/bin/bash

# --- CONFIGURATION ---
POD_IP="$1"
POD_PORT="$2"
LOCAL_DIR=~/gpu/gemm/
REMOTE_DIR="root@${POD_IP}:/root/gemm/"
KEY_PATH="~/.ssh/id_ed25519"
# ---------------------

# Define the RSYNC command with specific includes/excludes
# Order matters! Excludes for directories come FIRST, then includes, then final exclude

RSYNC_CMD="rsync -avz \
--exclude='.venv/' \
--exclude='__pycache__/' \
--exclude='.git/' \
--exclude='*.egg-info/' \
--exclude='build/' \
--exclude='dist/' \
--exclude='.pytest_cache/' \
--exclude='wandb/' \
--exclude='experiments/' \
--include='*/' \
--include='*.c' \
--include='*.yaml' \
--include='*.sh' \
--include='*.toml' \
--include='*.cpp' \
--include='*.cu' \
--include='*.cuh' \
--include='*.h' \
--include='*.py' \
--include='*.ini' \
--exclude='*' \
-e \"ssh -p ${POD_PORT} -i ${KEY_PATH}\" ${LOCAL_DIR} ${REMOTE_DIR}"

echo "🚀 Starting filtered sync to ${POD_IP}:${POD_PORT}..."

# 1. Run initial sync
eval $RSYNC_CMD

echo "👀 Watching for changes (Only .c, .h, .py, .ini)..."

# 2. Watcher loop
fswatch -o ${LOCAL_DIR} | xargs -n1 -I{} bash -c "$RSYNC_CMD"
