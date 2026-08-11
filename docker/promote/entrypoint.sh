#!/bin/bash
set -euo pipefail

# Default: dry-run (compare candidate vs production without applying change).
# Set PROMOTE_APPLY=1 in .env to actually move the @production alias.
ARGS=""
if [ "${PROMOTE_APPLY:-0}" = "1" ]; then
    ARGS="--promote"
    echo "[promote] mode: APPLY — will update @${WANDB_REGISTRY_ALIAS:-production} alias if candidate is better"
else
    echo "[promote] mode: DRY-RUN — set PROMOTE_APPLY=1 in .env to apply"
fi

exec python /workspace/scripts/register_model.py ${ARGS}
