#!/bin/bash
set -euo pipefail

# Allow git to operate on the mounted repo (container may run as root)
git config --global --add safe.directory /workspace

# Git identity for the automated commit created by publish_version()
git config --global user.name  "${GIT_USER_NAME:-airflow-bot}"
git config --global user.email "${GIT_USER_EMAIL:-airflow@localhost}"

# If a GitHub PAT is provided, configure HTTPS credentials so git push works.
# Without GH_PAT, --skip-push is expected to be set (INGEST_SKIP_PUSH=1).
if [ -n "${GH_PAT:-}" ]; then
    git config --global credential.helper \
        "!f() { echo \"username=x-token\"; echo \"password=${GH_PAT}\"; }; f"
    # Convert SSH remote (git@github.com:…) to HTTPS if needed
    REMOTE_URL=$(git -C /workspace remote get-url origin 2>/dev/null || true)
    if echo "${REMOTE_URL}" | grep -q "git@github.com:"; then
        HTTPS_URL=$(echo "${REMOTE_URL}" | sed 's|git@github.com:|https://github.com/|')
        git -C /workspace remote set-url origin "${HTTPS_URL}"
    fi
fi

# Build argument list for ingest_and_version_data.py
ARGS="--source=${INCOMING_SOURCE:-local}"

[ -n "${INCOMING_GDRIVE_FOLDER_ID:-}" ] && ARGS="${ARGS} --gdrive-folder-id ${INCOMING_GDRIVE_FOLDER_ID}"
[ -n "${INGEST_TAG:-}"                ] && ARGS="${ARGS} --tag ${INGEST_TAG}"

# Default: skip remote push (safe for local dev).
# Set INGEST_SKIP_PUSH=0 + GH_PAT to enable full DVC + git push.
if [ "${INGEST_SKIP_PUSH:-1}" = "1" ]; then
    ARGS="${ARGS} --skip-push"
fi

exec python /workspace/scripts/ingest_and_version_data.py ${ARGS}
