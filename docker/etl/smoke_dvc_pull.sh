#!/bin/bash
set -euo pipefail

echo "=== 1) Env check ==="
if [ -z "${GDRIVE_CLIENT_ID:-}" ] || [ -z "${GDRIVE_CLIENT_SECRET:-}" ]; then
  echo "FAIL: GDRIVE credentials missing"
  exit 1
fi
echo "GDRIVE credentials: OK"
echo "DATA_VERSION=${DATA_VERSION:-data-v1.0}"
echo "DVC_PULL=${DVC_PULL:-}"

test -f /dvc-config-ro && echo "dvc config mount: OK" || { echo "FAIL: missing /dvc-config-ro"; exit 1; }
test -f /gdrive-user-credentials.json && echo "gdrive token mount: OK" || { echo "FAIL: missing gdrive token"; exit 1; }
test -d /data && echo "data mount: OK" || { echo "FAIL: missing /data"; exit 1; }

echo "=== 2) Prepare DVC workspace ==="
export DVC_ROOT=/dvc-workspace
export DATA_VERSION=${DATA_VERSION:-data-v1.0}
mkdir -p "$DVC_ROOT"
ln -sfn /data "$DVC_ROOT/data"
cd "$DVC_ROOT"
git config --global --add safe.directory "$DVC_ROOT"

echo "=== 3) Configure GDrive ==="
mkdir -p .dvc/tmp
cp /dvc-config-ro .dvc/config
cp /gdrive-user-credentials.json .dvc/tmp/gdrive-user-credentials.json
dvc remote modify --local data gdrive_client_id "$GDRIVE_CLIENT_ID"
dvc remote modify --local data gdrive_client_secret "$GDRIVE_CLIENT_SECRET"
dvc remote modify --local data gdrive_user_credentials_file .dvc/tmp/gdrive-user-credentials.json

echo "=== 4) Resolve data.dvc from tag ==="
git show "${DATA_VERSION}:data.dvc" > data.dvc
grep -E "md5:|path:|nfiles:|size:" data.dvc || true

echo "=== 5) dvc pull ==="
dvc pull
echo "=== SUCCESS: dvc pull finished ==="
