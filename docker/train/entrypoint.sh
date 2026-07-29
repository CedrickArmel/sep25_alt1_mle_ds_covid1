#!/bin/bash
set -euo pipefail

: "${DVC_PULL:=0}"
: "${DATA_VERSION:=data-latest}"
: "${DVC_ROOT:=/dvc-workspace}"
: "${DVC_PULL_FORCE:=0}"

configure_dvc_gdrive() {
  if [[ -z "${GDRIVE_CLIENT_ID:-}" || -z "${GDRIVE_CLIENT_SECRET:-}" ]]; then
    echo "ERROR: set GDRIVE_CLIENT_ID and GDRIVE_CLIENT_SECRET in .env"
    exit 1
  fi
  mkdir -p "${DVC_ROOT}/.dvc/tmp"
  cp /dvc-config-ro "${DVC_ROOT}/.dvc/config"
  cp /gdrive-user-credentials.json \
    "${DVC_ROOT}/.dvc/tmp/gdrive-user-credentials.json"
  dvc remote modify --local data gdrive_client_id "${GDRIVE_CLIENT_ID}"
  dvc remote modify --local data gdrive_client_secret "${GDRIVE_CLIENT_SECRET}"
  dvc remote modify --local data gdrive_user_credentials_file \
    "${DVC_ROOT}/.dvc/tmp/gdrive-user-credentials.json"
}

if [[ "${DVC_PULL}" == "1" ]]; then
  echo "DVC — pull dataset (${DATA_VERSION})"
  mkdir -p "${DVC_ROOT}"
  ln -sfn /workspace/data "${DVC_ROOT}/data"
  cd "${DVC_ROOT}"
  git config --global --add safe.directory "${DVC_ROOT}"
  configure_dvc_gdrive
  git show "${DATA_VERSION}:data.dvc" > data.dvc
  if [[ "${DVC_PULL_FORCE}" == "1" ]]; then
    dvc pull --force
  else
    dvc pull
  fi
  cd /workspace
fi

exec radiocovid-train "$@"
