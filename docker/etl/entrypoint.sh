#!/bin/bash
set -euo pipefail

# Defaults — override via docker-compose environment
: "${DATA_DIR:=/data/01_raw/COVID-19_Radiography_Dataset}"
: "${MANIFEST_PATH:=/data/manifest.parquet}"
: "${TRAIN_FOLDER_DIR:=/data/train_folder}"
: "${DVC_PULL:=1}"
: "${DATA_VERSION:=data-latest}"
: "${DVC_ROOT:=/dvc-workspace}"
# ETL regenerates manifest/train_folder after pull — default force avoids
# blocking on local untracked files under data/ (e.g. manifest.parquet).
: "${DVC_PULL_FORCE:=1}"

configure_dvc_gdrive() {
  if [[ -z "${GDRIVE_CLIENT_ID:-}" || -z "${GDRIVE_CLIENT_SECRET:-}" ]]; then
    echo "ERROR: set GDRIVE_CLIENT_ID and GDRIVE_CLIENT_SECRET in .env"
    exit 1
  fi
  mkdir -p "${DVC_ROOT}/.dvc/tmp"
  cp /dvc-config-ro "${DVC_ROOT}/.dvc/config"
  # Token is mounted read-only; DVC may refresh it → copy to a writable path
  cp /gdrive-user-credentials.json \
    "${DVC_ROOT}/.dvc/tmp/gdrive-user-credentials.json"
  dvc remote modify --local data gdrive_client_id "${GDRIVE_CLIENT_ID}"
  dvc remote modify --local data gdrive_client_secret "${GDRIVE_CLIENT_SECRET}"
  dvc remote modify --local data gdrive_user_credentials_file \
    "${DVC_ROOT}/.dvc/tmp/gdrive-user-credentials.json"
}

pull_data() {
  if [[ "${DVC_PULL}" != "1" ]]; then
    echo "DVC_PULL=${DVC_PULL} — skipping dvc pull (using mounted ./data as-is)"
    return 0
  fi

  echo "========================================="
  echo " DVC — pull dataset (${DATA_VERSION})"
  echo "========================================="

  mkdir -p "${DVC_ROOT}"
  # data.dvc expects a relative path "data/" — link the compose mount here
  ln -sfn /data "${DVC_ROOT}/data"
  cd "${DVC_ROOT}"
  git config --global --add safe.directory "${DVC_ROOT}"

  configure_dvc_gdrive

  if ! git show "${DATA_VERSION}:data.dvc" > data.dvc; then
    echo "ERROR: cannot read data.dvc from ${DATA_VERSION}"
    echo "Check that .git is mounted and the tag exists (e.g. data-v1.0)."
    exit 1
  fi

  if [[ "${DVC_PULL_FORCE}" == "1" ]]; then
    echo "dvc pull --force (DVC_PULL_FORCE=1)"
    dvc pull --force
  else
    dvc pull
  fi
}

pull_data

echo "========================================="
echo " ETL — Étape 1 : nettoyage des images"
echo "========================================="
radiocovid-clean \
    data_dir="${DATA_DIR}" \
    output="${MANIFEST_PATH}"

echo "========================================="
echo " ETL — Étape 2 : construction du train folder"
echo "========================================="
rm -rf "${TRAIN_FOLDER_DIR}"
radiocovid-train-folder \
    manifest_path="${MANIFEST_PATH}" \
    dst_dir="${TRAIN_FOLDER_DIR}"

echo "========================================="
echo " ETL terminé avec succès"
echo "========================================="
