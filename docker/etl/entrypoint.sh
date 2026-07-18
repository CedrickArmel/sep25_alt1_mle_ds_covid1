#!/bin/bash
set -e

# Valeurs par défaut — surchargées via variables d'environnement dans docker-compose
: "${DATA_DIR:=/data/01_raw/COVID-19_Radiography_Dataset}"
: "${MANIFEST_PATH:=/data/manifest.parquet}"
: "${TRAIN_FOLDER_DIR:=/data/train_folder}"

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
