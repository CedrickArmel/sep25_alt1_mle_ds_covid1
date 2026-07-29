# Incoming images (local stand-in for Drive upload)

Drop new X-ray images here before running the versioning job.

## Layout (must match the raw dataset classes)

```text
incoming/
  COVID/
  Lung_Opacity/
  Normal/
  Viral Pneumonia/
```

Supported extensions: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff`

## Workflow (today — local)

1. Put new images into the class folders above.
2. Run:

```shell
make data-ingest
# or: python scripts/ingest_and_version_data.py
```

This will:
- copy files into `data/01_raw/COVID-19_Radiography_Dataset/<class>/`
- run `dvc add` + `dvc push`
- commit `data.dvc`, create an immutable tag `data-vX.Y`, move floating tag `data-latest`
- move processed files to `incoming/_processed/<timestamp>/`

3. ETL / train can then pull the latest version:

```shell
# DATA_VERSION=data-latest (default in docker-compose)
docker compose --profile etl up
```

## Later — Google Drive

When the shared Drive **incoming** folder is available, set in `.env`:

```env
INCOMING_SOURCE=gdrive
INCOMING_GDRIVE_FOLDER_ID=...
```

The same script will sync from Drive into `incoming/` first, then version as above.
Until then, keep `INCOMING_SOURCE=local` (default).
