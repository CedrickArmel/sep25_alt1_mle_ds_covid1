#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2025 @CedrickArmel, @samarita22, @TaxelleT & @Yeyecodes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Ingest images from incoming/ and publish a new DVC dataset version.

Supports two sources:
  - local  : drop files under incoming/<class>/images/ and /masks/
  - gdrive : sync from Drive incoming_images/ (set INCOMING_GDRIVE_FOLDER_ID)

See incoming/README.md.
"""

from __future__ import annotations

import argparse
import configparser
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INCOMING_DIR = REPO_ROOT / "incoming"
RAW_ROOT = REPO_ROOT / "data" / "01_raw" / "COVID-19_Radiography_Dataset"
CLASSES = ("COVID", "Lung_Opacity", "Normal", "Viral Pneumonia")
SUBFOLDERS = ("images", "masks")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
FLOATING_TAG = "data-latest"
VERSION_TAG_RE = re.compile(r"^data-v(\d+)\.(\d+)$")

# GDrive MIME type for folders
_GDRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def run(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        check=check,
        text=True,
        capture_output=False,
    )


def _read_dvc_config_local(key: str) -> str:
    """Read a key from .dvc/config.local (fallback when env var is not set)."""
    cfg_path = REPO_ROOT / ".dvc" / "config.local"
    if not cfg_path.exists():
        return ""
    parser = configparser.ConfigParser()
    parser.read(cfg_path)
    for section in parser.sections():
        if key in parser[section]:
            return parser[section][key]
    return ""


def _default_credentials_file() -> str:
    env = os.environ.get("DVC_GDRIVE_CREDENTIALS_PATH", "")
    if env:
        return env
    return str(REPO_ROOT / ".dvc" / "tmp" / "gdrive-user-credentials.json")


# ---------------------------------------------------------------------------
# GDrive sync (INFRA-04a)
# ---------------------------------------------------------------------------


def _gdrive_auth():
    """Authenticate with GDrive reusing the DVC OAuth token.

    Requires pydrive2 (bundled with dvc-gdrive, already installed).
    Falls back gracefully if credentials file is missing.
    """
    try:
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
    except ImportError:
        raise SystemExit(
            "pydrive2 is required but not installed.\n"
            "It is bundled with dvc-gdrive: `pip install dvc-gdrive`"
        )

    client_id = os.environ.get("GDRIVE_CLIENT_ID") or _read_dvc_config_local(
        "gdrive_client_id"
    )
    client_secret = os.environ.get("GDRIVE_CLIENT_SECRET") or _read_dvc_config_local(
        "gdrive_client_secret"
    )
    if not client_id or not client_secret:
        raise SystemExit(
            "GDRIVE_CLIENT_ID / GDRIVE_CLIENT_SECRET not set.\n"
            "Set them in .env or .dvc/config.local (same values used for dvc pull)."
        )

    creds_file = _default_credentials_file()
    if not Path(creds_file).exists():
        raise SystemExit(
            f"GDrive token not found: {creds_file}\n"
            "Run `dvc pull` once interactively (browser login) to create this file,\n"
            "then retry."
        )

    settings = {
        "client_config_backend": "settings",
        "client_config": {
            "client_id": client_id,
            "client_secret": client_secret,
        },
        "save_credentials": True,
        "save_credentials_backend": "file",
        "save_credentials_file": creds_file,
        "get_refresh_token": True,
        "oauth_scope": ["https://www.googleapis.com/auth/drive"],
    }

    gauth = GoogleAuth(settings=settings)
    gauth.LoadCredentialsFile(creds_file)

    if gauth.credentials is None:
        raise SystemExit(
            f"Could not load credentials from {creds_file}.\n"
            "Run `dvc pull` once interactively to refresh the token."
        )
    if gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()

    return GoogleDrive(gauth)


def _list_folder(drive, folder_id: str) -> list:
    """List direct children of a Drive folder (non-trashed)."""
    return drive.ListFile(
        {"q": f"'{folder_id}' in parents and trashed=false", "orderBy": "title"}
    ).GetList()


def sync_from_gdrive(folder_id: str, *, dry_run: bool = False) -> int:
    """Download new images from Drive incoming_images/ to local incoming/.

    Expected Drive layout:
        incoming_images/
          COVID/
            images/   ← *.png
            masks/    ← *.png
          Normal/
            images/
            masks/
          ...

    Files already present locally are skipped.
    Returns the number of files downloaded (or that would be downloaded).
    """
    print(f"[gdrive] authenticating (token: {_default_credentials_file()}) …")
    drive = _gdrive_auth()
    print(f"[gdrive] listing folder {folder_id} …")

    total = 0
    class_items = _list_folder(drive, folder_id)

    if not class_items:
        print("[gdrive] incoming_images/ folder is empty — nothing to download.")
        return 0

    for class_item in class_items:
        class_name = class_item["title"]
        if class_item["mimeType"] != _GDRIVE_FOLDER_MIME:
            print(f"  skip non-folder item: {class_name}")
            continue
        if class_name not in CLASSES:
            print(f"  skip unknown class folder: {class_name!r}")
            continue

        sub_items = _list_folder(drive, class_item["id"])
        for sub_item in sub_items:
            sub_name = sub_item["title"]
            if sub_item["mimeType"] != _GDRIVE_FOLDER_MIME:
                continue
            if sub_name not in SUBFOLDERS:
                print(f"  skip unknown subfolder: {class_name}/{sub_name!r}")
                continue

            file_items = _list_folder(drive, sub_item["id"])
            for file_item in file_items:
                if file_item["mimeType"] == _GDRIVE_FOLDER_MIME:
                    continue
                ext = Path(file_item["title"]).suffix.lower()
                if ext not in IMAGE_EXTS:
                    continue

                dest_dir = INCOMING_DIR / class_name / sub_name
                dest = dest_dir / file_item["title"]

                label = f"{class_name}/{sub_name}/{file_item['title']}"
                if dry_run:
                    print(f"  [dry-run] would download: {label}")
                    total += 1
                    continue

                if dest.exists():
                    print(f"  skip (exists locally): {label}")
                    continue

                dest_dir.mkdir(parents=True, exist_ok=True)
                print(f"  downloading: {label}")
                file_item.GetContentFile(str(dest))
                total += 1

    return total


# ---------------------------------------------------------------------------
# local helpers
# ---------------------------------------------------------------------------


def list_incoming_images() -> list[tuple[str, Path]]:
    """List all images under incoming/<class>/images/ and /masks/."""
    found: list[tuple[str, Path]] = []
    for class_name in CLASSES:
        class_dir = INCOMING_DIR / class_name
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                found.append((class_name, path))
    return found


def next_version_tag() -> str:
    result = subprocess.run(
        ["git", "tag", "-l", "data-v*"],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    found_any = False
    major, minor = 1, 0
    for line in result.stdout.splitlines():
        match = VERSION_TAG_RE.match(line.strip())
        if not match:
            continue
        found_any = True
        mjr, mnr = int(match.group(1)), int(match.group(2))
        if (mjr, mnr) >= (major, minor):
            major, minor = mjr, mnr
    if not found_any:
        return "data-v1.0"
    return f"data-v{major}.{minor + 1}"


def copy_into_raw(images: list[tuple[str, Path]]) -> list[tuple[str, Path, Path]]:
    """Copy images into data/01_raw/, preserving images/ and masks/ subfolders."""
    copied: list[tuple[str, Path, Path]] = []
    for class_name, src in images:
        class_incoming_dir = INCOMING_DIR / class_name
        # preserve sub-path (images/foo.png or masks/foo.png)
        rel = src.relative_to(class_incoming_dir)
        dest_dir = RAW_ROOT / class_name / rel.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name
        if dest.exists():
            stem, suffix = src.stem, src.suffix
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            dest = dest_dir / f"{stem}__{stamp}{suffix}"
        shutil.copy2(src, dest)
        print(f"copied {src.relative_to(REPO_ROOT)} -> {dest.relative_to(REPO_ROOT)}")
        copied.append((class_name, src, dest))
    return copied


def archive_incoming(images: list[tuple[str, Path]]) -> Path:
    """Move processed files to incoming/_processed/<timestamp>/."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_root = INCOMING_DIR / "_processed" / stamp
    for class_name, src in images:
        class_incoming_dir = INCOMING_DIR / class_name
        rel = src.relative_to(class_incoming_dir)
        dest_dir = archive_root / class_name / rel.parent
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest_dir / src.name))
    return archive_root


def publish_version(tag: str, *, skip_push: bool) -> None:
    run(["dvc", "add", "data/"])
    if not skip_push:
        run(["dvc", "push"])
    else:
        print("skipping dvc push (--skip-push)")

    run(["git", "add", "data.dvc"])
    status = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=REPO_ROOT,
    )
    if status.returncode == 0:
        raise SystemExit(
            "data.dvc did not change after dvc add — nothing to version.\n"
            "Are the new files inside data/01_raw/?."
        )

    run(["git", "commit", "--no-verify", "-m", f"chore: bump dataset to {tag}"])
    run(["git", "tag", "-a", tag, "-m", f"Dataset version {tag}"])
    run(["git", "tag", "-f", FLOATING_TAG, tag])

    if not skip_push:
        run(["git", "push", "origin", "HEAD", tag])
        run(["git", "push", "--force", "origin", FLOATING_TAG])
    else:
        print("skipping git push (--skip-push); tags exist only locally")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=os.environ.get("INCOMING_SOURCE", "local"),
        choices=("local", "gdrive"),
        help="Where new images come from (default: env INCOMING_SOURCE or local)",
    )
    parser.add_argument(
        "--gdrive-folder-id",
        default=os.environ.get("INCOMING_GDRIVE_FOLDER_ID", ""),
        help="Drive folder id for incoming_images/ (required when --source=gdrive)",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional explicit tag (default: auto-increment data-vX.Y)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List / preview actions without writing anything",
    )
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Skip dvc push / git push (local versioning only)",
    )
    args = parser.parse_args()

    # -- 1. sync from Drive if requested ------------------------------------
    if args.source == "gdrive":
        if not args.gdrive_folder_id:
            raise SystemExit(
                "INCOMING_GDRIVE_FOLDER_ID / --gdrive-folder-id is required "
                "for --source=gdrive"
            )
        n = sync_from_gdrive(args.gdrive_folder_id, dry_run=args.dry_run)
        if args.dry_run:
            print(f"[dry-run] {n} file(s) would be downloaded from Drive.")
            return 0
        if n == 0:
            print("No new files downloaded from Drive — nothing to version.")
            return 0

    # -- 2. collect local incoming files ------------------------------------
    images = list_incoming_images()
    if not images:
        print(f"No new images found under {INCOMING_DIR}/<class>/")
        print("Nothing to do.")
        return 0

    print(f"Found {len(images)} image(s) to ingest:")
    for class_name, path in images:
        print(f"  - [{class_name}] {path.relative_to(REPO_ROOT)}")

    if not RAW_ROOT.exists() and not args.dry_run:
        raise SystemExit(
            f"Missing raw dataset root: {RAW_ROOT}\n"
            "Run `dvc pull` first so data/01_raw/... exists locally."
        )

    tag = args.tag or next_version_tag()
    print(f"Target dataset tag: {tag} (+ {FLOATING_TAG})")

    if args.dry_run:
        print("[dry-run] no files copied, no dvc/git writes")
        return 0

    # -- 3. merge + version -------------------------------------------------
    copy_into_raw(images)
    publish_version(tag, skip_push=args.skip_push)
    archive = archive_incoming(images)
    print(f"Archived incoming files under {archive.relative_to(REPO_ROOT)}")
    print(f"Done. ETL can use DATA_VERSION={FLOATING_TAG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
