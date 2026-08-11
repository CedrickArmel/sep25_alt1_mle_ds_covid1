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

"""One-shot GDrive OAuth flow — saves token to .dvc/tmp/gdrive-user-credentials.json.

Run this ONCE from the repo root to (re)create the credentials file:
    python scripts/gdrive_auth.py

After success, the Airflow ingest task can reuse the saved token.
"""

import configparser
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CREDS_FILE = REPO_ROOT / ".dvc" / "tmp" / "gdrive-user-credentials.json"

cfg = configparser.ConfigParser()
cfg.read(REPO_ROOT / ".dvc" / "config.local")

client_id, client_secret = "", ""
for section in cfg.sections():
    client_id = client_id or cfg[section].get("gdrive_client_id", "")
    client_secret = client_secret or cfg[section].get("gdrive_client_secret", "")

# Fallback to env vars
client_id = client_id or os.environ.get("GDRIVE_CLIENT_ID", "")
client_secret = client_secret or os.environ.get("GDRIVE_CLIENT_SECRET", "")

if not client_id or not client_secret:
    sys.exit(
        "ERROR: GDRIVE_CLIENT_ID / GDRIVE_CLIENT_SECRET not found in .dvc/config.local or env."
    )

try:
    from pydrive2.auth import GoogleAuth
except ImportError:
    sys.exit("pydrive2 not installed. Run: pip install pydrive2")

CREDS_FILE.parent.mkdir(parents=True, exist_ok=True)

settings = {
    "client_config_backend": "settings",
    "client_config": {
        "client_id": client_id,
        "client_secret": client_secret,
    },
    "save_credentials": True,
    "save_credentials_backend": "file",
    "save_credentials_file": str(CREDS_FILE),
    "get_refresh_token": True,
    "oauth_scope": ["https://www.googleapis.com/auth/drive"],
}

print(f"client_id : {client_id}")
print(f"token file: {CREDS_FILE}")
print()
print("Opening browser for Google authentication...")
print("(A browser window should open. Authenticate and allow access.)")
print()

gauth = GoogleAuth(settings=settings)
gauth.LocalWebserverAuth()  # opens browser, waits for callback on localhost

print()
print(f"Token saved to {CREDS_FILE}")
print("You can now trigger the Airflow DAG — the ingest task will use this token.")
