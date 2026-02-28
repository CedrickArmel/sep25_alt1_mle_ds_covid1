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

import os
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig


def create_symlink(label: str, src: str, dst: Path):
    dst_dir = Path(dst) / str(label)
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_path = Path(src)
    dst_path = dst_dir / src_path.name
    if os.path.lexists(dst_path):
        raise FileExistsError(f"{dst_path} already exists")
    dst_path.symlink_to(src_path)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def make_image_folder(cfg: DictConfig):
    manifest = pd.read_parquet(cfg.symlink.manifest_path)
    if not cfg.symlink.keep_origin_classes:
        manifest["class"] = manifest["class"].map(cfg.symlink.classes)
    manifest = manifest.dropna(subset=["class", "image"])
    for row in manifest[["class", "image"]].itertuples(index=False):
        label, image_path = row
        create_symlink(label=label, src=image_path, dst=cfg.symlink.dst_dir)


if __name__ == "__main__":
    make_image_folder()
