# MIT License
#
# Copyright (c) 2026 @CedrickArmel, @TaxelleT, @Yeyecodes & @samarita22
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
from glob import glob
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from .preprocessings import remove_outliers


@hydra.main(version_base="1.3", config_path=".", config_name="clean")
def clean(cfg: DictConfig):
    data_dir = Path(cfg.data_dir)
    folders = cfg.folders
    records = []
    for rep in folders:
        masks = glob(os.path.join(data_dir / rep / "masks/*.png"))
        imgs = glob(os.path.join(data_dir / rep / "masks/*.png"))
        images = list(zip(imgs, masks))
        angles = (np.pi / 4) * np.arange(0, 4)
        dmax = list(range(1, cfg.clean.dmax))
        valid_images = remove_outliers(
            images=images,
            glcm_features=cfg.clean.features,
            glcm_angles=angles,
            glcm_distances=dmax,
            n_jobs=cfg.n_jobs,
            resize=cfg.resize,
            verbose=cfg.verbose,
        )
        records.extend(
            [
                {"class": Path(img).stem, "image": img, "mask": msk}
                for img, msk in valid_images
            ]
        )
    pd.DataFrame(records).to_parquet(cfg.output, index=False)


def create_symlink(label: str, src: str, dst: Path):
    dst_dir = Path(dst) / label
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_path = Path(src)
    dst_path = dst_dir / src_path.name
    if os.path.lexists(dst_path):
        raise FileExistsError(f"{dst_path} already exists")
    dst_path.symlink_to(src_path)


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def make_image_folder(cfg: DictConfig):
    manifest = pd.read_parquet(cfg.symlink.manifest_path)

    if not cfg.keep_origin_classes:
        manifest["class"] = manifest["class"].map(cfg.symlink.classes)

    manifest = manifest.dropna(subset=["class", "image"])

    for row in manifest[["class", "image"]].itertuples(index=False):
        label, image_path = row
        create_symlink(label=label, src=image_path, dst=cfg.symlink.dst_dir)
