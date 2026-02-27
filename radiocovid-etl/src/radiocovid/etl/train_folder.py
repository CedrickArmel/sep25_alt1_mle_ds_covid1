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
