import os
from glob import glob
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from .preprocessings import remove_outliers
from .utils import Logger

log = Logger(__name__)


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def clean(cfg: DictConfig):
    data_dir = Path(cfg.data_dir)
    folders = cfg.folders
    records = []
    for rep in folders:
        masks = glob(os.path.join(data_dir / rep / "masks/*.png"))
        imgs = glob(os.path.join(data_dir / rep / "images/*.png"))
        log.info(f"Number of images in {rep} before cleaning : {len(imgs)}")
        images = list(zip(imgs, masks))
        angles = (np.pi / 4) * np.arange(0, 4)
        dmax = list(range(1, cfg.clean.dmax))
        log.info("Removing outliers...")
        valid_images = remove_outliers(
            images=images,
            glcm_features=cfg.clean.features,
            glcm_angles=angles,
            glcm_distances=dmax,
            n_jobs=cfg.clean.n_jobs,
            resize=cfg.clean.resize,
            verbose=cfg.clean.verbose,
        )
        records.extend(
            [
                {"class": Path(img).stem.split("-")[0], "image": img, "mask": msk}
                for img, msk in valid_images
            ]
        )
        if x := (len(imgs) - len(valid_images)) > 0:
            log.info(f"Successfully removed {x} outliers from {rep}!")
        else:
            log.info(f"Cleaning completed successfully. No data removed from {rep}.")
        log.info(f"The remaining size is {len(valid_images)}.")
    log.info("Exporting manifest!")
    pd.DataFrame(records).to_parquet(cfg.clean.output, index=False)
    log.info("Cleaning completed successfully !")


if __name__ == "__main__":
    clean()
