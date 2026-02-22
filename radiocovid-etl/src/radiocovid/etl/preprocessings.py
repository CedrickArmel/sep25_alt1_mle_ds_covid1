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

from typing import Any

import cv2
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import entropy
from skimage.feature import graycomatrix, graycoprops


def lung_out_of_frame(path: str):
    """
    Détecte si un poumon touche les bords de l'image.

    Args:
        path: chemin vers le masque du poumon

    Returns:
        bool: True si un poumon touche les bords
    """
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask_binary = (mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_binary, connectivity=8
    )

    # Besoin d'au moins 2 poumons + background
    if num_labels < 3:
        return False

    # Récupérer les 2 plus grandes composantes (les poumons)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_indices = np.argsort(areas)[-2:] + 1

    # Vérifier si l'un des poumons touche les bords
    for idx in largest_indices:
        component_mask = labels == idx

        # Vérifier les 4 bords
        if (
            np.any(component_mask[0, :])  # Bord haut
            or np.any(component_mask[-1, :])  # Bord bas
            or np.any(component_mask[:, 0])  # Bord gauche
            or np.any(component_mask[:, -1])  # Bord droit
        ):
            return True

    return False


def compute_asymmetry(path: str):
    """
    Calcule l'asymétrie entre les deux poumons.

    Args:
        path: chemin vers le masque du poumon

    Returns:
        float: Ratio d'asymétrie entre 0 et 1
    """
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    num_labels, labels = cv2.connectedComponents(binary)

    # Calculer l'aire de chaque composante
    areas = {}
    for label in range(1, num_labels):
        areas[label] = np.sum(labels == label)

    # Prendre les 2 plus grandes (les poumons)
    lung_labels = sorted(areas, key=areas.get, reverse=True)[:2]  # type: ignore[arg-type]

    if len(lung_labels) < 2:
        return 1.0  # Asymétrie maximale si < 2 poumons

    area_1 = areas[lung_labels[0]]
    area_2 = areas[lung_labels[1]]

    return abs(area_1 - area_2) / (area_1 + area_2)


def get_valid_indices_iqr(values):
    """
    Filtre les outliers avec la méthode IQR.

    Args:
        values: Array numpy de valeurs

    Returns:
        Array d'indices valides
    """
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return np.where((values >= lower) & (values <= upper))[0]


def prepare_lung_roi(
    image_path: str, mask_path: str, size: tuple[int, int] = (256, 256)
):
    """resize les images en 256*256 et leur appliquer les masques
    Args:
        image_path (str): chemin vers l'image originale
        mask (str): chemin vers le masque binaire
        size (tuple): taille finale. Defaults to (256, 256).
    """

    image_resized = cv2.resize(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), size)
    mask_resized = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), size)

    mask_resized = (mask_resized > 0).astype(np.uint8)

    lung = image_resized * mask_resized

    return lung


def extract_texture_features_glcm(
    lung: np.ndarray, features: list, distances: list[int], angles: np.ndarray
):  # -> None | Any:
    """Extrait les features de texture à partir de la zone poumon uniquement
    Args:
        lung (np.ndarray): image du poumon
        features (list): liste des features de haralick à extraire
        distances (list[int]): distances pour le GLCM
        angles (np.ndarray): angles pour le GLCM

    Returns:
        np.ndarray: vecteur de features
    """

    # On garde uniquement la zone poumon
    pixels = lung[lung > 0]

    if len(pixels) == 0:
        return None

    # GLCM
    glcm = graycomatrix(
        lung,
        distances=distances,
        angles=angles,
        symmetric=True,
        normed=True,
    )

    feature_values = []

    for ft in features:
        props = graycoprops(glcm, prop=ft)
        feature_values.append(props.mean())

    # Mean / std / entropy sur pixels poumons uniquement
    mean_val = np.mean(pixels)
    std_val = np.std(pixels)

    hist, _ = np.histogram(pixels, bins=256, range=(0, 256), density=True)
    entropy_val = entropy(hist + 1e-10)

    feature_values.extend([mean_val, std_val, entropy_val])

    return feature_values


def extract_haralick_features(
    image_path: str,
    mask_path: str,
    features: list,
    distances: list[int],
    angles: np.ndarray,
    resize: tuple[int, int] = (256, 256),
):
    lung = prepare_lung_roi(image_path, mask_path, size=resize)
    return extract_texture_features_glcm(
        lung=lung, features=features, distances=distances, angles=angles
    )


def filter_iqr_multidimensional(feature_matrix):

    valid_indices = np.arange(len(feature_matrix))

    for col in range(feature_matrix.shape[1]):

        values = feature_matrix[:, col]

        Q1 = np.percentile(values, 25)
        Q3 = np.percentile(values, 75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        valid_col = np.where((values >= lower) & (values <= upper))[0]

        valid_indices = np.intersect1d(valid_indices, valid_col)

    return valid_indices


def remove_outliers(
    images: list[tuple[Any, Any]],
    glcm_features: list,
    glcm_distances: list[int],
    glcm_angles: np.array,
    n_jobs: int = -1,
    resize: tuple[int, int] = (256, 256),
    verbose=1,
):

    r = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(lung_out_of_frame)(msk) for _, msk in images
    )
    valid_frame = [item for i, item in enumerate(images) if r[i] is not False]
    if len(valid_frame) == 0:
        return valid_frame

    r = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(compute_asymmetry)(msk) for _, msk in valid_frame
    )

    valid_asym = get_valid_indices_iqr(r)

    valid_after_asym = [valid_frame[i] for i in valid_asym]

    if len(valid_after_asym) == 0:
        return valid_after_asym

    r = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(extract_haralick_features)(
            img, msk, glcm_features, glcm_distances, glcm_angles, resize
        )
        for img, msk in valid_after_asym
    )
    features = []
    valid_roi_frame = []
    for i, ft in enumerate(r):
        if ft is None:
            continue
        else:
            features.append(ft)
            valid_roi_frame.append(valid_after_asym[i])
    if len(features) == 0:
        return valid_roi_frame

    features = np.array(features)
    valid_texture = filter_iqr_multidimensional(features)
    final_valid_samples = [valid_roi_frame[i] for i in valid_texture]
    return final_valid_samples
