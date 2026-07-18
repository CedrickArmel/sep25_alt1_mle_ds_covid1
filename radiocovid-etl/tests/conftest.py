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

import cv2
import numpy as np
import pytest


def _write_mask(path, h, w, blobs):
    """Write a grayscale PNG mask with filled rectangles as blobs.

    blobs: list of (y, x, bh, bw) tuples defining each blob region.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    for y, x, bh, bw in blobs:
        mask[y : y + bh, x : x + bw] = 255
    cv2.imwrite(str(path), mask)


def _write_image(path, h, w):
    rng = np.random.default_rng(0)
    img = rng.integers(30, 200, (h, w), dtype=np.uint8)
    cv2.imwrite(str(path), img)


@pytest.fixture
def tmp_mask_centered(tmp_path):
    """Two centered blobs that do NOT touch any border. Returns mask path."""
    h, w = 64, 64
    path = tmp_path / "centered.png"
    _write_mask(path, h, w, [(10, 5, 20, 10), (10, 40, 20, 10)])
    return str(path)


@pytest.fixture
def tmp_mask_border(tmp_path):
    """Two blobs where one touches the top border. Returns mask path."""
    h, w = 64, 64
    path = tmp_path / "border.png"
    _write_mask(path, h, w, [(0, 5, 20, 10), (20, 40, 20, 10)])
    return str(path)


@pytest.fixture
def tmp_mask_single(tmp_path):
    """Only one blob — fewer than 3 connected components. Returns mask path."""
    h, w = 64, 64
    path = tmp_path / "single.png"
    _write_mask(path, h, w, [(20, 20, 20, 20)])
    return str(path)


@pytest.fixture
def tmp_mask_equal(tmp_path):
    """Two equal-area blobs for asymmetry test. Returns mask path."""
    h, w = 64, 64
    path = tmp_path / "equal.png"
    _write_mask(path, h, w, [(10, 5, 20, 20), (10, 38, 20, 20)])
    return str(path)


@pytest.fixture
def tmp_mask_unequal(tmp_path):
    """Two very unequal blobs for asymmetry test. Returns mask path."""
    h, w = 64, 64
    path = tmp_path / "unequal.png"
    _write_mask(path, h, w, [(5, 5, 30, 30), (5, 50, 4, 4)])
    return str(path)


@pytest.fixture
def tmp_image_mask_pairs(tmp_path):
    """Small set of (image_path, mask_path) pairs for remove_outliers tests.

    Returns a list of 4 pairs. The first pair has a border-touching mask
    (will be filtered by lung_out_of_frame); the rest have centered masks.
    """
    pairs = []
    h, w = 64, 64
    for i in range(4):
        img_path = tmp_path / f"img_{i}.png"
        msk_path = tmp_path / f"msk_{i}.png"
        _write_image(img_path, h, w)
        if i == 0:
            _write_mask(msk_path, h, w, [(0, 5, 20, 10), (20, 40, 20, 10)])
        else:
            _write_mask(msk_path, h, w, [(10, 5, 20, 10), (10, 40, 20, 10)])
        pairs.append((str(img_path), str(msk_path)))
    return pairs
