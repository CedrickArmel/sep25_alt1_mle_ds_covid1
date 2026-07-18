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

import numpy as np
from radiocovid.etl.preprocessings import (
    compute_asymmetry,
    extract_texture_features_glcm,
    filter_iqr_multidimensional,
    get_valid_indices_iqr,
    lung_out_of_frame,
    prepare_lung_roi,
    remove_outliers,
)

# --------------------------------------------------------------------------- #
# get_valid_indices_iqr                                                        #
# --------------------------------------------------------------------------- #


class TestGetValidIndicesIqr:
    def test_filters_outliers(self):
        # Values clustered around 10, one extreme outlier
        values = np.array([10.0, 10.1, 9.9, 10.2, 9.8, 1000.0])
        valid = get_valid_indices_iqr(values)
        assert 5 not in valid
        assert all(i in valid for i in range(5))

    def test_all_equal_keeps_all(self):
        values = np.ones(10)
        valid = get_valid_indices_iqr(values)
        assert len(valid) == 10

    def test_returns_indices_not_values(self):
        values = np.array([1.0, 2.0, 3.0])
        valid = get_valid_indices_iqr(values)
        assert all(isinstance(int(i), int) for i in valid)


# --------------------------------------------------------------------------- #
# filter_iqr_multidimensional                                                  #
# --------------------------------------------------------------------------- #


class TestFilterIqrMultidimensional:
    def test_outlier_in_any_column_drops_row(self):
        # Row 3 is an outlier in column 1 only
        rng = np.random.default_rng(0)
        mat = rng.uniform(0, 1, (10, 3))
        mat[3, 1] = 1000.0
        valid = filter_iqr_multidimensional(mat)
        assert 3 not in valid

    def test_all_inliers_keeps_all(self):
        rng = np.random.default_rng(0)
        mat = rng.uniform(0, 1, (20, 4))
        valid = filter_iqr_multidimensional(mat)
        assert len(valid) == 20

    def test_intersection_across_columns(self):
        rng = np.random.default_rng(1)
        mat = rng.uniform(0, 1, (10, 2))
        mat[0, 0] = 999.0  # outlier in col 0
        mat[9, 1] = 999.0  # outlier in col 1
        valid = filter_iqr_multidimensional(mat)
        assert 0 not in valid
        assert 9 not in valid


# --------------------------------------------------------------------------- #
# compute_asymmetry                                                            #
# --------------------------------------------------------------------------- #


class TestComputeAsymmetry:
    def test_equal_blobs_near_zero(self, tmp_mask_equal):
        asym = compute_asymmetry(tmp_mask_equal)
        assert asym < 0.05

    def test_unequal_blobs_near_one(self, tmp_mask_unequal):
        asym = compute_asymmetry(tmp_mask_unequal)
        assert asym > 0.7

    def test_single_blob_returns_one(self, tmp_mask_single):
        asym = compute_asymmetry(tmp_mask_single)
        assert asym == 1.0


# --------------------------------------------------------------------------- #
# lung_out_of_frame                                                            #
# --------------------------------------------------------------------------- #


class TestLungOutOfFrame:
    def test_border_blob_returns_true(self, tmp_mask_border):
        assert lung_out_of_frame(tmp_mask_border) is True

    def test_centered_blobs_returns_false(self, tmp_mask_centered):
        assert lung_out_of_frame(tmp_mask_centered) is False

    def test_single_component_returns_false(self, tmp_mask_single):
        assert lung_out_of_frame(tmp_mask_single) is False


# --------------------------------------------------------------------------- #
# prepare_lung_roi                                                             #
# --------------------------------------------------------------------------- #


class TestPrepareLungRoi:
    def test_output_shape(self, tmp_image_mask_pairs):
        img_path, msk_path = tmp_image_mask_pairs[1]
        size = (32, 32)
        roi = prepare_lung_roi(img_path, msk_path, size=size)
        assert roi.shape == size

    def test_outside_mask_zeroed(self, tmp_image_mask_pairs):
        img_path, msk_path = tmp_image_mask_pairs[1]
        import cv2

        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)
        roi = prepare_lung_roi(img_path, msk_path, size=mask.shape)
        binary = (mask > 0).astype("uint8")
        outside = roi[binary == 0]
        assert np.all(outside == 0)


# --------------------------------------------------------------------------- #
# extract_texture_features_glcm                                               #
# --------------------------------------------------------------------------- #


class TestExtractTextureFeatures:
    def _lung(self):
        rng = np.random.default_rng(0)
        lung = rng.integers(10, 200, (64, 64), dtype=np.uint8)
        return lung

    def test_returns_feature_list(self):
        lung = self._lung()
        features = ["contrast", "dissimilarity"]
        result = extract_texture_features_glcm(
            lung, features=features, distances=[1], angles=np.array([0])
        )
        assert isinstance(result, list)
        assert len(result) == len(features)

    def test_all_zero_lung_returns_none(self):
        lung = np.zeros((64, 64), dtype=np.uint8)
        result = extract_texture_features_glcm(
            lung, features=["contrast"], distances=[1], angles=np.array([0])
        )
        assert result is None


# --------------------------------------------------------------------------- #
# remove_outliers                                                              #
# --------------------------------------------------------------------------- #


class TestRemoveOutliers:
    _FEATURES = ["contrast", "dissimilarity"]
    _DISTANCES = [1]
    _ANGLES = np.array([0])

    def test_frame_check_failure_removed(self, tmp_image_mask_pairs):
        result = remove_outliers(
            tmp_image_mask_pairs,
            glcm_features=self._FEATURES,
            glcm_distances=self._DISTANCES,
            glcm_angles=self._ANGLES,
            n_jobs=1,
        )
        # The first pair (index 0) has a border-touching mask → filtered out
        assert len(result) < len(tmp_image_mask_pairs)

    def test_empty_input_returns_empty(self):
        result = remove_outliers(
            [],
            glcm_features=self._FEATURES,
            glcm_distances=self._DISTANCES,
            glcm_angles=self._ANGLES,
            n_jobs=1,
        )
        assert result == []

    def test_all_border_returns_empty(self, tmp_path):
        import cv2

        pairs = []
        for i in range(3):
            img_p = tmp_path / f"img_{i}.png"
            msk_p = tmp_path / f"msk_{i}.png"
            mask = np.zeros((64, 64), dtype=np.uint8)
            mask[0, 10:30] = 255  # touching top border
            mask[20, 40:55] = 255
            cv2.imwrite(str(msk_p), mask)
            rng = np.random.default_rng(i)
            cv2.imwrite(str(img_p), rng.integers(0, 200, (64, 64), dtype=np.uint8))
            pairs.append((str(img_p), str(msk_p)))
        result = remove_outliers(
            pairs,
            glcm_features=self._FEATURES,
            glcm_distances=self._DISTANCES,
            glcm_angles=self._ANGLES,
            n_jobs=1,
        )
        assert result == []
