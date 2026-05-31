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

from hashlib import sha256

import pytest
import torch
from radiocovid.core.data.datasets import (
    RadioCovidDataset,
    RadioCovidSubset,
    get_label_from_sample,
)
from torchvision.datasets.folder import default_loader

# --------------------------------------------------------------------------- #
# get_label_from_sample                                                        #
# --------------------------------------------------------------------------- #


class TestGetLabelFromSample:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("normal-001.png", 0),
            ("lung_opacity-001.png", 1),
            ("covid-001.png", 2),
            ("viral pneumonia-001.png", 3),
        ],
    )
    def test_known_classes(self, name, expected):
        assert get_label_from_sample(f"/some/dir/{name}") == expected

    def test_unknown_class_raises(self):
        with pytest.raises(KeyError):
            get_label_from_sample("/data/unknown-001.png")

    def test_case_insensitive(self):
        assert get_label_from_sample("/data/NORMAL-001.png") == 0
        assert get_label_from_sample("/data/COVID-001.png") == 2


# --------------------------------------------------------------------------- #
# RadioCovidDataset                                                            #
# --------------------------------------------------------------------------- #


class TestRadioCovidDataset:
    def test_getitem_returns_expected_keys(self, tmp_image_folder):
        ds = RadioCovidDataset(
            root=str(tmp_image_folder),
            loader=default_loader,
            extensions=(".png",),
        )
        item = ds[0]
        assert set(item.keys()) == {"id", "input", "target"}

    def test_id_is_deterministic_int(self, tmp_image_folder):
        ds = RadioCovidDataset(
            root=str(tmp_image_folder),
            loader=default_loader,
            extensions=(".png",),
        )
        path = ds.samples[0][0]
        expected_id = int(sha256(path.encode()).hexdigest()[:8], 16)
        assert ds[0]["id"] == expected_id

    def test_transform_applied(self, tmp_image_folder):
        from torchvision.transforms import v2

        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        ds = RadioCovidDataset(
            root=str(tmp_image_folder),
            loader=default_loader,
            extensions=(".png",),
            transform=transform,
        )
        item = ds[0]
        assert isinstance(item["input"], torch.Tensor)

    def test_target_is_int(self, tmp_image_folder):
        ds = RadioCovidDataset(
            root=str(tmp_image_folder),
            loader=default_loader,
            extensions=(".png",),
        )
        assert isinstance(ds[0]["target"], int)


# --------------------------------------------------------------------------- #
# RadioCovidSubset                                                             #
# --------------------------------------------------------------------------- #


class TestRadioCovidSubset:
    def _make_subset(self, tmp_image_folder):
        from torchvision.transforms import v2

        transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        ds = RadioCovidDataset(
            root=str(tmp_image_folder),
            loader=default_loader,
            extensions=(".png",),
            transform=transform,
        )
        return RadioCovidSubset(ds, indices=list(range(len(ds))))

    def test_int_index_returns_dict(self, tmp_image_folder):
        sub = self._make_subset(tmp_image_folder)
        item = sub[0]
        assert isinstance(item, dict)
        assert "input" in item and "target" in item

    def test_list_index_returns_list_of_dicts(self, tmp_image_folder):
        sub = self._make_subset(tmp_image_folder)
        items = sub[[0, 1]]
        assert isinstance(items, list)
        assert all(isinstance(i, dict) for i in items)

    def test_getitems_delegates(self, tmp_image_folder):
        sub = self._make_subset(tmp_image_folder)
        assert sub.__getitems__([0, 1]) == sub[[0, 1]]

    def test_subset_transform_applied(self, tmp_image_folder):
        from torchvision.transforms import v2

        base_ds = RadioCovidDataset(
            root=str(tmp_image_folder),
            loader=default_loader,
            extensions=(".png",),
        )
        extra_transform = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )
        sub = RadioCovidSubset(
            base_ds, indices=list(range(len(base_ds))), transform=extra_transform
        )
        item = sub[0]
        assert isinstance(item["input"], torch.Tensor)
