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


import importlib


def test_import_core_package():
    importlib.import_module("radiocovid.core")


def test_import_core_data():
    for mod in (
        "radiocovid.core.data",
        "radiocovid.core.data.datamodule",
        "radiocovid.core.data.datasets",
        "radiocovid.core.data.transforms",
    ):
        importlib.import_module(mod)


def test_import_core_losses():
    for mod in ("radiocovid.core.losses", "radiocovid.core.losses.focal_loss"):
        importlib.import_module(mod)


def test_import_core_models():
    for mod in (
        "radiocovid.core.models",
        "radiocovid.core.models.nets",
        "radiocovid.core.models.vanilla_model",
    ):
        importlib.import_module(mod)


def test_import_core_utils():
    for mod in (
        "radiocovid.core.utils",
        "radiocovid.core.utils.instantiators",
        "radiocovid.core.utils.logging_utils",
        "radiocovid.core.utils.pylogger",
        "radiocovid.core.utils.rich_utils",
        "radiocovid.core.utils.utils",
    ):
        importlib.import_module(mod)


def test_import_core_train():
    importlib.import_module("radiocovid.core.train")
