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
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class InferenceConfig:
    ckpt: str = os.environ.get("RADIOCOVID_CKPT", "models/model.ckpt")
    device: str = os.environ.get("RADIOCOVID_DEVICE", "cpu")
    classes: List[str] = field(
        default_factory=lambda: os.environ.get(
            "RADIOCOVID_CLASSES", "covid,normal"
        ).split(",")
    )
    host: str = os.environ.get("RADIOCOVID_HOST", "0.0.0.0")
    port: int = int(os.environ.get("RADIOCOVID_PORT", "8000"))

    @property
    def ckpt_path(self) -> Path:
        return Path(self.ckpt)


def get_config() -> InferenceConfig:
    return InferenceConfig()
