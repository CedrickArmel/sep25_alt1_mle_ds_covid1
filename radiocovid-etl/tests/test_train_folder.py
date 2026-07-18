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

import pytest
from radiocovid.etl.train_folder import create_symlink


class TestCreateSymlink:
    def test_creates_symlink(self, tmp_path):
        src = tmp_path / "src" / "img.png"
        src.parent.mkdir()
        src.write_bytes(b"\x89PNG")
        create_symlink("covid", str(src), tmp_path / "dst")
        link = tmp_path / "dst" / "covid" / "img.png"
        assert link.is_symlink()

    def test_symlink_resolves_to_source(self, tmp_path):
        src = tmp_path / "src" / "img.png"
        src.parent.mkdir()
        src.write_bytes(b"\x89PNG")
        create_symlink("covid", str(src), tmp_path / "dst")
        link = tmp_path / "dst" / "covid" / "img.png"
        assert link.resolve() == src.resolve()

    def test_creates_class_subdir(self, tmp_path):
        src = tmp_path / "img.png"
        src.write_bytes(b"")
        create_symlink("normal", str(src), tmp_path / "dst")
        assert (tmp_path / "dst" / "normal").is_dir()

    def test_duplicate_raises_file_exists_error(self, tmp_path):
        src = tmp_path / "img.png"
        src.write_bytes(b"")
        create_symlink("covid", str(src), tmp_path / "dst")
        with pytest.raises(FileExistsError):
            create_symlink("covid", str(src), tmp_path / "dst")

    def test_different_classes_no_conflict(self, tmp_path):
        src = tmp_path / "img.png"
        src.write_bytes(b"")
        create_symlink("covid", str(src), tmp_path / "dst")
        create_symlink("normal", str(src), tmp_path / "dst")
        assert (tmp_path / "dst" / "covid" / "img.png").is_symlink()
        assert (tmp_path / "dst" / "normal" / "img.png").is_symlink()
