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

import logging

import pytest
from lightning_utilities.core.rank_zero import rank_zero_only as rzo

from radiocovid.core.utils.pylogger import RankedLogger

# --------------------------------------------------------------------------- #
# RankedLogger                                                                 #
# --------------------------------------------------------------------------- #


class TestRankedLogger:
    def test_message_contains_rank_prefix(self, caplog):
        logger = RankedLogger("test.ranked", rank_zero_only=False)
        with caplog.at_level(logging.INFO, logger="test.ranked"):
            logger.info("hello")
        assert any("hello" in r.message for r in caplog.records)

    def test_rank_zero_only_logs_on_rank_zero(self, caplog):
        logger = RankedLogger("test.rzo_true", rank_zero_only=True)
        rzo.rank = 0
        with caplog.at_level(logging.INFO, logger="test.rzo_true"):
            logger.info("should appear")
        assert any("should appear" in r.message for r in caplog.records)

    def test_rank_zero_only_suppressed_on_nonzero_rank(self, caplog):
        logger = RankedLogger("test.rzo_suppress", rank_zero_only=True)
        rzo.rank = 1
        try:
            with caplog.at_level(logging.INFO, logger="test.rzo_suppress"):
                logger.info("should not appear")
            assert not any("should not appear" in r.message for r in caplog.records)
        finally:
            rzo.rank = 0

    def test_explicit_rank_kwarg_logs_on_matching_rank(self, caplog):
        logger = RankedLogger("test.explicit_rank", rank_zero_only=False)
        rzo.rank = 0
        with caplog.at_level(logging.INFO, logger="test.explicit_rank"):
            logger.info("rank0 msg", rank=0)
        assert any("rank0 msg" in r.message for r in caplog.records)

    def test_explicit_rank_kwarg_suppressed_on_non_matching(self, caplog):
        logger = RankedLogger("test.explicit_rank2", rank_zero_only=False)
        rzo.rank = 0
        with caplog.at_level(logging.INFO, logger="test.explicit_rank2"):
            logger.info("rank1 msg", rank=1)
        assert not any("rank1 msg" in r.message for r in caplog.records)

    def test_missing_rank_raises(self):
        logger = RankedLogger("test.missing_rank", rank_zero_only=False)
        rzo.rank = None
        try:
            with pytest.raises(RuntimeError, match="rank_zero_only.rank"):
                logger.info("will fail")
        finally:
            rzo.rank = 0
