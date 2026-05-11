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

from radiocovid.etl.utils import Logger


class TestLogger:
    def test_info_message_logged(self, caplog):
        logger = Logger("test.etl.info")
        with caplog.at_level(logging.INFO, logger="test.etl.info"):
            logger.info("hello from etl")
        assert any("hello from etl" in r.message for r in caplog.records)

    def test_debug_suppressed_at_info_level(self, caplog):
        logger = Logger("test.etl.debug")
        with caplog.at_level(logging.INFO, logger="test.etl.debug"):
            logger.debug("silent debug")
        assert not any("silent debug" in r.message for r in caplog.records)

    def test_warning_emitted(self, caplog):
        logger = Logger("test.etl.warn")
        with caplog.at_level(logging.WARNING, logger="test.etl.warn"):
            logger.warning("watch out")
        assert any("watch out" in r.message for r in caplog.records)

    def test_log_method_dispatches(self, caplog):
        logger = Logger("test.etl.log")
        with caplog.at_level(logging.INFO, logger="test.etl.log"):
            logger.log(logging.INFO, "via log()")
        assert any("via log()" in r.message for r in caplog.records)
