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

import warnings

import pytest
import torch
import torch.nn.functional as F
import torchvision.ops as tv_ops

from radiocovid.core.losses.focal_loss import (
    FocalLoss,
    sigmoid_focal_loss,
    softmax_focal_loss,
)

# --------------------------------------------------------------------------- #
# softmax_focal_loss                                                           #
# --------------------------------------------------------------------------- #


class TestSoftmaxFocalLoss:
    def _logits_target(self):
        torch.manual_seed(0)
        logits = torch.randn(2, 3)
        target = torch.zeros(2, 3)
        target[0, 1] = 1.0
        target[1, 0] = 1.0
        return logits, target

    def test_hand_computed_value(self):
        logits, target = self._logits_target()
        loss = softmax_focal_loss(logits, target, gamma=2.0, alpha=1.0)
        log_p = logits.log_softmax(1)
        expected = -(1 - log_p.exp()).pow(2) * log_p * target
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_gamma_zero_reduces_to_nll(self):
        logits, target = self._logits_target()
        loss = softmax_focal_loss(logits, target, gamma=0.0, alpha=None)
        expected = -logits.log_softmax(1) * target
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_alpha_scales_linearly(self):
        logits, target = self._logits_target()
        l1 = softmax_focal_loss(logits, target, gamma=2.0, alpha=1.0)
        l2 = softmax_focal_loss(logits, target, gamma=2.0, alpha=2.0)
        assert torch.allclose(l2, l1 * 2, atol=1e-6)

    def test_larger_gamma_shrinks_well_classified(self):
        # For a nearly perfect prediction, higher gamma → smaller loss
        logits = torch.tensor([[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]])
        target = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        l_low = softmax_focal_loss(logits, target, gamma=0.5, alpha=None).sum()
        l_high = softmax_focal_loss(logits, target, gamma=4.0, alpha=None).sum()
        assert l_high < l_low


# --------------------------------------------------------------------------- #
# sigmoid_focal_loss                                                           #
# --------------------------------------------------------------------------- #


class TestSigmoidFocalLoss:
    def _inputs(self):
        torch.manual_seed(1)
        inp = torch.randn(2, 4)
        tgt = torch.randint(0, 2, (2, 4)).float()
        return inp, tgt

    def test_matches_torchvision_oracle(self):
        inp, tgt = self._inputs()
        ours = sigmoid_focal_loss(inp, tgt, gamma=2.0, alpha=0.25)
        tv = tv_ops.sigmoid_focal_loss(
            inp, tgt, alpha=0.25, gamma=2.0, reduction="none"
        )
        assert torch.allclose(ours, tv, atol=1e-5)

    def test_alpha_none_skips_factor(self):
        inp, tgt = self._inputs()
        with_alpha = sigmoid_focal_loss(inp, tgt, gamma=2.0, alpha=0.5)
        without_alpha = sigmoid_focal_loss(inp, tgt, gamma=2.0, alpha=None)
        assert not torch.allclose(with_alpha, without_alpha)

    def test_gamma_zero_equals_bce(self):
        inp, tgt = self._inputs()
        loss = sigmoid_focal_loss(inp, tgt, gamma=0.0, alpha=None)
        bce = F.binary_cross_entropy_with_logits(inp, tgt, reduction="none")
        assert torch.allclose(loss, bce, atol=1e-5)


# --------------------------------------------------------------------------- #
# FocalLoss.forward                                                            #
# --------------------------------------------------------------------------- #


class TestFocalLossForward:
    def _make(self, **kw):
        return FocalLoss(**kw)

    def test_shape_mismatch_raises(self):
        fl = self._make()
        with pytest.raises(ValueError):
            fl(torch.randn(2, 3), torch.randn(2, 4))

    def test_to_onehot_y_converts_targets(self):
        fl = self._make(to_onehot_y=True, reduction="none")
        logits = torch.randn(4, 3)
        target = torch.tensor([0, 1, 2, 0])
        loss = fl(logits, target)
        assert loss.shape == (3,)

    def test_to_onehot_y_single_channel_warns(self):
        fl = self._make(to_onehot_y=True, reduction="none")
        logits = torch.randn(4, 1)
        target = torch.zeros(4, 1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            fl(logits, target)
        assert any("single channel" in str(warning.message) for warning in w)

    def test_reduction_sum(self):
        fl = self._make(reduction="sum")
        loss = fl(torch.randn(4, 3), torch.zeros(4, 3))
        assert loss.ndim == 0

    def test_reduction_mean(self):
        fl = self._make(reduction="mean")
        loss = fl(torch.randn(4, 3), torch.zeros(4, 3))
        assert loss.ndim == 0

    def test_reduction_none_shape(self):
        fl = self._make(reduction="none")
        loss = fl(torch.randn(4, 3), torch.zeros(4, 3))
        assert loss.shape == (3,)

    def test_custom_weight_applied(self):
        fl_uniform = self._make(weight=[1.0, 1.0, 1.0], reduction="none")
        fl_weighted = self._make(weight=[2.0, 1.0, 1.0], reduction="none")
        inp = torch.randn(4, 3)
        tgt = torch.zeros(4, 3)
        l_u = fl_uniform(inp, tgt)
        l_w = fl_weighted(inp, tgt)
        assert not torch.allclose(l_u, l_w)

    def test_wrong_weight_length_raises(self):
        fl = self._make(weight=[1.0, 2.0])
        with pytest.raises(ValueError):
            fl(torch.randn(4, 3), torch.zeros(4, 3))

    def test_scalar_weight_broadcast(self):
        fl = self._make(weight=[2.0], reduction="mean")
        loss = fl(torch.randn(4, 3), torch.zeros(4, 3))
        assert loss.ndim == 0

    def test_use_softmax_false(self):
        fl = self._make(use_softmax=False, reduction="mean")
        loss = fl(torch.randn(4, 3), torch.zeros(4, 3).float())
        assert torch.isfinite(loss)

    def test_spatial_input_reduced_over_spatial_dims(self):
        # ndim=4 input: (B, C, H, W)
        fl = self._make(reduction="none")
        logits = torch.randn(2, 3, 4, 4)
        target = torch.zeros(2, 3, 4, 4)
        loss = fl(logits, target)
        assert loss.shape == (3,)
