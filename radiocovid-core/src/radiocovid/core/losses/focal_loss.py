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

# TODO: Fine tune the loss function to deal with loader output

import warnings
from typing import Optional

import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(
        self,
        weight: list | None = None,
        alpha: float = 1.0,
        gamma: float = 2.0,
        reduction: str = "none",
        to_onehot_y: bool = False,
        use_softmax: bool = True,
    ):
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.gamma = gamma
        self.alpha = alpha
        self.use_softmax = use_softmax
        self.to_onehot_y = to_onehot_y

    def forward(self, input: torch.Tensor, target: torch.Tensor):

        C: int = input.shape[1] if input.ndim > 1 else 1

        if self.to_onehot_y:
            if C == 1:
                warnings.warn("single channel prediction, `to_onehot_y=True` ignored.")
            else:
                if target.ndim > 1:
                    target = target.squeeze(1)
                target = F.one_hot(target, num_classes=C)

        if target.shape != input.shape:
            raise ValueError(
                f"ground truth has different shape ({target.shape}) from input ({input.shape})"
            )

        loss: Optional[torch.Tensor] = None
        input = input.float()
        target = target.float()
        if self.use_softmax:
            loss = softmax_focal_loss(input, target, self.gamma, self.alpha)
        else:
            loss = sigmoid_focal_loss(input, target, self.gamma, self.alpha)

        weight = (
            torch.tensor(self.weight) if self.weight is not None else torch.ones(C)
        ).to(input.device)

        if weight.ndim == 0:
            weight = weight.repeat(C)
        elif weight.shape[0] != C:
            raise ValueError

        weight = weight.view([1, C] + [1] * (loss.ndim - 2))

        loss *= weight

        if loss.ndim >= 3:
            loss = loss.mean(dim=list(range(2, target.ndim)))

        loss = loss.mean(dim=0)

        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            pass
        return loss


def softmax_focal_loss(
    input: "torch.Tensor",
    target: "torch.Tensor",
    gamma: "float" = 2.0,
    alpha: "float | None" = None,
) -> "torch.Tensor":
    """
    FL(pt) = -alpha * (1 - pt)**gamma * log(pt)

    where p_i = exp(s_i) / sum_j exp(s_j), t is the target (ground truth) class, and
    s_j is the unnormalized score for class j.
    """

    input_ls: "torch.Tensor" = input.log_softmax(1)
    loss: "torch.Tensor" = -(1 - input_ls.exp()).pow(gamma) * input_ls * target

    if alpha:
        loss *= alpha

    return loss


def sigmoid_focal_loss(
    input: "torch.Tensor",
    target: "torch.Tensor",
    gamma: "float" = 2.0,
    alpha: "float | None" = None,
) -> "torch.Tensor":
    """
    FL(pt) = -alpha * (1 - pt)**gamma * log(pt)

    where p = sigmoid(x), pt = p if label is 1 or 1 - p if label is 0
    """
    loss: torch.Tensor = F.binary_cross_entropy_with_logits(
        input=input, target=target, reduction="none"
    )
    invprobs: "torch.Tensor" = F.logsigmoid(
        -input * (target * 2 - 1)
    )  # reduced chance of overflow
    loss = (invprobs * gamma).exp() * loss
    if alpha is not None:
        alpha_factor: "torch.Tensor" = target * alpha + (1 - target) * (1 - alpha)
        loss = alpha_factor * loss
    return loss
