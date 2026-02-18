# MIT License
#
# Copyright (c) 2026 @CedrickArmel, @TaxelleT, @Yeyecodes & @samarita22
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
from collections.abc import Sequence
from typing import Optional

import torch
import torch.nn.functional as F


class FocalLoss(torch.nn.Module):
    def __init__(
        self,
        to_onehot_y: bool = False,
        gamma: float = 2.0,
        alpha: float | Sequence[float] | None = None,
        reduction: str = "none",
        use_softmax: bool = False,
    ) -> None:
        super().__init__()

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f'Unsupported reduction: {reduction}, available options are ["mean", "sum", "none"].'
            )
        self.reduction = reduction
        self.to_onehot_y = to_onehot_y
        self.gamma = gamma
        self.use_softmax = use_softmax

        if alpha is not None:
            if alpha < 0.0 or min(alpha) < 0.0:  # type: ignore[operator,arg-type]
                raise ValueError(
                    "the value/values of the `alpha` should be no less than 0."
                )

        self.alpha = alpha

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BN, where N is the number of classes.
                The input should be the original logits since it will be transformed by
                a sigmoid/softmax in the forward function.
            target: the shape should be BN or B1, where N is the number of classes.

        Raises:
            ValueError: When input and target (after one hot transform if set)
                have different shapes.
            ValueError: When ``self.weight`` is a sequence and the length is not equal to the
                number of classes.
        """

        if self.to_onehot_y:
            if target.shape[1] == input.shape[1]:
                warnings.warn(
                    "Target and Input have same size, `to_onehot_y=True` ignored."
                )
            elif target.shape[1] == 1:
                target = F.one_hot(target.squeeze(1), num_classes=input.shape[1])
            else:
                raise ValueError(
                    f"Ground truth is {target.shape} with dim != 1. Ensure that Ground tructh dim 1 = 1 or Pred's dim 1."
                )

        if target.shape != input.shape:
            raise ValueError(
                f"ground truth has different shape ({target.shape}) from input ({input.shape})"
            )
        device = input.device
        loss: Optional[torch.Tensor] = None
        input = input.float()
        target = target.float()
        alpha = None
        if self.alpha is not None:
            alpha = (
                torch.tensor(self.alpha).to(device)
                if isinstance(self.alpha, (list, tuple))
                else self.alpha
            )

        loss = softmax_focal_loss(input, target, self.gamma, alpha)  # type: ignore[arg-type]

        loss = loss.mean(dim=0)
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "none":
            pass
        return loss


def softmax_focal_loss(
    input: "torch.Tensor",
    target: "torch.Tensor",
    gamma: "float" = 2.0,
    alpha: "float | torch.Tensor | None" = None,
) -> "torch.Tensor":
    """
    FL(pt) = -alpha * (1 - pt)**gamma * log(pt)

    where p_i = exp(s_i) / sum_j exp(s_j), t is the target (ground truth) class, and
    s_j is the unnormalized score for class j.
    """
    input_ls: "torch.Tensor" = input.log_softmax(1)
    num_classes: "int" = target.shape[1]
    loss: "torch.Tensor" = -(1 - input_ls.exp()).pow(gamma) * input_ls * target
    if alpha is not None:
        if isinstance(alpha, torch.Tensor):
            if alpha.numel() != num_classes:
                raise ValueError("")
        loss *= alpha
    return loss
