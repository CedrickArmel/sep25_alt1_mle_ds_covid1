from typing import Any

import torch
from torchvision.transforms.v2 import Transform

class Squeeze(Transform):
    """Convert images or videos to RGB (if they are already not RGB).
    """

    def __init__(self, dim: None | int = None):
        super().__init__()
        self.dim = dim
    
    def transform(self, inpt: Any, params: dict[str, Any]) -> Any:
        if isinstance(inpt, torch.Tensor):
            return inpt.squeeze() if not self.dim else inpt.squeeze(dim=self.dim)