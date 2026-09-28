import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen nn.Linear.

    Swapped in place of the original module, so the surrounding module tree,
    attribute paths (Pi3Trainer reaches for model.encoder) and forward signatures
    are all unchanged -- only this Linear's internals differ. lora_B starts at
    zero, so the adapted model is numerically identical to the original at init.
    """

    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / r

    def forward(self, x):
        delta = (x @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(0, 1)
        return self.base(x) + delta * self.scaling


def apply_lora(module: nn.Module, targets=('qkv', 'proj', 'fc1', 'fc2'),
               r: int = 16, alpha: int = 32) -> int:
    """Replace named Linear children with LoRA-adapted versions, recursively.

    Returns the number of layers adapted. Apply AFTER loading pretrained weights:
    wrapping renames parameters (`...qkv.weight` -> `...qkv.base.weight`), so a
    strict load against the original checkpoint would fail otherwise.
    """
    count = 0
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
            count += 1
        else:
            count += apply_lora(child, targets=targets, r=r, alpha=alpha)
    return count
