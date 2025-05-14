from __future__ import annotations

from dataclasses import dataclass, field

from typing import Type, Optional, Tuple

from nerfstudio.models.splatfacto import SplatfactoModelConfig, SplatfactoModel

import torch

@dataclass
class Gs2gsModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory= lambda: Gs2gsModel)


class Gs2gsModel(SplatfactoModel):
    config: Gs2gsModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        super().populate_modules()