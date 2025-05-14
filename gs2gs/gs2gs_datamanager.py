from __future__ import annotations

from dataclasses import dataclass, field

from typing import Type, Tuple, Dict, Union, Literal

from nerfstudio.cameras.cameras import Cameras

from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig
)

import torch

@dataclass
class Gs2gsDataManagerConfig(FullImageDatamanagerConfig):
    _target: Type = field(default_factory= lambda: Gs2gsDataManager)


class Gs2gsDataManager(FullImageDatamanager):
    config: Gs2gsDataManagerConfig

    def __init__(
        self,
        config: FullImageDatamanagerConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank, **kwargs)

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        return super().next_train(step)