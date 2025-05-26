# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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