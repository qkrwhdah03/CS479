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
from itertools import cycle
from typing import Type, Literal, Optional

from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig

@dataclass
class Gs2gsPipelineConfig(VanillaPipelineConfig):
    _target: Type = field(default_factory= lambda: Gs2gsPipeline)


class Gs2gsPipeline(VanillaPipeline):
    config: Gs2gsPipelineConfig

    def __init__(
        self,
        config: Gs2gsPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,   
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)

    
    def get_train_loss_dict(self, step: int):
            """
            Args:
                step: current iteration step to update sampler if using DDP (distributed)
            """

            camera, batch = self.datamanager.next_train(step) 
            model_outputs = self.model(camera)
            metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
            loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
            # print(loss_dict, end="\n")
            
            return model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError