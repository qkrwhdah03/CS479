from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.viewer.server.viewer_elements import ViewerButton
from nerfstudio.utils.decorators import check_main_thread

import torch

@dataclass
class Gs2gsTrainerConfig(TrainerConfig):
    _target: Type = field(default_factory=lambda: Gs2gsTrainer)


class Gs2gsTrainer(Trainer):
    def __init__(self, config: TrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        super().__init__(config, local_rank, world_size)

    def _clean_optimizer_states(self):
        """Remove optimizer states for parameters that no longer exist."""
        for optimizer in self.optimizers.optimizers.values():
            current_param_ids = {id(p) for p in self.pipeline.model.parameters()}
            state = optimizer.state
            to_delete = [k for k in state.keys() if isinstance(k, torch.Tensor) and id(k) not in current_param_ids]
            for k in to_delete:
                del state[k]

    def save_checkpoint(self, step: int):
        self._clean_optimizer_states()
        super().save_checkpoint(step)