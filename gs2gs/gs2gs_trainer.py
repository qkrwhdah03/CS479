from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.viewer.server.viewer_elements import ViewerButton
from nerfstudio.utils.decorators import check_main_thread

import os
import torch
from pathlib import Path

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

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.load_dir
        load_checkpoint = self.config.load_checkpoint
        if load_dir is not None:
            load_step = self.config.load_step
            if load_step is None:
                print("Loading latest Nerfstudio checkpoint from load_dir...")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path: Path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = torch.load(load_path, map_location="cpu")
            self._start_step = 0
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], 0)
            #self.optimizers.load_optimizers(loaded_state["optimizers"])
            #if "schedulers" in loaded_state and self.config.load_scheduler:
            #    self.optimizers.load_schedulers(loaded_state["schedulers"])
            #self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_path}")
        elif load_checkpoint is not None:
            assert load_checkpoint.exists(), f"Checkpoint {load_checkpoint} does not exist"
            loaded_state = torch.load(load_checkpoint, map_location="cpu")
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_checkpoint}")
        else:
            CONSOLE.print("No Nerfstudio checkpoint to load, so training from scratch.")
