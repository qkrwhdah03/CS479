from __future__ import annotations

from dataclasses import dataclass, field

from typing import Type, Optional, Tuple

from nerfstudio.models.splatfacto import SplatfactoModelConfig, SplatfactoModel

import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.model_components.losses import L1Loss, MSELoss


@dataclass
class Gs2gsModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory= lambda: Gs2gsModel)
    num_downscales: int = 0
    use_l1_loss: bool = False
    use_lpips_loss: bool = True
    lpips_loss_weight: float = 0.1
    num_random: int = 50000
    stop_split_at: int = 15000


class Gs2gsModel(SplatfactoModel):
    config: Gs2gsModelConfig

    def populate_modules(self):
        super().populate_modules()

        if self.config.use_l1_loss:
            self.rgb_loss = L1Loss()
        else:
            self.rgb_loss = MSELoss()

        self.lpips = LearnedPerceptualImagePatchSimilarity()

     
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        
        gt_image = batch["image"]
        pred_image = outputs["rgb"]  
        loss_dict["rgb_loss"] = self.rgb_loss(gt_image, pred_image)
        
        if self.config.use_lpips_loss:
            if pred_image.dim() == 3:
                pred_image = pred_image.unsqueeze(0)
            if gt_image.dim() == 3:
               gt_image = gt_image.unsqueeze(0)

            pred_image = pred_image.permute(0, 3, 1, 2)
            gt_image = gt_image.permute(0, 3, 1, 2)

            pred_image = (pred_image * 2 - 1).clamp(-1, 1)
            gt_image =  (gt_image * 2 - 1).clamp(-1, 1)

            loss_dict["lpips_loss"] = self.config.lpips_loss_weight * self.lpips(pred_image, gt_image)

        
        return loss_dict
        