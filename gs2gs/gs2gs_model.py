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

from typing import Type, Optional, Tuple

from nerfstudio.models.splatfacto import SplatfactoModelConfig, SplatfactoModel

import torch
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio
from pytorch_msssim import SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from nerfstudio.model_components.losses import L1Loss, MSELoss

from gs2gs.loss import VGG19


@dataclass
class Gs2gsModelConfig(SplatfactoModelConfig):
    _target: Type = field(default_factory= lambda: Gs2gsModel)
    num_downscales: int = 3

    warmup: bool = True
    # warmup
    use_l1_loss: bool = False
    use_lpips_loss: bool = False
    weight: float = 0.2

    # style transfer
    style_loss_weight: float = 1.0

    num_random: int = 10000
    stop_split_at: int = 15000
    refine_every: int = 100


class Gs2gsModel(SplatfactoModel):
    config: Gs2gsModelConfig

    def populate_modules(self):
        super().populate_modules()

        if self.config.warmup:

            if self.config.use_l1_loss:
                self.rgb_loss = L1Loss()
            else:
                self.rgb_loss = MSELoss()
            
            self.psnr = PeakSignalNoiseRatio(data_range=1.0)
            self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
            self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        
        else :
            device = next(self.parameters()).device
            self.vgg = VGG19().to(device)
            self.vgg.load_state_dict(torch.load("vgg19.pth")) # Load VGG19 weights
            self.vgg.eval()  

     
    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
          
        pred_images = outputs["rgb"].permute(2, 0, 1)
        if pred_images.dim() == 3:
            pred_images = pred_images.unsqueeze(0)

        gt_images = batch["image"].permute(2, 0, 1)
        if gt_images.dim() == 3:
            gt_images = gt_images.unsqueeze(0)
        
        if gt_images.size() != pred_images.size():
            gt_images = torch.nn.functional.interpolate(gt_images, size=pred_images.shape[2:], mode='bilinear', align_corners=False)

        if self.config.warmup:
            loss_dict["rgb_loss"] = self.rgb_loss(gt_images, pred_images)
            
            if self.config.use_lpips_loss:
                pred_images = (pred_images * 2 - 1).clamp(-1, 1)
                gt_images =  (gt_images * 2 - 1).clamp(-1, 1)

                loss_dict["lpips_loss"] = self.config.weight * self.lpips(pred_images, gt_images)
            
            else :
                loss_dict["dssim"] = self.config.weight *  (1 - self.ssim(pred_images, gt_images))
        
        else:
            loss_dict["vgg_loss"] = self.vgg.vgg_loss(gt_images, pred_images, self.config.style_loss_weight)

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)

        
        return loss_dict
        