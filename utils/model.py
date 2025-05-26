# Model for converting  style images

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionDepth2ImgPipeline
from torchvision.transforms import ToPILImage, ToTensor

from transformers import DPTImageProcessor, DPTForDepthEstimation

DEPTH_SOURCE =  "Intel/dpt-hybrid-midas"
SD2D_SOURCE = "stabilityai/stable-diffusion-2-depth"


class Model():
    def __init__(self, device=None):
        # Set up model
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
            SD2D_SOURCE,
            torch_dtype=torch.float16
        ).to(self.device)

        self.pipe.set_progress_bar_config(disable=True)

        self.to_pil = ToPILImage()
        self.to_tensor = ToTensor()

        self.prompt = (
            "brown bear in profile on mossy autumn forest floor, "
            "misty pine background under soft morning light, "
            "Van Gogh style with swirling brush strokes, "
            "earthy browns, deep greens, muted oranges, pale blues, "
            "diffused light, soft edges, no photorealism"
        )

        self.n_prompt = "bad anatomy, deformed, warped structure, ugly, low quality, unrealistic geometry, no text"


    def convert(self, images):
        # images : Tensor of shape [C, W, H] with pixel values of [0,1]
        # Return converted images of shape [C, W, H] with pixel values of [0,1]

        # images tensor to pil image
        pil_img = self.to_pil(images)

        # Estimate Depth
        #pil_depth = self.get_depthmap(pil_img)
        
        output = self.pipe(prompt=self.prompt, image=pil_img, negative_prompt=self.n_prompt).images[0]

        output = self.to_tensor(output).unsqueeze(0)  # shape: [1, C, H, W]

        # Resize to original shape
        output = F.interpolate(output, size=images.shape[1:], mode='bilinear', align_corners=False)
        return output.squeeze(0)  # shape: [C, H, W]
    
    def get_depthmap(self, pil_img):
        processor = DPTImageProcessor.from_pretrained(DEPTH_SOURCE)
        estimator = DPTForDepthEstimation.from_pretrained(DEPTH_SOURCE).to(self.device)

        inputs = processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            depth_map = estimator(**inputs).predicted_depth.unsqueeze(1)

        depth_map = torch.nn.functional.interpolate(depth_map, size=pil_img.size[::-1], mode="bilinear", align_corners=False)
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        assert depth_min != depth_max, "Divide by zero when normalizing depth map"
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
        depth_map = torch.cat([depth_map] * 3, dim=1)

        return self.to_pil(depth_map[0])



    
if __name__ == '__main__':
    model = Model()