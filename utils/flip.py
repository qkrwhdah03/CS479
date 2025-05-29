import glob
import torch
import torchvision.io as io

path = './data/duck-cd/images/*.jpg'
paths = sorted(glob.glob(path))

print(len(paths))

for img_path in paths:
    # [C, H, W], uint8 [0,255]
    img_tensor = io.read_image(img_path)

    # height <-> width
    transposed_tensor = img_tensor.permute(0, 2, 1).contiguous()
    transposed_tensor =  torch.flip(img_tensor, dims=[1]) 

    # 저장 (JPEG: 0~255 uint8, PNG도 가능)
    io.write_jpeg(transposed_tensor, img_path, quality=100)  # 덮어쓰기