# Stylize original images 

from parser import Parser
from model import Model

import torch
import torchvision.io as io
import torchvision.utils as vutils

import shutil

from tqdm import tqdm

if __name__ == '__main__':

    # Parsing Arguments 
    parser = Parser()
    args = parser.parse()

    # Load Data
    if args.data_dir.name != "images":
        data_parent_path = args.data_dir
        data_image_path = args.data_dir.joinpath("images")
    else :
        data_parent_path = args.data_dir.parent
        data_image_path = args.data_dir
    
    image_paths = list(data_image_path.glob("*"))
    num_images = len(image_paths)

    # Traget Directory
    converted_parent_path = args.target_dir
    converted_image_path = args.target_dir.joinpath("images")
    if not converted_image_path.exists():
           print(f"Target directory have no images directory. Creating: {args.target_dir}")
           converted_image_path.mkdir(parents=True, exist_ok=True)

    # Convert Images
    print("Total", num_images, "number of images detected.")
    print("Now start converting images...")

    model = Model()

    for path in tqdm(image_paths):
        image = io.read_image(str(path)) / 255.0  # [1, C, H, W] tensor with pixel values of [0,1]
        converted_image = model.convert(image)
        
        assert image.shape == converted_image.shape, "Converted Image Shape changed"
        
        image_name = path.name
        save_path = converted_image_path.joinpath(image_name)
        vutils.save_image(converted_image, save_path)

    print("Done Converting")

    # Copying transform.json file
    transform_json_path = data_parent_path / "transforms.json"
    target_transform_path = converted_parent_path / "transforms.json"

    if transform_json_path.exists():
        shutil.copy2(transform_json_path, target_transform_path)
        print(f"Saving transforms.json to {target_transform_path}")
    else:
        print("Warning: transforms.json not found in source data directory.")
