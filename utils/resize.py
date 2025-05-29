from parser import Parser
from pathlib import Path
import torchvision.transforms as T
import torchvision.utils as vutils
import torchvision.io as io
from tqdm import tqdm

if __name__ == '__main__':

    parser = Parser()
    args = parser.parse()

    # Determine image directories
    data_image_path = args.data_dir.joinpath("images") if args.data_dir.name != "images" else args.data_dir
    target_image_path = args.target_dir.parent if args.target_dir.name == "imags" else args.target_dir
    target_image_path.mkdir(parents=True, exist_ok=True)

    new_image_path = args.data_dir / "raw_images"
    data_image_path.rename(new_image_path)
    data_image_path = new_image_path

    target_image_path = args.target_dir.joinpath("images")
    target_image_path.mkdir(parents=True, exist_ok=True)

    # Target size as (H, W)
    if args.target_size is None:
        raise ValueError("You must provide --target_size WIDTH HEIGHT")
    target_size = tuple(args.target_size)

    # Resize transform
    resize = T.Resize(target_size, antialias=True)

    for path in tqdm(sorted(data_image_path.glob("*.jpg"))):
        # Read and preprocess image
        image = io.read_image(str(path)).float() / 255.0  # [C, H, W], float in [0,1]
        image = image.unsqueeze(0)  # [1, C, H, W]

        # Resize and save
        image_resized = resize(image)
        save_path = target_image_path / path.name
        vutils.save_image(image_resized, save_path)

        #print(f"Saved resized image to: {save_path}")