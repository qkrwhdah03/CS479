import argparse
from pathlib import Path


class Parser:
    def __init__(self,):
        self.parser = argparse.ArgumentParser()
        
        self.parser.add_argument(
            "--data_dir",
            type=Path,
            required=True,
            help="Path to the source data directory"
        )

        self.parser.add_argument(
            "--target_dir",
            type=Path,
            required=True,
            help="Path to the target output directory"
        )
        self.parser.add_argument(
            "--target_size",
            type= int,
            nargs=2,
            metavar=('WIDTH', 'HEIGHT'),
            help="Target size as two integers: WIDTH HEIGHT"
        )
    
    def parse(self,):
        args = self.parser.parse_args()

        if not args.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
        if not args.data_dir.is_dir():
            raise NotADirectoryError(f"Data directory is not a folder: {args.data_dir}")
    
        if not args.target_dir.exists():
            print(f"Target directory does not exist. Creating: {args.target_dir}")
            args.target_dir.mkdir(parents=True, exist_ok=True)

        return args

if __name__ == '__main__':
    parser = Parser()
    args = parser.parse()
    print("Data directory:", args.data_dir)
    print("Target directory:", args.target_dir)
    print(args.target_size)