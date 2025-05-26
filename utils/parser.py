import argparse
import os


class Parser:
    def __init__(self,):
        self.parser = argparse.ArgumentParser()
        
        self.parser.add_argument(
            "--data_dir",
            type=str,
            required=True,
            help="Path to the source data directory"
        )

        self.parser.add_argument(
            "--target_dir",
            type=str,
            required=True,
            help="Path to the target output directory"
        )
    
    def parse(self,):
        args = self.parser.parse_args()

        if not os.path.exists(args.data_dir):
             raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
        if not os.path.exists(args.target_dir):
            print(f"Target directory does not exist. Creating: {args.target_dir}")
            os.makedirs(args.target_dir)

        return args

if __name__ == '__main__':
    parser = Parser()
    args = parser.parse()
    print("Data directory:", args.data_dir)
    print("Target directory:", args.target_dir)