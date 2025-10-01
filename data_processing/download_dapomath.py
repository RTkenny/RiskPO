"""
Download DAPO-Math-17k (train) and AIME-2024 (test) parquet files using datasets.load_dataset, similar in style to download_easymath.py.
"""

import argparse
import os
import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download DAPO-Math-17k and AIME-2024 parquet files using datasets.load_dataset.")
    parser.add_argument("--output_dir", default="/home/rt/repo/RiskPO/", help="Target directory for downloaded parquet files.")
    parser.add_argument("--train_filename", default="dapo-math-17k.parquet", help="Output filename for train parquet.")
    parser.add_argument("--test_filename", default="aime-2024.parquet", help="Output filename for test parquet.")
    parser.add_argument(
        "--train_hf_repo",
        default="BytedTsinghua-SIA/DAPO-Math-17k",
        help="Huggingface repo for the DAPO-Math-17k dataset.",
    )
    parser.add_argument(
        "--test_hf_repo",
        default="BytedTsinghua-SIA/AIME-2024",
        help="Huggingface repo for the AIME-2024 dataset.",
    )

    args = parser.parse_args()
    save_dir = os.path.join(args.output_dir, "dapo_aime2024")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading {args.train_hf_repo} from Huggingface...", flush=True)
    train_dataset = datasets.load_dataset(args.train_hf_repo, split="train")
    train_path = os.path.join(save_dir, args.train_filename)
    print(f"Saving train split to {train_path}", flush=True)
    train_dataset.to_parquet(train_path)

    print(f"Loading {args.test_hf_repo} from Huggingface...", flush=True)
    test_dataset = datasets.load_dataset(args.test_hf_repo, split="train")
    test_path = os.path.join(save_dir, args.test_filename)
    print(f"Saving test split to {test_path}", flush=True)
    test_dataset.to_parquet(test_path)