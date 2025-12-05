import argparse
import os
import random
import torch as t
import numpy as np
from cfg import load_cfg
from huggingface_hub import login
import subprocess
from experiments import EXPERIMENTS

COUNT = subprocess.check_output(
    ["git", "rev-list", "--count", "HEAD"],
    text=True
).strip()

DATE = subprocess.check_output(
    ["git", "show", "-s", "--format=%cI", "HEAD"],
    text=True
).strip()


def main():

    print()
    print("Welcome to the experiments for our project!")
    print()
    print("=== Git commit count information ===")
    print()
    print("You are running git commit number:", COUNT, "from date:", DATE)

    parser = argparse.ArgumentParser(description="Run your pipeline with a config + seed")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config")
    parser.add_argument("--exp", choices=EXPERIMENTS.keys(), required=True,
                        help="Experiment to run")
    parser.add_argument("--set", nargs="*", help="Overrides like train.lr=1e-4")
    parser.add_argument("--hf_token", help="Hugging Face access token")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    token = args.hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("No Hugging Face token provided. "
                         "Set --hf_token or env var HF_TOKEN.")
    login(token)  # authenticates the session

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Check CUDA availability once
    cuda_available = t.cuda.is_available()
    if cuda_available:
        print(f"CUDA available")
    else:
        print("CUDA NOT available, using CPU")

    print(f"Running {args.exp} with {args.config} (seed={args.seed})")
    EXPERIMENTS[args.exp]()

if __name__ == "__main__":
    main()
