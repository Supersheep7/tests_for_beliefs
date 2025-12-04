import argparse
import os
import random
import numpy as np
from cfg import load_cfg
from huggingface_hub import login
from experiments import EXPERIMENTS

def main():
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

    print(f"Running {args.exp} with {args.config} (seed={args.seed})")
    EXPERIMENTS[args.exp]()

if __name__ == "__main__":
    main()
