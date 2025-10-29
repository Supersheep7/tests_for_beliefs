from gooey import Gooey, GooeyParser
import argparse, os, random, numpy as np
from cfg import load_cfg
from huggingface_hub import login
from experiments import EXPERIMENTS

@Gooey(program_name="InterpRunner", default_size=(600, 400))
def main():
    p = GooeyParser(description="Run your pipeline with a config + seed")
    p.add_argument("--config", default="configs/default.yaml",
                   widget="FileChooser", help="Path to YAML config")
    p.add_argument("--exp", choices=EXPERIMENTS.keys(), required=True)
    p.add_argument("--set", nargs="*", help="Overrides like train.lr=1e-4")
    p.add_argument("--hf_token", help="Hugging Face access token", widget="PasswordField")
    args = p.parse_args()

    token = args.hf_token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("No Hugging Face token provided. "
                         "Set --hf_token or env var HF_TOKEN.")
    login(token)  # authenticates the session

    cfg = load_cfg(args.config, args.set)
    common = cfg.get("common", {})
    random.seed(args.seed); np.random.seed(args.seed)
    print(f"Running with {args.config} (seed={args.seed})")
    exp_cfg = cfg.get(args.exp, {})
    EXPERIMENTS[args.exp](exp_cfg, common)

if __name__ == "__main__":
    main()
