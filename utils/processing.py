import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from src.cfg import load_cfg
from transformer_lens import HookedTransformer
import torch as t
cfg = load_cfg()

def get_data():
    return None

def get_activations(model: HookedTransformer, data, modality: str = 'residual'):
    return None