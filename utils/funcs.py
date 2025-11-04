import sys
from pathlib import Path
import torch
import einops
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from src.cfg import load_cfg
from datetime import datetime
from pathlib import Path
cfg = load_cfg()

def force_format(*items, format='tensor', device=cfg.device):
    
    results = []

    for item in items:
        # Tensor format
        if format == 'tensor':
            if not isinstance(item, torch.Tensor):
                item = torch.tensor(item)
            if device is not None:
                item = item.to(device)
            item = item.half()

        # NumPy format
        elif format == 'numpy':
            if isinstance(item, torch.Tensor):
                item = item.detach().cpu().numpy()
            if not isinstance(item, np.ndarray):
                item = np.array(item)
            item = item.astype(np.float16)

        else:
            raise ValueError(f"Unsupported format: {format}")

        results.append(item)

    return results if len(results) > 1 else results[0]

def decompose_mha(mha_batch):

    decomposed = einops.rearrange(mha_batch, 'n_batch batch_size n_head d_head -> n_head n_batch batch_size d_head')
    
    return [decomposed[i] for i in range(decomposed.shape[0])]

def save_results(item, datatype, modality='residual'):

    base_dir = ROOT / "results"
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    filename_map = {
        "accuracies": f"accuracies_{modality}.pkl",
        "directions": f"directions_{modality}.pkl",
        "probes": f"probes_{modality}.pkl",
        "intervention_scores": f"intervention_scores_{modality}.pkl",
        "coherence_scores": f"coherence_scores.pkl",
    }

    if datatype not in filename_map:
        raise ValueError(f"Unsupported datatype: {datatype}")

    if datatype == "intervention_scores":
        path = base_dir / cfg["common"]["model"] / cfg["probe"]["probe_type"] / "k"+cfg["intervention"]["k"] / "alpha"+cfg["intervention"]["alpha"] / filename_map[datatype]
    elif datatype == "coherence_scores":
        path = base_dir / cfg["common"]["model"] / cfg["coherence"]["estimator"] / filename_map[datatype]
    else:
        path = base_dir / cfg["common"]["model"] / cfg["probe"]["probe_type"] / filename_map[datatype]

    # If file already exists, append a timestamp to avoid clobbering
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = base_dir / f"{path.stem}_{ts}{path.suffix}"

    # Move tensors to cpu before saving to avoid device-specific state
    try:
        if isinstance(item, torch.Tensor):
            item_to_save = item.detach().cpu()
        else:
            item_to_save = item
    except Exception:
        item_to_save = item

    torch.save(item_to_save, path)
    return path

