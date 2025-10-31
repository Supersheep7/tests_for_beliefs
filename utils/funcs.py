import torch
import numpy as np
from cfg import load_cfg
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


