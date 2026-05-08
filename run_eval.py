# /data/maxshen/run_eval.py
import lerobot.policies.pretrained as _lp
from safetensors.torch import load_file as _load_file

_orig = _lp.load_model_as_safetensor


def _patched(model, filename, strict=True, device="cpu"):
    state_dict = _load_file(filename, device=device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return list(missing), list(unexpected)


_lp.load_model_as_safetensor = _patched

from lerobot.scripts.lerobot_eval import main

main()
