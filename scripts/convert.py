"""AutoTrust-JEV-27B v1.1 -> MLX (mlx-vlm).  python scripts/convert.py <autotrust folder or repo> <out> <recipe>
recipes: 8 (all 8-bit) | 6 | 4 (all 4-bit) | 4m (MLP 4-bit, attention + linear attention + embeddings 8-bit)
Always: lm_head (the decision read-out rows) 8-bit; vision tower untouched (bf16; the model is used for text only).
The decision files (vllm_readout.json, decision_config.json, jev_format.py) and the license files are copied along."""
import os, shutil, sys
from mlx_vlm.convert import convert
from mlx_vlm.utils import skip_multimodal_module
src, out, recipe = sys.argv[1], sys.argv[2], sys.argv[3]
if not os.path.isdir(src):
    from huggingface_hub import snapshot_download
    src = snapshot_download(src, ignore_patterns=["assets/*"])
mlp_only = recipe.endswith("m"); bits = int(recipe.rstrip("m"))

def predicate(path, module):
    if skip_multimodal_module(path) or not hasattr(module, "to_quantized"):
        return False
    if hasattr(module, "weight") and module.weight.shape[-1] % 64 != 0:
        return False
    if path.endswith("lm_head"):
        return {"group_size": 64, "bits": 8}
    if mlp_only:
        return {"group_size": 64, "bits": bits if ".mlp." in path else 8}
    return {"group_size": 64, "bits": bits}

convert(src, out, quantize=True, q_bits=bits, q_group_size=64, quant_predicate=predicate)
for f in ("vllm_readout.json", "decision_config.json", "jev_format.py", "LICENSE", "NOTICE.base", "LICENSE.base-apache-2.0"):
    if os.path.exists(os.path.join(src, f)):
        shutil.copy(os.path.join(src, f), os.path.join(out, f))
print("converted", out)
