"""Independent reference: transformers' Qwen3.5 implementation, bf16 weights, attention computed in float32, with the
full-attention layers' causal mask removed through a custom attention function (Gated DeltaNet layers untouched).

  python scripts/reference_torch.py <autotrust folder> requests.jsonl out.json [mps|cpu] [--causal]

Each request line: {"id", "prompt", "n"} (from jevbench_to_requests.py). Output: {id: probabilities}.
Needs torch and transformers >= 5.17; holds the bf16 weights (about 55 GB) in memory.
"""
import json, math, os, sys, time
import torch
import torch.nn.functional as F
from transformers import AttentionInterface, AutoConfig, AutoModelForImageTextToText, AutoTokenizer

path, req_file, out_file = sys.argv[1], sys.argv[2], sys.argv[3]
device = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else "mps"
causal = "--causal" in sys.argv
MAX_S = float(os.environ.get("REF_MAX_S", "2400"))
CALLS = {"n": 0}


def bidir_attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kw):
    # query [B, H, L, D], key/value [B, Hkv, L, D]; single unpadded row -> no mask at all
    CALLS["n"] += 1
    q, k, v = query.float(), key.float(), value.float()
    rep = q.shape[1] // k.shape[1]
    if rep > 1:
        k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scaling)
    return out.to(query.dtype).transpose(1, 2).contiguous(), None


AttentionInterface.register("bidir_ref", bidir_attention)
tok = AutoTokenizer.from_pretrained(path)
t0 = time.time()
model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="bidir_ref", device_map=device)
model.eval()
print(f"loaded in {time.time()-t0:.0f}s on {device}; causal={causal}; {type(model).__name__}", flush=True)
rd = json.load(open(os.path.join(path, "vllm_readout.json")))
ids_all, T = rd["readout_token_ids"], float(rd.get("temperature", 1.0))
W = model.lm_head.weight[torch.tensor(ids_all, device=device)].float()
body = model.model.language_model

out = json.load(open(out_file)) if os.path.exists(out_file) else {}
reqs = [json.loads(l) for l in open(req_file) if l.strip()]
for i, r in enumerate(reqs):
    if r["id"] in out:
        continue
    if time.time() - t0 > MAX_S:
        print(f"time limit reached after {len(out)} prompts"); break
    ids = tok.encode(r["prompt"], add_special_tokens=False)
    with torch.no_grad():
        CALLS["n"] = 0
        h = body(input_ids=torch.tensor([ids], device=device)).last_hidden_state[0, -1].float()
        assert CALLS["n"] == 16, f"custom attention ran {CALLS['n']} times, expected 16 full-attention layers"
        z = (W[: r["n"]] @ h) / T
        p = torch.softmax(z, 0).tolist()
    out[r["id"]] = p
    if device == "mps": torch.mps.empty_cache()
    json.dump(out, open(out_file, "w"))
    print(f"{i+1}/{len(reqs)} {r['id']} tokens={len(ids)} top={max(range(len(p)), key=p.__getitem__)} {time.time()-t0:.0f}s", flush=True)
print("done")
