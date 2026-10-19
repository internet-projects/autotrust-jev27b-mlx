"""All requests through an MLX build: python scripts/run.py <model> requests.jsonl out.json
Prints accuracy against "expected", median latency and the largest peak memory (per prompt length)."""
import json, statistics, sys, time
from jev27b_mlx import Jev27
model, req_file, out_file = sys.argv[1:4]
j = Jev27(model)
reqs = sorted((json.loads(l) for l in open(req_file) if l.strip()), key=lambda r: r["tokens"])
out, lat, ok, peak = {}, [], 0, (0, 0)
for r in reqs:
    t0 = time.perf_counter(); p, _ = j._probs(r["prompt"], r["n"]); lat.append(time.perf_counter() - t0)
    out[r["id"]] = p; ok += r["keys"][max(range(len(p)), key=p.__getitem__)] == r.get("expected")
    peak = max(peak, (j.last_peak_gb, r["tokens"]))
json.dump(out, open(out_file, "w"))
print(f"{model}: {ok}/{len(reqs)} match expected; median {statistics.median(lat)*1000:.0f} ms; "
      f"peak memory {peak[0]:.1f} GB (at a {peak[1]}-token prompt); prompt attention {j.check['attention']}")
