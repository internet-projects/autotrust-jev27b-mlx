"""python scripts/speed.py <model> requests.jsonl [n=120]: latency per decision, one at a time, two passes."""
import json, statistics, sys, time
from jev27b_mlx import Jev27
j = Jev27(sys.argv[1]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 120
reqs = [json.loads(l) for l in open(sys.argv[2]) if l.strip()][: n + 3]
for r in reqs[:3]: j._probs(r["prompt"], r["n"])
for p in (1, 2):
    lat = []
    for r in reqs[3:]:
        t0 = time.perf_counter(); j._probs(r["prompt"], r["n"]); lat.append(time.perf_counter() - t0)
    lat.sort(); toks = sorted(r["tokens"] for r in reqs[3:])
    print(f"pass {p}: n={len(lat)} median {statistics.median(lat)*1000:.0f} ms, p95 {lat[int(len(lat)*.95)]*1000:.0f} ms (median prompt {toks[len(toks)//2]} tokens)")
