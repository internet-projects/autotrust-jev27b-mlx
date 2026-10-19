"""python scripts/compare.py <reference probs.json> <probs.json> [...]: same top answer and KL(ref||x), over the ids both have."""
import json, math, sys
ref = json.load(open(sys.argv[1]))
for f in sys.argv[2:]:
    x = json.load(open(f)); ids = [k for k in ref if k in x]; same = 0; kls = []; dmax = 0
    for k in ids:
        a, b = ref[k], x[k]
        same += max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__)
        kls.append(sum(p * math.log(max(p, 1e-12) / max(q, 1e-12)) for p, q in zip(a, b)))
        dmax = max(dmax, max(abs(p - q) for p, q in zip(a, b)))
    kls.sort(); n = len(kls)
    print(f"{f.split('/')[-1]} vs {sys.argv[1].split('/')[-1]}: n={n} same top {same}/{n}, KL mean {sum(kls)/n:.4f} median {kls[n//2]:.4f} p95 {kls[int(n*.95)]:.4f}, max |dp| {dmax:.3f}")
