"""Render the JevBench public tasks (https://github.com/fstandhartinger/jevbench, datasets/public/*.jsonl) into this
model's prompts with its own jev_format.  python scripts/jevbench_to_requests.py <model folder> easy.jsonl ... > requests.jsonl
Score questions (a list of levels) become a choice over "0".."n-1"; noul answers yes/no map to true/false."""
import json, sys
from transformers import AutoTokenizer
path = sys.argv[1]; sys.path.insert(0, path)
import jev_format
tok = AutoTokenizer.from_pretrained(path)
codes = json.load(open(f"{path}/vllm_readout.json"))["codes"]
for f in sys.argv[2:]:
    for line in open(f):
        if not line.strip(): continue
        t = json.loads(line); q = t["question"]
        if q["type"] == "score":
            q = {"type": "choice", "instructions": q["instructions"], "criteria": {str(i): c for i, c in enumerate(q["criteria"])}}
            exp = str(t["labels"].index(t["expected"])) if t["expected"] in t["labels"] else str(t["expected"])
        else:
            q = {k: v for k, v in q.items() if k in ("type", "criteria", "instructions")}
            exp = {"yes": "true", "no": "false"}.get(t["expected"], t["expected"]) if q["type"] == "noul" else t["expected"]
        keys = jev_format.options(q)[0]; prompt = jev_format.render(tok, t["state"], q, codes)
        print(json.dumps({"id": t["id"], "prompt": prompt, "n": len(keys), "keys": keys, "expected": exp,
                          "tokens": len(tok.encode(prompt, add_special_tokens=False))}, ensure_ascii=False))
