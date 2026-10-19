# jev27b-mlx

[autotrust/AutoTrust-JEV-27B-v1.1](https://huggingface.co/autotrust/AutoTrust-JEV-27B-v1.1) on Apple Silicon with [MLX](https://github.com/ml-explore/mlx): one-pass decisions with calibrated option probabilities, read the way the model was trained, with bidirectional attention over the prompt.

AutoTrust-JEV-27B v1.1 is a text decision model (Decision Index 0.3 public score 67.31 on its card), built on [vllm-sr/d3](https://huggingface.co/vllm-sr/d3) and Qwen3.8-27B. Each call is one question about a state: a `choice` among up to 255 options you supply, or a yes/no (`noul`). The answer is a probability for every option. It does not chat or generate text.

Published by [NetProjects](https://netprojects.com) (Internet Projects Ltd). Not affiliated with AutoTrust AI.

## Why a dedicated engine

The model was trained to read its whole prompt **bidirectionally** in its 16 full-attention layers, while its 48 Gated DeltaNet layers stay causal. autotrust serve it on vLLM with [StartLux's prefix-bidirectional plugin](https://github.com/StartLuxLabs/StartLux-Decision/blob/main/docs/vllm_bidirectional.md). A plain MLX conversion reads the prompt left to right and gives a different model without any error: in our test the same weights read causally changed 5 of 39 answers and moved probabilities by up to 0.72.

`jev27b_mlx` runs the layers itself, the full-attention ones without a mask, following the approach of StartLux-Decision's MLX backend. It uses the model's own prompt renderer (`jev_format.py`, shipped with the weights) and its read-out: the logits of the 255 answer-code tokens at the last position, in float32, divided by the temperature in `vllm_readout.json`. At load, a probe checks that a change of the last token reaches the first position.

## Builds

| Hugging Face | size | memory in use (longest prompt) | same answer as the original bf16 | probability drift, KL mean / p95 | median per decision |
|---|---|---|---|---|---|
| [netprojects/AutoTrust-JEV-27B-v1.1-mlx-8bit](https://huggingface.co/netprojects/AutoTrust-JEV-27B-v1.1-mlx-8bit) | 28 GB | 31 GB | 231 / 231 | 0.0002 / 0.0004 | 318 ms |
| [netprojects/AutoTrust-JEV-27B-v1.1-mlx-4bit](https://huggingface.co/netprojects/AutoTrust-JEV-27B-v1.1-mlx-4bit) | 16 GB | 18 GB | 227 / 231 | 0.004 / 0.019 | 285 ms |

The decision read-out rows in `lm_head` are kept at 8 bits in both builds. Pick by memory: 8-bit for 48 GB Macs and up, 4-bit for 24-32 GB.

## Install

```bash
pip install "git+https://github.com/internet-projects/jev27b-mlx"
```

Apple Silicon, Python 3.10+, `mlx`, `mlx-vlm` 0.7.6 or later and `transformers` (installed with the package).

## Use from Python

```python
import jev27b_mlx

jev = jev27b_mlx.load("netprojects/AutoTrust-JEV-27B-v1.1-mlx-4bit")

q = jev.question("choice", "Which team should handle this ticket?",
                 ["billing", "shipping", "technical support", "account security"])
r = jev.decide("Customer message: my card was charged twice for the same order.", q)
print(r["choice"], r["probabilities"])          # billing [0.99..., ...]

q = jev.question("noul", "Does the customer report damage?")
print(jev.decide("Order 1182 arrived with a cracked screen.", q)["probabilities"][1])   # P(true)
```

Options can carry descriptions, and questions can be written in the model's own format directly:
`{"type": "choice", "instructions": "...", "criteria": {"billing": "charges, invoices, refunds", ...}}`.
`state` is text or any JSON-serialisable object. Text only: the model is not used with images.

## Use as a server

```bash
python -m jev27b_mlx serve netprojects/AutoTrust-JEV-27B-v1.1-mlx-4bit --port 8000

curl localhost:8000/v1/decide -H 'Content-Type: application/json' -d '{
  "state": "Customer message: my card was charged twice for the same order.",
  "kind": "choice", "question": "Which team should handle this ticket?",
  "options": ["billing", "shipping", "technical support", "account security"]}'
```

The response has `options`, `probabilities`, `choice`, `choice_index`, `usage` and `latency_ms`. `question` may also be a full question object. Requests are handled one at a time.

## Verification

We checked the MLX engine against an independent PyTorch reference (`scripts/reference_torch.py`: transformers' own Qwen3.5 code, bf16 weights, attention in float32 with the mask removed in the full-attention layers only), on 39 of the 231 prompts:

| | same answer | KL mean / p95 | largest probability difference |
|---|---|---|---|
| MLX, original bf16 weights | 39 / 39 | 0.0001 / 0.0008 | 0.037 |
| MLX 8-bit | 39 / 39 | 0.0001 / 0.0007 | 0.031 |
| MLX 4-bit | 39 / 39 | 0.0035 / 0.027 | 0.112 |
| the original weights read causally (control) | 34 / 39 | 0.088 / 0.57 | 0.72 |

On all 231 JevBench public decisions the original weights on MLX answer 209 correctly (easy 48/48, standard 72/72, hard 89/111); the 8-bit build 209, the 4-bit build 210. Details: [`results/results.json`](results/results.json).

```bash
python scripts/jevbench_to_requests.py <autotrust folder> jevbench/datasets/public/{easy,original,hard}.jsonl > requests.jsonl
python scripts/run.py netprojects/AutoTrust-JEV-27B-v1.1-mlx-8bit requests.jsonl probs-8bit.json
python scripts/reference_torch.py <autotrust folder> requests.jsonl probs-torch.json mps
python scripts/compare.py probs-torch.json probs-8bit.json
python scripts/speed.py netprojects/AutoTrust-JEV-27B-v1.1-mlx-4bit requests.jsonl
```

Speed: one decision at a time, prompts of about 150 tokens, other models running on the same Mac. Prompts of 3,000 to 4,000 tokens take 3.5 to 6 s. The engine caps MLX's buffer cache at 1 GB and evaluates one layer at a time, so working memory stays at about 2.5 GB above the weights even on a 4,000-token prompt.

## Convert it yourself

```bash
python scripts/convert.py autotrust/AutoTrust-JEV-27B-v1.1 ./AutoTrust-JEV-27B-v1.1-mlx-8bit 8     # recipes: 8, 6, 4, 4m
```

The original folder also loads directly: `jev27b_mlx.load("<autotrust folder>")` runs it in bf16 (about 57 GB in use).

## More from NetProjects

- [JEV-35B on MLX](https://github.com/internet-projects/jev35b-mlx), autotrust's mixture-of-experts decision model, with vision
- [H2O-Lightning-4B on MLX](https://github.com/internet-projects/h2o-lightning-mlx)
- [NVIDIA Nemotron-Labs Dory 1.0 4B on MLX](https://github.com/internet-projects/dory-mlx)
- [NetProjects on Hugging Face](https://huggingface.co/netprojects)

## Credits and license

Model: [autotrust/AutoTrust-JEV-27B-v1.1](https://huggingface.co/autotrust/AutoTrust-JEV-27B-v1.1) by AutoTrust AI (Apache-2.0), built on [vllm-sr/d3](https://huggingface.co/vllm-sr/d3) by the vLLM Semantic Router team (Apache-2.0) and [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0). Bidirectional prompt reading: the rule and MLX approach of [StartLux-Decision](https://github.com/StartLuxLabs/StartLux-Decision) (Apache-2.0). Runtime: [mlx-vlm](https://github.com/Blaizzy/mlx-vlm). Code in this repository: Apache-2.0. Port by Paris Michalas, Partner, NetProjects (Internet Projects Ltd). Not affiliated with AutoTrust AI, the vLLM Semantic Router team or StartLux.
