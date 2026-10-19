"""AutoTrust-JEV-27B v1.1 on MLX: one-pass decisions with bidirectional prompt attention.

Prompt    : the model's own jev_format.render (shipped with the weights): system turn, user turn with the state, the
            question and one coded line per option ("A: ...", "B: ..."), chat template with thinking off.
Attention : the model was trained to read the whole one-pass prompt bidirectionally in its 16 full-attention layers;
            the 48 Gated DeltaNet layers and their short convolutions stay causal (the rule of StartLux's
            prefix-bidirectional vLLM plugin, which autotrust's serve_vllm.sh enables). The layers are run here
            directly, the full-attention ones without a mask; approach adapted from StartLux-Decision's MLX backend
            (Apache-2.0). A probe at load checks that a change of the last token reaches the first position.
Read-out  : hidden state at the last position x lm_head rows of the 255 answer-code tokens, float32, divided by the
            temperature in vllm_readout.json, softmax over the codes of the question's options.
"""
from __future__ import annotations

import json, os, sys, time
from pathlib import Path

import mlx.core as mx
import numpy as np

CACHE_LIMIT = 1 << 30   # cap MLX's buffer cache: freed GPU buffers otherwise pile up across prompt lengths


def _local_path(path_or_repo: str) -> str:
    p = os.path.expanduser(path_or_repo)
    if os.path.isdir(p):
        return p
    from huggingface_hub import snapshot_download
    return snapshot_download(path_or_repo)


class Jev27:
    """Load once, then call decide(). Calls must be serialised (the server does)."""

    def __init__(self, path_or_repo: str, causal: bool = False, memory_limit_gb: float | None = None):
        from mlx_vlm.utils import load_model
        from transformers import AutoTokenizer
        mx.set_cache_limit(CACHE_LIMIT)
        if memory_limit_gb:
            mx.set_memory_limit(int(memory_limit_gb * 1e9))
        self.path, self.causal = _local_path(path_or_repo), causal
        t0 = time.perf_counter()
        self.model = load_model(Path(self.path))
        self.load_s = time.perf_counter() - t0
        self.lm = self.model.language_model
        self.body = self.lm.model
        self.layers = list(self.body.layers)
        self.tok = AutoTokenizer.from_pretrained(self.path)
        if self.path not in sys.path:
            sys.path.insert(0, self.path)
        import jev_format                                    # the model's prompt contract, shipped with the weights
        self.fmt = jev_format
        rd = json.load(open(os.path.join(self.path, "vllm_readout.json")))
        self.codes, self.code_ids = rd["codes"], rd["readout_token_ids"]
        self.temperature = float(rd.get("temperature", 1.0))
        codes, ids = jev_format.answer_codes(self.tok)
        if codes != self.codes or ids != self.code_ids:
            raise RuntimeError("the tokenizer's answer codes differ from vllm_readout.json")
        head = self.lm.lm_head
        idx = mx.array(self.code_ids)
        w = head.weight[idx]
        if hasattr(head, "scales"):
            b = head.biases[idx] if getattr(head, "biases", None) is not None else None
            w = mx.dequantize(w, head.scales[idx], b, head.group_size, head.bits, mode=getattr(head, "mode", "affine"))
        self.rows = w.astype(mx.float32)
        mx.eval(self.rows)
        self.last_peak_gb = 0.0
        self.check = self.mask_probe()

    # ------------------------------------------------------------------ forward
    def _forward(self, ids) -> mx.array:
        L = len(ids)
        h = self.body.embed_tokens(mx.array(np.asarray(ids, dtype=np.int32))[None])
        pos = mx.array(np.broadcast_to(np.arange(L, dtype=np.int32), (3, 1, L)).copy())
        first = next(l for l in self.layers if not l.is_linear)
        shared = None if first.self_attn.rotary_emb.fused_apply else first.self_attn.rotary_emb(h, pos)
        fa_mask = "causal" if self.causal else None
        for layer in self.layers:
            if layer.is_linear:
                h = layer(h, mask=None, cache=None)
            else:
                h = layer(h, mask=fa_mask, cache=None, position_ids=pos, position_embeddings=shared)
            mx.eval(h)                                        # one layer at a time keeps working memory small
        return self.body.norm(h)

    def mask_probe(self, n: int = 24) -> dict:
        rng = np.random.default_rng(1234)
        a = rng.integers(1000, 50000, size=n).astype(np.int32)
        b = a.copy(); b[-1] += 1
        h0, h1 = (self._forward(list(x))[0, 0].astype(mx.float32) for x in (a, b))
        reach = float(mx.abs(h0 - h1).max())
        mx.clear_cache()
        if (reach > 0) == self.causal:
            raise RuntimeError(f"prompt attention is not {'causal' if self.causal else 'bidirectional'} (reach {reach})")
        return {"attention": "causal" if self.causal else "bidirectional", "first_position_change": reach,
                "full_attention_layers": sum(not l.is_linear for l in self.layers), "layers": len(self.layers)}

    def _probs(self, prompt: str, n: int):
        ids = self.tok.encode(prompt, add_special_tokens=False)
        mx.reset_peak_memory()
        h = self._forward(ids)[0, -1].astype(mx.float32)
        p = mx.softmax((self.rows[:n] @ h) / self.temperature)
        mx.eval(p)
        self.last_peak_gb = mx.get_peak_memory() / 1e9
        out = p.tolist()
        del h, p
        mx.clear_cache()
        return out, len(ids)

    # ------------------------------------------------------------------ public
    @staticmethod
    def question(kind: str, question: str, options=None, descriptions=None) -> dict:
        """Build the model's question object: kind "choice" (options, optional descriptions) or "noul" (yes/no)."""
        if kind == "noul":
            return {"type": "noul", "instructions": question,
                    **({"criteria": dict(zip(["false", "true"], descriptions))} if descriptions else {})}
        if kind != "choice":
            raise ValueError("kind must be choice or noul")
        descriptions = descriptions or [None] * len(options or [])
        return {"type": "choice", "instructions": question, "criteria": dict(zip(options, descriptions))}

    def decide(self, state, question: dict) -> dict:
        """state: text or a JSON-serialisable object. question: {"type": "choice", "instructions": ...,
        "criteria": {option: description or None, ...}} or {"type": "noul", "instructions": ..., "criteria":
        {"false": ..., "true": ...} (optional)}. Returns options, probabilities, choice, choice_index, usage."""
        keys = self.fmt.options(question)[0]
        if not 1 <= len(keys) <= len(self.codes):
            raise ValueError(f"a question needs 1 to {len(self.codes)} options")
        p, n_tok = self._probs(self.fmt.render(self.tok, state, question, self.codes), len(keys))
        i = max(range(len(p)), key=p.__getitem__)
        return {"options": keys, "probabilities": p, "choice": keys[i], "choice_index": i,
                "usage": {"prompt_tokens": n_tok, "completion_tokens": 0}}


def load(path_or_repo: str, **kw) -> Jev27:
    """jev = jev27b_mlx.load("netprojects/AutoTrust-JEV-27B-v1.1-mlx-4bit")  (Hugging Face repo id or local folder)"""
    return Jev27(path_or_repo, **kw)
