"""Minimal local-LLM wrapper (Hugging Face transformers) used across labs."""
from __future__ import annotations

import re
import time

from .env import Runtime


class ChatModel:
    def __init__(self, model_id: str, rt: Runtime):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()
        dtype = torch.bfloat16 if rt.supports_bf16 else (torch.float16 if rt.device == "cuda" else torch.float32)
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(rt.device).eval()
        self.device = rt.device
        self.model_id = model_id
        print(f"Loaded {model_id} on {rt.device} ({dtype}) in {time.time() - t0:.0f}s")

    @classmethod
    def from_model(cls, model, tok, rt: Runtime) -> "ChatModel":
        """Wrap an already-loaded model (e.g. one you just fine-tuned)."""
        obj = cls.__new__(cls)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        obj.tok, obj.model, obj.device, obj.model_id = tok, model.eval(), rt.device, getattr(model, "name_or_path", "model")
        return obj

    def chat(self, prompts: list[str], system: str | None = None, max_new_tokens: int = 384,
             temperature: float = 0.7, batch_size: int = 8) -> list[str]:
        import torch

        outs: list[str] = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            texts = []
            for p in batch:
                msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": p}]
                texts.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(
                    **enc, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None, top_p=0.9 if temperature > 0 else None,
                    pad_token_id=self.tok.pad_token_id,
                )
            for row in gen[:, enc["input_ids"].shape[1]:]:
                outs.append(self.tok.decode(row, skip_special_tokens=True).strip())
            print(f"  generated {min(i + batch_size, len(prompts))}/{len(prompts)}")
        return outs

    def next_token_logprobs(self, texts: list[str], batch_size: int = 8):
        """Log-probabilities of the NEXT token after each raw text (no chat template).

        Returns a float32 CPU tensor of shape [len(texts), vocab_size].
        """
        import torch

        rows = []
        for i in range(0, len(texts), batch_size):
            enc = self.tok(texts[i:i + batch_size], return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits[:, -1, :]   # left padding -> last position is real
            rows.append(torch.log_softmax(logits.float(), dim=-1).cpu())
        return torch.cat(rows)

    def chat_next_token_logprobs(self, prompts: list[str], system: str | None = None, batch_size: int = 8):
        """Like next_token_logprobs, but wraps each prompt in the chat template first.

        Use it to read a model's first-token preference, e.g. "Yes" vs "No".
        """
        texts = []
        for p in prompts:
            msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": p}]
            texts.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        return self.next_token_logprobs(texts, batch_size=batch_size)

    def token_id(self, text: str) -> int:
        """Id of a string that must encode to exactly one token, e.g. ' A'."""
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"{text!r} is {len(ids)} tokens for this tokenizer")
        return ids[0]

    def unload(self) -> None:
        """Free GPU memory so another model can be loaded."""
        import gc
        import torch

        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


QA_RE = re.compile(r"Q\s*\d*\s*[:.)-]\s*(?P<q>.+?)\s*\n+\s*A\s*\d*\s*[:.)-]\s*(?P<a>.+?)(?=\n\s*Q\s*\d*\s*[:.)-]|\Z)", re.S | re.I)


def parse_qa(text: str) -> list[tuple[str, str]]:
    """Parse 'Q: ... / A: ...' blocks from a generation."""
    pairs = []
    for m in QA_RE.finditer(text):
        q = " ".join(m.group("q").split()).strip("* ")
        a = " ".join(m.group("a").split()).strip("* ")
        if q and a:
            pairs.append((q, a))
    if not pairs:
        # Fallback for small models that write "Question? Answer." on one line.
        for line in text.splitlines():
            line = re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", line).strip()
            if "?" in line:
                q, _, a = line.partition("?")
                if len(q.split()) >= 3 and len(a.split()) >= 2:
                    pairs.append((q.strip() + "?", a.strip()))
    return pairs


def perplexity(model, tok, texts: list[str], device: str, batch_size: int = 8) -> float:
    """Token-weighted perplexity of `model` on `texts` (lower = the text is less surprising)."""
    import math
    import torch

    total_nll, total_tokens = 0.0, 0
    old_side = tok.padding_side
    tok.padding_side = "right"
    model.eval()
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True).to(device)
        labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
        with torch.no_grad():
            logits = model(**enc).logits[:, :-1].float()
        tgt = labels[:, 1:]
        nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                                ignore_index=-100, reduction="sum")
        total_nll += nll.item()
        total_tokens += (tgt != -100).sum().item()
    tok.padding_side = old_side
    return math.exp(total_nll / max(total_tokens, 1))
