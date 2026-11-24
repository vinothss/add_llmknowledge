"""Reference solutions for Module 4 (quantization, depth pruning, distillation). Try the exercises first!"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F


# ------------------------------------------------------------ Exercise 1: quantization
def quantize(w: torch.Tensor, bits: int = 8, group_size: int | None = None):
    """Symmetric absmax quantization.

    group_size=None -> one scale per output row (per-channel)
    group_size=g    -> one scale per block of g consecutive weights in each row (per-group)
    Returns (q, scale) with q an int8 tensor in [-qmax, qmax].
    """
    qmax = 2 ** (bits - 1) - 1
    rows, cols = w.shape
    g = cols if group_size is None else group_size
    assert cols % g == 0, "row length must be divisible by group_size"
    wg = w.float().reshape(rows, cols // g, g)
    scale = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.round(wg / scale).clamp(-qmax, qmax).to(torch.int8)
    return q.reshape(rows, cols), scale


def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, cols = q.shape
    n_groups = scale.shape[1]
    return (q.float().reshape(rows, n_groups, cols // n_groups) * scale).reshape(rows, cols)


def e4m3_round(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest FP8 E4M3 value (4 exponent bits, 3 mantissa bits, max 448).

    Pure arithmetic, so it runs on any device. Matches torch's float8_e4m3fn cast exactly.
    """
    a = x.abs().clamp(max=448.0)
    e = torch.floor(torch.log2(a.clamp(min=2 ** -9))).clamp(min=-6)   # -6 = smallest normal exponent
    step = torch.pow(2.0, e - 3)                                         # 3 mantissa bits -> 8 steps per power of 2
    return torch.sign(x) * (torch.round(a / step) * step).clamp(max=448.0)


def fp8_roundtrip(w: torch.Tensor) -> torch.Tensor:
    """Simulate FP8 (E4M3) weights with a per-row scale."""
    w32 = w.detach().float()
    scale = w32.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 448.0
    return (e4m3_round(w32 / scale) * scale).to(w.dtype)


@torch.no_grad()
def fake_quantize_model(model, scheme: str):
    """Replace every transformer Linear weight with its quantize->dequantize version.

    'Fake' (simulated) quantization measures the ACCURACY effect of a format on any
    hardware. Real memory/speed gains need kernels that compute in that format.
    scheme: 'int8' | 'int4-channel' | 'int4-g128' | 'fp8'
    """
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
            w = mod.weight.data
            if scheme == "fp8":
                mod.weight.data = fp8_roundtrip(w)
                continue
            bits, group = {"int8": (8, None), "int4-channel": (4, None), "int4-g128": (4, 128)}[scheme]
            q, s = quantize(w, bits, group)
            mod.weight.data = dequantize(q, s).to(w.dtype)
    return model


BITS_PER_WEIGHT = {"fp32": 32, "fp16": 16, "int8": 8 + 32 / 896, "fp8": 8 + 32 / 896,
                   "int4-channel": 4 + 32 / 896, "int4-g128": 4 + 32 / 128}


def linear_weight_count(model) -> int:
    return sum(m.weight.numel() for n, m in model.named_modules()
               if isinstance(m, torch.nn.Linear) and "lm_head" not in n)


# ------------------------------------------------------------ Exercise 2: depth pruning
@torch.no_grad()
def layer_importance(model, tok, texts: list[str], device: str, max_length: int = 256) -> list[float]:
    """importance_l = 1 - mean cosine(h_in, h_out) over real tokens, for each decoder layer."""
    model.eval()
    sims, count = None, 0
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states   # L+1 tensors [1, T, H]
        cos = torch.stack([F.cosine_similarity(hs[i][0].float(), hs[i + 1][0].float(), dim=-1).mean()
                           for i in range(len(hs) - 1)])
        sims = cos if sims is None else sims + cos
        count += 1
    return (1 - sims / count).tolist()


def prune_layers(model, drop: list[int]):
    """Remove decoder layers in-place and fix the bookkeeping the KV cache relies on."""
    keep = [i for i in range(len(model.model.layers)) if i not in set(drop)]
    model.model.layers = torch.nn.ModuleList([model.model.layers[i] for i in keep])
    for new_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, "self_attn"):
            layer.self_attn.layer_idx = new_idx
    model.config.num_hidden_layers = len(keep)
    if getattr(model.config, "layer_types", None):
        model.config.layer_types = [model.config.layer_types[i] for i in keep]
    return model


def choose_layers_to_drop(importance: list[float], n_drop: int, protect_ends: bool = True) -> list[int]:
    candidates = list(range(len(importance)))
    if protect_ends:   # first and last layers are almost always critical
        candidates = candidates[1:-1]
    return sorted(sorted(candidates, key=lambda i: importance[i])[:n_drop])


# ------------------------------------------------------------ Exercise 3: distillation
def distill_loss(student_logits, teacher_logits, labels, T: float = 2.0, alpha: float = 0.5):
    """alpha * T^2 * KL(teacher || student) on softened distributions + (1 - alpha) * CE on labels.

    Logits: [B, S, V] already aligned so position t predicts token t+1. labels: [B, S] with -100 = ignore.
    """
    mask = labels != -100
    s = student_logits[mask].float()
    t = teacher_logits[mask].float()
    kl = F.kl_div(F.log_softmax(s / T, dim=-1), F.log_softmax(t / T, dim=-1),
                  log_target=True, reduction="batchmean") * T * T
    ce = F.cross_entropy(s, labels[mask])
    return alpha * kl + (1 - alpha) * ce


def distill(student, teacher, tok, texts: list[str], rt, steps: int = 200, lr: float = 5e-5,
            batch_size: int = 4, max_length: int = 192, T: float = 2.0, alpha: float = 0.5,
            loss_fn=distill_loss, log_every: int = 20) -> list[float]:
    """Plain PyTorch distillation loop (full fine-tuning of the student).

    Memory note: student + teacher + two [batch, seq, 151k-vocab] logit tensors. Batch 4 x 192 tokens
    keeps this inside a 16 GB T4; on NVIDIA GPUs 8-bit Adam saves a further ~2 GB.
    """
    teacher.eval()
    student.train()
    opt = None
    if rt.device == "cuda":
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(student.parameters(), lr=lr, weight_decay=0.0)
        except ImportError:
            pass
    opt = opt or torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0)
    use_amp = rt.device == "cuda"
    amp_dtype = torch.bfloat16 if rt.supports_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    tok.padding_side = "right"
    losses, g = [], torch.Generator().manual_seed(0)
    for step in range(steps):
        idx = torch.randint(len(texts), (batch_size,), generator=g).tolist()
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length).to(rt.device)
        labels = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)[:, 1:]
        with torch.autocast(rt.device, dtype=amp_dtype, enabled=use_amp):
            with torch.no_grad():
                t_logits = teacher(**enc).logits[:, :-1]
            s_logits = student(**enc).logits[:, :-1]
            loss = loss_fn(s_logits, t_logits, labels, T=T, alpha=alpha)
            del t_logits
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
        if (step + 1) % log_every == 0:
            print(f"  step {step + 1}/{steps}  loss {sum(losses[-log_every:]) / log_every:.3f}")
    student.eval()
    tok.padding_side = "left"
    return losses


# ------------------------------------------------------------ benchmarking
@torch.no_grad()
def tokens_per_second(model, tok, device: str, n_new: int = 64, prompt: str = "Explain how a home battery works.") -> float:
    enc = tok(prompt, return_tensors="pt").to(device)
    model.generate(**enc, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)   # warm-up
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    out = model.generate(**enc, max_new_tokens=n_new, min_new_tokens=n_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    if device == "cuda":
        torch.cuda.synchronize()
    return (out.shape[1] - enc["input_ids"].shape[1]) / (time.time() - t)
