"""Generate notebooks/04_optimization.ipynb from readable Python source.

Edit the cells here, then run:  python scripts/make_m04_notebook.py
"""
from pathlib import Path

import nbformat as nbf

REPO_SLUG = "vinothss/add_llmknowledge"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells += [md(f"""# Module 4 - Optimising LLMs for deployment

<a href="https://colab.research.google.com/github/{REPO_SLUG}/blob/main/notebooks/04_optimization.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Scenario.** Lumora's assistant works, and now finance wants it cheaper and faster. You'll try the
three main compression techniques on the student model and measure what each one costs in accuracy:

| Part | Technique | Question |
|---|---|---|
| 1 | **Quantization** (INT8, FP8, INT4) | How few bits per weight can we get away with? |
| 2 | **Depth pruning** | Which layers can we delete? |
| 3 | **Knowledge distillation** | Can the original model repair the pruned one? |
| 4 | **Trade-off analysis** | Which variant should we ship? |

We measure **perplexity** on held-out Wikipedia text (general language ability, lower is better),
an **MMLU** slice (knowledge), **tokens/second**, and **size**.

| | Free track (T4) | Pro bonus (L4 / A100) |
|---|---|---|
| Model | Qwen2.5-0.5B-Instruct | Qwen2.5-1.5B-Instruct |
| Extra | - | Real FP8 hardware on L4 (see the end) |
| Time | ~35 min | ~45 min |

> Runtime -> Change runtime type -> **T4 GPU**. Run the check cells until every line says PASS.
> Stuck? Set `USE_SOLUTIONS = True`."""),
]

cells += [md("## 0. Setup"), code(f"""import os, sys
REPO_URL = "https://github.com/{REPO_SLUG}"
USE_DRIVE = False
USE_SOLUTIONS = os.environ.get("LAB_USE_SOLUTIONS") == "1"

if "google.colab" in sys.modules:
    if not os.path.exists("/content/llm-knowledge-workshop"):
        !git clone -q {{REPO_URL}} /content/llm-knowledge-workshop
    %cd /content/llm-knowledge-workshop
    !pip install -q -r requirements/m04.txt
else:
    if os.path.basename(os.getcwd()) == "notebooks":
        os.chdir("..")
    sys.path.insert(0, os.getcwd())"""),
    code("""import gc, json, time, warnings
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import mlflow
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import labkit
from labkit import checks, progress
from labkit.llm import ChatModel, perplexity
from labkit.solutions.m02 import build_prompt, predict_letters, LETTERS
from labkit.solutions.m04 import fp8_roundtrip, tokens_per_second, prune_layers, distill, linear_weight_count

warnings.filterwarnings("ignore")
rt = labkit.env.setup(use_drive=USE_DRIVE)
OUT = rt.workdir / "m04"; OUT.mkdir(parents=True, exist_ok=True)
SMOKE = os.environ.get("LAB_SMOKE") == "1"
STRICT = not SMOKE
MODEL_ID = rt.model("student")
print("Model:", MODEL_ID)

def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

tok = AutoTokenizer.from_pretrained(MODEL_ID)
def load_model():
    return AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(rt.device).eval()"""),
    code("""wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
def paragraphs(split, n, min_words=40):
    return [t.strip() for t in wt[split]["text"] if len(t.split()) >= min_words and not t.strip().startswith("=")][:n]
TEST_TEXT = paragraphs("test", 20 if SMOKE else 60)        # held-out: perplexity
TRAIN_TEXT = paragraphs("train", 2000)                      # distillation data

MMLU = {s: load_dataset("cais/mmlu", s) for s in ["astronomy", "marketing"]}
MMLU = {s: {"test": list(d["test"].select(range(8 if SMOKE else 25))), "dev": list(d["dev"])} for s, d in MMLU.items()}

RESULTS = {}
def evaluate(name, model, bits_per_weight=32.0, speed=True):
    cm = ChatModel.from_model(model, tok, rt)
    ppl = perplexity(model, tok, TEST_TEXT, rt.device)
    prompts = [build_prompt(e, d["dev"], s) for s, d in MMLU.items() for e in d["test"]]
    gold = [LETTERS[e["answer"]] for d in MMLU.values() for e in d["test"]]
    mmlu = sum(p == g for p, g in zip(predict_letters(cm, prompts), gold)) / len(gold)
    n_params = sum(p.numel() for p in model.parameters())
    n_linear = linear_weight_count(model)
    # size if the transformer Linear weights were stored at `bits_per_weight` (embeddings stay 16-bit)
    size_gb = (n_linear * bits_per_weight + (n_params - n_linear) * 16) / 8 / 1e9
    tps = tokens_per_second(model, tok, rt.device) if speed else float("nan")
    RESULTS[name] = {"ppl": ppl, "mmlu": mmlu, "params_M": n_params / 1e6, "size_GB": size_gb, "tok_s": tps}
    print(f"[{name:14s}] ppl={ppl:6.2f}  mmlu={mmlu:.0%}  params={n_params/1e6:.0f}M  size~{size_gb:.2f} GB  {tps:.1f} tok/s")

model = load_model()
evaluate("fp16 (base)", model, bits_per_weight=16)
ORIGINAL = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}   # CPU copy, to undo fake quantization""")]

cells += [md("""## 1. Exercise 1 - Quantization

Implement **symmetric absmax quantization** for a weight matrix `w` (shape `[rows, cols]`):

$$s = \\frac{\\max |w|}{q_{max}}, \\quad q = \\mathrm{round}(w / s) \\in [-q_{max}, q_{max}], \\quad \\hat w = s \\cdot q, \\qquad q_{max} = 2^{bits-1} - 1$$

* `group_size=None`: one scale per **row** (per-channel)
* `group_size=g`: one scale per block of `g` consecutive weights in each row (**per-group**)

Hint: reshape to `[rows, cols // g, g]` and take `amax` over the last dimension."""),
    code("""def quantize(w: torch.Tensor, bits: int = 8, group_size: int = None):
    qmax = 2 ** (bits - 1) - 1
    rows, cols = w.shape
    g = cols if group_size is None else group_size
    wg = w.float().reshape(rows, cols // g, g)
    scale = torch.ones(rows, cols // g, 1)            # TODO: max |w| per group / qmax (clamp to >= 1e-8)
    q = wg.to(torch.int8)                              # TODO: round(wg / scale), clamp to [-qmax, qmax], int8
    return q.reshape(rows, cols), scale

def dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, cols = q.shape
    n_groups = scale.shape[1]
    # TODO: multiply each group by its scale and reshape back to [rows, cols]
    return q.float()

if USE_SOLUTIONS:
    from labkit.solutions.m04 import quantize, dequantize"""),
    code("""checks.check_quantizer(quantize, dequantize)"""),
    md("""### Look at one real weight matrix

Before quantizing a whole model, look at a single layer. LLM weights are roughly bell-shaped,
but a few **outliers** stretch the range. With one scale per row, those outliers force a coarse grid
on everything else, and that's why per-group scales matter so much at 4 bits."""),
    code("""w = model.model.layers[5].mlp.down_proj.weight.detach().float().cpu()
errs = {}
for name, (bits, g) in {"INT8 per-channel": (8, None), "INT4 per-channel": (4, None), "INT4 group-128": (4, 128)}.items():
    q, s = quantize(w, bits, g)
    errs[name] = ((dequantize(q, s) - w).norm() / w.norm()).item()
errs["FP8 E4M3 per-channel"] = ((fp8_roundtrip(w) - w).norm() / w.norm()).item()

fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
ax[0].hist(w.flatten().numpy(), bins=200, color="#7e57c2"); ax[0].set_yscale("log")
ax[0].set_title(f"Layer 5 down_proj weights (max |w| = {w.abs().max():.3f}, std = {w.std():.4f})")
ax[1].barh(list(errs), [v * 100 for v in errs.values()], color="#ffb300"); ax[1].set_xlabel("relative error %")
ax[1].set_title("Quantization error on this matrix"); plt.tight_layout(); plt.show()"""),
    md("""### Quantize the whole model

We apply each scheme to every transformer `Linear` layer and measure the result. This is
**simulated ("fake") quantization**: weights are quantized and immediately de-quantized, so we see
the *accuracy* effect on any GPU. Real memory and speed gains need kernels that compute directly in
the low-precision format (bitsandbytes, TensorRT-LLM). See the end of this section."""),
    code("""SCHEMES = {"int8": (8, None, 8 + 32 / 896), "int4-channel": (4, None, 4 + 32 / 896), "int4-g128": (4, 128, 4 + 32 / 128)}

@torch.no_grad()
def fake_quantize(model, scheme):
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and "lm_head" not in name:
            w = mod.weight.data
            if scheme == "fp8":
                mod.weight.data = fp8_roundtrip(w)
            else:
                bits, g, _ = SCHEMES[scheme]
                q, s = quantize(w, bits, g)
                mod.weight.data = dequantize(q, s).to(w.dtype)

ppl_by_scheme = {"fp16": RESULTS["fp16 (base)"]["ppl"]}
for scheme, bpw in [("int8", SCHEMES["int8"][2]), ("fp8", 8.25), ("int4-channel", SCHEMES["int4-channel"][2]),
                    ("int4-g128", SCHEMES["int4-g128"][2])]:
    fake_quantize(model, scheme)
    evaluate(scheme, model, bits_per_weight=bpw, speed=False)
    ppl_by_scheme[scheme] = RESULTS[scheme]["ppl"]
    model.load_state_dict(ORIGINAL); free()
checks.check_quant_results(ppl_by_scheme)"""),
    md("""**Read the table.** In our test run: FP16 24.1, INT8 24.0, FP8 24.2, INT4 per-channel 60.9,
INT4 group-128 35.5. Eight bits are essentially free. At four bits, one scale per row destroys the
model, while a scale per 128 weights recovers most of it. That's why GPTQ, AWQ and NF4 all use small
groups. (FP8 shows "speed = nan" because simulation doesn't make anything faster.)

**Real low-bit inference (NVIDIA GPUs).** The cell below loads the model with `bitsandbytes` in 8-bit
and 4-bit (NF4) and measures *actual* memory. Notice that tokens/second often goes **down**: these
kernels de-quantize on the fly, which saves memory but adds work at batch size 1. Production serving
stacks such as TensorRT-LLM use fused low-precision kernels to turn the smaller weights into speed."""),
    code("""if rt.device == "cuda":
    from transformers import BitsAndBytesConfig
    for label, cfg in [("bnb-int8", BitsAndBytesConfig(load_in_8bit=True)),
                       ("bnb-nf4", BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                      bnb_4bit_compute_dtype=torch.float16))]:
        try:
            qm = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=cfg, device_map={"": 0})
            print(f"{label}: memory {qm.get_memory_footprint() / 1e9:.2f} GB, "
                  f"{tokens_per_second(qm, tok, 'cuda'):.1f} tok/s")
            del qm; free()
        except Exception as e:
            print(f"{label}: skipped ({type(e).__name__}: {e})")
else:
    print("bitsandbytes needs an NVIDIA GPU - skipped on", rt.device)""")]

cells += [md("""## 2. Exercise 2 - Depth pruning

Each decoder layer *adds* a correction to the residual stream. If a layer's output hidden state is
almost the same as its input, the layer isn't doing much:

$$\\text{importance}_\\ell = 1 - \\overline{\\cos\\big(h_\\ell^{in},\\, h_\\ell^{out}\\big)}$$

`model(..., output_hidden_states=True).hidden_states` returns `L + 1` tensors: the embeddings, then
the output of each layer. So layer `i` maps `hidden_states[i]` to `hidden_states[i + 1]`.

Implement the importance score and the selection rule. Never drop the **first or last** layer:
they translate between tokens and the model's internal representation."""),
    code("""@torch.no_grad()
def layer_importance(model, tok, texts, device, max_length=256):
    sims, count = None, 0
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        # TODO: for each layer i, mean cosine similarity between hs[i][0] and hs[i+1][0] over tokens (dim=-1)
        cos = torch.zeros(len(hs) - 1)
        sims = cos if sims is None else sims + cos
        count += 1
    return (1 - sims / count).tolist()

def choose_layers_to_drop(importance, n_drop):
    # TODO: among layers 1 .. L-2, return the n_drop with the LOWEST importance (sorted ascending by index)
    return list(range(1, n_drop + 1))

if USE_SOLUTIONS:
    from labkit.solutions.m04 import layer_importance, choose_layers_to_drop"""),
    code("""N_LAYERS = model.config.num_hidden_layers
N_DROP = N_LAYERS // 4                                   # remove 25% of layers
importance = layer_importance(model, tok, TEST_TEXT[:32], rt.device)
drop = choose_layers_to_drop(importance, N_DROP)

plt.figure(figsize=(11, 3))
plt.bar(range(N_LAYERS), importance, color=["#e53935" if i in drop else "#7e57c2" for i in range(N_LAYERS)])
plt.xlabel("layer"); plt.ylabel("1 - cos(in, out)"); plt.yscale("log")
plt.title(f"Layer importance (red = dropped: {drop})"); plt.show()

pruned = prune_layers(load_model(), drop)    # fresh copy; prune_layers also fixes the KV-cache bookkeeping
checks.check_pruning(importance, drop, pruned, N_LAYERS, N_DROP)
evaluate(f"pruned -{N_DROP}L", pruned, bits_per_weight=16)"""),
    md("""**Read the plot.** In our test run the first layer scored about 0.97 and the last about 0.61,
while the middle layers scored 0.04-0.14. The middle of the network makes small refinements, so
that's where we cut. Removing 6 of 24 layers raised perplexity from about 24 to about 76, and MMLU
fell from 68% to about 24%, which is chance level for 4-option questions. The model is about 18%
smaller (much of a 0.5B model is its embedding table, which pruning layers doesn't touch) and
generates noticeably faster.""")]

cells += [md("""## 3. Exercise 3 - Heal with knowledge distillation

The unpruned model becomes the **teacher**; the pruned model is the **student**. At every token the
student learns to match the teacher's *whole probability distribution*, softened with a
temperature $T$, plus the usual next-token loss:

$$\\mathcal{L} = \\alpha\\, T^2\\, \\mathrm{KL}\\big(\\mathrm{softmax}(z_t/T)\\,\\|\\,\\mathrm{softmax}(z_s/T)\\big) + (1-\\alpha)\\,\\mathrm{CE}(z_s, y)$$

Implement it. Logits are already aligned (position $t$ predicts token $t+1$); ignore positions where
`labels == -100` (padding).

Hint: `F.kl_div(F.log_softmax(s / T, -1), F.log_softmax(t / T, -1), log_target=True, reduction="batchmean")`."""),
    code("""def distill_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    mask = labels != -100
    s = student_logits[mask].float()
    t = teacher_logits[mask].float()
    ce = F.cross_entropy(s, labels[mask])
    kl = torch.zeros((), device=s.device)   # TODO: temperature-softened KL(teacher || student), times T*T
    return alpha * kl + (1 - alpha) * ce

if USE_SOLUTIONS:
    from labkit.solutions.m04 import distill_loss"""),
    code("""checks.check_distill_loss(distill_loss)"""),
    code("""STEPS = 10 if SMOKE else 200
# Memory knobs: batch 4 x 192 tokens fits a 16 GB T4. Out of memory? Halve DISTILL_BATCH.
DISTILL_BATCH = int(os.environ.get("LAB_DISTILL_BATCH", 4))
MAX_LEN = int(os.environ.get("LAB_DISTILL_LEN", 192))
del ORIGINAL; free()                                 # no longer needed
teacher = model.half()                               # original model; 16-bit is plenty for forward passes
t = time.time()
losses = distill(pruned, teacher, tok, TRAIN_TEXT, rt, steps=STEPS, lr=5e-5, T=2.0, alpha=0.5,
                 batch_size=DISTILL_BATCH, max_length=MAX_LEN,
                 loss_fn=distill_loss, log_every=max(STEPS // 8, 1))
print(f"Distillation: {STEPS} steps in {time.time() - t:.0f}s")
plt.figure(figsize=(8, 2.5)); plt.plot(pd.Series(losses).rolling(10, min_periods=1).mean())
plt.title("Distillation loss"); plt.xlabel("step"); plt.show()
free()
evaluate("pruned+distilled", pruned, bits_per_weight=16)
checks.check_recovery(RESULTS["fp16 (base)"]["ppl"], RESULTS[f"pruned -{N_DROP}L"]["ppl"],
                      RESULTS["pruned+distilled"]["ppl"], strict=STRICT)"""),
    md("""**Read the result carefully.** In our test run, distillation brought perplexity from about
76 back to about 43, recovering **64% of the damage** in two minutes. But MMLU did **not** come back:
it stayed at chance level (24% pruned, 12% healed; on 50 questions both are within noise of 25%).
Fluency returns quickly because the teacher's next-token distributions are a rich signal for "how
English works". Knowledge spread across the deleted layers needs far more training: NVIDIA's
Minitron work distilled on billions of tokens, and we used about 100,000.

**Experiment (optional):** set `N_DROP = 2` in Exercise 2 and re-run Exercises 2 and 3. A gentler
prune keeps much more knowledge. Choosing how much to cut is the real engineering decision.""")]

cells += [md("""## 4. Which variant should we ship?

Plot accuracy against size. A variant that is both bigger *and* worse than another is **dominated**.
The rest form the **Pareto frontier**: the only sensible choices, depending on your budget."""),
    code("""table = pd.DataFrame(RESULTS).T.round(3)
display(table)
fig, ax = plt.subplots(figsize=(8, 4.5))
for name, r in RESULTS.items():
    ax.scatter(r["size_GB"], r["ppl"], s=80)
    ax.annotate(name, (r["size_GB"], r["ppl"]), textcoords="offset points", xytext=(6, 4))
ax.set_xlabel("approx. weight size (GB)"); ax.set_ylabel("perplexity (lower = better)")
ax.set_yscale("log"); ax.set_title("Size vs quality"); ax.grid(alpha=0.3); plt.show()"""),
    md("""**Typical conclusion:** INT8 or FP8 roughly halves the transformer weights at essentially no cost,
so do it first. INT4 with small groups shrinks them about 4x at a real accuracy cost (MMLU 68% to 50%
in our run). Aggressive pruning on a model this small destroys knowledge that a short distillation
can't restore; it pays off on large models with a big distillation budget, where removing layers cuts
**latency** on every token. NVIDIA's Minitron recipe combines pruning, distillation and quantization at
that scale. Always confirm on your
**domain** benchmark (Module 2) before shipping."""),
    code("""mlflow.set_tracking_uri(f"sqlite:///{OUT}/mlflow.db")
mlflow.set_experiment("lumora-optimisation")
for name, r in RESULTS.items():
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"model": MODEL_ID, "variant": name, "tier": rt.tier})
        mlflow.log_metrics({k: float(v) for k, v in r.items() if v == v})   # skip NaN
(OUT / "results.json").write_text(json.dumps(RESULTS, indent=2))
progress.summary("m04")"""),
    md("""### Pro bonus: real FP8 on an L4

The L4 (Ada) has FP8 tensor cores. To go from simulated to *real* FP8 inference, the standard NVIDIA
path is:

1. Quantize with **NVIDIA TensorRT Model Optimizer** (`pip install nvidia-modelopt`): FP8 post-training
   quantization with a calibration set (use real support prompts, not random text).
2. Export a checkpoint and build a **TensorRT-LLM** engine.
3. Benchmark throughput against the FP16 engine at realistic batch sizes.

This changes often between releases, so follow the current ModelOpt "LLM PTQ" example for your
installed version. The numbers you measured above tell you what accuracy to expect from FP8.

### Key takeaways
* Decoding is **memory-bandwidth-bound**, so fewer bytes per weight means faster tokens, *if* the
  kernels support it.
* **8-bit (INT8 / FP8) is nearly free; 4-bit needs per-group scales.**
* **Depth pruning** removes low-importance middle layers. It hurts at first, and **distillation**
  from the original model recovers a large part of the damage.
* Judge variants on a **size/speed vs accuracy** chart and keep only the Pareto frontier.

**Next:** Module 5 - the assessment.""")]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
})
out = Path(__file__).resolve().parents[1] / "notebooks" / "04_optimization.ipynb"
nbf.write(nb, out)
print("wrote", out)
