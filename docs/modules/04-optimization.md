# Module 4: Optimising LLMs for deployment

**Time:** about 40 min reading + 35 min lab &nbsp;|&nbsp; **Track:** Free (T4), with a Pro bonus

## Learning objectives

By the end of this module you should be able to:

- [ ] Explain what limits LLM inference speed and memory, and why smaller weights help
- [ ] Describe number formats (FP32, BF16, FP16, FP8, INT8, INT4) and their trade-offs
- [ ] Explain how post-training quantization works: scales, granularity, calibration, outliers
- [ ] Distinguish unstructured, structured and depth pruning, and how layer importance is measured
- [ ] Explain knowledge distillation and why it is used to "heal" a pruned model
- [ ] Evaluate an optimised model on size, speed *and* accuracy, and pick a point on the trade-off curve

---

## 1. Why optimise?

Your Module 3 model works. Deploying it means paying for every token it generates, for every user,
indefinitely. Optimisation targets four things:

| Goal | Why it matters |
|---|---|
| **Memory** | Fit on a cheaper GPU, or serve more users per GPU |
| **Latency** | Time to first token, and time per token, determine how responsive it feels |
| **Throughput** | Tokens per second across all users determines cost per request |
| **Energy and cost** | Fewer GPU-hours for the same traffic |

The constraint: **accuracy must stay acceptable on the tasks you care about.** Every technique in
this module trades some accuracy for efficiency, and the job is to measure that trade-off.

### Where inference time goes

Generating text has two phases:

1. **Prefill:** the whole prompt is processed in parallel. This is **compute-bound**: big matrix
   multiplications keep the GPU busy.
2. **Decode:** tokens are generated one at a time. For each new token the GPU must read **every
   weight** from memory, but does relatively little maths with it. This is **memory-bandwidth-bound**.

For a 7B model in FP16 (14 GB), each decoded token means reading 14 GB of weights. A GPU with
about 1 TB/s of memory bandwidth can therefore do at most about 70 tokens per second for a single
user, however fast its compute units are. **Halve the bytes per weight and you roughly double the
decode speed limit.** That's why quantization is the first optimisation almost everyone applies.

### The KV cache

During generation the model stores each past token's attention **keys and values** so it doesn't
recompute them. That cache grows with sequence length and batch size:

$$\text{KV bytes} = 2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{tokens} \times \text{batch} \times \text{bytes per value}$$

For Qwen2.5-0.5B (24 layers, 2 KV heads of dimension 64) in FP16 that's about 12 KB per token,
so a batch of 32 conversations of 4,000 tokens each needs about 1.5 GB. For large models with long
contexts the KV cache can exceed the weights. It can be quantized too (e.g. FP8 KV cache), and
architectures such as grouped-query attention (GQA, used by Qwen and Llama) exist to shrink it.

---

## 2. Number formats

A floating-point number has a **sign**, an **exponent** (range) and a **mantissa** (precision):

| Format | Bits (sign / exponent / mantissa) | Max value | Bytes | 7B model weights |
|---|---|---|---|---|
| FP32 | 1 / 8 / 23 | about 3.4e38 | 4 | 28 GB |
| FP16 | 1 / 5 / 10 | 65,504 | 2 | 14 GB |
| BF16 | 1 / 8 / 7 | about 3.4e38 | 2 | 14 GB |
| FP8 E4M3 | 1 / 4 / 3 | 448 | 1 | 7 GB |
| FP8 E5M2 | 1 / 5 / 2 | 57,344 | 1 | 7 GB |
| INT8 | 8-bit integer + scale | 127 x scale | 1 | 7 GB |
| INT4 / NF4 | 4-bit integer + scale | 7 x scale | 0.5 | 3.5 GB |

- **BF16** keeps FP32's range with less precision. It's the standard training format on modern GPUs
  because it rarely overflows. The T4 doesn't support it in hardware, which is why the labs use FP16
  there.
- **FP16** has more precision but a small range: values above 65,504 overflow, which is why FP16
  training needs loss scaling.
- **FP8** comes in two flavours: E4M3 (more precision, usually for weights and activations) and E5M2
  (more range, often for gradients). It's supported natively by NVIDIA **Hopper** (H100) and
  **Ada** (L4, L40S, RTX 40-series) tensor cores, and not by the T4 or A100.
- **INT8 / INT4** store integers plus a floating-point **scale** that maps them back to real values.

---

## 3. Quantization

### The core operation

Quantization maps high-precision values onto a small set of levels. The simplest scheme,
**symmetric absmax INT8**, for a group of weights $w$:

$$s = \frac{\max |w|}{127}, \qquad q = \text{round}\!\left(\frac{w}{s}\right) \in [-127, 127], \qquad \hat{w} = s \cdot q$$

Each weight is stored as one byte $q$; the group shares one scale $s$. The reconstruction $\hat{w}$ is
close to $w$, and the difference is **quantization error**.

!!! example "Quantizing four weights"
    $w = [0.12, -0.50, 0.33, 0.04]$, so $\max|w| = 0.50$ and $s = 0.50/127 \approx 0.003937$.
    Then $q = [30, -127, 84, 10]$ and $\hat{w} = [0.118, -0.500, 0.331, 0.039]$.
    The error is under 0.002 per weight, using a quarter of FP32's memory.

### Granularity: who shares a scale?

| Granularity | One scale per | Accuracy | Overhead |
|---|---|---|---|
| Per-tensor | Whole matrix | Lowest: one outlier ruins everyone's precision | Negligible |
| Per-channel | Output row | Good | Small |
| Per-group | Block of e.g. 128 weights | Best (standard for INT4) | A few % extra storage |

### The outlier problem

LLM **activations** contain rare but huge outlier values in a few channels. Dettmers et al. (2022,
*LLM.int8()*) showed these systematic outliers emerge in models of a few billion parameters and up,
and that naive INT8 quantization of activations then collapses accuracy, because one outlier
stretches the scale and every other value gets rounded to zero. Solutions:

- **Mixed precision:** keep outlier channels in FP16 (LLM.int8()).
- **SmoothQuant:** mathematically move the difficulty from activations into weights, which quantize
  more easily.
- **Weight-only quantization:** quantize weights (INT4/INT8) and keep activations in FP16. This is
  called **W4A16** or **W8A16**, and is great for memory-bound decoding.
- **FP8:** its exponent bits handle a wide dynamic range far better than INT8, so FP8 weight *and*
  activation quantization (**W8A8-FP8**) is often close to lossless for LLMs.

### PTQ vs QAT

| | Post-training quantization (PTQ) | Quantization-aware training (QAT) |
|---|---|---|
| How | Quantize a finished model, using a small **calibration set** to measure activation ranges | Simulate quantization during training so the model adapts |
| Cost | Minutes | A training run |
| Accuracy | Excellent at 8-bit; can degrade at 4-bit | Best at low bit-widths |
| Methods | absmax/minmax, **GPTQ** (second-order error correction), **AWQ** (protect important weights) | Fake-quant + straight-through estimator |

The **calibration set** should look like real traffic: a few hundred samples of the prompts the
model will actually see. Calibrating on random web text and deploying on Lumora support questions
is a classic mistake.

---

## 4. Pruning

Quantization makes each weight smaller. **Pruning removes weights entirely.**

| Type | Removes | Real speed-up? |
|---|---|---|
| **Unstructured** | Individual weights (set to zero) | Rarely: dense hardware still multiplies the zeros |
| **Semi-structured (2:4)** | 2 of every 4 consecutive weights | Yes on NVIDIA Ampere and later (sparse tensor cores) |
| **Structured: width** | Attention heads, MLP neurons, embedding channels | Yes: matrices genuinely shrink |
| **Structured: depth** | Whole transformer layers | Yes: fewer sequential steps per token |

### Depth pruning: which layers can go?

Transformer layers update a **residual stream**: each layer adds a correction to the hidden state.
If a layer's output is almost identical to its input, it isn't contributing much. A simple
importance score is the **[cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity)** between a layer's input and output hidden states,
averaged over calibration text:

$$\text{importance}_\ell = 1 - \cos\big(h_\ell^{\text{in}}, h_\ell^{\text{out}}\big)$$

Low-importance layers are pruning candidates. Studies (Men et al., 2024, *ShortGPT*; Gromov et al.,
2024, *The Unreasonable Ineffectiveness of the Deeper Layers*) found that a substantial fraction of
deeper layers can be removed with modest damage, especially if the model is briefly retrained
afterwards. The first and last layers are almost always critical.

### Pruning hurts, then you heal

Removing a quarter of the layers *will* drop accuracy, often sharply at first. The damage is
largely recoverable with a short retraining phase, and the most efficient way to retrain is
**distillation** from the original model. NVIDIA's **Minitron** work (2024) pruned Nemotron and Llama
models in depth and width, then distilled from the unpruned teacher, reaching strong smaller models
with a small fraction of the training tokens needed to train them from scratch.

---

## 5. Knowledge distillation

### The idea

A large **teacher** model's output carries more information than the correct label alone. For
"The Vault warranty lasts ___", a teacher might put 70% on "12", 15% on "twelve" and small amounts
on other numbers. That distribution, the **soft targets**, tells the student which mistakes are
"nearly right". Hinton et al. (2015) called this the teacher's *dark knowledge*.

### The loss

Soften both distributions with a **temperature** $T > 1$ (dividing the logits $z$ by $T$ before the
softmax) and make the student match the teacher with [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence), usually mixed with the normal
loss on real data:

$$\mathcal{L} = \alpha\, T^2\, \mathrm{KL}\!\left(\mathrm{softmax}(z_t / T)\,\big\|\,\mathrm{softmax}(z_s / T)\right) + (1 - \alpha)\,\mathcal{L}_{\text{CE}}$$

- $z_t$, $z_s$: teacher and student logits
- $T$: higher values spread probability out, revealing more of the teacher's ranking (2-4 is common)
- $T^2$: keeps gradient sizes comparable as $T$ changes
- $\alpha$: balance between imitating the teacher and fitting the data

### Flavours

| Flavour | Student learns from | Notes |
|---|---|---|
| **Logit distillation** | Teacher's full token distributions | Needs teacher and student to share a tokenizer; strongest signal |
| **Sequence-level distillation** | Text the teacher generated | Works across tokenizers; this is what "train on GPT-4 outputs" is |
| **Feature distillation** | Teacher's hidden states | Used in pruning recovery when architectures align |

Module 1's synthetic Q&A generation was, in effect, sequence-level distillation from the 1.5B model
into training data.

---

## 6. Evaluating the trade-off

Never report "4x smaller" without "and lost X points on Y". For each variant, measure:

| Axis | Metric |
|---|---|
| Size | GB on disk, GPU memory at batch 1 and at your target batch |
| Latency | Time to first token (TTFT), time per output token |
| Throughput | Tokens/second at realistic concurrency |
| General accuracy | MMLU slice (from Module 2) |
| **Domain accuracy** | **Lumora QA with the calibrated judge** (from Module 2) |

Then plot accuracy against cost. Variants that are both slower *and* less accurate than another
variant are **dominated** and can be discarded. The rest form the **Pareto frontier**, and you pick
the point that meets your latency or cost budget.

Typical rules of thumb, which you should always verify on your own model:

- **FP8 / INT8** weight quantization: usually within about a point on most benchmarks.
- **INT4** weight-only (GPTQ/AWQ): usually small losses; larger on maths, code and small models.
- **Depth pruning without healing:** large losses. **With a short distillation:** fluency (perplexity)
  recovers quickly, knowledge much less. Recovering knowledge needs a large distillation budget.
- **Small models degrade more** under every technique: there's less redundancy to remove.

---

## 7. Deployment stack: TensorRT-LLM

**[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)** is NVIDIA's open-source library for optimised LLM inference. It compiles a model
into an engine with fused kernels, in-flight (continuous) batching, paged KV cache, and support for
FP8, INT8 and INT4 quantization. **NVIDIA TensorRT Model Optimizer (ModelOpt)** performs the
quantization (including FP8 PTQ with calibration), pruning and distillation steps that produce the
model you then build into an engine. NeMo models export into this path for serving, for example
behind NVIDIA NIM or Triton.

---

## Lab

<div class="lab-card" markdown>
**Lab 4: Make the student model smaller and faster, and measure what it costs**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinothss/add_llmknowledge/blob/main/notebooks/04_optimization.ipynb)

| Exercise | Auto-check |
|---|---|
| 1. Absmax quantizer (per-channel and per-group) | integer range, round-trip error, outlier protection with groups; INT8 within 5% of FP16 perplexity, INT4 group-128 better than per-channel |
| 2. Layer importance + depth pruning | one score per layer, first/last layers kept, least important dropped, pruned model still runs |
| 3. Distillation loss + healing | KL term is 0 for identical models, temperature scaling correct; distillation recovers at least 30% of the pruning damage |
| Pro bonus | Qwen2.5-1.5B student; pointers for real FP8 with TensorRT Model Optimizer + TensorRT-LLM |

What our test run found (Qwen2.5-0.5B; perplexity on held-out Wikipedia text, lower is better;
MMLU on 50 questions, chance = 25%):

| Variant | Perplexity | MMLU |
|---|---|---|
| Original | 24.1 | 68% |
| INT8 | 24.0 | 68% |
| FP8 (simulated) | 24.2 | 66% |
| INT4, one scale per row | 60.9 | 28% |
| INT4, one scale per 128 weights | 35.5 | 50% |
| 6 of 24 layers removed | 76.4 | 24% |
| ...then distilled for 200 steps | 43.1 | 12% |

Distillation recovered 64% of the perplexity damage in about two minutes, but MMLU stayed at chance
level. Fluency comes back quickly; knowledge spread across deleted layers needs a far bigger
distillation budget (Minitron used billions of tokens; the lab uses about 100,000). The lab asks you
to try a gentler prune and see the difference.
</div>

<div class="lab-complete" data-id="m04"></div>

## Glossary

| Term | Meaning |
|---|---|
| **Prefill / decode** | Processing the prompt in parallel / generating tokens one by one |
| **Memory-bandwidth-bound** | Speed limited by how fast weights can be read, not by maths |
| **KV cache** | Stored attention keys and values for past tokens |
| [**Scale**](https://en.wikipedia.org/wiki/Quantization_(signal_processing)) | Floating-point factor mapping quantized integers back to real values |
| **Calibration set** | Representative data used to measure activation ranges for PTQ |
| **W4A16** | 4-bit weights, 16-bit activations (weight-only quantization) |
| **Outlier features** | Rare, very large activation values that break naive quantization |
| [**Structured pruning**](https://en.wikipedia.org/wiki/Pruning_(artificial_neural_network)) | Removing whole components (heads, neurons, layers) so matrices shrink |
| [**Soft targets**](https://en.wikipedia.org/wiki/Knowledge_distillation) | Teacher's full probability distribution used in distillation |
| [**Temperature**](https://en.wikipedia.org/wiki/Softmax_function) | Logit divisor that flattens a distribution |
| [**Pareto frontier**](https://en.wikipedia.org/wiki/Pareto_front) | Set of options where no other option is better on every axis |

## Further reading

**Papers**

- Dettmers et al. (2022), [*LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale*](https://arxiv.org/abs/2208.07339)
- Frantar et al. (2023), [*GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers*](https://arxiv.org/abs/2210.17323)
- Lin et al. (2024), [*AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration*](https://arxiv.org/abs/2306.00978)
- Xiao et al. (2023), [*SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs*](https://arxiv.org/abs/2211.10438)
- Micikevicius et al. (2022), [*FP8 Formats for Deep Learning*](https://arxiv.org/abs/2209.05433)
- Hinton, Vinyals & Dean (2015), [*Distilling the Knowledge in a Neural Network*](https://arxiv.org/abs/1503.02531)
- Men et al. (2024), [*ShortGPT: Layers in Large Language Models are More Redundant Than You Expect*](https://arxiv.org/abs/2403.03853)
- Gromov et al. (2024), [*The Unreasonable Ineffectiveness of the Deeper Layers*](https://arxiv.org/abs/2403.17887)
- Muralidharan et al. (2024), [*Compact Language Models via Pruning and Knowledge Distillation*](https://arxiv.org/abs/2407.14679) (Minitron)

**Concepts and tools**

- [Half-precision floating point, FP16 (Wikipedia)](https://en.wikipedia.org/wiki/Half-precision_floating-point_format)
- [bfloat16 floating-point format (Wikipedia)](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
- [Quantization in signal processing (Wikipedia)](https://en.wikipedia.org/wiki/Quantization_(signal_processing))
- [Cosine similarity (Wikipedia)](https://en.wikipedia.org/wiki/Cosine_similarity)
- [Pruning artificial neural networks (Wikipedia)](https://en.wikipedia.org/wiki/Pruning_(artificial_neural_network))
- [Knowledge distillation (Wikipedia)](https://en.wikipedia.org/wiki/Knowledge_distillation)
- [Kullback-Leibler divergence (Wikipedia)](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
- [Softmax function and temperature (Wikipedia)](https://en.wikipedia.org/wiki/Softmax_function)
- [Pareto front (Wikipedia)](https://en.wikipedia.org/wiki/Pareto_front)
- [bitsandbytes documentation (Hugging Face)](https://huggingface.co/docs/bitsandbytes/main/en/index)
- [NVIDIA TensorRT-LLM (GitHub)](https://github.com/NVIDIA/TensorRT-LLM)
- [NVIDIA TensorRT Model Optimizer (GitHub)](https://github.com/NVIDIA/TensorRT-Model-Optimizer)

## Knowledge check

<script type="application/json" class="quiz" data-id="m04">
[
  {"q": "Why does quantizing weights from FP16 to INT8 speed up token-by-token generation?",
   "options": ["INT8 maths is always more accurate", "Decoding is memory-bandwidth-bound: half the bytes to read per token means up to about twice the speed", "It reduces the number of layers", "It shortens the prompt"],
   "answer": 1, "why": "Each decoded token requires reading every weight; fewer bytes means less time waiting on memory."},
  {"q": "Why can naive per-tensor INT8 quantization of activations fail in larger LLMs?",
   "options": ["INT8 has no negative numbers", "Rare huge outlier activations stretch the scale so most values round to zero", "Activations are always zero", "INT8 isn't supported on GPUs"],
   "answer": 1, "why": "This is the outlier-feature problem; fixes include mixed precision, SmoothQuant, weight-only schemes and FP8."},
  {"q": "Which GPU in Colab supports FP8 tensor-core quantization?",
   "options": ["T4", "A100", "L4", "None of them"],
   "answer": 2, "why": "FP8 needs Ada (L4) or Hopper (H100) architecture; T4 (Turing) and A100 (Ampere) lack it."},
  {"q": "Why does unstructured pruning (zeroing individual weights) rarely speed up inference on its own?",
   "options": ["It removes too many weights", "Dense hardware still multiplies the zeros; matrices don't actually shrink", "It changes the tokenizer", "It only works on CPUs"],
   "answer": 1, "why": "Real speed-ups need structured removal (heads, neurons, layers) or hardware-supported 2:4 sparsity."},
  {"q": "In distillation, why use the teacher's full probability distribution rather than just the correct token?",
   "options": ["It's smaller to store", "It carries information about which wrong answers are nearly right (dark knowledge), a richer training signal", "It avoids needing a teacher", "It removes the need for a loss function"],
   "answer": 1, "why": "Soft targets teach the student the teacher's similarity structure, not just the argmax."},
  {"q": "Variant X is slower and less accurate than variant Y. What should you do with X?",
   "options": ["Deploy it as a backup", "Discard it: it's dominated and not on the Pareto frontier", "Quantize it further", "Retrain Y"],
   "answer": 1, "why": "Only non-dominated variants are worth considering."}
]
</script>
