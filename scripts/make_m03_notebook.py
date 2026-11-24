"""Generate notebooks/03_customization.ipynb from readable Python source.

Edit the cells here, then run:  python scripts/make_m03_notebook.py
"""
from pathlib import Path

import nbformat as nbf

REPO_SLUG = "vinothss/add_llmknowledge"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells += [md(f"""# Module 3 - Customising LLMs: CPT, SFT and DPO

<a href="https://colab.research.google.com/github/{REPO_SLUG}/blob/main/notebooks/03_customization.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Scenario.** Module 2 showed that our small model knows almost nothing about Lumora closed-book.
Leadership wants an assistant that **knows** Lumora's products (knowledge), **answers questions
in a support format** (skill), and has a **friendly tone** (behaviour). Three goals, three techniques:

| Technique | Teaches | Data | Loss on | Typical method |
|---|---|---|---|---|
| **CPT** - continued pretraining | *knowledge* | raw domain text | every token | full fine-tune |
| **SFT** - supervised fine-tuning | *skills / format* | prompt -> answer pairs | answer tokens only | LoRA |
| **DPO** - direct preference optimisation | *behaviour / style* | prompt + chosen + rejected | preference margin | LoRA |

**Follow along with the Module 3 page in the portal: the step numbers match.** Step 1 measures the
base model; Part A (Step 2) teaches the facts with CPT; Part B (Step 3) teaches the answer format
with SFT; Part C (Step 4) teaches Lumora's voice with DPO; Part D (Step 5) grades everything.

You'll train five models and compare them on the Module 2 benchmark with the calibrated judge:

```
base ──CPT──> cpt ──SFT──> cpt+sft          base ──SFT──> sft          base ──DPO──> dpo (style)
```

| | Free track (T4) | Pro bonus (L4 / A100) |
|---|---|---|
| Student model | Qwen2.5-0.5B-Instruct | Qwen2.5-1.5B-Instruct (+ gradient checkpointing for CPT) |
| Judge / preference-data writer | Qwen2.5-1.5B-Instruct | Qwen2.5-7B-Instruct |
| Time | ~45 min | ~60 min |

> Runtime -> Change runtime type -> **T4 GPU**. Run the check cells until every line says PASS.
> Stuck? Set `USE_SOLUTIONS = True`."""),
]

cells += [md("## Setup"), code(f"""import os, sys
REPO_URL = "https://github.com/{REPO_SLUG}"
USE_DRIVE = False
USE_SOLUTIONS = os.environ.get("LAB_USE_SOLUTIONS") == "1"

if "google.colab" in sys.modules:
    if not os.path.exists("/content/llm-knowledge-workshop"):
        !git clone -q {{REPO_URL}} /content/llm-knowledge-workshop
    %cd /content/llm-knowledge-workshop
    !pip install -q -r requirements/m03.txt
else:
    if os.path.basename(os.getcwd()) == "notebooks":
        os.chdir("..")
    sys.path.insert(0, os.getcwd())"""),
    code("""import gc, json, re, time, warnings
import pandas as pd
import torch
import mlflow
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DPOTrainer
import labkit
from labkit import checks, progress
from labkit.llm import ChatModel, perplexity
from labkit.solutions.m02 import judge_scores, pick_threshold, build_prompt, predict_letters, LETTERS

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
pd.set_option("display.max_colwidth", 110)
rt = labkit.env.setup(use_drive=USE_DRIVE)
OUT = rt.workdir / "m03"; OUT.mkdir(parents=True, exist_ok=True)
SMOKE = os.environ.get("LAB_SMOKE") == "1"
STRICT = not SMOKE

STUDENT, JUDGE = rt.model("student"), rt.model("generator")
print("Student:", STUDENT, "| Judge:", JUDGE)

def load_student():
    \"\"\"Fresh copy of the base student in fp32 (mixed precision is handled by the trainer).\"\"\"
    tok = AutoTokenizer.from_pretrained(STUDENT)
    model = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.float32).to(rt.device)
    return model, tok

def free():
    \"\"\"Release memory after `del`-ing models/trainers you no longer need.\"\"\"
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()"""),
    code("""lumora_eval = pd.read_json(rt.repo_dir / "data/eval/lumora_eval.jsonl", lines=True)
cpt_corpus = pd.read_json(rt.repo_dir / "data/train/cpt_corpus.jsonl", lines=True)
sft_raw = pd.read_json(rt.repo_dir / "data/train/sft_qa.jsonl", lines=True)
style_prompts = pd.read_json(rt.repo_dir / "data/train/style_prompts.jsonl", lines=True)
if SMOKE:
    lumora_eval = lumora_eval.head(6)
EVAL_Q = list(lumora_eval.question)
# held-out "fact sentences" for perplexity: the eval questions + reference answers (never trained on)
FACT_TEXTS = [f"{q} {r}" for q, r in zip(lumora_eval.question, lumora_eval.reference)]
SYSTEM_PROMPT = "You are Lumora Energy's support assistant. Answer in one short sentence."

# MMLU slice to detect catastrophic forgetting
MMLU = {s: load_dataset("cais/mmlu", s) for s in ["astronomy", "marketing"]}
MMLU = {s: {"test": list(d["test"].select(range(8 if SMOKE else 25))), "dev": list(d["dev"])} for s, d in MMLU.items()}

print(f"CPT corpus: {len(cpt_corpus)} texts ({(cpt_corpus.kind != 'original').sum()} paraphrases)")
print(f"SFT pairs : {len(sft_raw)}   | style prompts: {len(style_prompts)}   | eval questions: {len(lumora_eval)}")
cpt_corpus.groupby("kind").size()"""),
    md("""> **Using your own Module 1 outputs?** Replace `sft_raw` with your `sft_qa.jsonl` and add your
> `curated_corpus.jsonl` texts to `cpt_corpus`. The shipped files were produced by the same pipeline,
> plus LLM **paraphrases** of each document (kept only if every number survived). Paraphrasing is a
> standard trick: seeing a fact in several phrasings helps the model *extract* it later, not just
> recite it. Skim a few paraphrases: some contain small inaccuracies. Augmented data needs QA too.""")]

cells += [md("""## Step 1 - Measure the baseline

Every customised model is compared with the untouched student. For each model we record:
* **answers** to the 30 held-out Lumora questions (graded by the judge at the end)
* **perplexity** on held-out Lumora fact sentences (how surprising the facts are to the model)
* **MMLU** accuracy on a small slice (did we break general knowledge?)"""),
    code("""RESULTS = {}

def evaluate(name, model, tok):
    cm = ChatModel.from_model(model, tok, rt)
    t = time.time()
    answers = cm.chat(EVAL_Q, system=SYSTEM_PROMPT, temperature=0, max_new_tokens=50, batch_size=16)
    ppl = perplexity(model, tok, FACT_TEXTS, rt.device)
    prompts = [build_prompt(e, d["dev"], s) for s, d in MMLU.items() for e in d["test"]]
    gold = [LETTERS[e["answer"]] for d in MMLU.values() for e in d["test"]]
    mmlu = sum(p == g for p, g in zip(predict_letters(cm, prompts), gold)) / len(gold)
    RESULTS[name] = {"answers": answers, "ppl": ppl, "mmlu": mmlu}
    print(f"[{name}] perplexity={ppl:.1f}  mmlu={mmlu:.0%}  ({time.time() - t:.0f}s)")

model, tok = load_student()
evaluate("base", model, tok)
free()
for q, a in list(zip(EVAL_Q, RESULTS["base"]["answers"]))[:3]:
    print(f"  Q: {q}\\n  A: {a}")""")]

cells += [md("""# Part A: Teach the facts

## Step 2 - Continued pretraining (Exercise 1)

CPT is plain next-token prediction on domain text, the same objective as pretraining but on your
corpus. Two decisions matter:

1. **Data format:** one document per row, ending in the EOS token so the model learns where
   documents stop.
2. **Full fine-tuning vs LoRA.** LoRA updates a tiny low-rank slice of the weights. That's great for
   format and style, but it **absorbs new facts poorly** (in testing, LoRA-CPT reached about 27% on
   our benchmark versus about 50% for full CPT). A 0.5B model fits full fine-tuning on a T4:
   ~2 GB weights + 2 GB gradients + 4 GB Adam states (on a GPU the lab uses **8-bit Adam**, shrinking that last item to ~1 GB for headroom). Full fine-tuning needs a **much smaller learning
   rate** than LoRA (around 1e-5 to 5e-5, versus around 2e-4).

Complete the dataset builder and the training config."""),
    code("""from trl import SFTConfig

def make_cpt_dataset(texts: list, eos_token: str) -> Dataset:
    # TODO: one row per document, each ending with eos_token
    return Dataset.from_dict({"text": texts})

def make_cpt_config(output_dir: str, rt, epochs: int = 6) -> SFTConfig:
    return SFTConfig(
        output_dir=output_dir, dataset_text_field="text", max_length=512, packing=False,
        completion_only_loss=None,     # TODO: CPT learns EVERY token - set this explicitly
        learning_rate=2e-4,            # TODO: that's a LoRA learning rate - pick one for FULL fine-tuning
        num_train_epochs=epochs, lr_scheduler_type="cosine", warmup_steps=2,
        per_device_train_batch_size=4, gradient_accumulation_steps=2,   # effective batch 8, less memory
        logging_steps=5, save_strategy="no", report_to="none",
        fp16=rt.device == "cuda" and not rt.supports_bf16, bf16=rt.supports_bf16,
        optim="adamw_bnb_8bit" if rt.device == "cuda" else "adamw_torch",   # 8-bit Adam saves ~3 GB on GPU
        gradient_checkpointing=rt.is_pro,
    )

if USE_SOLUTIONS:
    from labkit.solutions.m03 import make_cpt_dataset, make_cpt_config"""),
    code("""CPT_EPOCHS = 1 if SMOKE else 6
cpt_ds = make_cpt_dataset(list(cpt_corpus.text), tok.eos_token)
cpt_cfg = make_cpt_config(str(OUT / "cpt"), rt, epochs=CPT_EPOCHS)
checks.check_cpt_setup(cpt_ds, cpt_cfg, tok.eos_token)"""),
    code("""free()
t = time.time()
trainer = SFTTrainer(model=model, args=cpt_cfg, train_dataset=cpt_ds, processing_class=tok)   # no peft_config = full FT
trainer.train()
print(f"CPT finished in {time.time() - t:.0f}s")
model = trainer.model
del trainer; free()
evaluate("cpt", model, tok)
checks.check_cpt_result(RESULTS["base"]["ppl"], RESULTS["cpt"]["ppl"], strict=STRICT)""")]

cells += [md("""# Part B: Teach it to answer

## Step 3 - Supervised fine-tuning with LoRA (Exercise 2)

The CPT model *knows* more, but it was trained to continue documents, not to answer questions.
SFT teaches the **question -> short answer** behaviour. Three details:

* **Prompt/completion format.** TRL then masks the prompt and computes loss only on the answer.
  We want the model to learn to *answer*, not to generate questions.
* **Decontamination.** Our SFT pairs were generated from the same documents the eval questions were
  written from. Some generated questions are near-copies of eval questions, and training on them
  would inflate the score. Drop any training question whose content-word Jaccard similarity with an
  eval question is >= 0.6.
* **LoRA.** Small adapters on the attention *and* MLP projections are plenty for teaching format,
  and they barely disturb the knowledge CPT just added."""),
    code("""from peft import LoraConfig

def make_lora_config(r: int = 16) -> LoraConfig:
    return LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "v_proj"])   # TODO: also adapt k/o and the MLP (gate/up/down)

STOP = {"the", "a", "an", "of", "to", "in", "on", "for", "is", "are", "what", "how", "does", "do",
        "can", "which", "and", "or", "by", "with", "be", "it", "its", "at", "lumora"}

def content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9\\-]+", text.lower()) if w not in STOP}

def is_contaminated(question: str, eval_questions: list, threshold: float = 0.6) -> bool:
    # TODO: True if Jaccard(content_words(question), content_words(e)) >= threshold for ANY eval question e
    return False

def to_prompt_completion(question: str, answer: str, system: str = SYSTEM_PROMPT) -> dict:
    # TODO: return {"prompt": [system msg, user msg], "completion": [assistant msg]}
    #       messages look like {"role": "user", "content": "..."}
    return {"prompt": question, "completion": answer}

if USE_SOLUTIONS:
    from labkit.solutions.m03 import make_lora_config, is_contaminated, to_prompt_completion"""),
    code("""flags = [is_contaminated(q, EVAL_Q) for q in sft_raw.question]
sft_clean = sft_raw[[not f for f in flags]].reset_index(drop=True)
print(f"Dropped {sum(flags)} contaminated pairs, e.g.:")
for q in sft_raw.question[flags].head(4):
    print("  -", q)
sft_records = [to_prompt_completion(q, a) for q, a in zip(sft_clean.question, sft_clean.answer)]
lora_cfg = make_lora_config()
checks.check_sft_setup(sft_records, list(sft_clean.question), EVAL_Q, lora_cfg, n_raw=len(sft_raw))"""),
    code("""from labkit.solutions.m03 import make_sft_config
SFT_EPOCHS = 1 if SMOKE else 6
sft_ds = Dataset.from_list(sft_records)

def run_sft(model, tok, tag):
    t = time.time()
    tr = SFTTrainer(model=model, args=make_sft_config(str(OUT / tag), rt, epochs=SFT_EPOCHS),
                    train_dataset=sft_ds, processing_class=tok, peft_config=make_lora_config())
    tr.train()
    tr.model.print_trainable_parameters()
    merged = tr.model.merge_and_unload()     # fold the adapter back into the weights
    print(f"SFT ({tag}) finished in {time.time() - t:.0f}s")
    del tr; free()
    return merged

model = run_sft(model, tok, "cpt_sft")       # SFT on top of the CPT model
evaluate("cpt+sft", model, tok)
del model; free()

model, tok = load_student()                  # SFT straight from the base model, for comparison
model = run_sft(model, tok, "sft")
evaluate("sft", model, tok)
del model; free()""")]

cells += [md("""# Part C: Teach it Lumora's voice

## Step 4 - Give it Lumora's voice with DPO (Exercise 3)

We want a warmer, more casual support voice. Writing "ideal" replies by hand is slow. Judging
*which of two replies is better* is fast, and that's the data DPO uses: `(prompt, chosen, rejected)`.
DPO raises the likelihood of *chosen* relative to *rejected*, measured against a frozen reference
model so the policy doesn't drift too far. `beta` sets how strongly it stays anchored.

We'll synthesise the pairs with the judge model: the same prompt answered once with a casual-style
instruction (chosen) and once with a formal one (rejected). The key detail: **the style
instruction must not appear in the training prompt.** The model has to learn the style itself,
not learn to follow an instruction."""),
    code("""CHOSEN_STYLE = ("Reply in a warm, casual, friendly tone, like a helpful friend texting back. "
                "Use contractions (you're, it's, don't) and keep it to 2-3 short sentences.")
REJECTED_STYLE = ("Reply in a formal, impersonal, corporate tone. Do not use contractions. "
                  "Keep it to 2-3 sentences.")

n_train = 8 if SMOKE else 60
train_prompts = list(style_prompts.prompt[:n_train])
test_prompts = list(style_prompts.prompt[60:66] if SMOKE else style_prompts.prompt[60:])

writer = ChatModel(JUDGE, rt)
t = time.time()
chosen = writer.chat(train_prompts, system=CHOSEN_STYLE, temperature=0.7, max_new_tokens=120, batch_size=16)
rejected = writer.chat(train_prompts, system=REJECTED_STYLE, temperature=0.7, max_new_tokens=120, batch_size=16)
print(f"Generated {len(chosen)} preference pairs in {time.time() - t:.0f}s\\n")
print("PROMPT  :", train_prompts[0]); print("CHOSEN  :", chosen[0]); print("REJECTED:", rejected[0])"""),
    code("""from trl import DPOConfig

def to_preference(prompt: str, chosen: str, rejected: str) -> dict:
    # TODO: the prompt currently leaks the style instruction - remove it!
    return {
        "prompt": [{"role": "user", "content": CHOSEN_STYLE + "\\n\\n" + prompt}],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }

def make_dpo_config(output_dir: str, rt, epochs: int = 3) -> DPOConfig:
    return DPOConfig(
        output_dir=output_dir, beta=0.1,          # try 0.05 (more drift) and 0.5 (more anchored) later
        learning_rate=5e-5, num_train_epochs=epochs, lr_scheduler_type="cosine", warmup_steps=2,
        per_device_train_batch_size=4, gradient_accumulation_steps=2, max_length=512,
        logging_steps=5, save_strategy="no", report_to="none",
        fp16=rt.device == "cuda" and not rt.supports_bf16, bf16=rt.supports_bf16,
    )

if USE_SOLUTIONS:
    from labkit.solutions.m03 import to_preference, make_dpo_config"""),
    code("""pref_records = [to_preference(p, c, r) for p, c, r in zip(train_prompts, chosen, rejected)]
dpo_cfg = make_dpo_config(str(OUT / "dpo"), rt, epochs=1 if SMOKE else 3)
checks.check_preferences(pref_records, dpo_cfg, style_markers=["casual, friendly tone", "corporate tone"])"""),
    code("""from labkit.solutions.m03 import casualness

def style_eval(model, tok):
    outs = ChatModel.from_model(model, tok, rt).chat(test_prompts, temperature=0, max_new_tokens=120, batch_size=16)
    return outs, sum(map(casualness, outs)) / len(outs)

model, tok = load_student()
style_before, casual_before = style_eval(model, tok)

t = time.time()
dpo = DPOTrainer(model=model, args=dpo_cfg, train_dataset=Dataset.from_list(pref_records),
                 processing_class=tok, peft_config=make_lora_config())   # LoRA: the reference = adapter switched off
dpo.train()
print(f"DPO finished in {time.time() - t:.0f}s")
logs = [l for l in dpo.state.log_history if "rewards/accuracies" in l]
reward_acc = logs[-1]["rewards/accuracies"] if logs else 0.0
print({k: round(v, 3) for k, v in logs[-1].items() if k.startswith("rewards")} if logs else "no logs")
model = dpo.model.merge_and_unload(); del dpo; free()

style_after, casual_after = style_eval(model, tok)
evaluate("dpo", model, tok)
del model; free()
pd.DataFrame({"prompt": test_prompts, "before DPO": style_before, "after DPO": style_after}).head(5)"""),
    code("""checks.check_dpo_result(casual_before, casual_after, reward_acc, strict=STRICT)"""),
    md("""**Experiment (optional, about 2 min):** re-run the DPO cell with `learning_rate=2e-4`. In testing,
replies collapsed into content-free cheerfulness ("Hey there! Here's some fun ideas!"): the
policy **over-optimised** the style signal and dropped helpfulness. That's why we watch both the
style metric *and* sample outputs, and why `beta` and the learning rate matter.""")]

cells += [md("""# Part D: Compare the results

## Step 5 - Grade everything with Module 2's judge (Exercise 4)

We re-calibrate the Module 2 judge (same method, same calibration data) and grade all five models'
answers on the held-out Lumora benchmark."""),
    code("""writer.unload(); del writer; free()
judge = ChatModel(JUDGE, rt)
calib = pd.read_json(rt.repo_dir / "data/eval/judge_calibration.jsonl", lines=True)
cal = calib[calib.split == "calib"].to_dict("records")
THRESHOLD = pick_threshold(judge_scores(judge, cal), [r["human_label"] for r in cal])
print(f"Judge threshold: {THRESHOLD:.2f}")

for name, r in RESULTS.items():
    rows = [{"question": q, "reference": ref, "candidate": a}
            for q, ref, a in zip(lumora_eval.question, lumora_eval.reference, r["answers"])]
    r["lumora_acc"] = sum(s >= THRESHOLD for s in judge_scores(judge, rows)) / len(rows)

order = ["base", "sft", "cpt", "cpt+sft", "dpo"]
summary = pd.DataFrame({n: {"lumora_acc": RESULTS[n]["lumora_acc"], "fact_ppl": RESULTS[n]["ppl"],
                            "mmlu": RESULTS[n]["mmlu"]} for n in order}).T.round(3)
summary"""),
    code("""checks.check_knowledge_gain({n: RESULTS[n]["lumora_acc"] for n in RESULTS}, strict=STRICT)
pd.DataFrame({"question": EVAL_Q, "reference": lumora_eval.reference,
              "base": RESULTS["base"]["answers"], "sft": RESULTS["sft"]["answers"],
              "cpt+sft": RESULTS["cpt+sft"]["answers"]}).head(8)"""),
    md("""**Interpret your table.** Results from our test run (yours will vary by a few points):

| model | Lumora acc | fact perplexity | MMLU | what happened |
|---|---|---|---|---|
| base | 10% | 86 | 68% | doesn't know Lumora |
| sft | 20% | 50 | 70% | right *format*, invented facts. SFT alone teaches style, not knowledge |
| cpt | 50% | 15 | 70% | knows the facts, but answers can ramble |
| cpt+sft | **70%** | **11** | 70% | knowledge **plus** answer format |
| dpo | 3% | 86 | 68% | casual tone, no Lumora knowledge: it now chats instead of stating facts |

Two things to notice. **Perplexity on held-out facts fell from 86 to 11**, so the facts became
unsurprising to the model even though it never saw these exact sentences. And **MMLU didn't move**
on our 50-question slice, so there's no measurable forgetting at this scale. Full fine-tuning on a
larger corpus, or for longer, can cost general knowledge, so always check. Production recipes mix in
general data to limit it.""")]

cells += [md("## Log to MLflow and save"),
    code("""mlflow.set_tracking_uri(f"sqlite:///{OUT}/mlflow.db")
mlflow.set_experiment("lumora-customisation")
for name in order:
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"student": STUDENT, "cpt": "cpt" in name, "sft": "sft" in name, "dpo": name == "dpo",
                           "cpt_method": "full", "sft_method": "lora-r16", "judge": JUDGE, "tier": rt.tier})
        mlflow.log_metrics({"lumora_acc": RESULTS[name]["lumora_acc"], "fact_ppl": RESULTS[name]["ppl"],
                            "mmlu": RESULTS[name]["mmlu"]})
    print("logged", name)
(OUT / "results.json").write_text(json.dumps({n: {k: v for k, v in r.items()} for n, r in RESULTS.items()}, indent=2))
progress.summary("m03")"""),
    md("""### Key takeaways
* **CPT adds knowledge, SFT adds skills, DPO shapes behaviour.** Use the right tool for the goal,
  and stack them: CPT -> SFT -> DPO.
* **LoRA vs full fine-tuning:** LoRA is cheap and safe for format and style, but weak at absorbing
  new facts.
* **Decontaminate** training data against your eval set, or your scores will lie.
* **Prompt/completion masking** makes SFT learn the answer, not the question.
* **DPO needs a reference and a leash (`beta`).** Over-optimisation produces fluent, empty replies.
* **Always re-check general ability** (MMLU) after customisation.

### Pro bonus
On an L4/A100 the student is **Qwen2.5-1.5B** (CPT uses 8-bit Adam + gradient checkpointing) and
the judge/writer is **7B**. Compare your Pro table with the Free one: bigger students absorb more
from the same data.

### How this maps to NVIDIA NeMo
| This lab (TRL + PEFT) | NeMo Framework / NeMo-RL |
|---|---|
| `SFTTrainer` with text data (CPT) | NeMo 2.0 pretraining recipe resumed from a checkpoint |
| `SFTTrainer` prompt/completion + LoRA | NeMo `finetune` recipe with `peft="lora"` |
| `DPOTrainer` (`beta`, reference model) | NeMo-RL DPO (`dpo.reference_policy_kl_penalty`, ...) |
| `merge_and_unload()` | NeMo LoRA merge -> export for deployment |

**Next:** Module 4 - make the customised model smaller and faster (quantisation, pruning, distillation).""")]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
})
out = Path(__file__).resolve().parents[1] / "notebooks" / "03_customization.ipynb"
nbf.write(nb, out)
print("wrote", out)
