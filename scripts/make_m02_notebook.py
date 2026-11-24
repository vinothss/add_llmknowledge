"""Generate notebooks/02_evaluation.ipynb from readable Python source.

Edit the cells here, then run:  python scripts/make_m02_notebook.py
"""
from pathlib import Path

import nbformat as nbf

REPO_SLUG = "vinothss/add_llmknowledge"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells += [md(f"""# Module 2 - Evaluating Large Language Models

<a href="https://colab.research.google.com/github/{REPO_SLUG}/blob/main/notebooks/02_evaluation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Scenario.** Before Lumora spends GPU budget on fine-tuning, leadership asks two questions:
*"How good are the candidate base models in general?"* and *"How much do they know about **our**
products?"* You need numbers you can trust, and a record of every experiment.

| Part | Technique | Question it answers |
|---|---|---|
| 1 | Eyeballing | Does the model even understand the task? |
| 2-3 | **MMLU**, zero-shot vs few-shot (log-likelihood scoring) | General knowledge and reasoning |
| 4 | **Lumora QA benchmark**, closed-book vs open-book | Domain knowledge gap |
| 5 | **LLM-as-a-judge**, threshold calibrated on human labels | Scalable grading of free-text answers |
| 6 | **Pairwise comparison + Elo** | Ranking systems head-to-head |
| 7 | **MLflow** | Tracking and comparing every run |

| | Free track (T4) | Pro bonus (L4 / A100) |
|---|---|---|
| Models compared | Qwen2.5-0.5B vs 1.5B | Qwen2.5-1.5B vs 7B |
| Judge | Qwen2.5-1.5B | Qwen2.5-7B |
| Extra | - | Cross-check with `lm-evaluation-harness` |
| Time | ~45 min | ~50 min |

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
    !pip install -q -r requirements/m02.txt
else:
    if os.path.basename(os.getcwd()) == "notebooks":
        os.chdir("..")
    sys.path.insert(0, os.getcwd())"""),
    code("""import json, re, warnings
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import labkit
from labkit import checks, progress
from labkit.llm import ChatModel

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 100)
rt = labkit.env.setup(use_drive=USE_DRIVE)
OUT = rt.workdir / "m02"; OUT.mkdir(parents=True, exist_ok=True)
SMOKE = os.environ.get("LAB_SMOKE") == "1"

STUDENT, GENERATOR = rt.model("student"), rt.model("generator")
print("Small model :", STUDENT)
print("Larger model:", GENERATOR, "(also our judge)")

mlflow.set_tracking_uri(f"sqlite:///{OUT}/mlflow.db")
mlflow.set_experiment("lumora-eval")
print("MLflow tracking ->", mlflow.get_tracking_uri())"""),
    code("""lumora_eval = pd.read_json(rt.repo_dir / "data/eval/lumora_eval.jsonl", lines=True)
calibration = pd.read_json(rt.repo_dir / "data/eval/judge_calibration.jsonl", lines=True)
corpus = pd.read_json(rt.repo_dir / "data/raw/lumora_corpus.jsonl", lines=True)
docs_by_title = {r.title: r.text for r in corpus.itertuples() if r.source.startswith("docs/")}
if SMOKE:
    lumora_eval = lumora_eval.sample(6, random_state=0).reset_index(drop=True)
print(f"{len(lumora_eval)} held-out Lumora questions, {len(calibration)} human-labelled judge items")
lumora_eval.head()""")]

cells += [md("""## 1. Eyeballing

Always start by *reading* outputs. Ask the small model some Lumora questions with no context
(**closed-book**). What do you notice?"""),
    code("""student = ChatModel(STUDENT, rt)
CLOSED_BOOK = "Answer this question about Lumora Energy in one short sentence.\\nQuestion: {question}"
peek = lumora_eval.head(5)
for q, ref, a in zip(peek.question, peek.reference,
                     student.chat([CLOSED_BOOK.format(question=q) for q in peek.question], temperature=0, max_new_tokens=60)):
    print(f"Q: {q}\\n   model : {a}\\n   truth : {ref}\\n")"""),
    md("""**Think about it:** the model answers fluently and confidently, but it has never seen Lumora
data. This is **hallucination**, and it's why eyeballing alone isn't enough: you can't eyeball
10,000 answers, and a fluent wrong answer is easy to miss. We need systematic metrics.""")]

cells += [md("""## 2. Exercise 1 - Build MMLU prompts (zero-shot and few-shot)

[MMLU](https://huggingface.co/datasets/cais/mmlu) has 57 subjects of 4-option multiple-choice
questions. The classic prompt format is:

```
The following are multiple choice questions (with answers) about astronomy.

<shot question>
A. ...
B. ...
C. ...
D. ...
Answer: B

<test question>
A. ...
...
Answer:
```

**Zero-shot** = no solved examples. **Few-shot** = k solved examples from the `dev` split first
(*in-context learning*). Complete `format_example` and `build_prompt`."""),
    code("""from datasets import load_dataset

SUBJECTS = ["astronomy", "high_school_physics", "marketing", "business_ethics"]
N_PER_SUBJECT = 5 if SMOKE else (50 if rt.is_pro else 25)
if SMOKE:
    SUBJECTS = SUBJECTS[:2]

mmlu = {}
for s in SUBJECTS:
    ds = load_dataset("cais/mmlu", s)
    mmlu[s] = {"test": list(ds["test"].select(range(N_PER_SUBJECT))), "dev": list(ds["dev"])}
ex = mmlu["astronomy"]["test"][0]
ex"""),
    code("""LETTERS = ["A", "B", "C", "D"]

def format_example(ex: dict, include_answer: bool) -> str:
    lines = [ex["question"].strip()]
    # TODO: add one line per choice: "A. <choice>", "B. <choice>", ...
    # TODO: add the final line "Answer:" - followed by " <letter>" only if include_answer
    return "\\n".join(lines)

def build_prompt(test_ex: dict, shots: list, subject: str) -> str:
    header = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\\n\\n"
    # TODO: format each shot WITH its answer, then the test question WITHOUT, joined by blank lines
    return header + format_example(test_ex, include_answer=False)

if USE_SOLUTIONS:
    from labkit.solutions.m02 import format_example, build_prompt"""),
    code("""checks.check_mmlu_prompt(build_prompt, ex, mmlu["astronomy"]["dev"], "astronomy")
print(build_prompt(ex, mmlu["astronomy"]["dev"][:2], "astronomy"))""")]

cells += [md("""## 3. Exercise 2 - Log-likelihood scoring

Generating text and regex-parsing "the answer is B" is fragile. Benchmark harnesses instead ask:
*which of the tokens `" A"`, `" B"`, `" C"`, `" D"` does the model think is most likely to come
next?* One forward pass, no parsing, and fully deterministic.

You have two helpers:
* `model.next_token_logprobs(prompts)` -> tensor `[n_prompts, vocab_size]` of log-probabilities
* `model.token_id(" A")` -> the vocabulary id of `" A"` (**note the leading space**)"""),
    code("""def predict_letters(model, prompts: list) -> list:
    logprobs = model.next_token_logprobs(prompts)
    # TODO 1: letter_ids = [model.token_id(" A"), ...] for each letter
    # TODO 2: select those columns, take argmax per row, map index -> letter
    return ["A"] * len(prompts)

if USE_SOLUTIONS:
    from labkit.solutions.m02 import predict_letters"""),
    code("""checks.check_predict(predict_letters, student, mmlu)"""),
    code("""def eval_mmlu(model, k: int) -> dict:
    \"\"\"Accuracy per subject and overall for k-shot prompting.\"\"\"
    per_subject, correct, total = {}, 0, 0
    for s in SUBJECTS:
        tests, shots = mmlu[s]["test"], mmlu[s]["dev"][:k]
        preds = predict_letters(model, [build_prompt(e, shots, s) for e in tests])
        hits = sum(p == LETTERS[e["answer"]] for p, e in zip(preds, tests))
        per_subject[s] = hits / len(tests)
        correct += hits; total += len(tests)
    return {"mmlu_acc": correct / total, **{f"mmlu_{s}": v for s, v in per_subject.items()}}

mmlu_results = {}
for k in (0, 5):
    mmlu_results[(STUDENT, k)] = eval_mmlu(student, k)
    print(f"{STUDENT}  {k}-shot  acc={mmlu_results[(STUDENT, k)]['mmlu_acc']:.1%}")
print("(random guessing = 25%)")""")]

cells += [md("""## 4. The Lumora QA benchmark - closed-book vs open-book

MMLU says nothing about Lumora. Our held-out benchmark (`data/eval/lumora_eval.jsonl`, written
by humans and **never used for training**) does. We test three systems:

| System | Prompt contains |
|---|---|
| small, closed-book | just the question |
| large, closed-book | just the question |
| large, **open-book** | the question **plus the source document** (knowledge supplied *in context*) |

Open-book is the "upper bound" for knowledge injection: it shows what the model could do if it
*knew* the facts. Modules 3 (CPT/SFT) try to close that gap without putting docs in the prompt."""),
    code("""OPEN_BOOK = ("Use the document to answer the question in one short sentence.\\n\\n"
             "Document:\\n{doc}\\n\\nQuestion: {question}")

answers = {}
answers["small-closed"] = student.chat([CLOSED_BOOK.format(question=q) for q in lumora_eval.question],
                                       temperature=0, max_new_tokens=60)
student.unload(); del student

big = ChatModel(GENERATOR, rt)
for k in (0, 5):
    mmlu_results[(GENERATOR, k)] = eval_mmlu(big, k)
    print(f"{GENERATOR}  {k}-shot  acc={mmlu_results[(GENERATOR, k)]['mmlu_acc']:.1%}")

answers["large-closed"] = big.chat([CLOSED_BOOK.format(question=q) for q in lumora_eval.question],
                                   temperature=0, max_new_tokens=60)
answers["large-open"] = big.chat([OPEN_BOOK.format(doc=docs_by_title[t], question=q)
                                  for q, t in zip(lumora_eval.question, lumora_eval.source_title)],
                                 temperature=0, max_new_tokens=60)
SYSTEMS = list(answers)
pd.DataFrame({"question": lumora_eval.question, "reference": lumora_eval.reference, **answers}).head(6)"""),
    md("""### A cheap automatic metric first: token F1

Token-overlap F1 between answer and reference is fast and free. Look at the scores, then at the
answers above. Does F1 reward the *right* things?"""),
    code("""def token_f1(pred: str, ref: str) -> float:
    p, r = re.findall(r"\\w+", pred.lower()), re.findall(r"\\w+", ref.lower())
    common = sum(min(p.count(w), r.count(w)) for w in set(p))
    if not p or not r or common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)

f1 = {s: sum(token_f1(a, r) for a, r in zip(answers[s], lumora_eval.reference)) / len(lumora_eval) for s in SYSTEMS}
pd.Series(f1, name="token_f1").round(3)"""),
    md("""F1 punishes correct paraphrases ("twelve years" vs "12 years") and rewards wrong answers that
reuse the question's words. We need a grader that understands meaning: **LLM-as-a-judge**.""")]

cells += [md("""## 5. Exercise 3 - LLM-as-a-judge, calibrated

A judge is itself a model, so **measure it before trusting it**. `judge_calibration.jsonl` holds
155 answers (mostly *real* outputs of small models, hallucinations included) that a human
labelled correct or incorrect.

Small judges are **lenient**: asked "Is it correct? Yes/No" they often say *Yes* to fluent
nonsense. Instead of trusting the Yes/No, we read the judge's **confidence**:

$$\\text{score} = \\log P(\\text{"Yes"}) - \\log P(\\text{"No"})$$

and choose the **decision threshold** from the human labels. To avoid fooling ourselves, the
threshold is tuned on questions q01-q15 (`calib`) and reported on q16-q30 (`test`), with no question
in both. This is the same train/test discipline as any ML model.

Implement `judge_scores` and `pick_threshold`. Optionally improve `JUDGE_PROMPT` too."""),
    code("""JUDGE_PROMPT = \"\"\"Question: {question}
Reference answer: {reference}
Candidate answer: {candidate}

Is the candidate answer correct? Answer Yes or No.\"\"\"

def judge_scores(judge, rows: list, prompt: str = JUDGE_PROMPT) -> list:
    prompts = [prompt.format(question=r["question"], reference=r["reference"], candidate=r["candidate"]) for r in rows]
    lp = judge.chat_next_token_logprobs(prompts)     # [n_rows, vocab] log-probs of the first answer token
    # TODO: return log P("Yes") - log P("No") for each row (use judge.token_id("Yes") / ("No"))
    return [0.0] * len(rows)

def pick_threshold(scores: list, labels: list) -> float:
    # TODO: try every observed score t as a threshold (predict "correct" if score >= t)
    #       and return the t with the highest agreement with labels.
    #       On ties prefer the HIGHER t - accepting hallucinations is the costlier mistake.
    return 0.0

if USE_SOLUTIONS:
    from labkit.solutions.m02 import judge_scores, pick_threshold"""),
    code("""cal = calibration[calibration.split == "calib"].to_dict("records")
test = calibration[calibration.split == "test"].to_dict("records")
cal_scores, test_scores = judge_scores(big, cal), judge_scores(big, test)
cal_labels, test_labels = [r["human_label"] for r in cal], [r["human_label"] for r in test]

def judge_report(scores, labels, t):
    agree = sum((s >= t) == l for s, l in zip(scores, labels)) / len(labels)
    fpr = sum(s >= t and not l for s, l in zip(scores, labels)) / max(sum(not l for l in labels), 1)
    return {"agreement": round(agree, 2), "hallucinations_accepted": round(fpr, 2)}

THRESHOLD = pick_threshold(cal_scores, cal_labels)
display(pd.DataFrame({
    ("naive t=0", "calib"): judge_report(cal_scores, cal_labels, 0),
    ("naive t=0", "test"): judge_report(test_scores, test_labels, 0),
    (f"calibrated t={THRESHOLD:.2f}", "calib"): judge_report(cal_scores, cal_labels, THRESHOLD),
    (f"calibrated t={THRESHOLD:.2f}", "test"): judge_report(test_scores, test_labels, THRESHOLD),
}).T)

plt.figure(figsize=(8, 3))
plt.hist([s for s, l in zip(cal_scores, cal_labels) if l], bins=20, alpha=0.6, label="human: correct", color="#43a047")
plt.hist([s for s, l in zip(cal_scores, cal_labels) if not l], bins=20, alpha=0.6, label="human: incorrect", color="#e53935")
plt.axvline(0, c="grey", ls=":", label="naive (0)"); plt.axvline(THRESHOLD, c="k", ls="--", label="calibrated")
plt.xlabel("judge score  log P(Yes) - log P(No)"); plt.legend(); plt.title("Judge scores on the calib split"); plt.show()"""),
    md("""Two things to notice in the table: the naive judge accepts many hallucinations, and the
calibrated judge scores a little **lower on test than on calib**. That's overfitting the threshold,
and it's why you always report the held-out number."""),
    code("""checks.check_judge(pick_threshold, test_scores, test_labels, THRESHOLD, rt.tier)"""),
    md("Now grade the three systems with your calibrated judge:"),
    code("""lumora_scores = {}
for s in SYSTEMS:
    rows = [{"question": q, "reference": r, "candidate": a}
            for q, r, a in zip(lumora_eval.question, lumora_eval.reference, answers[s])]
    v = [x >= THRESHOLD for x in judge_scores(big, rows)]
    lumora_eval[f"correct_{s}"] = v
    lumora_scores[s] = sum(v) / len(v)
summary = pd.DataFrame({"judge_acc": lumora_scores, "token_f1": f1}).round(3)
summary"""),
    code("""lumora_eval.groupby("category")[[f"correct_{s}" for s in SYSTEMS]].mean().round(2)"""),
    md("""**Interpret:** closed-book accuracy is low for *both* model sizes. Scale doesn't buy
knowledge the model never saw. Open-book is high, so the information is useful once present.
That gap is what Module 3 closes. Note the remaining bias: the large model is judging its *own*
answers (self-preference). In production, use a different, stronger judge and re-calibrate it.""")]

cells += [md("""## 6. Exercise 4 - Pairwise comparison and Elo

Absolute scores need a reference answer. When you don't have one (chat quality, style), compare
two answers side by side and ask *which is better*. Chatbot Arena ranks models this way, turning
thousands of pairwise votes into **Elo ratings**:

$$E_A = \\frac{1}{1 + 10^{(R_B - R_A)/400}} \\qquad R_A' = R_A + K\\,(S_A - E_A)$$

where $S_A$ is 1 for a win, 0.5 for a tie and 0 for a loss. Implement both functions."""),
    code("""def elo_expected(r_a: float, r_b: float) -> float:
    # TODO: expected score of A against B
    return 0.5

def elo_update(r_a: float, r_b: float, score_a: float, k: float = 32) -> tuple:
    # TODO: return the new (r_a, r_b); the update must be zero-sum
    return r_a, r_b

if USE_SOLUTIONS:
    from labkit.solutions.m02 import elo_expected, elo_update"""),
    code("""checks.check_elo(elo_expected, elo_update)"""),
    md("""Now run a tournament: every pair of systems on every question. Each comparison is judged
**twice with the answers swapped**, because LLM judges tend to prefer whichever answer is shown
first. The match score is the judge's averaged probability that *a* is better (a number between 0
and 1), which Elo handles just like a win, loss or draw."""),
    code("""import random
from labkit.solutions.m02 import pairwise_scores, all_pairs

items, meta = [], []
for a_sys, b_sys in all_pairs(SYSTEMS):
    for i, row in lumora_eval.iterrows():
        items.append({"question": row.question, "reference": row.reference,
                      "a": answers[a_sys][i], "b": answers[b_sys][i]})
        meta.append((a_sys, b_sys))
scores, p_first = pairwise_scores(big, items)
matches = [(a, b, s) for (a, b), s in zip(meta, scores)]
print(f"{len(matches)} matches. Position bias: the judge prefers whichever answer is shown FIRST "
      f"{p_first:.0%} of the time (50% = unbiased) - swapping the order cancels this.")

def run_elo(matches, systems, k=32, init=1000, permutations=50, seed=0):
    rng, totals = random.Random(seed), {s: 0.0 for s in systems}
    for _ in range(permutations):           # Elo depends on match order -> average many orders
        ratings = {s: init for s in systems}
        order = matches[:]; rng.shuffle(order)
        for a, b, s in order:
            ratings[a], ratings[b] = elo_update(ratings[a], ratings[b], s, k)
        for s in systems:
            totals[s] += ratings[s] / permutations
    return dict(sorted(totals.items(), key=lambda kv: -kv[1]))

elo = run_elo(matches, SYSTEMS)
pd.Series(elo, name="elo").round(0)""")]

cells += [md("""## 7. Exercise 5 - Track everything with MLflow

You've produced a dozen numbers across models, prompts and metrics. Next week you'll fine-tune a
model and need to compare against *exactly* these baselines. MLflow records each run's **params**
(what you did) and **metrics** (what happened). Complete `log_run`, then log:
* one run per (model, k-shot) for MMLU, with metric `mmlu_acc` (+ per-subject metrics)
* one run per Lumora system, with metrics `lumora_acc`, `token_f1`, `elo`"""),
    code("""def log_run(run_name: str, params: dict, metrics: dict, tags: dict = None) -> str:
    with mlflow.start_run(run_name=run_name) as run:
        # TODO: log params, metrics and (optional) tags
        return run.info.run_id

if USE_SOLUTIONS:
    from labkit.solutions.m02 import log_run"""),
    code("""for (model, k), m in mmlu_results.items():
    log_run(f"mmlu-{model.split('/')[-1]}-{k}shot",
            params={"model": model, "benchmark": "mmlu", "k_shot": k, "n_per_subject": N_PER_SUBJECT},
            metrics=m, tags={"tier": rt.tier})
for s in SYSTEMS:
    log_run(f"lumora-{s}",
            params={"model": STUDENT if s.startswith("small") else GENERATOR, "benchmark": "lumora_qa",
                    "context": "open-book" if s.endswith("open") else "closed-book", "judge": GENERATOR},
            metrics={"lumora_acc": lumora_scores[s], "token_f1": f1[s], "elo": elo[s]},
            tags={"tier": rt.tier})
checks.check_mlflow("lumora-eval", ["mmlu_acc", "lumora_acc", "token_f1", "elo"], min_runs=len(mmlu_results) + len(SYSTEMS))"""),
    code("""runs = mlflow.search_runs(experiment_names=["lumora-eval"])
assert "params.benchmark" in runs, "No runs logged yet - complete log_run above first."
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
m = runs[runs["params.benchmark"] == "mmlu"].copy()
m["label"] = m["params.model"].str.split("/").str[-1] + " " + m["params.k_shot"] + "-shot"
m.sort_values("label").plot.barh(x="label", y="metrics.mmlu_acc", ax=axes[0], legend=False, color="#7e57c2")
axes[0].axvline(0.25, ls="--", c="grey"); axes[0].set_title("MMLU accuracy (dashed = chance)"); axes[0].set_ylabel("")
l = runs[runs["params.benchmark"] == "lumora_qa"].copy()
l["label"] = l["tags.mlflow.runName"].str.replace("lumora-", "")
l.sort_values("label").plot.barh(x="label", y="metrics.lumora_acc", ax=axes[1], legend=False, color="#ffb300")
axes[1].set_title("Lumora QA accuracy (judge)"); axes[1].set_ylabel("")
plt.tight_layout(); plt.show()
runs[["tags.mlflow.runName", "params.model", "metrics.mmlu_acc", "metrics.lumora_acc", "metrics.elo"]]"""),
    md("""**Want the MLflow UI?** Copy `mlflow.db` from your Drive / workdir to your laptop and run
`mlflow ui --backend-store-uri sqlite:///mlflow.db`, then open http://127.0.0.1:5000.""")]

cells += [md("""## 8. Pro bonus - cross-check with `lm-evaluation-harness`

*Runs on Colab Pro.* Our MMLU numbers come from code you wrote. Production teams use standard
harnesses so results are comparable across papers. EleutherAI's `lm-evaluation-harness` (which
NeMo Evaluator can also run) implements the same log-likelihood method. Your number for
`high_school_physics` 5-shot should be close to the harness's on the same questions. Small
differences come from prompt whitespace, which is itself a lesson in how fragile benchmarks are."""),
    code("""RUN_HARNESS = rt.is_pro
if RUN_HARNESS:
    !pip install -q "lm-eval>=0.4.5"
    !lm_eval --model hf --model_args pretrained={GENERATOR},dtype=bfloat16 \\
        --tasks mmlu_high_school_physics --num_fewshot 5 --limit {N_PER_SUBJECT} \\
        --batch_size 8 --device cuda:0
    print("Your implementation:", round(mmlu_results[(GENERATOR, 5)]["mmlu_high_school_physics"], 3))
else:
    print("Skipped - switch to an L4/A100 runtime (Colab Pro), or set RUN_HARNESS = True to try it on T4 (slow).")""")]

cells += [md("## 9. Save baselines and wrap up"),
    code("""baselines = {
    "mmlu": {f"{m}|{k}shot": v["mmlu_acc"] for (m, k), v in mmlu_results.items()},
    "lumora_judge_acc": lumora_scores, "lumora_token_f1": f1, "elo": elo,
    "judge_model": GENERATOR, "judge_prompt": JUDGE_PROMPT, "judge_threshold": THRESHOLD,
}
(OUT / "baselines.json").write_text(json.dumps(baselines, indent=2))
print("Saved", OUT / "baselines.json", "- Module 3 compares fine-tuned models against these.")
progress.summary("m02")"""),
    md("""### Key takeaways
* **Eyeball first, then measure.** Fluent does not mean correct.
* **Log-likelihood scoring** makes multiple-choice benchmarks cheap, deterministic and parse-free.
* **Few-shot prompts** help small models follow formats (MMLU *and* the judge).
* **Calibrate your judge** against human labels on a held-out split: use its *confidence* and
  pick the threshold from data. All judges have position and self-preference biases.
* **Pairwise + Elo** ranks systems when there is no single right answer.
* **Track every run.** You'll need these exact baselines in Module 3.

### How this maps to NVIDIA NeMo Evaluator
| This lab | NeMo Evaluator |
|---|---|
| hand-written MMLU loop / `lm-evaluation-harness` | academic benchmark jobs (MMLU, GSM8K, ...) via API |
| few-shot vs zero-shot | `num_fewshot` in the evaluation config |
| custom judge prompt + calibration | LLM-as-a-judge evaluation with custom metrics |
| Lumora QA benchmark | custom dataset evaluation |
| MLflow tracking | results store + MLflow integration |

**Next:** Module 3 - inject Lumora knowledge with CPT, teach skills with SFT, and align style with DPO.""")]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
})
out = Path(__file__).resolve().parents[1] / "notebooks" / "02_evaluation.ipynb"
nbf.write(nb, out)
print("wrote", out)
