"""Auto-checks for lab exercises.

Each check prints a short report, records the result in progress.json and
returns True/False. Hints point at the concept, not the answer.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Iterable

from . import progress

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?){2,4}\d{2,4}")


def _report(ex_id: str, results: list[tuple[bool, str, str]]) -> bool:
    ok = all(r[0] for r in results)
    print(f"--- {ex_id} ---")
    for passed, msg, hint in results:
        print(f"  {'PASS' if passed else 'FAIL'}  {msg}")
        if not passed and hint:
            print(f"        hint: {hint}")
    print("  >>> All checks passed!" if ok else "  >>> Not yet - fix the FAIL lines and re-run.")
    progress.mark(ex_id, ok, f"{sum(r[0] for r in results)}/{len(results)}")
    return ok


def _key(repo_dir: Path) -> dict:
    return json.loads((Path(repo_dir) / "data" / "raw" / "answer_key.json").read_text())


# ---------------------------------------------------------------- Module 1
def check_clean_text(clean_text: Callable[[str], str]) -> bool:
    cases = [
        ("<p>The Vault weighs <b>98 kg</b>.</p>", "The Vault weighs 98 kg.",
         "Strips HTML tags", "Use a regex like <[^>]+> or BeautifulSoup.get_text()."),
        ("<script>trackPage()</script><p>Rated IP55.</p>", "Rated IP55.",
         "Drops <script>/<style> contents, not just the tags", "Remove whole <script>...</script> blocks before stripping tags."),
        ("Usable   capacity\n\n\nis  10.2 kWh", "Usable capacity is 10.2 kWh",
         "Collapses repeated whitespace", "re.sub(r'\\s+', ' ', text).strip()"),
        ("Fish &amp; chips &lt;3", "Fish & chips <3",
         "Decodes HTML entities", "html.unescape()"),
        ("Home | Products | Support | Find an Installer | Cookie settings\nSkip to main content\nThe Bridge is 10 kW.",
         "The Bridge is 10 kW.", "Removes navigation boilerplate lines",
         "Drop known boilerplate lines (see BOILERPLATE_PATTERNS) before collapsing whitespace."),
        ("Lumora Vault​ 10", "Lumora Vault 10",
         "Normalises unicode spaces / zero-width chars", "unicodedata.normalize('NFKC', ...) and remove \\u200b."),
    ]
    results = []
    for raw, expected, msg, hint in cases:
        try:
            got = clean_text(raw)
        except Exception as e:  # noqa: BLE001
            results.append((False, f"{msg}  (raised {type(e).__name__}: {e})", hint))
            continue
        results.append((got == expected, f"{msg}" + ("" if got == expected else f"  got {got!r}"), hint))
    return _report("m01.ex1_clean_text", results)


def check_quality_filter(df, repo_dir: Path) -> bool:
    key = _key(repo_dir)
    ids = set(df["id"])
    bad = {"junk": key["junk"], "spam": key["spam"], "non_english": key["non_english"]}
    must_keep = key["clean"] + key["pii"]
    results = []
    for kind, bad_ids in bad.items():
        left = [i for i in bad_ids if i in ids]
        results.append((not left, f"Removed all {kind} documents ({len(bad_ids) - len(left)}/{len(bad_ids)})",
                        {"junk": "A minimum word count filter catches these.",
                         "spam": "Look at the ratio of unique words to total words.",
                         "non_english": "Use a language-ID model (langdetect here; fastText in NeMo Curator)."}[kind]))
    lost = [i for i in must_keep if i not in ids]
    results.append((not lost, f"Kept all good documents ({len(must_keep) - len(lost)}/{len(must_keep)})",
                    "A filter is too aggressive - print the removed docs to see which rule fired."))
    return _report("m01.ex2_quality_filter", results)


def check_dedup(df, repo_dir: Path) -> bool:
    key = _key(repo_dir)
    tickets = set(key["pii"])
    non_ticket = df[~df["id"].isin(tickets)]
    titles = list(non_ticket["title"])
    dup_titles = sorted({t for t in titles if titles.count(t) > 1})
    results = [
        (not dup_titles, f"No duplicated content left ({len(dup_titles)} titles still duplicated)",
         "Exact hashing misses near-duplicates and HTML copies; use MinHash with a lower threshold."),
        (len(non_ticket) == len(key["clean"]), f"{len(non_ticket)} unique knowledge docs kept (expected {len(key['clean'])})",
         "If fewer, your threshold is too low and merges distinct docs."),
        (tickets <= set(df["id"]), "All support tickets kept", "Tickets are unique - they should survive dedup."),
    ]
    return _report("m01.ex3_dedup", results)


def check_pii(df, repo_dir: Path) -> bool:
    key = _key(repo_dir)
    tickets = df[df["id"].isin(key["pii"])]
    emails = sum(len(EMAIL_RE.findall(t)) for t in tickets["text"])
    phones = sum(len([m for m in PHONE_RE.findall(t) if sum(c.isdigit() for c in m) >= 9]) for t in tickets["text"])
    names = ["Maria Jansen", "David Okafor", "Sofia Romano", "Peter Lindqvist", "Anna Kowalski", "Jean Dupont"]
    name_hits = [n for n in names if any(n in t for t in tickets["text"])]
    codes_kept = all(any(c in t for t in tickets["text"]) for c in ["E-204", "E-101", "E-317"])
    results = [
        (emails == 0, f"Emails redacted ({emails} remaining)", "Presidio entity EMAIL_ADDRESS."),
        (phones == 0, f"Phone numbers redacted ({phones} remaining)", "Presidio entity PHONE_NUMBER; check international formats."),
        (not name_hits, f"Person names redacted ({len(name_hits)} remaining: {name_hits[:3]})", "Presidio entity PERSON."),
        (codes_kept, "Error codes like E-204 preserved (no over-redaction)",
         "Do not redact everything that looks like a number - restrict entities."),
    ]
    return _report("m01.ex4_pii", results)


def check_qa(pairs: Iterable[dict], min_pairs: int = 20) -> bool:
    pairs = list(pairs)
    schema_ok = all(isinstance(p, dict) and {"question", "answer", "context_id"} <= p.keys() for p in pairs)
    qs = [p.get("question", "").strip().lower() for p in pairs]
    unique = len(set(qs)) == len(qs)
    qmarks = sum(q.endswith("?") for q in qs) / max(len(qs), 1)
    # Re-verify grounding with the reference filter (don't trust the learner's own flag).
    from .solutions.m01 import is_grounded as _ref_grounded
    grounded = sum(_ref_grounded(p.get("answer", ""), p.get("context", "")) for p in pairs) / max(len(pairs), 1)
    results = [
        (len(pairs) >= min_pairs, f"{len(pairs)} QA pairs (need >= {min_pairs})",
         "Generate more questions per chunk, or loosen your grounding filter slightly."),
        (schema_ok, "Every pair has question / answer / context_id", "Keep the dict keys consistent."),
        (unique, "No duplicate questions", "Deduplicate on the lower-cased question text."),
        (qmarks >= 0.9, f"{qmarks:.0%} of questions end with '?'", "Filter out malformed generations."),
        (grounded >= 0.9, f"{grounded:.0%} of answers grounded in their source chunk",
         "Implement is_grounded() and keep only pairs where it returns True."),
    ]
    return _report("m01.ex5_synthetic_qa", results)


# ---------------------------------------------------------------- Module 2
def check_mmlu_prompt(build_prompt, example: dict, shots: list[dict], subject: str) -> bool:
    results = []
    for k in (0, len(shots)):
        try:
            p = build_prompt(example, shots[:k], subject)
        except Exception as e:  # noqa: BLE001
            results.append((False, f"{k}-shot prompt raised {type(e).__name__}: {e}", ""))
            continue
        answered = re.findall(r"\nAnswer: [ABCD]\b", p)
        last = p.rstrip().split("\n\n")[-1]
        results += [
            (p.rstrip().endswith("Answer:"), f"{k}-shot: ends with 'Answer:' (no answer leaked)",
             "The test question must NOT include its answer."),
            (len(answered) == k, f"{k}-shot: contains {k} solved examples ({len(answered)} found)",
             "Each shot should end with 'Answer: <letter>'."),
            (all(f"{L}. " in last for L in "ABCD"), f"{k}-shot: test question lists choices as 'A. ...' to 'D. ...'",
             "Label choices with letters so the model can answer with one token."),
            (subject.replace("_", " ") in p.split("\n")[0], f"{k}-shot: header names the subject",
             "Start with 'The following are multiple choice questions (with answers) about <subject>.'"),
        ]
    return _report("m02.ex1_mmlu_prompt", results)


def check_predict(predict_letters, model, mmlu: dict, n_per_subject: int = 4) -> bool:
    """Compare against reference log-likelihood scoring on prompts built by the reference builder."""
    from .solutions.m02 import build_prompt, predict_letters as ref
    prompts = [build_prompt(e, d["dev"], s) for s, d in mmlu.items() for e in d["test"][:n_per_subject]]
    try:
        got = predict_letters(model, prompts)
    except Exception as e:  # noqa: BLE001
        return _report("m02.ex2_scoring", [(False, f"raised {type(e).__name__}: {e}", "")])
    want = ref(model, prompts)
    valid = got is not None and len(got) == len(prompts) and all(g in list("ABCD") for g in got)
    agree = sum(g == w for g, w in zip(got or [], want)) / len(want)
    results = [
        (valid, "Returns one letter A-D per prompt", "Map the argmax index back to a letter."),
        (valid and agree >= 0.95, f"Matches log-likelihood scoring on {agree:.0%} of {len(prompts)} prompts",
         "Compare the log-probs of the tokens ' A', ' B', ' C', ' D' (note the leading space)."),
    ]
    return _report("m02.ex2_scoring", results)


JUDGE_TARGET = {"pro": (0.85, 0.10), "free": (0.80, 0.10), "mps": (0.80, 0.10), "cpu": (0.65, 0.25)}


def check_judge(pick_threshold, test_scores: list[float], test_labels: list[bool], threshold: float, tier: str) -> bool:
    results = []
    # unit test for pick_threshold on a tiny separable example
    try:
        t = pick_threshold([-3, -1, 0.5, 2, 4], [False, False, False, True, True])
        results.append((0.5 < t <= 2, f"pick_threshold separates a toy example (got {t})",
                        "Try every observed score as a threshold; keep the one with the best agreement."))
    except Exception as e:  # noqa: BLE001
        results.append((False, f"pick_threshold raised {type(e).__name__}: {e}", ""))
    preds = [s >= threshold for s in test_scores]
    agree = sum(p == l for p, l in zip(preds, test_labels)) / len(test_labels)
    negatives = sum(not l for l in test_labels)
    fpr = sum(p and not l for p, l in zip(preds, test_labels)) / max(negatives, 1)
    target, max_fpr = JUDGE_TARGET.get(tier, (0.8, 0.10))
    results += [
        (agree >= target, f"Held-out agreement with humans {agree:.0%} (target {target:.0%})",
         "Pick the threshold on the calib split, then apply it unchanged to the test split."),
        (fpr <= max_fpr, f"Hallucinations accepted as correct: {fpr:.0%} (max {max_fpr:.0%})",
         "A threshold of 0 trusts the judge's own Yes/No - small judges are lenient. Calibrate it."),
    ]
    return _report("m02.ex3_judge", results)


def check_elo(elo_expected, elo_update) -> bool:
    def close(a, b, tol=0.01):
        return abs(a - b) <= tol

    results = []
    try:
        results.append((close(elo_expected(1000, 1000), 0.5), "Equal ratings -> expected score 0.5", "1 / (1 + 10 ** ((rb - ra) / 400))"))
        results.append((close(elo_expected(1400, 1000), 0.909), "400-point favourite -> expected ~0.91", "1 / (1 + 10 ** ((rb - ra) / 400))"))
        a, b = elo_update(1000, 1000, 1.0, k=32)
        results.append((close(a, 1016) and close(b, 984), "Win between equals with K=32 -> 1016 / 984", "delta = k * (score_a - expected_a)"))
        a, b = elo_update(1200, 1000, 0.5, k=32)
        results.append((a < 1200 and b > 1000, "A draw against a weaker player costs rating", "Draws score 0.5."))
        results.append((close(a + b, 2200), "Updates are zero-sum", "Subtract the same delta from B that you add to A."))
    except Exception as e:  # noqa: BLE001
        results.append((False, f"raised {type(e).__name__}: {e}", ""))
    return _report("m02.ex4_elo", results)


def check_mlflow(experiment: str, required_metrics: list[str], min_runs: int) -> bool:
    import mlflow

    runs = mlflow.search_runs(experiment_names=[experiment])
    cols = set(runs.columns)
    missing = [m for m in required_metrics if f"metrics.{m}" not in cols]
    results = [
        (len(runs) >= min_runs, f"{len(runs)} runs logged in '{experiment}' (need >= {min_runs})",
         "Call log_run once per model x condition."),
        (not missing, "Required metrics logged" + (f" (missing {missing})" if missing else ""),
         "Use mlflow.log_metrics({...}) inside the run."),
        ("params.model" in cols, "Each run records its model as a param", "mlflow.log_params({'model': ...})"),
    ]
    return _report("m02.ex5_mlflow", results)


# ---------------------------------------------------------------- Module 3
def check_cpt_setup(dataset, config, eos_token: str) -> bool:
    results = []
    cols = getattr(dataset, "column_names", [])
    results.append(("text" in cols, "Dataset has a 'text' column", "Dataset.from_dict({'text': [...]})"))
    if "text" in cols:
        texts = dataset["text"]
        results.append((all(t.endswith(eos_token) for t in texts), "Every document ends with the EOS token",
                        "Append tokenizer.eos_token so the model learns where documents end."))
        results.append((len(texts) >= 20, f"{len(texts)} training texts (originals + paraphrases)", "Use the whole CPT corpus."))
    lr = getattr(config, "learning_rate", None)
    results += [
        (getattr(config, "completion_only_loss", None) is False, "completion_only_loss=False (CPT learns every token)",
         "CPT is plain language modelling - there is no prompt to mask."),
        (lr is not None and 5e-6 <= lr <= 1e-4, f"Learning rate {lr} suits full fine-tuning (5e-6 .. 1e-4)",
         "Full fine-tuning needs a much smaller LR than LoRA (~1e-5 .. 5e-5)."),
        (getattr(config, "num_train_epochs", 0) >= 3, f"{getattr(config, 'num_train_epochs', 0)} epochs (>= 3 for a tiny corpus)",
         "A 70-document corpus needs several passes to be absorbed."),
    ]
    return _report("m03.ex1_cpt_setup", results)


def check_cpt_result(ppl_before: float, ppl_after: float, strict: bool = True) -> bool:
    drop = 1 - ppl_after / ppl_before
    results = [(drop >= 0.3 or not strict, f"Perplexity on held-out Lumora facts: {ppl_before:.1f} -> {ppl_after:.1f} ({drop:.0%} lower)",
                "Train longer or with a higher learning rate; check the loss went down.")]
    return _report("m03.ex1_cpt_result", results)


def check_sft_setup(records: list, kept_questions: list, eval_questions: list, lora_config, n_raw: int) -> bool:
    from .solutions.m03 import is_contaminated as ref_contam
    results = []
    r0 = records[0] if records else {}
    ok_shape = (isinstance(r0.get("prompt"), list) and isinstance(r0.get("completion"), list)
                and [m.get("role") for m in r0.get("prompt", [])] == ["system", "user"]
                and [m.get("role") for m in r0.get("completion", [])] == ["assistant"])
    results.append((ok_shape, "Records are {'prompt': [system, user], 'completion': [assistant]}",
                    "Prompt/completion format makes TRL compute loss on the answer only."))
    leaked = [q for q in kept_questions if ref_contam(q, eval_questions)]
    results.append((not leaked, f"No near-duplicates of eval questions in training data ({len(leaked)} found)",
                    "Compare content-word Jaccard against every eval question and drop matches >= 0.6."))
    results.append((len(kept_questions) >= 0.7 * n_raw, f"Kept {len(kept_questions)}/{n_raw} pairs (not over-filtered)",
                    "Your contamination filter is too aggressive."))
    targets = set(getattr(lora_config, "target_modules", None) or [])
    results.append(({"q_proj", "v_proj"} <= targets and bool(targets & {"gate_proj", "up_proj", "down_proj"}),
                    "LoRA targets attention AND MLP projections", "Add the MLP layers: gate_proj, up_proj, down_proj."))
    r = getattr(lora_config, "r", 0)
    results.append((4 <= r <= 128, f"LoRA rank r={r}", "Typical ranks are 8-64."))
    return _report("m03.ex2_sft_setup", results)


def check_preferences(records: list, dpo_config, style_markers: list) -> bool:
    from .solutions.m03 import casualness
    results = []
    r0 = records[0] if records else {}
    results.append((all(k in r0 for k in ("prompt", "chosen", "rejected")), "Records have prompt / chosen / rejected", ""))
    prompts_txt = " ".join(m["content"] for r in records for m in r.get("prompt", []))
    leaked = [s for s in style_markers if s.lower() in prompts_txt.lower()]
    results.append((not leaked, "Style instructions are NOT in the prompt",
                    "If the prompt says 'be casual', the model learns to obey instructions, not the style."))
    gap = (sum(casualness(r["chosen"][0]["content"]) for r in records)
           - sum(casualness(r["rejected"][0]["content"]) for r in records)) / max(len(records), 1)
    results.append((gap >= 0.2, f"Chosen replies are clearly more casual than rejected (gap {gap:.2f})",
                    "Did you swap chosen and rejected?"))
    beta = getattr(dpo_config, "beta", None)
    results.append((beta is not None and 0.05 <= beta <= 0.5, f"beta={beta}", "beta controls how far the policy may drift from the reference; 0.1 is a common start."))
    return _report("m03.ex3_preferences", results)


def check_dpo_result(casual_before: float, casual_after: float, reward_acc: float, strict: bool = True) -> bool:
    results = [
        (reward_acc >= 0.8 or not strict, f"Final reward accuracy {reward_acc:.0%} (chosen preferred over rejected)",
         "Train for more steps or check the dataset."),
        (casual_after >= casual_before + 0.3 or not strict, f"Casualness on unseen prompts {casual_before:.2f} -> {casual_after:.2f}",
         "The policy barely moved: raise the learning rate a little or train another epoch."),
    ]
    return _report("m03.ex3_dpo_result", results)


def check_knowledge_gain(scores: dict, strict: bool = True) -> bool:
    base, best = scores.get("base", 0), scores.get("cpt+sft", 0)
    results = [
        (best >= base + 0.15 or not strict, f"CPT+SFT improves closed-book Lumora accuracy: {base:.0%} -> {best:.0%}",
         "Check that CPT used full fine-tuning and that SFT started from the CPT model."),
        (best >= scores.get("sft", 0) or not strict, f"CPT+SFT ({best:.0%}) >= SFT alone ({scores.get('sft', 0):.0%})",
         "SFT alone teaches the format; CPT supplies the knowledge."),
    ]
    return _report("m03.ex4_knowledge", results)


# ---------------------------------------------------------------- Module 4
def check_quantizer(quantize, dequantize) -> bool:
    import torch
    torch.manual_seed(0)
    w = torch.randn(64, 256) * 0.02
    w_out = w.clone(); w_out[:, 3] = 1.0          # one outlier column
    results = []
    try:
        q, s = quantize(w, 8, None)
        rel = ((dequantize(q, s) - w).norm() / w.norm()).item()
        results += [
            (q.shape == w.shape, "Quantized tensor keeps the weight shape", ""),
            (q.dtype in (torch.int8, torch.int16, torch.int32) and int(q.abs().max()) <= 127,
             "INT8 values are integers in [-127, 127]", "round(w / scale) then clamp to +/-qmax"),
            (rel < 0.01, f"INT8 per-channel round-trip error {rel:.2%} (< 1%)", "scale = max|w| / 127 per row"),
        ]
        q4, s4 = quantize(w, 4, None)
        results.append((int(q4.abs().max()) <= 7, "INT4 values are in [-7, 7]", "qmax = 2**(bits-1) - 1"))
        e_ch = ((dequantize(*quantize(w_out, 4, None)) - w_out)[:, 4:].norm() / w_out[:, 4:].norm()).item()
        e_g = ((dequantize(*quantize(w_out, 4, 32)) - w_out)[:, 4:].norm() / w_out[:, 4:].norm()).item()
        results.append((e_g < e_ch * 0.6, f"Per-group scales protect normal weights from an outlier ({e_ch:.0%} -> {e_g:.0%} error)",
                        "With group_size, compute one scale per block of group_size weights in each row."))
    except Exception as e:  # noqa: BLE001
        results.append((False, f"raised {type(e).__name__}: {e}", ""))
    return _report("m04.ex1_quantizer", results)


def check_quant_results(ppl: dict) -> bool:
    base = ppl["fp16"]
    results = [
        (ppl["int8"] <= base * 1.05, f"INT8 perplexity within 5% of FP16 ({base:.2f} -> {ppl['int8']:.2f})",
         "Per-channel INT8 should be almost lossless - check your scales."),
        (ppl["int4-g128"] < ppl["int4-channel"], f"INT4 per-group beats per-channel ({ppl['int4-channel']:.2f} -> {ppl['int4-g128']:.2f})",
         "Smaller groups = less damage from outliers."),
    ]
    return _report("m04.ex1_quant_results", results)


def check_pruning(importance: list, drop: list, model, n_layers: int, n_drop: int) -> bool:
    import torch
    spread = max(importance) - min(importance) if importance else 0
    results = [
        (len(importance) == n_layers, f"One importance score per layer ({len(importance)}/{n_layers})", "Use hidden_states[i] -> hidden_states[i+1]."),
        (spread > 0.05 and all(0 <= x <= 2 for x in importance),
         f"Importance scores vary across layers (range {min(importance):.3f}-{max(importance):.3f})" if importance else "Importance scores computed",
         "Compute 1 - mean cosine similarity between each layer's input and output hidden states."),
        (len(drop) == n_drop and 0 not in drop and n_layers - 1 not in drop,
         f"Dropping {len(drop)} layers, keeping the first and last", "Never drop layer 0 or the final layer."),
    ]
    inner = [i for i in range(1, n_layers - 1)]
    lowest = sorted(sorted(inner, key=lambda i: importance[i])[:n_drop])
    results.append((sorted(drop) == lowest, "Dropped the least important layers", "Sort candidate layers by importance, ascending."))
    results.append((len(model.model.layers) == n_layers - n_drop and model.config.num_hidden_layers == n_layers - n_drop,
                    f"Model now has {len(model.model.layers)} layers and the config agrees", "Update model.config.num_hidden_layers too."))
    try:
        dev = next(model.parameters()).device
        with torch.no_grad():
            model(input_ids=torch.tensor([[1, 2, 3]], device=dev), use_cache=True)
        results.append((True, "Pruned model still runs a forward pass with the KV cache", ""))
    except Exception as e:  # noqa: BLE001
        results.append((False, f"Forward pass failed: {type(e).__name__}: {e}", "Re-number self_attn.layer_idx for the kept layers."))
    return _report("m04.ex2_pruning", results)


def check_distill_loss(loss_fn) -> bool:
    import torch
    torch.manual_seed(0)
    s = torch.randn(2, 5, 11); t = torch.randn(2, 5, 11); y = torch.randint(0, 11, (2, 5)); y[0, 0] = -100
    import torch.nn.functional as F
    ce = F.cross_entropy(s[y != -100], y[y != -100]).item()
    results = []
    try:
        same = loss_fn(s, s.clone(), y, T=2.0, alpha=1.0).item()
        results.append((abs(same) < 1e-4, f"Identical student/teacher -> KL term is 0 (got {same:.4f})", "KL(p || p) = 0"))
        only_ce = loss_fn(s, t, y, T=2.0, alpha=0.0).item()
        results.append((abs(only_ce - ce) < 1e-3, "alpha=0 gives plain cross-entropy on the labels", "(1 - alpha) * CE"))
        kd = loss_fn(s, t, y, T=2.0, alpha=1.0).item()
        results.append((kd > 0, "KL term is positive when distributions differ", ""))
        ref = (F.kl_div(F.log_softmax(s[y != -100] / 2, -1), F.log_softmax(t[y != -100] / 2, -1),
                        log_target=True, reduction="batchmean") * 4).item()
        results.append((abs(kd - ref) < 1e-3, "KL uses temperature-softened distributions scaled by T^2",
                        "Divide logits by T before softmax and multiply the KL by T*T; ignore -100 positions."))
    except Exception as e:  # noqa: BLE001
        results.append((False, f"raised {type(e).__name__}: {e}", ""))
    return _report("m04.ex3_distill_loss", results)


def check_recovery(ppl_base: float, ppl_pruned: float, ppl_healed: float, strict: bool = True) -> bool:
    gap = ppl_pruned - ppl_base
    rec = (ppl_pruned - ppl_healed) / gap if gap > 0 else 0
    results = [(rec >= 0.3 or not strict, f"Distillation recovered {rec:.0%} of the pruning damage "
                f"(ppl {ppl_base:.1f} -> pruned {ppl_pruned:.1f} -> healed {ppl_healed:.1f})",
                "Train for more steps, or check the loss is going down."),
               # Guards against "recovery" from a model that pruning broke completely.
               (ppl_healed <= 3 * ppl_base or not strict,
                f"Healed model is within 3x of the original perplexity ({ppl_healed:.1f} vs {ppl_base:.1f})",
                "Check your pruning (Exercise 2) and your distillation loss: the healed model should be usable.")]
    return _report("m04.ex3_recovery", results)
