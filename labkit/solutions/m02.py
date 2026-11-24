"""Reference solutions for Module 2 (evaluation). Try the exercises first!"""
from __future__ import annotations

import itertools
import random


LETTERS = ["A", "B", "C", "D"]


# ------------------------------------------------------------ Exercise 1: MMLU prompts
def format_example(ex: dict, include_answer: bool) -> str:
    lines = [ex["question"].strip()]
    lines += [f"{L}. {c}" for L, c in zip(LETTERS, ex["choices"])]
    lines.append("Answer:" + (f" {LETTERS[ex['answer']]}" if include_answer else ""))
    return "\n".join(lines)


def build_prompt(test_ex: dict, shots: list[dict], subject: str) -> str:
    header = f"The following are multiple choice questions (with answers) about {subject.replace('_', ' ')}.\n\n"
    blocks = [format_example(s, include_answer=True) for s in shots]
    blocks.append(format_example(test_ex, include_answer=False))
    return header + "\n\n".join(blocks)


# ------------------------------------------------------------ Exercise 2: scoring
def predict_letters(model, prompts: list[str]) -> list[str]:
    """Pick the answer letter with the highest next-token log-probability."""
    logprobs = model.next_token_logprobs(prompts)
    letter_ids = [model.token_id(f" {L}") for L in LETTERS]
    best = logprobs[:, letter_ids].argmax(dim=-1).tolist()
    return [LETTERS[i] for i in best]


# ------------------------------------------------------------ Exercise 3: LLM-as-a-judge
JUDGE_PROMPT = """Question: {question}
Reference answer: {reference}
Candidate answer: {candidate}

Is the candidate answer correct? Answer Yes or No."""


def judge_scores(judge, rows: list[dict], prompt: str = JUDGE_PROMPT) -> list[float]:
    """Judge confidence per row: log P("Yes") - log P("No") for the first answer token.

    A continuous score lets us pick the decision threshold from human labels,
    instead of trusting the judge's own (usually lenient) Yes/No.
    """
    prompts = [prompt.format(question=r["question"], reference=r["reference"], candidate=r["candidate"]) for r in rows]
    lp = judge.chat_next_token_logprobs(prompts)
    yes, no = judge.token_id("Yes"), judge.token_id("No")
    return (lp[:, yes] - lp[:, no]).tolist()


def pick_threshold(scores: list[float], labels: list[bool]) -> float:
    """Threshold t (predict correct if score >= t) that maximises agreement with humans.

    Ties are broken towards the higher (stricter) threshold, because accepting a
    hallucination is usually worse than rejecting a correct answer.
    """
    candidates = sorted(set(scores)) + [max(scores) + 1e-6]
    best_t, best_acc = candidates[0], -1.0
    for t in candidates:
        acc = sum((s >= t) == l for s, l in zip(scores, labels)) / len(labels)
        if acc >= best_acc:
            best_t, best_acc = t, acc
    return best_t


def judge_correct(judge, rows: list[dict], threshold: float, prompt: str = JUDGE_PROMPT) -> list[bool]:
    return [s >= threshold for s in judge_scores(judge, rows, prompt)]


# ------------------------------------------------------------ Exercise 4: pairwise + Elo
PAIRWISE_PROMPT = """Question: {question}
Reference answer: {reference}

Answer A: {a}
Answer B: {b}

Which answer is more accurate with respect to the reference answer? Reply A or B."""


def pairwise_scores(judge, items: list[dict]) -> tuple[list[float], float]:
    """Score for system 'a' in each item, in [0, 1], plus the judge's position bias.

    Each pair is judged in both orders. P(a wins) = mean of P("A") when a is shown
    first and P("B") when a is shown second, which cancels position bias.
    """
    import torch

    fwd = [PAIRWISE_PROMPT.format(question=i["question"], reference=i["reference"], a=i["a"], b=i["b"]) for i in items]
    rev = [PAIRWISE_PROMPT.format(question=i["question"], reference=i["reference"], a=i["b"], b=i["a"]) for i in items]
    lp = judge.chat_next_token_logprobs(fwd + rev)
    ab = torch.softmax(lp[:, [judge.token_id("A"), judge.token_id("B")]], dim=-1)  # P(A), P(B)
    n = len(items)
    p_first = ab[:, 0].mean().item()  # how often the first-shown answer is preferred
    scores = [((ab[i, 0] + ab[n + i, 1]) / 2).item() for i in range(n)]
    return scores, p_first


def elo_expected(r_a: float, r_b: float) -> float:
    return 1 / (1 + 10 ** ((r_b - r_a) / 400))


def elo_update(r_a: float, r_b: float, score_a: float, k: float = 32) -> tuple[float, float]:
    e_a = elo_expected(r_a, r_b)
    delta = k * (score_a - e_a)
    return r_a + delta, r_b - delta


def run_elo(matches: list[tuple[str, str, float]], systems: list[str], k: float = 32,
            init: float = 1000, permutations: int = 50, seed: int = 0) -> dict[str, float]:
    """Average Elo over several random match orders (Elo is order-dependent)."""
    rng = random.Random(seed)
    totals = {s: 0.0 for s in systems}
    for _ in range(permutations):
        ratings = {s: init for s in systems}
        order = matches[:]
        rng.shuffle(order)
        for a, b, s in order:
            ratings[a], ratings[b] = elo_update(ratings[a], ratings[b], s, k)
        for s in systems:
            totals[s] += ratings[s] / permutations
    return dict(sorted(totals.items(), key=lambda kv: -kv[1]))


def all_pairs(systems: list[str]):
    return list(itertools.combinations(systems, 2))


# ------------------------------------------------------------ Exercise 5: MLflow
def log_run(run_name: str, params: dict, metrics: dict, tags: dict | None = None) -> str:
    import mlflow

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)
        return run.info.run_id
