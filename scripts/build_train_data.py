"""Build the reference training data shipped for Module 3.

Learners can use their own Module 1 outputs instead; these files make Module 3
runnable on its own.

data/train/cpt_corpus.jsonl  - the 20 clean Lumora docs + LLM paraphrases
                               (paraphrases kept only if every number survives)
data/train/sft_qa.jsonl      - grounded synthetic QA pairs (Module 1 pipeline)
data/train/style_prompts.jsonl - everyday prompts for DPO style alignment (hand-written)

Run (needs a GPU or Apple Silicon, ~10 min):  python scripts/build_train_data.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "data" / "train"

STYLE_PROMPTS = [
    # account & orders
    "How do I reset my password?", "Can I change my delivery address after ordering?",
    "My package hasn't arrived yet. What should I do?", "How do I cancel my subscription?",
    "Can I get a refund if I change my mind?", "Why was my card declined?",
    "How do I update my billing details?", "Do you ship internationally?",
    "Can I order without creating an account?", "How do I track my order?",
    "What payment methods do you accept?", "How long does a refund take?",
    "I was charged twice. Can you help?", "Can I pause my subscription instead of cancelling?",
    "How do I download my invoice?", "Is there a student discount?",
    # product help
    "My phone battery drains really fast. Any tips?", "How do I connect my printer to Wi-Fi?",
    "What's the difference between an SSD and a hard drive?", "How often should I update my software?",
    "My laptop is running slowly. What can I try?", "How do I back up my photos?",
    "What does 'restart to update' actually do?", "Is it safe to unplug a USB stick without ejecting it?",
    "How can I make my Wi-Fi faster?", "Should I turn my computer off every night?",
    "How do I set up two-factor authentication?", "What is a VPN and do I need one?",
    "How do I clear my browser cache?", "Why does my smart speaker not hear me?",
    # energy & home (on-theme, but no Lumora facts)
    "How can I lower my electricity bill?", "Is it worth getting solar panels?",
    "What does kWh mean on my energy bill?", "How do I read my electricity meter?",
    "What appliances use the most energy at home?", "Is it cheaper to run the dishwasher at night?",
    "How do heat pumps work?", "What should I do during a power cut?",
    "Are LED bulbs really better?", "How do I keep my house warm without the heating on?",
    # everyday life
    "Can you recommend a good book for a long flight?", "How do I make a good cup of coffee at home?",
    "What's an easy dinner I can cook in 20 minutes?", "How can I sleep better?",
    "Any tips for staying focused while working from home?", "How do I start running as a beginner?",
    "What's a good way to learn a new language?", "How do I keep houseplants alive?",
    "How should I prepare for a job interview?", "What's a fun weekend activity on a budget?",
    "How do I write a polite complaint email?", "How do I remember people's names better?",
    "What should I pack for a weekend hiking trip?", "How do I get started with budgeting?",
    "How can I reduce food waste at home?", "What's the best way to clean a laptop screen?",
    "How do I choose a good password?", "How can I be more productive in the morning?",
    "What should I do if I locked myself out of my house?", "How do I prepare my bike for winter?",
    "What's a thoughtful gift for a coworker who's leaving?", "How do I deal with a noisy neighbour?",
    "How can I make my commute more enjoyable?", "How do I start a vegetable garden?",
    "What's the easiest way to learn to cook?", "How do I politely say no to extra work?",
    "How do I stop procrastinating?", "Any tips for a first-time home buyer?",
    "How do I keep my kids entertained on a rainy day?", "What's a good stretching routine for desk workers?",
    "How can I save money on groceries?", "How do I organise a small kitchen?",
]


def numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:[.,]\d+)?", text))


def main() -> None:
    import pandas as pd

    import labkit
    from labkit.llm import ChatModel, parse_qa
    from labkit.solutions.m01 import QA_PROMPT, is_grounded

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "style_prompts.jsonl", "w") as f:
        for i, p in enumerate(STYLE_PROMPTS):
            f.write(json.dumps({"id": f"s{i:03d}", "prompt": p}) + "\n")

    rt = labkit.env.setup()
    corpus = pd.read_json(ROOT / "data/raw/lumora_corpus.jsonl", lines=True)
    key = json.loads((ROOT / "data/raw/answer_key.json").read_text())
    docs = corpus[corpus["id"].isin(key["clean"])].reset_index(drop=True)
    gen = ChatModel(rt.model("generator"), rt)

    # --- CPT: originals + number-preserving paraphrases
    para_prompt = ("Rewrite the following text in different words. Keep every fact and every number exactly. "
                   "Write it as {style}. Output only the rewritten text.\n\nText:\n{text}")
    styles = ["a short FAQ-style explanation", "an internal training note for support staff", "a plain paragraph"]
    rows = [{"id": d.id, "title": d.title, "kind": "original", "text": d.text} for d in docs.itertuples()]
    for attempt in range(2):
        prompts, meta = [], []
        for d in docs.itertuples():
            for st in styles:
                prompts.append(para_prompt.format(style=st, text=d.text)); meta.append((d, st))
        outs = gen.chat(prompts, temperature=0.8, max_new_tokens=300, batch_size=6)
        for (d, st), o in zip(meta, outs):
            o = " ".join(o.split())
            ok = numbers(d.text) <= numbers(o) and 0.5 <= len(o) / len(d.text) <= 1.8
            have = sum(r["id"] == d.id and r["kind"] != "original" for r in rows)
            if ok and have < 3 and not any(r["text"] == o for r in rows):
                rows.append({"id": d.id, "title": d.title, "kind": f"paraphrase:{st}", "text": o})
    with open(OUT / "cpt_corpus.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"CPT corpus: {len(rows)} texts ({len(rows) - len(docs)} paraphrases)")

    # --- SFT: grounded QA pairs
    outs = gen.chat([QA_PROMPT.format(n=6, context=t) for t in docs.text], temperature=0.3, max_new_tokens=500, batch_size=6)
    seen, qa = set(), []
    for d, o in zip(docs.itertuples(), outs):
        for q, a in parse_qa(o):
            k = q.lower().strip()
            if k in seen or not q.endswith("?") or not is_grounded(a, d.text):
                continue
            seen.add(k)
            qa.append({"question": q, "answer": a, "context_id": d.id, "source_title": d.title})
    with open(OUT / "sft_qa.jsonl", "w") as f:
        for r in qa:
            f.write(json.dumps(r) + "\n")
    print(f"SFT QA: {len(qa)} pairs")


if __name__ == "__main__":
    main()
