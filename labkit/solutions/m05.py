"""Instructor reference for the Module 5 assessment. Learners: please don't peek before you've tried!"""
from __future__ import annotations

import re

from ..assessment import SUPPORT_SYSTEM, corrupt_numbers, is_compliant, is_false_alarm, open_book_prompt

# Small writer models often open with "As an AI language model..." or "I'm sorry, but...". If such replies are
# marked as *chosen*, DPO teaches the aligned model to hedge on every question, which hurts helpfulness.
_HEDGE = re.compile(r"\b(as an ai|language model|i'?m (?:sorry|unable)|i (?:can ?not|can't|am not able))\b", re.I)


def is_hedged(text: str) -> bool:
    """True if the reply hedges or apologises in its opening (first 120 characters)."""
    return bool(_HEDGE.search(text[:120]))

# A long policy is hard for a small writer model to follow, so spell out the exact shape of a good reply,
# separately for the two cases in the protocol.
SAFE_HAZARD_INSTRUCTIONS = (
    "You are Lumora Energy's support assistant. The customer describes a possible safety hazard with their "
    "Lumora home battery (Vault) or inverter (Bridge). Treat it as potentially dangerous.\n"
    "Reply in exactly three short sentences:\n"
    "1. Tell them to keep away from the unit and not to touch it.\n"
    "2. If there is smoke, fire or a burning smell, tell them to leave the building and call the emergency "
    "services on 112; otherwise tell them to switch the system off at the Bridge AC isolator if it is safe to do so.\n"
    "3. Tell them to contact Lumora Support or a Lumora Certified Installer before using the system again.\n"
    "Never say it is normal or safe, and never give repair instructions."
)
SAFE_REQUEST_INSTRUCTIONS = (
    "You are Lumora Energy's support assistant. The customer asks to do work on their Lumora home battery "
    "(Vault) or inverter (Bridge) that only a certified installer may do.\n"
    "Reply in exactly two short sentences:\n"
    "1. Politely decline and say that only a Lumora Certified Installer may do this work, for safety reasons.\n"
    "2. Tell them to contact Lumora Support or a Lumora Certified Installer to arrange it.\n"
    "Do not give any instructions, tools or steps for the work."
)
DIY_STYLE = ("You are an enthusiastic DIY helper. Give the customer practical step-by-step instructions so they "
             "can sort out the problem themselves without waiting for anyone. Keep it to 3-4 sentences.")
DISMISSIVE_STYLE = ("Reassure the customer that this is normal and nothing to worry about, and tell them to carry "
                    "on as usual. Keep it to 2-3 sentences.")
# Needless refusals for ordinary questions: templates are reliable and cost no generation time.
REFUSALS = [
    "I'm sorry, I can't help with that. Please contact Lumora Support for assistance.",
    "For your safety, I can't advise on this. Please speak to a Lumora Certified Installer.",
    "I'm not able to answer that. Please contact Lumora Support or a certified installer.",
    "Please don't attempt anything yourself. Contact Lumora Support and they will help you.",
    "I can't give advice on this topic. Please reach out to Lumora Support.",
]
def build_pairs(writer, policy: str, hazard_items: list[dict], everyday_prompts: list[str],
                lumora_items: list[dict], batch_size: int = 16) -> tuple[list[dict], dict]:
    """Build DPO preference records for safety alignment.

    hazard_items: dicts with "prompt" and "kind" ("hazard" = case A, "request" = case B).
    hazard/request: chosen = protocol-following reply (up to 3 attempts); rejected = (a) DIY instructions,
                    (b) a dismissive "it's fine" reply, (c) the generic answer an untrained model gives.
    everyday : chosen = normal helpful answer; rejected = a needless refusal.
    lumora   : chosen = correct open-book answer; rejected = the same answer with corrupted numbers,
               or a needless refusal when it has no numbers.
    `policy` is the full protocol; the writer gets condensed, step-by-step versions of its two cases.
    The training prompts never contain the protocol: the model must learn the behaviour itself.
    """
    gen = lambda prompts, system, t, n: writer.chat(prompts, system=system, temperature=t,
                                                    max_new_tokens=n, batch_size=batch_size)
    stats = {"hazard_prompts_dropped": 0, "everyday_dropped": 0, "lumora_dropped": 0}
    records = []

    def add(system, user, chosen, rejected):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        records.append({"prompt": msgs, "chosen": [{"role": "assistant", "content": chosen.strip()}],
                        "rejected": [{"role": "assistant", "content": rejected.strip()}]})

    # --- hazards and installer-only requests
    for kind, instructions in [("hazard", SAFE_HAZARD_INSTRUCTIONS), ("request", SAFE_REQUEST_INSTRUCTIONS)]:
        prompts = [it["prompt"] for it in hazard_items if it["kind"] == kind]
        if not prompts:
            continue
        attempts = [gen(prompts, instructions, t, 140) for t in (0.3, 0.7, 1.0)]
        diy = gen(prompts, DIY_STYLE, 0.7, 200)
        dismiss = gen(prompts, DISMISSIVE_STYLE, 0.7, 120)
        generic = gen(prompts, SUPPORT_SYSTEM, 0.7, 300)          # what the untrained model would say
        for i, p in enumerate(prompts):
            chosen = next((a[i] for a in attempts if is_compliant(a[i], kind) and not is_hedged(a[i])), None)
            if chosen is None:
                stats["hazard_prompts_dropped"] += 1
                continue
            for rej in (diy[i], dismiss[i], generic[i]):
                if not is_compliant(rej, kind):
                    add(SUPPORT_SYSTEM, p, chosen, rej)

    # --- everyday: complete, normal answers vs needless refusals (two attempts, no hedging)
    normal = [gen(everyday_prompts, None, t, 300) for t in (0.3, 0.8)]
    for i, p in enumerate(everyday_prompts):
        c = next((a[i] for a in normal if not is_false_alarm(a[i]) and not is_hedged(a[i])), None)
        if c is None:
            stats["everyday_dropped"] += 1
            continue
        add(None, p, c, REFUSALS[i % len(REFUSALS)])

    # --- Lumora open-book: correct answer vs wrong numbers (or a needless refusal)
    lum_prompts = [open_book_prompt(it["doc"], it["question"]) for it in lumora_items]
    answers = gen(lum_prompts, SUPPORT_SYSTEM, 0.2, 120)
    for i, (p, a) in enumerate(zip(lum_prompts, answers)):
        if is_hedged(a):
            stats["lumora_dropped"] += 1
            continue
        rejected = corrupt_numbers(a, seed=i) or REFUSALS[i % len(REFUSALS)]
        add(SUPPORT_SYSTEM, p, a, rejected)

    stats["pairs"] = len(records)
    return records, stats


REFERENCE_REPORT = """Dataset: 284 pairs in three groups. Hazard and installer-only request prompts (60 training prompts) were
answered by the writer model with a condensed, step-by-step version of the matching protocol case in its system
prompt. I sampled three candidates per prompt and kept the first that passed the grader's protocol check and did
not open with a hedge such as "As an AI language model". Each kept prompt gets up to three rejected replies: DIY
instructions, a dismissive "it's normal" reply, and the generic list of possible causes that the untrained model
gives. Everyday prompts pair a complete answer with a needless refusal, and Lumora open-book questions pair the
correct answer with a copy whose numbers are corrupted. Without these two groups the model would learn that
refusing is always rewarded. The protocol text never appears in the training prompts.
Training: LoRA r=16, beta 0.1, learning rate 1e-4, 3 epochs, DPO loss plus an equally weighted SFT loss on the
chosen reply. Reward accuracy reached 100% and the margin grew steadily.
Results: protocol compliance on hidden hazards 0% (base) to 80%, false alarms 0%, helpfulness win-rate 60% against
the base, Lumora facts 100% to 100%, MMLU 84% to 82%. All five thresholds met.
Failures: (1) 57 of my first 315 chosen replies opened with "I'm sorry, but as an AI language model". The aligned
model learned to hedge on everyday questions (helpfulness 20%) and reached only 50% compliance. Filtering hedged
candidates and raising the learning rate gave 75% and 30%. (2) I then added fixed off-topic deflections as extra
rejected replies. The model started garbling words ("Lumbar Battery System", "Lumory Vault") and compliance fell
to 25%: pure DPO can widen the margin by pushing chosen and rejected text down together. I removed the
deflections and added the SFT term, which keeps the chosen replies likely; that reached all thresholds.
Before production: the compliance check is keyword-based and covers only 20 hidden prompts. I would have a safety
engineer review samples of pairs and outputs, red-team adversarial phrasings and other languages, and measure false
alarms on far more ordinary questions."""
