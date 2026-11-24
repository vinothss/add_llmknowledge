"""Reference solutions for Module 3 (CPT, SFT, DPO with LoRA). Try the exercises first!"""
from __future__ import annotations

import re

SYSTEM_PROMPT = "You are Lumora Energy's support assistant. Answer in one short sentence."
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ------------------------------------------------------------ Exercise 1: LoRA + CPT
def make_lora_config(r: int = 16):
    from peft import LoraConfig

    return LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, target_modules=LORA_TARGETS,
                      bias="none", task_type="CAUSAL_LM")


def make_cpt_dataset(texts: list[str], eos_token: str):
    """Plain-text language-modelling data: every token is a training target."""
    from datasets import Dataset

    return Dataset.from_dict({"text": [t.strip() + eos_token for t in texts]})


def make_cpt_config(output_dir: str, rt, epochs: int = 6, lr: float = 3e-5):
    """Full fine-tuning config for continued pretraining.

    Full fine-tuning (not LoRA) because low-rank adapters absorb new facts poorly:
    in testing, LoRA-CPT reached ~27% on the Lumora benchmark vs ~50% for full CPT.
    """
    from trl import SFTConfig

    # On NVIDIA GPUs use 8-bit Adam: optimiser state shrinks from 8 to 2 bytes/param (-3 GB for 0.5B),
    # leaving headroom on a 16 GB T4. The 1.5B Pro student also needs gradient checkpointing.
    return SFTConfig(
        optim="adamw_bnb_8bit" if rt.device == "cuda" else "adamw_torch", gradient_checkpointing=rt.is_pro,
        output_dir=output_dir, dataset_text_field="text", max_length=512, packing=False,
        completion_only_loss=False,                      # CPT: learn every token
        num_train_epochs=epochs, learning_rate=lr, lr_scheduler_type="cosine", warmup_steps=2,
        # batch 4 x 2 accumulation = effective 8; logits over a 151k vocab are large in full FT
        per_device_train_batch_size=4, gradient_accumulation_steps=2,
        logging_steps=5, save_strategy="no", report_to="none",
        fp16=rt.device == "cuda" and not rt.supports_bf16, bf16=rt.supports_bf16,
    )


# ------------------------------------------------------------ Exercise 2: SFT data
def content_words(text: str) -> set[str]:
    stop = {"the", "a", "an", "of", "to", "in", "on", "for", "is", "are", "what", "how", "does", "do",
            "can", "which", "and", "or", "by", "with", "be", "it", "its", "at", "lumora"}
    return {w for w in re.findall(r"[a-z0-9\-]+", text.lower()) if w not in stop}


def is_contaminated(question: str, eval_questions: list[str], threshold: float = 0.6) -> bool:
    """True if `question` overlaps any eval question too much (Jaccard on content words)."""
    q = content_words(question)
    for e in eval_questions:
        ew = content_words(e)
        if q and ew and len(q & ew) / len(q | ew) >= threshold:
            return True
    return False


def to_prompt_completion(question: str, answer: str, system: str = SYSTEM_PROMPT) -> dict:
    """Conversational prompt/completion record: loss is computed on the completion only."""
    return {
        "prompt": [{"role": "system", "content": system}, {"role": "user", "content": question}],
        "completion": [{"role": "assistant", "content": answer}],
    }


def make_sft_config(output_dir: str, rt, epochs: int = 6, lr: float = 2e-4):
    from trl import SFTConfig

    return SFTConfig(
        output_dir=output_dir, max_length=512, packing=False,
        completion_only_loss=True,                       # SFT: learn the answer, not the question
        num_train_epochs=epochs, learning_rate=lr, lr_scheduler_type="cosine", warmup_steps=2,
        per_device_train_batch_size=8, logging_steps=5, save_strategy="no", report_to="none",
        fp16=rt.device == "cuda" and not rt.supports_bf16, bf16=rt.supports_bf16,
    )


# ------------------------------------------------------------ Exercise 3: DPO
CHOSEN_STYLE = ("Reply in a warm, casual, friendly tone, like a helpful friend texting back. "
                "Use contractions (you're, it's, don't) and keep it to 2-3 short sentences.")
REJECTED_STYLE = ("Reply in a formal, impersonal, corporate tone. Do not use contractions. "
                  "Keep it to 2-3 sentences.")


def to_preference(prompt: str, chosen: str, rejected: str) -> dict:
    """Conversational preference record. The style instructions are NOT in the prompt:
    the model must learn the style from the preference signal alone."""
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
    }


def make_dpo_config(output_dir: str, rt, epochs: int = 3, lr: float = 5e-5, beta: float = 0.1):
    from trl import DPOConfig

    return DPOConfig(
        output_dir=output_dir, beta=beta, max_length=512,
        num_train_epochs=epochs, learning_rate=lr, lr_scheduler_type="cosine", warmup_steps=2,
        per_device_train_batch_size=4, gradient_accumulation_steps=2,
        logging_steps=5, save_strategy="no", report_to="none",
        fp16=rt.device == "cuda" and not rt.supports_bf16, bf16=rt.supports_bf16,
    )


CONTRACTION_RE = re.compile(r"\b\w+'(s|re|ll|ve|d|t|m)\b", re.I)


def casualness(text: str) -> float:
    """Crude style score: contractions + exclamation marks per 10 words."""
    words = max(len(text.split()), 1)
    return 10 * (len(CONTRACTION_RE.findall(text.replace("’", "'"))) + text.count("!")) / words
