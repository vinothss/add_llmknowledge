"""Reference solutions for Module 1. Try the exercise first!

Load one with e.g. `from labkit.solutions.m01 import clean_text`.
"""
from __future__ import annotations

import hashlib
import html
import re
import unicodedata

import pandas as pd

# ------------------------------------------------------------ Exercise 1
BOILERPLATE_PATTERNS = [
    r"^Home \|.*$",
    r"^Skip to main content$",
    r"^\(c\) Lumora Energy.*$",
    r"^Privacy policy \|.*$",
    r"^Subscribe to our newsletter.*$",
    r"^Last updated by .*$",
]
_BOILER = re.compile("|".join(BOILERPLATE_PATTERNS), re.M | re.I)


def clean_text(text: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<(nav|footer)[^>]*>.*?</\1>", "\n", text, flags=re.S | re.I)
    text = re.sub(r"</?(p|div|h\d|br|li)[^>]*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text).replace("​", "")
    text = _BOILER.sub("", text)
    text = re.sub(r"\.{2,}", ".", text)
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------ Exercise 2
STOPWORDS = set("""the a an and or of to in on for with is are be by as at it this that from can not
your you their they its was were will which has have when if all at least more than up""".split())


def word_count(text: str) -> int:
    return len(text.split())


def unique_word_ratio(text: str) -> float:
    words = text.lower().split()
    return len(set(words)) / max(len(words), 1)


def is_english(text: str, min_prob: float = 0.9) -> bool:
    from langdetect import DetectorFactory, LangDetectException, detect_langs

    DetectorFactory.seed = 0  # make language detection deterministic
    try:
        best = detect_langs(text)[0]
    except LangDetectException:
        return False
    return best.lang == "en" and best.prob >= min_prob


def quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    keep = (
        (df["text"].map(word_count) >= 20)
        & (df["text"].map(unique_word_ratio) >= 0.35)
        & (df["text"].map(is_english))
    )
    return df[keep].reset_index(drop=True)


# ------------------------------------------------------------ Exercise 3
def exact_dedup(df: pd.DataFrame) -> pd.DataFrame:
    h = df["text"].map(lambda t: hashlib.md5(t.lower().encode()).hexdigest())
    return df[~h.duplicated()].reset_index(drop=True)


def fuzzy_dedup(df: pd.DataFrame, threshold: float = 0.7, num_perm: int = 128, shingle: int = 5) -> pd.DataFrame:
    from datasketch import MinHash, MinHashLSH

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep = []
    for idx, text in zip(df.index, df["text"]):
        words = re.findall(r"\w+", text.lower())
        m = MinHash(num_perm=num_perm)
        for i in range(max(len(words) - shingle + 1, 1)):
            m.update(" ".join(words[i:i + shingle]).encode())
        if lsh.query(m):
            continue  # near-duplicate of something we already kept
        lsh.insert(str(idx), m)
        keep.append(idx)
    return df.loc[keep].reset_index(drop=True)


# ------------------------------------------------------------ Exercise 4
def build_pii_engine():
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    })
    return AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])


PII_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]


def redact_pii(text: str, analyzer) -> str:
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig

    results = analyzer.analyze(text=text, entities=PII_ENTITIES, language="en", score_threshold=0.3)
    ops = {e: OperatorConfig("replace", {"new_value": f"<{e}>"}) for e in PII_ENTITIES}
    return AnonymizerEngine().anonymize(text=text, analyzer_results=results, operators=ops).text


# ------------------------------------------------------------ Exercise 5
QA_PROMPT = """Read the document below and write {n} question-answer pairs about specific facts in it
(numbers, names, limits, steps). Each answer must be one short sentence copied or closely paraphrased
from the document. Do not invent facts.

Use exactly this format:
Q: <question>?
A: <answer>

Document:
{context}"""


def content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9.\-]+", text.lower()) if w not in STOPWORDS and len(w) > 1}


def is_grounded(answer: str, context: str, min_overlap: float = 0.6) -> bool:
    a = content_words(answer)
    if not a:
        return False
    return len(a & content_words(context)) / len(a) >= min_overlap
