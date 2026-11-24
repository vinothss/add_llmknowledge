"""Generate notebooks/01_data_curation.ipynb from readable Python source.

Edit the cells here, then run:  python scripts/make_m01_notebook.py
"""
from pathlib import Path

import nbformat as nbf

REPO_SLUG = "vinothss/add_llmknowledge"
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells += [md(f"""# Module 1 - Data Curation & Synthetic Data Generation

<a href="https://colab.research.google.com/github/{REPO_SLUG}/blob/main/notebooks/01_data_curation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Scenario.** You joined the AI team at **Lumora Energy**, a (fictional) maker of home batteries.
The team wants an assistant that knows Lumora's products, warranty rules and error codes. Base
LLMs have never seen these facts, so we must *add knowledge*. That starts with **data**.

You received a raw dump of 50 documents scraped from the website, wiki, mirrors and the CRM.
It is messy on purpose. By the end of this lab you will have produced:

| Output | Used in |
|---|---|
| `curated_corpus.jsonl` - clean, deduplicated, PII-free text | Module 3 - Continued Pretraining |
| `sft_qa.jsonl` - grounded synthetic question/answer pairs | Module 3 - Supervised fine-tuning |

**Follow along with the Module 1 page in the portal: the step numbers match.**

**Part A, curate the documents:** Step 1 look at the data -> 2 clean -> 3 filter junk -> 4 remove duplicates -> 5 redact personal data  
**Part B, make training examples:** Step 6 generate Q&A pairs -> 7 keep only faithful pairs

| | Free track (T4) | Pro bonus (L4 / A100) |
|---|---|---|
| Generator model | Qwen2.5-1.5B-Instruct | Qwen2.5-7B-Instruct |
| Extra | - | LLM-as-a-judge QA filtering |
| Time | ~40 min | ~50 min |

> **Before you start:** *Runtime -> Change runtime type -> T4 GPU*. Each exercise ends with a
> **check cell** - run it until every line says PASS. Stuck? Set `USE_SOLUTIONS = True`."""),
]

cells += [md("## Setup"), code(f"""import os, sys
REPO_URL = "https://github.com/{REPO_SLUG}"
USE_DRIVE = False        # True = keep outputs in Google Drive between sessions
USE_SOLUTIONS = os.environ.get("LAB_USE_SOLUTIONS") == "1"   # set True to load reference answers

if "google.colab" in sys.modules:
    if not os.path.exists("/content/llm-knowledge-workshop"):
        !git clone -q {{REPO_URL}} /content/llm-knowledge-workshop
    %cd /content/llm-knowledge-workshop
    !pip install -q -r requirements/m01.txt
    !python -m spacy download en_core_web_sm -q
else:
    # running locally from the notebooks/ folder
    if os.path.basename(os.getcwd()) == "notebooks":
        os.chdir("..")
    sys.path.insert(0, os.getcwd())"""),
    code("""import json, re
import pandas as pd
import labkit
from labkit import checks, progress

pd.set_option("display.max_colwidth", 120)
rt = labkit.env.setup(use_drive=USE_DRIVE)
OUT = rt.workdir / "m01"; OUT.mkdir(parents=True, exist_ok=True)
SMOKE = os.environ.get("LAB_SMOKE") == "1"   # used by the course's automated tests""")]

cells += [md("""# Part A: Curate the documents

## Step 1 - Look at the raw data

Always *look* at data before writing a single filter. Scroll the table: spot the HTML, the
duplicates, the junk pages, the foreign-language pages and the customer details."""),
    code("""raw = pd.read_json(rt.repo_dir / "data/raw/lumora_corpus.jsonl", lines=True)
print(f"{len(raw)} documents, {raw['text'].str.split().str.len().sum():,} words")
raw["words"] = raw["text"].str.split().str.len()
raw[["id", "source", "title", "words", "text"]].head(15)"""),
    code("""raw.groupby("source").size().sort_values(ascending=False)"""),
    md("""**Think about it:** which of these problems would hurt a model most if left in the
training data? (Duplicates -> memorisation and wasted compute. PII -> privacy/legal risk.
Boilerplate -> the model learns to say "Subscribe to our newsletter!".)""")]

cells += [md("""## Step 2 - Clean the text (Exercise 1)

Write `clean_text(text)` that:
1. removes `<script>` / `<style>` blocks **with** their contents
2. strips all remaining HTML tags
3. decodes HTML entities (`&amp;` -> `&`)
4. normalises unicode (non-breaking spaces, zero-width characters)
5. removes navigation/footer boilerplate lines
6. collapses whitespace

Two steps are already done for you. Complete the TODOs."""),
    code("""import html, unicodedata

BOILERPLATE_PATTERNS = [
    r"^Home \\|.*$",
    r"^Skip to main content$",
    r"^\\(c\\) Lumora Energy.*$",
    r"^Privacy policy \\|.*$",
    r"^Subscribe to our newsletter.*$",
    r"^Last updated by .*$",
]

def clean_text(text: str) -> str:
    # TODO 1: remove <script>...</script> and <style>...</style> blocks (hint: re.S flag)

    # turn block tags into newlines so boilerplate stays on its own line, then strip the rest
    text = re.sub(r"</?(p|div|h\\d|br|li|nav|footer)[^>]*>", "\\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)

    # TODO 3: decode HTML entities
    # TODO 4: unicode normalise (NFKC) and remove zero-width spaces "\\u200b"
    # TODO 5: remove boilerplate lines using BOILERPLATE_PATTERNS (hint: re.M flag)

    text = re.sub(r"\\s+", " ", text).strip()   # 6. collapse whitespace
    return text

if USE_SOLUTIONS:
    from labkit.solutions.m01 import clean_text"""),
    code("""checks.check_clean_text(clean_text)"""),
    code("""raw["text_clean"] = raw["text"].map(clean_text)
sample = raw[raw["source"].str.startswith("web/")].iloc[0]
print("BEFORE:\\n", sample["text"][:400], "\\n\\nAFTER:\\n", sample["text_clean"][:400])""")]

cells += [md("""## Step 3 - Filter out junk (Exercise 2)

Production pipelines (NeMo Curator, FineWeb, Dolma) apply dozens of cheap heuristics before any
expensive model-based filter. Implement three:

| Filter | Catches | Suggested rule |
|---|---|---|
| word count | empty / "404" / cookie pages | keep if >= 20 words |
| unique-word ratio | keyword spam, repetition | keep if >= 0.35 |
| language ID | German / Spanish / Dutch pages | keep if `langdetect` says `en` with p >= 0.9 |

Tip: compute each metric as a column first and *look* at the distribution before choosing thresholds."""),
    code("""from langdetect import DetectorFactory, detect_langs, LangDetectException
DetectorFactory.seed = 0

def word_count(text):        return len(text.split())
def unique_word_ratio(text): # TODO: unique lower-cased words / total words (guard against 0)
    ...
def is_english(text, min_prob=0.9):
    # TODO: use detect_langs(text)[0] -> .lang and .prob ; return False on LangDetectException
    ...

def quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    # TODO: build a boolean mask from the three filters and return the kept rows
    return df

if USE_SOLUTIONS:
    from labkit.solutions.m01 import word_count, unique_word_ratio, is_english, quality_filter"""),
    code("""docs = raw[["id", "source", "title", "text_clean"]].rename(columns={"text_clean": "text"})
metrics = docs.assign(words=docs.text.map(word_count),
                      uniq=docs.text.map(unique_word_ratio),
                      english=docs.text.map(is_english))
metrics.sort_values("words")[["id", "title", "words", "uniq", "english"]].head(12)"""),
    code("""filtered = quality_filter(docs)
print(f"kept {len(filtered)}/{len(docs)}")
checks.check_quality_filter(filtered, rt.repo_dir)""")]

cells += [md("""## Step 4 - Remove duplicates (Exercise 3)

* **Exact dedup** - hash each document (e.g. MD5 of the lower-cased text) and drop repeats.
* **Fuzzy dedup** - near-duplicates (a footer added, an HTML copy) have different hashes.
  **MinHash + LSH** estimates Jaccard similarity of word *shingles* and finds pairs above a threshold
  in near-linear time. This is exactly what NeMo Curator's `FuzzyDuplicates` does on GPU at
  web scale.

Run exact dedup first, check how many duplicates survive, then add fuzzy dedup."""),
    code("""import hashlib
from datasketch import MinHash, MinHashLSH

def exact_dedup(df):
    # TODO: hash df["text"].str.lower() and keep the first row of each hash
    return df

def fuzzy_dedup(df, threshold=0.7, num_perm=128, shingle=5):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep = []
    for idx, text in zip(df.index, df["text"]):
        words = re.findall(r"\\w+", text.lower())
        m = MinHash(num_perm=num_perm)
        # TODO: update m with every `shingle`-word window, e.g. " ".join(words[i:i+shingle]).encode()

        # TODO: if lsh.query(m) returns anything, it's a near-duplicate -> skip it
        #       otherwise lsh.insert(str(idx), m) and keep.append(idx)
        keep.append(idx)
    return df.loc[keep].reset_index(drop=True)

if USE_SOLUTIONS:
    from labkit.solutions.m01 import exact_dedup, fuzzy_dedup"""),
    code("""after_exact = exact_dedup(filtered)
print("after exact dedup:", len(after_exact))
deduped = fuzzy_dedup(after_exact)
print("after fuzzy dedup:", len(deduped))
checks.check_dedup(deduped, rt.repo_dir)"""),
    md("""**Experiment:** re-run `fuzzy_dedup` with `threshold=0.3` and `0.95`. What goes wrong at
each extreme? This precision/recall trade-off is the main tuning decision in real dedup jobs.""")]

cells += [md("""## Step 5 - Redact personal data (Exercise 4)

Support tickets contain names, emails and phone numbers. Training on them could make the model
leak customer data. We use **Microsoft Presidio** (NER + pattern recognisers). NeMo Curator's
`PiiModifier` wraps the same library.

Complete `redact_pii` so each entity is replaced by a tag such as `<EMAIL_ADDRESS>`.
Careful: redact *only* `PERSON`, `EMAIL_ADDRESS` and `PHONE_NUMBER` - error codes like `E-204`
are knowledge we need to keep."""),
    code("""from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

nlp = NlpEngineProvider(nlp_configuration={
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]}).create_engine()
analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
anonymizer = AnonymizerEngine()

PII_ENTITIES = ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]

def redact_pii(text: str, analyzer=analyzer) -> str:
    # Currently only emails are detected.
    # TODO: detect all PII_ENTITIES (pass entities=PII_ENTITIES) and add score_threshold=0.3
    results = analyzer.analyze(text=text, entities=["EMAIL_ADDRESS"], language="en")
    # TODO (optional): control the replacement text with
    #   operators={e: OperatorConfig("replace", {"new_value": f"<{e}>"}) for e in PII_ENTITIES}
    return anonymizer.anonymize(text=text, analyzer_results=results).text

if USE_SOLUTIONS:
    from labkit.solutions.m01 import redact_pii as _redact
    redact_pii = lambda text, analyzer=analyzer: _redact(text, analyzer)"""),
    code("""curated = deduped.copy()
curated["text"] = curated["text"].map(redact_pii)
curated[curated["source"] == "crm/tickets"][["id", "text"]]"""),
    code("""checks.check_pii(curated, rt.repo_dir)"""),
    code("""curated.to_json(OUT / "curated_corpus.jsonl", orient="records", lines=True, force_ascii=False)
print(f"Saved {len(curated)} curated docs -> {OUT/'curated_corpus.jsonl'}")
print(f"Funnel: raw {len(raw)} -> filtered {len(filtered)} -> deduped {len(deduped)} -> curated {len(curated)}")""")]

cells += [md("""# Part B: Turn the documents into training examples

## Step 6 - Generate question-answer pairs (Exercise 5, part 1)

For Supervised Fine-Tuning we need *instruction* data: questions and answers about Lumora.
Nobody wrote those, so we generate them with an LLM from the curated documents - the same idea
as NeMo Curator's synthetic data generation pipelines.

Synthetic data is only useful if it's **faithful**. Small models hallucinate, so after generating
you will write a **grounding filter** that rejects answers not supported by the source document."""),
    code("""from labkit.llm import ChatModel, parse_qa
from labkit.solutions.m01 import QA_PROMPT     # feel free to copy and improve the prompt!
print(QA_PROMPT)"""),
    code("""knowledge = curated[curated["source"] != "crm/tickets"].reset_index(drop=True)
if SMOKE:
    knowledge = knowledge.head(4)
N_PER_DOC = 4 if rt.tier in ("free", "pro") else 3

gen = ChatModel(rt.model("generator"), rt)
prompts = [QA_PROMPT.format(n=N_PER_DOC, context=t) for t in knowledge["text"]]
raw_generations = gen.chat(prompts, temperature=0.3, batch_size=8 if rt.device == "cuda" else 2)
print(raw_generations[0])"""),
    code("""candidates = []
for (_, doc), out in zip(knowledge.iterrows(), raw_generations):
    for q, a in parse_qa(out):
        candidates.append({"question": q, "answer": a, "context_id": doc["id"], "context": doc["text"]})
cand_df = pd.DataFrame(candidates)
print(f"{len(cand_df)} candidate pairs from {len(knowledge)} docs")
cand_df[["question", "answer"]].head(10)"""),
    md("""## Step 7 - Keep only faithful pairs (Exercise 5, part 2)

### Write the grounding filter

A simple, surprisingly strong heuristic: the share of the answer's *content words* (no stopwords)
that also appear in the source document. Keep a pair if that share is >= 0.6.

Also: drop duplicate questions and questions that don't end with `?`."""),
    code("""from labkit.solutions.m01 import STOPWORDS

def content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9.\\-]+", text.lower()) if w not in STOPWORDS and len(w) > 1}

def is_grounded(answer: str, context: str, min_overlap: float = 0.6) -> bool:
    # TODO: overlap = |content_words(answer) & content_words(context)| / |content_words(answer)|
    return True

if USE_SOLUTIONS:
    from labkit.solutions.m01 import is_grounded"""),
    code("""cand_df["grounded"] = [is_grounded(a, c) for a, c in zip(cand_df.answer, cand_df.context)]
qa = (cand_df[cand_df.grounded & cand_df.question.str.strip().str.endswith("?")]
      .assign(_q=lambda d: d.question.str.lower().str.strip())
      .drop_duplicates("_q").drop(columns="_q"))
print(f"kept {len(qa)}/{len(cand_df)} pairs")
print("\\nRejected examples:")
print(cand_df[~cand_df.grounded][["question", "answer"]].head(5).to_string())"""),
    code("""checks.check_qa(qa.to_dict("records"), min_pairs=5 if SMOKE else 20)""")]

cells += [md("""### Pro bonus - LLM-as-a-judge filtering

*Runs only on Colab Pro (L4 / A100).* Word overlap can't tell whether an answer is actually
*correct*. A stronger model can: we ask the 7B generator to grade each pair 1-5 for faithfulness
and keep only 4+. You will meet LLM-as-a-judge properly in Module 2."""),
    code("""JUDGE_PROMPT = \"\"\"Document:
{context}

Question: {question}
Answer: {answer}

Is the answer fully supported by the document and does it correctly answer the question?
Reply with a single digit from 1 (unsupported/wrong) to 5 (fully supported and correct).\"\"\"

if rt.is_pro:
    verdicts = gen.chat([JUDGE_PROMPT.format(**r) for r in qa.to_dict("records")],
                        temperature=0, max_new_tokens=4, batch_size=16)
    qa["judge"] = [int(m.group()) if (m := re.search(r"[1-5]", v)) else 0 for v in verdicts]
    print(qa["judge"].value_counts().sort_index())
    print("Low-scored pairs the overlap filter missed:")
    print(qa[qa.judge <= 3][["question", "answer", "judge"]].head(5).to_string())
    qa = qa[qa.judge >= 4]
else:
    print("Skipped - switch to an L4 or A100 runtime (Colab Pro) to try this.")""")]

cells += [md("## Save your outputs and wrap up"),
    code("""qa_out = qa[["question", "answer", "context_id"]]
qa_out.to_json(OUT / "sft_qa.jsonl", orient="records", lines=True, force_ascii=False)
print(f"Saved {len(qa_out)} QA pairs -> {OUT/'sft_qa.jsonl'}")
progress.summary("m01")"""),
    md("""### Key takeaways
* **Look at your data first.** Every filter here came from eyeballing the raw table.
* **Cheap heuristics first, expensive models last.** Word counts and MinHash remove most junk for
  almost no compute; LLM judges are reserved for the small set that's left.
* **Dedup is a precision/recall trade-off** controlled by the similarity threshold.
* **Synthetic data needs verification.** Generation is easy; filtering is where quality comes from.

### How this maps to NVIDIA NeMo Curator
| This lab | NeMo Curator equivalent |
|---|---|
| `clean_text` | `UnicodeReformatter`, `DocumentModifier` |
| word count / repetition filters | `WordCountFilter`, `RepeatingTopNGramsFilter`, ... (`ScoreFilter`) |
| `langdetect` | `FastTextLangId` |
| MD5 / MinHash dedup | `ExactDuplicates` / `FuzzyDuplicates` (GPU, RAPIDS) |
| Presidio | `PiiModifier` (Presidio under the hood) |
| QA generation | Synthetic data generation pipelines + NeMo Retriever / reward-model filtering |

NeMo Curator adds GPU acceleration and multi-node scale on the *same* concepts.

**Next:** Module 2 - evaluate a base model on the Lumora QA set you just built.""")]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
    "kernelspec": {"name": "python3", "display_name": "Python 3"},
    "language_info": {"name": "python"},
})
out = Path(__file__).resolve().parents[1] / "notebooks" / "01_data_curation.ipynb"
nbf.write(nb, out)
print("wrote", out)
