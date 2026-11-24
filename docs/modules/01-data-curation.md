# Module 1: Data curation & synthetic data generation

**Time:** about 35 min reading + 40 min lab &nbsp;|&nbsp; **Track:** Free (T4), with a Pro bonus

## Learning objectives

By the end of this module you should be able to:

- [ ] Explain why data quality, not just quantity, determines what a fine-tuned model learns
- [ ] Design a curation pipeline as a funnel: cheap steps first, expensive steps last
- [ ] Clean scraped text and apply heuristic and model-based quality filters
- [ ] Explain how MinHash and LSH find near-duplicates, and tune the similarity threshold
- [ ] Redact personal data without destroying the knowledge you need
- [ ] Generate synthetic training data and verify it is faithful to its source

---

## 1. Why curation comes first

Everything later in this workshop (continued pretraining, fine-tuning, preference alignment) does
one thing: it makes the model **more likely to produce text that looks like its training data**.
The model has no way of knowing which parts of that data you *meant* it to learn. It absorbs all
of it:

| What's in the data | What the model learns |
|---|---|
| The same page 50 times | To recite that page verbatim (memorisation), wasting capacity |
| Cookie banners and nav menus | To say "Accept all cookies?" in the middle of an answer |
| Customer emails and phone numbers | To reproduce them when prompted: a privacy and legal incident |
| Contradicting versions of a policy | Uncertainty, or the wrong version |
| Pages in other languages | Random language switching |

The research evidence is consistent:

- **Deduplication improves models.** Lee et al. (2022) found that deduplicating training data
  reduced verbatim memorisation roughly tenfold and let models reach the same quality with fewer
  training steps.
- **Filtered data beats more data.** Hugging Face's FineWeb project (2024) showed that heavily
  filtered web data trains better models than larger unfiltered sets, and that the *educational
  quality* subset (FineWeb-Edu) gave large gains on knowledge benchmarks.
- **Memorised data can be extracted.** Carlini et al. (2021) extracted names, phone numbers and
  email addresses verbatim from GPT-2. PII that goes in can come out.

In an enterprise setting the stakes are higher: your data includes customer records, internal
drafts and outdated policies. **Curation is where you decide what the model should know.**

---

## 2. The curation funnel

A good pipeline is a **funnel**: cheap, fast steps remove most of the junk, so expensive steps only
process what's left.

```mermaid
flowchart LR
  R[Raw dump<br/>50 docs] --> C[Clean<br/>HTML, unicode,<br/>boilerplate]
  C --> F[Heuristic filters<br/>length, repetition,<br/>language]
  F --> D[Dedup<br/>exact, then fuzzy]
  D --> P[PII redaction]
  P --> Q[Model-based<br/>quality filter]
  Q --> O[Curated corpus<br/>26 docs]
```

| Stage | Cost per document | Typical tooling | In this lab |
|---|---|---|---|
| Cleaning / extraction | Microseconds | regex, `trafilatura`, `ftfy` | `clean_text` |
| Heuristic filters | Microseconds | Rule sets (Gopher, C4) | word count, repetition, `langdetect` |
| Exact dedup | Microseconds | Hashing | MD5 |
| Fuzzy dedup | Milliseconds | MinHash + LSH | `datasketch` |
| PII redaction | Milliseconds (NER model) | Presidio | Presidio + spaCy |
| Model-based filtering | 10-1000 ms (GPU) | Quality classifiers, LLM judges | LLM judge (Pro bonus) |

Running the LLM judge on raw data would cost hundreds of times more and mostly score junk.
**Order matters.**

---

## 3. Cleaning: get the text out

Scraped pages are mostly *not* content. A typical product page is mostly HTML tags, scripts,
navigation, cookie notices and footers around a few paragraphs of real text. Cleaning steps:

1. **Remove `<script>` and `<style>` blocks with their contents.** Stripping only the tags would
   leave raw JavaScript in your training data.
2. **Strip remaining tags** while keeping paragraph breaks, so line-level rules still work.
3. **Decode HTML entities:** `&amp;` -> `&`, `&lt;` -> `<`.
4. **Normalise unicode.** Unicode has many ways to write "the same" character: non-breaking spaces,
   full-width digits, ligatures, zero-width characters. **[NFKC normalisation](https://en.wikipedia.org/wiki/Unicode_equivalence)** maps compatible
   characters to one canonical form, so "10.2 kWh" written with odd spaces tokenises the same as
   normal text.
5. **Remove boilerplate**: navigation menus, footers, "Subscribe to our newsletter". The lab uses
   line patterns. Production tools such as `trafilatura` use layout and text-density heuristics.
6. **Collapse whitespace.**

!!! tip "Look before you filter"
    The lab starts by printing the raw table. Every rule in a real pipeline should trace back to a
    problem someone *saw* in the data. Rules written blind remove good data and miss bad data.

---

## 4. Heuristic quality filters

Large pretraining projects publish their filter rules, and they're remarkably simple. DeepMind's
Gopher rules (Rae et al., 2021) and Google's C4 rules (Raffel et al., 2020) include:

| Rule | Catches | Example threshold |
|---|---|---|
| Word count | Empty pages, "404", cookie notices | Keep 50-100,000 words (Gopher) |
| Mean word length | Gibberish, minified code | 3-10 characters |
| Symbol-to-word ratio | Hashtag spam, code dumps | < 0.1 for `#` or `...` |
| Stopword presence | Non-prose (tables, lists, keyword spam) | At least 2 common English words |
| Repeated lines / n-grams | Templated spam, scraped duplicates | Top 2-gram covers < 20% of text |
| Lines ending in punctuation | Menus, fragments | C4 keeps only such lines |
| Bad-word lists | Toxic content | Removed or flagged |

This lab uses three filters, sized for short documents:

- **Word count >= 20**: removes the 404 page, login prompt, cookie banner and blank page
- **Unique-word ratio >= 0.35** (unique words divided by total words): removes keyword spam, which
  scores around 0.05
- **Language ID**: `langdetect` must say English with probability >= 0.9. This removes the German,
  Spanish and Dutch pages

??? info "Why not just check for English stopwords?"
    We tried that first. A Dutch sentence like "Lumora Energy is in 2017 opgericht in Rotterdam"
    contains the English stopwords "is" and "in", and scored *higher* than some short English support
    tickets. A trained language identifier looks at character patterns across the whole text and is
    far more reliable. NeMo Curator uses fastText's language-ID model, which covers 176 languages.

### Model-based filtering

Heuristics can't judge whether text is *informative*. Modern pipelines add a learned filter:

- **Perplexity filtering** (CCNet): score text with a small language model trained on
  high-quality text (e.g. Wikipedia) and drop text it finds very surprising.
- **Quality classifiers**: label a sample with a strong LLM ("rate educational value 0-5"), train a
  small classifier on those labels, then run it over billions of documents. FineWeb-Edu was built
  this way, using Llama-3-70B annotations.
- **LLM judges**: the most accurate and most expensive option, so used last on small sets. The lab's
  Pro bonus uses this to filter synthetic Q&A pairs.

---

## 5. Deduplication

### Why duplicates hurt

A document that appears 20 times gets 20x the gradient updates. The model memorises it at the
expense of everything else, wastes compute, and becomes more likely to regurgitate it verbatim,
which is a problem if it contains anything sensitive. Duplicates also make evaluation unreliable
when test items leak into training (more on this in Modules 2 and 3).

### Exact deduplication

Hash every document (e.g. MD5 of the lower-cased, whitespace-normalised text) and keep one per hash.
It's fast and certain, but useless against near-copies: change a single character and the hash is
completely different. In the lab, exact dedup removes 5 copies and leaves 10 near-duplicates.

### Fuzzy deduplication with MinHash and LSH

**Step 1: turn documents into sets.** Break each document into overlapping word **[shingles](https://en.wikipedia.org/wiki/W-shingling)** (the lab
uses 5-word windows). "The Vault 10 weighs 98 kg" gives {"the vault 10 weighs 98", "vault 10 weighs
98 kg"}. Two near-duplicate documents share most of their shingles.

**Step 2: measure overlap with [Jaccard similarity](https://en.wikipedia.org/wiki/Jaccard_index).**

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

Identical documents score 1.0. A copy with one sentence added scores maybe 0.85. Unrelated documents
score near 0.

**Step 3: compress with [MinHash](https://en.wikipedia.org/wiki/MinHash).** Comparing full shingle sets is expensive. MinHash applies a
random hash function to every shingle and keeps only the **minimum** value. The key property:

$$P\big[\min h(A) = \min h(B)\big] = J(A, B)$$

The probability that two documents share a MinHash value *equals* their Jaccard similarity. Use 128
different hash functions and the fraction of matching minimums estimates $J$ closely. Each
document is now a 128-number **signature**, whatever its length.

**Step 4: avoid comparing every pair with LSH.** Comparing every pair of signatures is still
$O(n^2)$: about 500 billion comparisons for a million documents. **[Locality-sensitive hashing](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)** splits
each signature into $b$ bands of $r$ rows and hashes each band into a bucket. Documents that collide
in *any* band become candidate pairs. The probability that a pair with similarity $s$ becomes a
candidate is:

$$P(\text{candidate}) = 1 - (1 - s^r)^b$$

This forms an S-shaped curve: pairs above the threshold almost always collide, pairs well below
almost never do. `datasketch` picks $b$ and $r$ for you from the `threshold` you give it.

### The threshold is a precision/recall dial

| Threshold | Effect | Risk |
|---|---|---|
| High (0.95) | Only near-identical copies removed | HTML copies and lightly edited versions slip through (low recall) |
| **Medium (0.7)** | Catches HTML copies and edited versions | **Lab default** |
| Low (0.3) | Very aggressive | Distinct documents that share boilerplate get merged (low precision) |

The lab's check enforces both sides: no duplicated content left, *and* exactly 20 unique knowledge
documents kept.

??? info "Semantic deduplication"
    MinHash finds *lexical* near-duplicates. Two documents saying the same thing in different words
    have low Jaccard similarity. **Semantic dedup** (e.g. SemDeDup) embeds documents with a sentence
    encoder, clusters them, and removes items that are very close in embedding space. NeMo Curator
    supports exact, fuzzy and semantic dedup on GPU.

---

## 6. PII redaction

### What counts as PII

Personally identifiable information is anything that identifies a person: names, email
addresses, phone numbers, postal addresses, account and ID numbers, IP addresses, and sometimes
combinations of harmless-looking details. Regulations such as GDPR restrict how it may be processed,
and a model that memorises it can leak it.

### How Presidio detects it

[Microsoft Presidio](https://microsoft.github.io/presidio/) combines several detectors and scores each finding:

| Detector type | Finds | Example |
|---|---|---|
| **NER model** (spaCy) | Names, locations, organisations | "Maria Jansen" -> PERSON |
| **Pattern + validation** | Structured identifiers | Emails; phone numbers parsed with `phonenumbers`; credit cards with a checksum |
| **Context words** | Boosts confidence | "call me on ..." raises a nearby number's score |

You then choose an **operator** for each entity type:

| Operator | Output | Use when |
|---|---|---|
| Replace | `<PERSON>` | Training data: keeps sentence structure, removes identity (lab choice) |
| Mask | `M**** J*****` | Display to humans |
| Hash | `a91f3c...` | You need to link records without revealing identity |
| Synthetic replacement | "Anna Smith" | You want natural text (be careful about consistency) |

### Over-redaction is a real failure

PII detection is a trade-off between **missing PII** (privacy risk) and **redacting non-PII**
(destroying knowledge). Run Presidio with every entity type and it may tag "E-204" or "24 months"
as an identifier or date. Those are exactly the facts we want the model to learn. The lab restricts
detection to `PERSON`, `EMAIL_ADDRESS` and `PHONE_NUMBER`, and its check fails if error codes
disappear.

!!! warning "Small NER models miss things"
    The lab uses spaCy's small English model, which is fast but imperfect. Names in unusual formats,
    or from languages it rarely saw, can slip through. Production pipelines use larger models,
    domain-specific patterns, and **sample audits by humans**.

---

## 7. Synthetic data generation

### Why generate data?

Fine-tuning needs examples of the behaviour you want, such as questions paired with ideal answers.
Enterprises usually have documents but no Q&A pairs, and writing thousands by hand is slow and
expensive. A capable LLM can turn documents into training examples in minutes.

### Common approaches

| Approach | How it works | Used for |
|---|---|---|
| **Grounded generation** | Give the LLM a source document; ask for Q&A pairs about it | Domain knowledge (this lab) |
| **Self-Instruct** | Seed tasks -> LLM writes new tasks and answers -> filter -> repeat | General instruction data |
| **Evol-Instruct** | LLM rewrites instructions to be harder or more diverse | Complex reasoning data |
| **Persona-driven** | Ask the same thing "as" different user personas | Diversity |
| **Distillation** | A strong "teacher" model answers prompts for a smaller "student" | Transferring capability (Module 4) |

### Risks, and the filters that address them

| Risk | What it looks like | Mitigation |
|---|---|---|
| **Hallucination** | Answer states a fact the document doesn't contain | **Grounding filter**, LLM judge |
| **Low diversity** | 80 variations of "What is X?" | Prompt for varied question types, dedup questions |
| **Format errors** | Missing question marks, merged pairs | Parsing + schema validation |
| **Model collapse** | Training models on model output over generations degrades them (Shumailov et al., 2024) | Keep real data in the mix; verify synthetic data |
| **Licence terms** | Some model licences restrict using outputs to train other models | Check the generator's terms |

### The grounding filter

For each generated answer, take its **content words** (dropping "the", "is" and other stopwords) and
measure what fraction also appears in the source document:

$$\text{overlap} = \frac{|\text{content}(\text{answer}) \cap \text{content}(\text{document})|}{|\text{content}(\text{answer})|}$$

Keep the pair if overlap >= 0.6. It's crude, since a paraphrase may use different words, but it
reliably catches answers that bring in facts from nowhere. In the lab's test run it rejected
"Anyone who has observed unusual conditions should check for ventilation gaps": plausible-sounding,
and not in the source.

**Word overlap can't check correctness**, though. "The warranty is 15 years" overlaps well with a
document saying "12-year warranty". That's why the Pro bonus adds an **LLM judge** that grades
faithfulness, and why Module 2 spends a whole section on making judges trustworthy.

---

## 8. What the lab pipeline does, by the numbers

| Stage | Documents left | Removed |
|---|---|---|
| Raw dump | 50 | |
| After quality filters | 41 | 4 junk, 2 spam, 3 non-English |
| After exact dedup | 36 | 5 exact copies |
| After fuzzy dedup | 26 | 5 near-copies, 5 HTML copies |
| After PII redaction | 26 | 0 removed, 6 tickets redacted |
| **Curated corpus** | **26** | **48% of the raw dump discarded** |

Discarding half the raw data is normal. Large web pipelines often discard far more.

---

## Lab

<div class="lab-card" markdown>
**Lab 1: Curate the Lumora corpus**

You start with 50 messy documents and finish with a clean corpus (about 26 docs) and a grounded Q&A
set, which you'll reuse in Modules 2 and 3.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinothss/add_llmknowledge/blob/main/notebooks/01_data_curation.ipynb)

| Exercise | Auto-check |
|---|---|
| 1. `clean_text` | 6 unit tests |
| 2. Quality filters | junk, spam and foreign pages removed; all good docs kept |
| 3. Exact + fuzzy dedup | exactly 20 unique knowledge docs remain |
| 4. PII redaction | no emails, phones or names; error codes kept |
| 5. Synthetic Q&A | at least 20 pairs, at least 90% grounded |
| Pro bonus | LLM-as-a-judge filtering |
</div>

<div class="lab-complete" data-id="m01"></div>

## How this maps to NVIDIA NeMo Curator

The lab uses lightweight open-source libraries so it runs on a free GPU.
[NeMo Curator](https://github.com/NVIDIA/NeMo-Curator) implements the *same* stages with GPU
acceleration (RAPIDS) and multi-node scaling for terabyte-scale corpora:

| Lab | NeMo Curator |
|---|---|
| `clean_text` | `UnicodeReformatter`, custom `DocumentModifier` |
| word-count / repetition filters | `WordCountFilter`, `RepeatingTopNGramsFilter` and other heuristic filters via `ScoreFilter` |
| `langdetect` | `FastTextLangId` |
| MD5 / MinHash | `ExactDuplicates` / `FuzzyDuplicates`, plus semantic dedup |
| Presidio | `PiiModifier` |
| Q&A generation + filtering | Synthetic data generation pipelines, quality classifiers, reward-model filtering |

## Common pitfalls checklist

- [ ] Writing filters without looking at the data first
- [ ] Stripping `<script>` tags but leaving their contents
- [ ] Exact dedup only: near-copies and HTML mirrors survive
- [ ] Dedup threshold never tuned: either duplicates survive or distinct docs get merged
- [ ] Redacting every entity type: knowledge such as error codes and numbers disappears
- [ ] Trusting synthetic data without a faithfulness check
- [ ] Not recording what each stage removed, so you can't explain or reproduce the dataset

## Glossary

| Term | Meaning |
|---|---|
| **Boilerplate** | Repeated non-content text: navigation, footers, cookie notices |
| [**NFKC**](https://en.wikipedia.org/wiki/Unicode_equivalence) | Unicode normalisation form that maps compatible characters to a canonical form |
| [**Shingle**](https://en.wikipedia.org/wiki/W-shingling) | Overlapping window of k consecutive words used to represent a document as a set |
| [**Jaccard similarity**](https://en.wikipedia.org/wiki/Jaccard_index) | Size of the intersection divided by size of the union of two sets |
| [**MinHash**](https://en.wikipedia.org/wiki/MinHash) | Compact signature whose per-position match rate estimates Jaccard similarity |
| [**LSH**](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) | Bucketing scheme that finds likely-similar pairs without comparing all pairs |
| [**PII**](https://en.wikipedia.org/wiki/Personal_data) | Personally identifiable information |
| [**NER**](https://en.wikipedia.org/wiki/Named-entity_recognition) | Named-entity recognition: a model that tags names, places, organisations |
| **Grounding** | Whether a generated answer is supported by its source document |
| [**Model collapse**](https://en.wikipedia.org/wiki/Model_collapse) | Quality degradation from repeatedly training on model-generated data |

## Further reading

**Papers**

- Lee et al. (2022), [*Deduplicating Training Data Makes Language Models Better*](https://arxiv.org/abs/2107.06499)
- Penedo et al. (2024), [*The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*](https://arxiv.org/abs/2406.17557)
- Rae et al. (2021), [*Scaling Language Models: Methods, Analysis & Insights from Training Gopher*](https://arxiv.org/abs/2112.11446) (Appendix: quality filters)
- Broder (1997), [*On the Resemblance and Containment of Documents*](https://doi.org/10.1109/SEQUEN.1997.666900) (MinHash)
- Carlini et al. (2021), [*Extracting Training Data from Large Language Models*](https://arxiv.org/abs/2012.07805)
- Wang et al. (2023), [*Self-Instruct: Aligning Language Models with Self-Generated Instructions*](https://arxiv.org/abs/2212.10560)
- Shumailov et al. (2024), [*AI models collapse when trained on recursively generated data*](https://doi.org/10.1038/s41586-024-07566-y)

**Concepts and tools**

- [Jaccard index (Wikipedia)](https://en.wikipedia.org/wiki/Jaccard_index)
- [w-shingling (Wikipedia)](https://en.wikipedia.org/wiki/W-shingling)
- [MinHash (Wikipedia)](https://en.wikipedia.org/wiki/MinHash)
- [Locality-sensitive hashing (Wikipedia)](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)
- [Leskovec, Rajaraman & Ullman, *Mining of Massive Datasets*, chapter 3: Finding Similar Items (free book; the clearest explanation of shingling, MinHash and LSH)](http://www.mmds.org/)
- [Unicode equivalence and normalisation forms, including NFKC (Wikipedia)](https://en.wikipedia.org/wiki/Unicode_equivalence)
- [Language identification (Wikipedia)](https://en.wikipedia.org/wiki/Language_identification)
- [Named-entity recognition (Wikipedia)](https://en.wikipedia.org/wiki/Named-entity_recognition)
- [Personal data and GDPR (Wikipedia)](https://en.wikipedia.org/wiki/Personal_data)
- [Model collapse (Wikipedia)](https://en.wikipedia.org/wiki/Model_collapse)
- [datasketch: MinHash LSH documentation](https://ekzhu.com/datasketch/lsh.html)
- [Microsoft Presidio documentation](https://microsoft.github.io/presidio/)
- [langdetect (PyPI)](https://pypi.org/project/langdetect/)
- [trafilatura: web text extraction](https://trafilatura.readthedocs.io/)

## Knowledge check

<script type="application/json" class="quiz" data-id="m01">
[
  {"q": "Why should cheap heuristic filters run before model-based quality filters?",
   "options": ["They are more accurate", "They remove most junk for little compute, so expensive models only see what is left", "Model-based filters cannot read HTML", "It doesn't matter which runs first"],
   "answer": 1, "why": "Curation is a funnel: spend compute only on the data that survives the cheap stages."},
  {"q": "An HTML copy of a page still has a different MD5 hash from the clean page after cleaning. Which step catches it?",
   "options": ["Exact deduplication", "Language identification", "Fuzzy deduplication with MinHash", "PII redaction"],
   "answer": 2, "why": "Small differences change the hash completely; MinHash estimates the overlap of shingles instead."},
  {"q": "What property makes MinHash useful?",
   "options": ["It encrypts documents", "The probability that two documents share a MinHash value equals their Jaccard similarity", "It always finds exact duplicates", "It translates documents into English"],
   "answer": 1, "why": "Matching rate over many hash functions estimates similarity from a fixed-size signature."},
  {"q": "You lower the MinHash LSH threshold from 0.7 to 0.3. What is the most likely effect?",
   "options": ["More near-duplicates slip through", "Distinct documents start being removed as 'duplicates'", "Nothing changes", "Dedup becomes faster and more accurate"],
   "answer": 1, "why": "Lower threshold raises recall but lowers precision: unrelated docs that share boilerplate get merged."},
  {"q": "Why restrict Presidio to PERSON, EMAIL_ADDRESS and PHONE_NUMBER in this lab?",
   "options": ["Other entities are not supported", "To avoid over-redacting useful knowledge such as error codes and product numbers", "It makes Presidio run on GPU", "Because the documents are in English"],
   "answer": 1, "why": "Over-redaction destroys the knowledge we are trying to teach (E-204, 10.2 kWh...)."},
  {"q": "A synthetic answer says the Vault warranty is 15 years, but the source says 12. Will the word-overlap grounding filter reliably catch it?",
   "options": ["Yes, always", "No: most of the answer's words still overlap, which is why an LLM judge is added", "Yes, because numbers are weighted heavily", "Only if the answer is long"],
   "answer": 1, "why": "Overlap catches invented content, not subtly wrong values. Correctness needs a judge."}
]
</script>
