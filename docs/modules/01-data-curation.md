# Module 1: Data curation & synthetic data generation

**Time:** about 75 minutes, reading and lab together &nbsp;|&nbsp; **Track:** Free (T4), with a Pro bonus

## The situation

Lumora Energy, the home-battery company in this workshop's case study, wants an assistant that
answers questions about its products: the Vault batteries, the Bridge inverter and the Lumora Home
app. A general model knows nothing about them. Later modules teach it, but every one of those
methods learns from **data**, so this module builds that data.

Lumora has handed you a dump of **50 documents** pulled from its website, internal wiki and support
system. As in any real company, it's a mess:

| What's in the dump | How many |
|---|---|
| Useful product, policy and support documents | 20 |
| Exact copies of some of those documents | 5 |
| Lightly edited copies (a footer added, a line changed) | 5 |
| Raw HTML copies of web pages, with scripts and menus | 5 |
| Junk: a 404 page, a login prompt, a cookie banner, a blank page | 4 |
| Keyword spam | 2 |
| Pages in German, Spanish and Dutch | 3 |
| Support tickets containing customer names, emails and phone numbers | 6 |

By the end of the module you'll have turned it into the two kinds of data that Module 3 trains on:

1. **A curated corpus**: clean, unique documents with no personal data. Module 3 uses it for
   *continued pretraining*, which teaches the model Lumora's facts.
2. **A set of question-answer pairs** generated from those documents. Module 3 uses it for
   *supervised fine-tuning*, which teaches the model to answer questions.

Module 3 ships a larger version of both, built with this same pipeline, so it works even if you
skip ahead; you can swap in your own files from this lab.

### Why this matters

Every training method in this workshop does the same thing: it makes the model **more likely to
produce text that looks like its training data**. The model can't tell which parts you *meant* it
to learn. Leave the dump as it is and here's what it learns:

| Left in the data | What the model learns |
|---|---|
| The same page five times | To recite that page word for word, at the expense of everything else |
| Cookie banners and menus | To say "Accept all cookies?" in the middle of an answer |
| Customer emails and phone numbers | To repeat them when prompted: a privacy incident |
| German and Dutch pages | To switch language at random |

Research backs this up. Deduplicating training data cut word-for-word memorisation roughly tenfold
(Lee et al., 2022). Heavily filtered web data trains better models than larger unfiltered data
(Hugging Face's FineWeb, 2024). And personal data that goes into a model can be pulled back out:
researchers extracted names, phone numbers and emails from GPT-2 (Carlini et al., 2021).
**Curation is where you decide what the model knows.**

### The plan: a funnel

A good pipeline is a **funnel**. Cheap, fast steps remove most of the junk first, so slower steps
only handle what's left. This module follows the funnel from top to bottom:

```mermaid
flowchart TB
  subgraph A["Part A: curate the documents"]
    direction LR
    R[Raw dump<br/>50 docs] --> C[2. Clean] --> F[3. Filter junk] --> D[4. Remove<br/>duplicates] --> P[5. Redact<br/>personal data] --> O[Curated corpus<br/>26 docs]
  end
  subgraph B["Part B: turn them into training examples"]
    direction LR
    G[6. Generate<br/>Q&A pairs] --> V[7. Keep only<br/>faithful pairs] --> Q[Q&A set]
  end
  A --> B
```

**Part A** (steps 1-5) curates the documents; Step 1 is looking at the raw data. **Part B**
(steps 6-7) turns the curated documents into training examples.

<div class="lab-card" markdown>
**Open the lab now and work through it alongside this page**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinothss/add_llmknowledge/blob/main/notebooks/01_data_curation.ipynb)

The notebook uses the same step numbers as this page. Each step below ends with a
**▶ Lab** box that tells you which exercise to do. Every exercise has an automatic check.

| Exercise | Step | The check confirms |
|---|---|---|
| 1. `clean_text` | 2 | 6 unit tests pass |
| 2. Quality filters | 3 | junk, spam and foreign pages removed; all good documents kept |
| 3. Deduplication | 4 | exactly 20 unique knowledge documents remain |
| 4. PII redaction | 5 | no emails, phones or names left; error codes kept |
| 5. Synthetic Q&A | 6-7 | at least 20 pairs, all grounded in their source |
| Pro bonus | 7 | LLM-as-a-judge filtering (Colab Pro only) |
</div>

---

## Part A: Curate the documents

### Step 1: Look at the raw data

Before writing a single rule, **look at the data**. Every rule in a real pipeline should trace
back to a problem someone *saw*. Rules written without looking remove good data and miss bad data.

!!! example "▶ Lab: Step 1"
    Run the setup cells, then scroll through the raw table. Try to spot each problem from the
    table above: the HTML, the copies, the junk pages, the foreign languages and the customer
    details. Which would hurt the model most if left in?

The first thing you'll notice is that many documents are mostly *not* text: they're HTML. That's
the first job.

---

### Step 2: Clean the text

A scraped product page is mostly tags, scripts, menus, cookie notices and footers wrapped around
a few paragraphs of real content. Cleaning extracts the content:

1. **Remove `<script>` and `<style>` blocks, including their contents.** Removing only the tags
   leaves raw JavaScript in your training data.
2. **Strip the remaining tags**, keeping line breaks so later rules can work line by line.
3. **Decode HTML entities:** `&amp;` becomes `&`, `&lt;` becomes `<`.
4. **Normalise unicode.** There are many ways to write "the same" character: non-breaking spaces,
   full-width digits, zero-width characters. **[NFKC normalisation](https://en.wikipedia.org/wiki/Unicode_equivalence)**
   maps them to one standard form, so "10.2 kWh" with an odd space is treated like normal text.
5. **Remove boilerplate lines** such as menus, footers and "Subscribe to our newsletter". The lab
   uses simple line patterns. Production tools such as `trafilatura` use page layout and text
   density instead.
6. **Collapse repeated whitespace.**

!!! example "▶ Lab: Exercise 1, write `clean_text`"
    Implement the six steps, run the check, and compare a document before and after cleaning.

Clean text shows the next problem clearly: some documents have no useful content at all.

---

### Step 3: Filter out junk

With the HTML gone, you can judge documents by their text. Large pretraining projects publish
their filter rules, and they're surprisingly simple. DeepMind's Gopher rules (Rae et al., 2021)
and Google's C4 rules (Raffel et al., 2020) include:

| Rule | Catches | Example threshold |
|---|---|---|
| Word count | Empty pages, "404", cookie notices | Keep 50-100,000 words (Gopher) |
| Mean word length | Gibberish, minified code | 3-10 characters |
| Symbol-to-word ratio | Hashtag spam, code dumps | Under 0.1 for `#` or `...` |
| Stopword presence | Tables, lists, keyword spam | At least 2 common English words |
| Repeated lines or word pairs | Templated spam | Top 2-gram covers under 20% of text |
| Lines ending in punctuation | Menus, fragments | C4 keeps only such lines |

Lumora's documents are short, so the lab uses three rules sized for them:

| Rule | Removes from the Lumora dump |
|---|---|
| **Word count of at least 20** | The 404 page, login prompt, cookie banner and blank page |
| **Unique-word ratio of at least 0.35** (unique words ÷ total words) | The keyword spam, which scores around 0.05 |
| **Language ID**: `langdetect` says English with probability of at least 0.9 | The German, Spanish and Dutch pages |

??? info "Why use a language identifier instead of checking for English words?"
    Checking for common English words sounds simpler, but it fails on short texts. A Dutch sentence
    like "Lumora Energy is in 2017 opgericht in Rotterdam" contains the English words "is" and
    "in", and can score higher than a short English support ticket. A trained language identifier
    looks at character patterns across the whole text and is far more reliable. NeMo Curator uses
    fastText's language-ID model, which covers 176 languages.

!!! example "▶ Lab: Exercise 2, write `quality_filter`"
    Implement the three rules. The check confirms every junk, spam and foreign page is removed
    *and* every good document is kept. Removing good data is a failure too.

??? info "Beyond the lab: model-based filters"
    Rules can tell that a page is short or repetitive, but not whether it's *informative*. Large
    pipelines add a learned filter after the rules:

    - **Perplexity filtering** (CCNet): a small language model trained on high-quality text, such
      as Wikipedia, scores each document, and text it finds very surprising is dropped.
    - **Quality classifiers**: a strong LLM rates a sample of documents (for example "educational
      value 0-5"), a small fast classifier learns from those ratings, and then it scores billions
      of documents. FineWeb-Edu was built this way.
    - **LLM judges**: the most accurate and most expensive option, so used last on small sets.
      You'll use one in Step 7 to judge Q&A pairs.

    Lumora's 50 documents don't need this, which is why it isn't in the funnel above.

What's left is all real content, but much of it is the same content more than once.

---

### Step 4: Remove duplicates

#### Why duplicates hurt

A document that appears five times gets five times the training. The model memorises it at the
expense of everything else, wastes compute, and becomes more likely to repeat it word for word.
Duplicates also make evaluation unreliable when test questions leak into training data, a problem
you'll meet again in Modules 2 and 3.

The Lumora dump has three kinds of copy, and they need two techniques.

#### Exact copies: hashing

Turn each document into a **hash**, a short fingerprint of its text (the lab uses MD5 of the
lower-cased text), and keep one document per fingerprint. It's fast and certain, but it only
catches *identical* text: change a single character and the fingerprint is completely different.

#### Near copies: MinHash

Lumora's edited copies and HTML mirrors aren't identical, so hashing misses them. To catch them
you need a measure of *how similar* two documents are, and a way to compute it quickly.

1. **Turn each document into a set of shingles.** A **[shingle](https://en.wikipedia.org/wiki/W-shingling)**
   is a run of consecutive words; the lab uses 5. "The Vault 10 weighs 98 kg" gives {"the vault 10
   weighs 98", "vault 10 weighs 98 kg"}. Near copies share most of their shingles.
2. **Measure the overlap with [Jaccard similarity](https://en.wikipedia.org/wiki/Jaccard_index):**
   the number of shingles two documents share, divided by the number of distinct shingles they have
   between them. Identical documents score 1.0, a copy with one sentence added scores about 0.85,
   and unrelated documents score close to 0.
3. **Estimate it cheaply with [MinHash](https://en.wikipedia.org/wiki/MinHash).** Comparing full
   shingle sets is slow, so MinHash shrinks every document to a fixed signature of 128 numbers.
   The fraction of positions where two signatures match estimates their Jaccard similarity.
4. **Avoid comparing every pair with [LSH](https://en.wikipedia.org/wiki/Locality-sensitive_hashing).**
   Locality-sensitive hashing sorts signatures into buckets so that only likely-similar documents
   are compared at all.

??? info "How it works: the maths behind MinHash and LSH"
    **Jaccard similarity** of shingle sets $A$ and $B$:

    $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

    **MinHash** applies a random hash function $h$ to every shingle and keeps only the minimum
    value. The key property is:

    $$P\big[\min h(A) = \min h(B)\big] = J(A, B)$$

    The chance that two documents share a MinHash value *equals* their Jaccard similarity. Repeat
    with 128 different hash functions and the fraction of matches estimates $J$ closely, whatever
    the documents' length.

    **LSH.** Comparing every pair of signatures is still $O(n^2)$: about 500 billion comparisons for
    a million documents. LSH splits each signature into $b$ bands of $r$ rows and hashes each band
    into a bucket. Documents that share a bucket in *any* band become candidate pairs. A pair with
    similarity $s$ becomes a candidate with probability

    $$P(\text{candidate}) = 1 - (1 - s^r)^b$$

    This is an S-shaped curve: pairs above the threshold almost always collide, and pairs well
    below almost never do. `datasketch` picks $b$ and $r$ for you from the `threshold` you give it.
    Chapter 3 of the free book *Mining of Massive Datasets* (in Further reading) explains this step
    by step.

#### Choosing the threshold

The similarity threshold decides what counts as a duplicate, and it's a trade-off:

| Threshold | Effect | Risk |
|---|---|---|
| High (0.95) | Only near-identical copies removed | Edited copies and HTML mirrors slip through |
| **Medium (0.7)** | Catches edited copies and HTML mirrors | **Lab default** |
| Low (0.3) | Very aggressive | Different documents that share boilerplate get merged |

!!! example "▶ Lab: Exercise 3, write `exact_dedup` and `fuzzy_dedup`"
    The check enforces both sides of the trade-off: no duplicated content left, *and* exactly 20
    unique knowledge documents kept. Then try thresholds of 0.3 and 0.95 and see what goes wrong
    at each extreme.

??? info "Beyond the lab: semantic deduplication"
    MinHash finds documents that share *words*. Two documents that say the same thing in
    different words have low Jaccard similarity. **Semantic deduplication** (for example SemDeDup)
    turns each document into an embedding, groups similar ones, and removes items that are very
    close together. NeMo Curator supports exact, fuzzy and semantic deduplication on GPU.

The corpus is now clean and unique. One problem is left, and it's the riskiest: the support
tickets contain real customers' details.

---

### Step 5: Redact personal data

#### What counts as personal data

Personally identifiable information (**PII**) is anything that identifies a person: names, email
addresses, phone numbers, postal addresses, account numbers, and sometimes combinations of
harmless-looking details. Laws such as GDPR restrict how it may be processed, and a model that
memorises it can leak it.

The support tickets are worth keeping: they show how customers describe problems. So rather than
delete them, you **redact** them, replacing the personal details and keeping the rest.

#### How Presidio finds it

[Microsoft Presidio](https://microsoft.github.io/presidio/) combines several detectors and gives
each finding a confidence score:

| Detector | Finds | Example |
|---|---|---|
| **Named-entity recognition** (a spaCy model) | Names, places, organisations | "Maria Jansen" is a PERSON |
| **Patterns with validation** | Structured identifiers | Emails; phone numbers checked with `phonenumbers` |
| **Context words** | Raises confidence | "call me on ..." makes a nearby number more likely to be a phone number |

You then choose what to replace each finding with:

| Option | Output | Use when |
|---|---|---|
| Replace | `<PERSON>` | Training data: keeps the sentence shape, removes the identity (lab choice) |
| Mask | `M**** J*****` | Showing records to people |
| Hash | `a91f3c...` | You need to link records without revealing who they are |
| Fake replacement | "Anna Smith" | You want natural-looking text (keep it consistent) |

#### Don't redact too much

Redaction is a trade-off between **missing personal data** (a privacy risk) and **redacting
things that aren't personal** (destroying knowledge). Turn on every entity type and Presidio may
tag "E-204" or "24 months" as an identifier or a date, and those are exactly the facts the model
should learn. The lab limits detection to `PERSON`, `EMAIL_ADDRESS` and `PHONE_NUMBER`.

!!! example "▶ Lab: Exercise 4, redact the support tickets"
    The check confirms that no emails, phone numbers or names remain, and that error codes such as
    E-204 survive.

!!! warning "Small models miss things"
    The lab uses spaCy's small English model, which is fast but imperfect. Names in unusual
    formats, or from languages it rarely saw, can slip through. Production pipelines use larger
    models, extra patterns for their own data, and **spot checks by people**.

### Part A result

Here's what the funnel did to Lumora's dump:

| Step | Documents left | Removed |
|---|---|---|
| Raw dump | 50 | |
| After filtering junk | 41 | 4 junk, 2 spam, 3 foreign-language |
| After exact deduplication | 36 | 5 exact copies |
| After near-duplicate removal | 26 | 5 edited copies, 5 HTML mirrors |
| After redaction | 26 | none removed; 6 tickets redacted |
| **Curated corpus** | **26** | **48% of the dump discarded** |

Throwing away half the raw data is normal; large web pipelines discard far more. The notebook
saves this corpus as `curated_corpus.jsonl`.

---

## Part B: Turn the documents into training examples

### Step 6: Generate question-answer pairs

#### Why generate data?

The curated corpus teaches the model *facts*. But Lumora wants an assistant that *answers
questions*, and for that the model needs examples of questions paired with good answers. Nobody
at Lumora has written those, and writing thousands by hand is slow and expensive. A capable LLM
can turn documents into question-answer pairs in minutes. This is called **synthetic data**.

#### Common approaches

| Approach | How it works | Used for |
|---|---|---|
| **Grounded generation** | Give the LLM a source document and ask for Q&A pairs about it | Domain knowledge (this lab) |
| **Self-Instruct** | Start from seed tasks; the LLM writes new tasks and answers; filter; repeat | General instruction data |
| **Evol-Instruct** | The LLM rewrites instructions to be harder or more varied | Complex reasoning data |
| **Persona-driven** | Ask the same thing as different kinds of user | Variety |
| **Distillation** | A strong "teacher" model answers prompts for a smaller "student" | Transferring ability (Module 4) |

The lab uses grounded generation: the model reads each knowledge document (not the support
tickets) and writes four question-answer pairs about specific facts in it.

!!! example "▶ Lab: Exercise 5, part 1: generate candidates"
    Read the generation prompt, run it over the knowledge documents, and look at a few of the
    candidate pairs it produces.

Read a few candidates and you'll see why the next step exists: some answers sound right but
aren't in the document.

---

### Step 7: Keep only faithful pairs

#### What can go wrong with synthetic data

| Risk | What it looks like | What to do |
|---|---|---|
| **Hallucination** | The answer states a fact the document doesn't contain | **Grounding filter**, LLM judge |
| **Low variety** | 80 versions of "What is X?" | Ask for varied question types; remove duplicate questions |
| **Format errors** | Missing question marks, two pairs merged into one | Parse and validate the format |
| **Model collapse** | Models trained on model output, generation after generation, get worse (Shumailov et al., 2024) | Keep real data in the mix; check synthetic data |
| **Licence terms** | Some model licences restrict using their output to train other models | Check the generator's terms |

#### The grounding filter

For each generated answer, take its **content words** (dropping "the", "is" and other common
words) and measure what fraction also appear in the source document:

$$\text{overlap} = \frac{|\text{content}(\text{answer}) \cap \text{content}(\text{document})|}{|\text{content}(\text{answer})|}$$

Keep the pair if the overlap is at least 0.6. It's crude, since a paraphrase may use different
words, but it reliably catches answers that bring in facts from nowhere, such as "Anyone who has
observed unusual conditions should check for ventilation gaps" when the document says nothing
about it.

!!! example "▶ Lab: Exercise 5, part 2: write the grounding filter"
    Implement `is_grounded`, drop duplicate questions, and run the check. Look at the rejected
    pairs: would you want the model to learn them?

#### When word overlap isn't enough

Word overlap can't check that an answer is *correct*. "The warranty is 15 years" overlaps well
with a document that says "12-year warranty". Catching that needs a model that reads the
document and the answer together: an **LLM judge**.

!!! example "▶ Lab: Pro bonus, LLM-as-a-judge (Colab Pro only)"
    On an L4 or A100 runtime, a 7B model grades each pair from 1 to 5 for faithfulness, and you
    keep only pairs scored 4 or 5. On the free T4 the notebook skips this cell. Module 2 spends a
    whole section on making judges like this trustworthy.

The notebook saves the Q&A set as `sft_qa.jsonl`.

<div class="lab-complete" data-id="m01"></div>

---

## Wrap-up

### Key takeaways

- **Look at the data first.** Every rule in this module came from a problem visible in the raw
  table.
- **Cheap steps first, expensive steps last.** Rules and hashing remove most of the junk for almost
  no compute; LLMs only see what survives.
- **Every filter is a trade-off.** Too strict and you lose good data; too loose and junk gets
  through. The checks in this lab test both sides.
- **Synthetic data needs checking.** Generating pairs is easy; making sure they're faithful to
  the source is the real work.

### At production scale: NVIDIA NeMo Curator

The lab uses small open-source libraries so it runs on a free GPU.
[NeMo Curator](https://github.com/NVIDIA/NeMo-Curator) runs the *same* steps on GPUs and across
many machines, for terabytes of data:

| Lab | NeMo Curator |
|---|---|
| `clean_text` | `UnicodeReformatter`, custom `DocumentModifier` |
| Word-count and repetition rules | `WordCountFilter`, `RepeatingTopNGramsFilter` and other filters via `ScoreFilter` |
| `langdetect` | `FastTextLangId` |
| MD5 / MinHash | `ExactDuplicates` / `FuzzyDuplicates`, plus semantic deduplication |
| Presidio | `PiiModifier` |
| Q&A generation and filtering | Synthetic data pipelines, quality classifiers, reward-model filtering |

### Common pitfalls

- [ ] Writing filters without looking at the data first
- [ ] Removing `<script>` tags but leaving the code inside them
- [ ] Removing only exact copies, so edited copies and HTML mirrors survive
- [ ] Never tuning the deduplication threshold
- [ ] Redacting every entity type, so error codes and numbers disappear
- [ ] Trusting synthetic data without checking it against the source
- [ ] Not recording what each step removed, so you can't explain or reproduce the dataset

### Knowledge check

<script type="application/json" class="quiz" data-id="m01">
[
  {"q": "Why should cheap rule-based filters run before model-based quality filters?",
   "options": ["They are more accurate", "They remove most junk for little compute, so expensive models only see what is left", "Model-based filters cannot read HTML", "It doesn't matter which runs first"],
   "answer": 1, "why": "Curation is a funnel: spend compute only on the data that survives the cheap steps."},
  {"q": "After cleaning, an HTML copy of a page still has a different MD5 hash from the original. Which step catches it?",
   "options": ["Exact deduplication", "Language identification", "Near-duplicate removal with MinHash", "PII redaction"],
   "answer": 2, "why": "Small differences change the hash completely; MinHash estimates the overlap of shingles instead."},
  {"q": "What property makes MinHash useful?",
   "options": ["It encrypts documents", "The probability that two documents share a MinHash value equals their Jaccard similarity", "It always finds exact duplicates", "It translates documents into English"],
   "answer": 1, "why": "The match rate over many hash functions estimates similarity from a fixed-size signature."},
  {"q": "You lower the MinHash threshold from 0.7 to 0.3. What is the most likely effect?",
   "options": ["More near-duplicates slip through", "Different documents start being removed as 'duplicates'", "Nothing changes", "Deduplication becomes faster and more accurate"],
   "answer": 1, "why": "A lower threshold catches more copies but also merges different documents that share boilerplate."},
  {"q": "Why does the lab limit Presidio to PERSON, EMAIL_ADDRESS and PHONE_NUMBER?",
   "options": ["Other entities are not supported", "To avoid redacting useful knowledge such as error codes and product numbers", "It makes Presidio run on GPU", "Because the documents are in English"],
   "answer": 1, "why": "Over-redaction destroys the knowledge we're trying to teach (E-204, 10.2 kWh...)."},
  {"q": "A synthetic answer says the Vault warranty is 15 years, but the source says 12. Will the word-overlap grounding filter reliably catch it?",
   "options": ["Yes, always", "No: most of the answer's words still overlap, which is why an LLM judge is added", "Yes, because numbers are weighted heavily", "Only if the answer is long"],
   "answer": 1, "why": "Overlap catches invented content, not subtly wrong values. Correctness needs a judge."}
]
</script>

### Glossary

| Term | Meaning |
|---|---|
| **Boilerplate** | Repeated non-content text: menus, footers, cookie notices |
| [**NFKC**](https://en.wikipedia.org/wiki/Unicode_equivalence) | A unicode normalisation form that maps equivalent characters to one standard form |
| [**Shingle**](https://en.wikipedia.org/wiki/W-shingling) | A run of k consecutive words, used to represent a document as a set |
| [**Jaccard similarity**](https://en.wikipedia.org/wiki/Jaccard_index) | Shared items divided by all distinct items across two sets |
| [**MinHash**](https://en.wikipedia.org/wiki/MinHash) | A compact signature whose match rate estimates Jaccard similarity |
| [**LSH**](https://en.wikipedia.org/wiki/Locality-sensitive_hashing) | A bucketing scheme that finds likely-similar pairs without comparing every pair |
| [**PII**](https://en.wikipedia.org/wiki/Personal_data) | Personally identifiable information |
| [**NER**](https://en.wikipedia.org/wiki/Named-entity_recognition) | Named-entity recognition: a model that tags names, places and organisations |
| **Grounding** | Whether a generated answer is supported by its source document |
| [**Model collapse**](https://en.wikipedia.org/wiki/Model_collapse) | Quality loss from repeatedly training on model-generated data |

### Further reading

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
