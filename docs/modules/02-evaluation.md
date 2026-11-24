# Module 2: Evaluating LLMs

**Time:** about 40 min reading + 45 min lab &nbsp;|&nbsp; **Track:** Free (T4), with a Pro bonus

## Learning objectives

By the end of this module you should be able to:

- [ ] Explain why evaluating LLMs is harder than evaluating classic ML models
- [ ] Choose between benchmarks, reference metrics, LLM judges and pairwise ranking for a given question
- [ ] Score multiple-choice benchmarks by log-likelihood, zero-shot and few-shot
- [ ] Build a domain benchmark and measure a knowledge gap (closed-book vs open-book)
- [ ] Calibrate an LLM judge against human labels, and report its error rates honestly
- [ ] Compute Elo ratings from pairwise comparisons and control for position bias
- [ ] Judge whether a difference between two models is real or noise
- [ ] Track experiments so results can be compared and reproduced

---

## 1. Why evaluating LLMs is hard

A spam classifier outputs "spam" or "not spam", and you compare against labels: done. An LLM outputs
**free text**, and that breaks the simple recipe in several ways:

| Challenge | Example |
|---|---|
| **Many correct answers** | "12 years", "twelve years" and "The warranty lasts 12 years" are all right |
| **Many dimensions** | Correct? Faithful to the source? Helpful? Safe? Right tone? Concise? |
| **Fluent wrong answers** | "The Vault 10 has 13.5 kWh" reads exactly like a correct answer |
| **Prompt sensitivity** | Changing the prompt format can move benchmark scores by several points |
| **Contamination** | Benchmark questions leak onto the web and into training data |
| **Goodhart's law** | "When a measure becomes a target, it ceases to be a good measure" |

No single number captures model quality. Good evaluation **combines methods**, each answering a
specific question, and is honest about each method's error.

### The evaluation ladder

| Method | Question it answers | Cost | Main weakness |
|---|---|---|---|
| **Eyeballing** | Does it work at all? Is the format right? | Your time | Doesn't scale; fluent errors fool you |
| **Benchmarks** (MMLU, GSM8K...) | How capable is the model in general? | Low | Not *your* domain; contamination; saturation |
| **Reference metrics** (exact match, F1, ROUGE) | Does the answer match a known answer? | Very low | Punish paraphrase, reward word overlap |
| **LLM-as-a-judge** | Is this free-text answer correct, helpful, faithful? | Medium | Biases: leniency, position, verbosity, self-preference |
| **Pairwise + Elo** | Which system is better overall? | Medium | Order effects; needs many comparisons |
| **Human evaluation** | Ground truth for all of the above | High | Slow, expensive, raters disagree too |

---

## 2. Benchmarks: MMLU

### What MMLU is

**MMLU** (Massive Multitask Language Understanding, Hendrycks et al., 2021) has about 14,000
four-option multiple-choice questions across **57 subjects**, from elementary maths and
astronomy to law, medicine and marketing. Random guessing scores 25%. For years it was the standard
headline number for general knowledge and reasoning.

It has well-known limitations, and you should be aware of them when you read model cards:

- **Saturation:** frontier models score near the top, so differences become meaningless.
  Harder successors such as MMLU-Pro (10 options, more reasoning) exist for this reason.
- **Label errors:** independent reviews found that a small percentage of questions have wrong or
  ambiguous answers.
- **Contamination:** the questions have been public since 2020 and appear in web crawls.
- **Not your domain:** a high MMLU score says nothing about Lumora warranty terms.

In this workshop, MMLU's job is to detect **regressions**: if fine-tuning on Lumora data damages
general ability, MMLU drops (Module 3).

### Scoring by log-likelihood

You could generate an answer and parse "The answer is B" with a regex, but models write "B)", "(B)",
"Option B", or explain first and answer last. Parsing is fragile and sampling adds noise.

Evaluation harnesses instead feed the prompt ending in `Answer:` and look at the model's probability
distribution over the **next token**:

```
...
C. It would be harder since the truck is lighter on Mars.
D. It would be the same no matter where you are.
Answer:      ->  P(" A") = 0.08   P(" B") = 0.71   P(" C") = 0.05   P(" D") = 0.16   -> predict B
```

This takes one forward pass per question, needs no parsing, and gives the same answer every run.

??? info "Why the leading space in ' A'?"
    Tokenizers usually treat a word and the same word preceded by a space as **different tokens**.
    After "Answer:" the natural continuation is " B" (with a space). Comparing the probability of
    "B" without the space measures something the model rarely produces. Details like this are why
    different harnesses report slightly different numbers for the same model.

### Zero-shot vs few-shot (in-context learning)

**Zero-shot**: the question alone. **Few-shot**: *k* solved examples first, typically 5 from the
benchmark's dev split. Brown et al. (2020, the GPT-3 paper) showed that large models improve
markedly from a few examples in the prompt **without any weight update**. This is
*in-context learning*. The examples show the format ("answer with a single letter") and prime the
task. Small models usually gain the most, because they're the least able to infer the format.

The same trick reappears in this module: few-shot examples are one way to improve an LLM judge.

---

## 3. Your own benchmark: closed-book vs open-book

Public benchmarks can't tell you what matters most for an enterprise: **does the model know *our*
domain?** For that you need your own test set. The lab ships one: 30 human-written Lumora questions
with reference answers, covering products, warranty, support codes, policies and engineering notes.

!!! warning "Rules for a trustworthy domain benchmark"
    1. **Held out:** never used for training, prompting examples or tuning.
    2. **Written independently** of your training data, ideally by different people.
    3. **Covers the categories you care about**, with enough questions per category to mean something.
    4. **Unambiguous reference answers.** Avoid yes/no questions: a model guesses those correctly
       50% of the time. (We replaced one such question during development for this reason.)
    5. **Versioned**, so scores from different months are comparable.

The lab compares three setups:

| Setup | Prompt contains | Measures |
|---|---|---|
| Small model, closed-book | The question | What the model *knows* |
| Large model, closed-book | The question | Whether scale alone buys domain knowledge |
| Large model, **open-book** | The question **plus the source document** | Whether the model can *use* the knowledge when it has it |

Typical result: both closed-book setups score low, and open-book is near 100%. Two lessons:
**scale doesn't create facts the model never saw**, and **the gap between closed-book and open-book
is the size of the knowledge problem.** RAG closes it by supplying documents at question time;
Module 3 closes it by training the knowledge in.

---

## 4. Reference metrics, and why they're not enough

The cheapest way to grade a free-text answer is to compare it with the reference:

| Metric | How | Fails on |
|---|---|---|
| **Exact match** | Answer string equals reference (after normalising) | Any paraphrase: "twelve years" vs "12 years." |
| **Token F1** | Word overlap ([F-score](https://en.wikipedia.org/wiki/F-score)): harmonic mean of precision and recall (used in SQuAD) | Wrong answers that reuse the question's words |
| **ROUGE / BLEU** | n-gram overlap, from summarisation and translation | Same, plus length effects |
| **Embedding similarity** | Cosine similarity of sentence embeddings | "12 years" and "15 years" embed almost identically |

!!! example "Token F1 rewarding the wrong answer"
    Reference: *"12 years."*
    Answer A: *"The Lumora Vault warranty lasts for 12 years."* -> F1 = 0.40 (correct, but wordy)
    Answer B: *"The Lumora Vault warranty typically lasts 5 years."* -> F1 = 0.20 (**wrong**, still gets credit)
    Answer C: *"Twelve years."* -> F1 = 0.50 (correct, but "twelve" doesn't match "12")

    The two correct answers score only 0.4-0.5, and the wrong one gets half as much credit as the
    correct one. The metric's numbers don't track correctness. Reference metrics are fine for short, canonical
    answers (numbers, names), but free-text answers need a grader that understands meaning.

---

## 5. LLM-as-a-judge

### The idea

Use an LLM to grade answers: give it the question, the reference answer and the candidate, and ask
"Is the candidate correct?". Zheng et al. (2023, the MT-Bench / Chatbot Arena paper) found that a
strong judge such as GPT-4 agreed with human preferences about 80% of the time, roughly as often as
two humans agreed with each other. That made LLM judging the standard way to evaluate open-ended
output at scale.

Three common setups:

| Setup | Judge sees | Use when |
|---|---|---|
| **Reference-guided grading** | Question + reference + candidate | You have a correct answer (the lab's Lumora QA) |
| **Single-answer scoring** | Question + candidate, with a rubric (1-10) | No reference, e.g. helpfulness |
| **Pairwise comparison** | Question + two candidates | Ranking systems (section 6) |

### Known biases

| Bias | What happens | Mitigation |
|---|---|---|
| **Leniency** | Says "correct" to fluent nonsense, especially small judges | Calibrate the threshold on human labels (below) |
| **Position bias** | Prefers whichever answer is shown first (or second) | Judge both orders and average |
| **Verbosity bias** | Prefers longer answers | Rubrics that reward concision; length-controlled comparisons |
| **Self-preference** | Rates its own outputs, or its model family's, higher | Use a different, stronger judge |

### A judge is a classifier, so evaluate it like one

When the judge decides "correct" or "incorrect" it's a binary classifier, and the human labels are
the ground truth. Build the confusion matrix:

| | Human: correct | Human: incorrect |
|---|---|---|
| **Judge: correct** | True positive | **False positive: hallucination accepted** |
| **Judge: incorrect** | False negative: correct answer rejected | True negative |

Two numbers matter most:

- **Agreement** (accuracy): how often the judge matches the human.
- **False-positive rate**: the share of genuinely wrong answers the judge accepts. This is the
  dangerous error, because it **inflates your model's score** and hides hallucinations.

### What happened when we built the lab's judge

This story is worth knowing in detail, because it's what happens in real projects:

1. **Hand-written calibration examples looked fine.** On 30 invented answers, a 1.5B judge agreed
   with the human labels about 90% of the time.
2. **Real model outputs told a different story.** On actual closed-book answers from the models
   under test, the same judge accepted about **half** of them as correct, when a careful human
   reading found only about 10% were. Real hallucinations are fluent, specific and plausible: "The
   warranty typically lasts 5 years". Invented test cases rarely look like that.
3. **Lesson: calibrate on the distribution you'll actually judge.** The lab's calibration set now has
   155 items, 120 of them real outputs from the models being evaluated, each labelled by a human.

### Calibrating with the judge's confidence

Instead of trusting the judge's Yes/No, read how *confident* it is from the probability of its first
token:

$$\text{score} = \log P(\text{"Yes"}) - \log P(\text{"No"})$$

A positive score means the judge leans Yes. Then **choose the threshold from the human labels**:
try every candidate threshold on the calibration split and keep the one with the best agreement.
On ties, prefer the stricter threshold, because accepting hallucinations is the costlier mistake.

**Never tune and report on the same data.** The lab splits by *question* (q01-q15 for tuning,
q16-q30 for reporting) so no question's answers appear on both sides:

| Judge | Calib agreement | **Test agreement** | **Test hallucinations accepted** |
|---|---|---|---|
| Naive (threshold 0, i.e. trust the Yes/No) | 88% | 86% | **21%** |
| Calibrated (threshold about 4.0) | 95% | **84%** | **3%** |

Notice two things. Agreement barely changes, but **false acceptances drop from about 1 in 5 to
about 1 in 30**, and that's what protects your scores. And the calibrated judge does better on calib
than on test: the threshold slightly **overfits** the data it was tuned on, which is exactly why
you report the held-out number.

??? info "Beyond accuracy: Cohen's kappa"
    If 90% of answers are wrong, a judge that always says "incorrect" scores 90% agreement while
    learning nothing. **[Cohen's kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa)** corrects for chance agreement:
    $\kappa = \frac{p_o - p_e}{1 - p_e}$, where $p_o$ is observed agreement and $p_e$ is the agreement
    expected by chance given each rater's label frequencies. Values above about 0.6 are usually read
    as substantial agreement. The lab's calibration set is roughly balanced, so accuracy is
    meaningful there, but use kappa when your classes are skewed.

---

## 6. Pairwise comparison and Elo

### Why pairwise?

For style, helpfulness or chat quality there's no single reference answer, and absolute scores
drift ("is this a 7 or an 8?"). People, and LLM judges, are more consistent at **relative**
judgements: "Is A or B better?". Chatbot Arena ranks models this way from millions of crowd votes.

### The Elo system

Each system has a rating $R$ (see [Elo rating system](https://en.wikipedia.org/wiki/Elo_rating_system)). Before a match, the **expected score** of A against B is:

$$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

After the match, with actual score $S_A$ (1 win, 0.5 tie, 0 loss, or any probability in between):

$$R_A' = R_A + K\,(S_A - E_A), \qquad R_B' = R_B - K\,(S_A - E_A)$$

| Rating gap | Expected win rate of the higher-rated system |
|---|---|
| 0 | 50% |
| 100 | 64% |
| 200 | 76% |
| 400 | 91% |

$K$ (the lab uses 32) sets how much one match moves the ratings. Updates are **zero-sum**: what A
gains, B loses. An upset (a low-rated system beating a high-rated one) moves ratings a lot; an
expected result barely moves them.

### Practical details

- **Order dependence:** Elo processes matches one at a time, so shuffling the match order changes
  final ratings. The lab averages over 50 random orders. Chatbot Arena switched to the
  **[Bradley-Terry](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)** model, which fits all matches at once and doesn't depend on order.
- **Position bias:** the lab judges every pair twice, with A and B swapped, and averages the judge's
  probability that each system wins. The lab prints the measured bias. In our test run the 1.5B judge
  preferred whichever answer came *first* only 36% of the time, a clear lean towards the second
  answer that swapping cancels out.
- **Uncertainty:** ratings from a few dozen matches are noisy. Report confidence intervals
  (bootstrap: resample matches, recompute, repeat) when you compare close systems.

---

## 7. Is the difference real? Statistics for small evals

With 30 questions, every question is worth 3.3 percentage points. If a model's true accuracy is 50%,
the 95% [confidence interval](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval) on a 30-question measurement is roughly:

$$\pm 1.96\sqrt{\frac{p(1-p)}{n}} = \pm 1.96\sqrt{\frac{0.5 \times 0.5}{30}} \approx \pm 18 \text{ points}$$

So **"57% vs 50%" on 30 questions is not evidence of a difference.** "10% vs 70%" is. Practical
guidance:

- Use a few hundred questions before trusting single-digit differences.
- Compare models on the **same** questions (paired comparison) and look at which questions changed.
- Run generation with temperature 0 (as the lab does), or average several seeds.
- Treat the judge's error rate as part of the noise.

---

## 8. Experiment tracking with MLflow

Without tracking, three weeks later nobody remembers which prompt, model revision or judge
threshold produced "72%". MLflow organises results as:

| Concept | Meaning | Lab example |
|---|---|---|
| **Experiment** | A group of related runs | `lumora-eval` |
| **Run** | One evaluation of one configuration | `mmlu-Qwen2.5-1.5B-Instruct-5shot` |
| **Params** | Inputs: what you did | model, benchmark, k_shot, judge |
| **Metrics** | Outputs: what happened | mmlu_acc, lumora_acc, token_f1, elo |
| **Tags** | Free-form labels | tier = free / pro |
| **Artifacts** | Files | predictions, plots, the judge prompt |

For reproducibility, also record the model **revision** (commit hash on Hugging Face), the dataset
version, prompt templates, seeds and library versions. Module 3 compares fine-tuned models against
the baselines you log here.

---

## Lab

<div class="lab-card" markdown>
**Lab 2: How good are our candidate models, and how much do they know about Lumora?**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinothss/add_llmknowledge/blob/main/notebooks/02_evaluation.ipynb)

| Exercise | Auto-check |
|---|---|
| 1. Build MMLU prompts | 0-shot and 5-shot structure, no leaked answers |
| 2. Log-likelihood scoring | matches reference scoring on at least 95% of prompts |
| 3. Calibrate the judge | held-out agreement at least 80% and at most 10% hallucinations accepted |
| 4. Elo | 5 unit tests (expected score, update, zero-sum) |
| 5. MLflow tracking | all MMLU and Lumora runs logged with params and metrics |
| Pro bonus | cross-check with `lm-evaluation-harness` |

Output: `baselines.json` plus an MLflow database. Module 3 compares fine-tuned models against these.
</div>

<div class="lab-complete" data-id="m02"></div>

## How this maps to NVIDIA NeMo Evaluator

| Lab | NeMo Evaluator |
|---|---|
| MMLU loop / `lm-evaluation-harness` | Academic benchmark jobs (MMLU, GSM8K, ...) submitted via API |
| `k_shot` | `num_fewshot` in the evaluation config |
| Calibrated judge | LLM-as-a-judge evaluation with custom prompts and metrics |
| Lumora QA benchmark | Custom dataset evaluation |
| MLflow | Results store and MLflow integration |

## Common pitfalls checklist

- [ ] Trusting fluent answers without a systematic metric
- [ ] Using MMLU (or any public benchmark) as evidence of *domain* knowledge
- [ ] Yes/no questions in your benchmark (50% by guessing)
- [ ] Trusting an LLM judge without measuring it against human labels
- [ ] Calibrating the judge on invented examples instead of real model outputs
- [ ] Tuning and reporting on the same split
- [ ] Pairwise judging in one order only (position bias)
- [ ] Declaring a winner from a few points' difference on a small test set
- [ ] Not recording prompts, model revisions and thresholds

## Glossary

| Term | Meaning |
|---|---|
| **Benchmark** | A fixed, public test set for comparing models |
| **Log-likelihood scoring** | Picking the option to which the model assigns the highest probability |
| [**In-context learning**](https://en.wikipedia.org/wiki/Prompt_engineering) | Improving behaviour with examples in the prompt, without weight updates |
| **Closed-book / open-book** | Answering from memory vs with the source document in the prompt |
| **Calibration** | Aligning a judge's decisions with human labels, e.g. by choosing a threshold |
| [**False-positive rate**](https://en.wikipedia.org/wiki/Confusion_matrix) | Share of truly incorrect answers that the judge accepts |
| **Position bias** | A judge's preference for an answer because of where it appears |
| [**Elo**](https://en.wikipedia.org/wiki/Elo_rating_system) | Rating system that updates scores from pairwise results |
| [**Confidence interval**](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval) | Range likely to contain the true value given sampling noise |
| **Contamination** | Test items present in training data |

## Further reading

**Papers**

- Hendrycks et al. (2021), [*Measuring Massive Multitask Language Understanding*](https://arxiv.org/abs/2009.03300)
- Brown et al. (2020), [*Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165)
- Zheng et al. (2023), [*Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*](https://arxiv.org/abs/2306.05685)
- Liang et al. (2022), [*Holistic Evaluation of Language Models (HELM)*](https://arxiv.org/abs/2211.09110)
- Chiang et al. (2024), [*Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference*](https://arxiv.org/abs/2403.04132)
- EleutherAI, [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) (GitHub)

**Concepts and tools**

- [MMLU dataset card (Hugging Face)](https://huggingface.co/datasets/cais/mmlu)
- [F-score / F1 (Wikipedia)](https://en.wikipedia.org/wiki/F-score)
- [Confusion matrix (Wikipedia)](https://en.wikipedia.org/wiki/Confusion_matrix)
- [Cohen's kappa (Wikipedia)](https://en.wikipedia.org/wiki/Cohen%27s_kappa)
- [Binomial proportion confidence interval (Wikipedia)](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval)
- [Bootstrapping in statistics (Wikipedia)](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))
- [Elo rating system (Wikipedia)](https://en.wikipedia.org/wiki/Elo_rating_system)
- [Bradley-Terry model (Wikipedia)](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model)
- [Goodhart's law (Wikipedia)](https://en.wikipedia.org/wiki/Goodhart%27s_law)
- [In-context learning and prompt engineering (Wikipedia)](https://en.wikipedia.org/wiki/Prompt_engineering)
- [MLflow Tracking documentation](https://mlflow.org/docs/latest/tracking.html)

## Knowledge check

<script type="application/json" class="quiz" data-id="m02">
[
  {"q": "Why do benchmark harnesses score MMLU with next-token log-probabilities instead of generating an answer?",
   "options": ["It is more creative", "One forward pass, no answer parsing, and deterministic results", "Generation is not possible for small models", "It makes the model smarter"],
   "answer": 1, "why": "Comparing P(' A')...P(' D') avoids fragile regex parsing and sampling noise."},
  {"q": "A 1.5B and a 0.5B model both score near 0% closed-book on Lumora questions but about 100% open-book. What does this tell you?",
   "options": ["The benchmark is broken", "Bigger models always know more", "The models lack the knowledge, not the ability to use it once supplied", "The judge is too strict"],
   "answer": 2, "why": "Scale doesn't create facts the model never saw. Supplying them, in context or via training, does."},
  {"q": "Calibration barely changed the judge's agreement (86% to 84%) but cut accepted hallucinations from 21% to 3%. Why does the second number matter more?",
   "options": ["It doesn't", "Accepted hallucinations inflate a model's score and hide exactly the errors you're trying to measure", "False negatives are impossible", "Agreement is always 50%"],
   "answer": 1, "why": "A lenient judge makes bad models look good. The false-positive rate is the dangerous error."},
  {"q": "Your judge agrees with humans 95% on the calibration split and 84% on the held-out split. What is going on?",
   "options": ["The held-out labels are wrong", "The threshold slightly overfits the calibration split, so report the held-out number", "The judge got worse while running", "Agreement can't be compared across splits"],
   "answer": 1, "why": "Any parameter tuned on data looks best on that data. The held-out score is the honest estimate."},
  {"q": "Why is each pairwise comparison judged twice with the answers swapped?",
   "options": ["To double the dataset", "LLM judges tend to prefer one position; averaging both orders cancels that bias", "Elo requires an even number of games", "To check the answers are different"],
   "answer": 1, "why": "Position bias is one of the best-documented LLM-judge biases."},
  {"q": "Two systems have Elo 1100 and 1000. Roughly how often should the first one win a comparison?",
   "options": ["50%", "64%", "91%", "100%"],
   "answer": 1, "why": "E = 1 / (1 + 10^(-100/400)), about 0.64."},
  {"q": "Model A scores 57% and model B 50% on your 30-question benchmark. What can you conclude?",
   "options": ["A is clearly better", "Not much: the 95% confidence interval is about ±18 points at this size", "B is better", "The benchmark is contaminated"],
   "answer": 1, "why": "Small test sets are noisy. You'd need hundreds of questions to trust a 7-point gap."}
]
</script>
