# Data

All four detectors in this project are trained and tested on the **same files** in `data/splits/`. Don't re-split, re-shuffle or sub-sample them. Keeping the data identical means any difference in results comes from the method, not from different data.

## What the data is

We use [HaluEval](https://github.com/RUCAIBox/HaluEval), a benchmark of AI responses where each example comes with one **correct** response and one **hallucinated** (made-up) response. Example from the QA file:

> **Knowledge:** Arthur's Magazine (1844–1846) was an American literary periodical… First for Women is a woman's magazine published by Bauer Media Group…
> **Question:** Which magazine was started first, Arthur's Magazine or First for Women?
> **Correct answer:** Arthur's Magazine
> **Hallucinated answer:** First for Women was started first.

HaluEval has three tasks we use (10,000 examples each):

| Task | The "query" is… | The "source" is… | The "response" is… |
|---|---|---|---|
| QA | a question | Wikipedia knowledge | an answer |
| Dialogue | the conversation so far | knowledge-graph facts | the chatbot's next reply |
| Summarization | *(empty)* | a news article | a summary of it |

## The splits

Our research question is whether detectors trained on labelled data beat ones that need no training, **and whether they still win on a different type of hallucination.** So trained models learn from **QA only**, and dialogue and summarization serve as unseen hallucination types.

| File | Task | Rows | Pairs | Purpose |
|---|---|---|---|---|
| `train` | QA | 16,000 | 8,000 | Fit models (TF-IDF, DistilBERT only) |
| `val` | QA | 2,000 | 1,000 | Pick thresholds / hyperparameters (**everyone**) |
| `test_qa` | QA | 2,000 | 1,000 | Final score: same type as training |
| `test_dialogue` | Dialogue | 2,000 | 1,000 | Final score: **new** hallucination type |
| `test_summarization` | Summarization | 2,000 | 1,000 | Final score: **new** hallucination type |

How it was made (see `make_splits.py`):
- Each HaluEval example becomes **2 rows**: the correct response (`label = 0`) and the hallucinated one (`label = 1`). Every split is exactly 50/50.
- Both rows of an example (same `pair_id`) are **always in the same split**. Otherwise a model could see a question during training and be tested on its twin.
- QA was shuffled and divided 80/10/10. For dialogue and summarization, 1,000 random examples each were taken (2 duplicated summarization articles removed first).
- The random seed is fixed (42), so running `python make_splits.py` again gives identical files. `manifest.json` lists each file's md5 checksum if you want to verify.

## Columns

| Column | Meaning |
|---|---|
| `id` | unique row id, e.g. `qa_03841_hall` |
| `pair_id` | the original HaluEval example; the correct and hallucinated rows share it |
| `task` | `qa`, `dialogue` or `summarization` |
| `split` | which file the row is in |
| `source` | the evidence the response *should* be based on (knowledge text or article) |
| `query` | the question / dialogue history (empty for summarization) |
| `response` | **the text you classify** |
| `label` | **1 = hallucinated, 0 = correct** |

Each split is saved twice with the same content: `.csv` (opens in Excel) and `.jsonl` (safer for code, since articles contain commas and newlines).

## How to load it

From the repo root:

```python
from data_utils import load_split

train = load_split("train")               # pandas DataFrame
val   = load_split("val")
tests = {name: load_split(name) for name in ["test_qa", "test_dialogue", "test_summarization"]}
```

Or without the helper: `pd.read_json("data/splits/train.jsonl", lines=True)` or `pd.read_csv("data/splits/train.csv", keep_default_na=False)`.

## How each person uses it

### Nick: TF-IDF + logistic regression
- Fit the vectorizer **and** the classifier on `train` only (never `fit` on val/test, only `transform`).
- Input text: `response` (optionally `query + " " + response`).
- Tune `C` / n-grams on `val`, then score all three test files.

```python
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
X_train = vec.fit_transform(train["response"])
clf = LogisticRegression(max_iter=1000).fit(X_train, train["label"])
probs = clf.predict_proba(vec.transform(tests["test_dialogue"]["response"]))[:, 1]
```

### CK: fine-tuned DistilBERT / DeBERTa
- Fine-tune on `train`, early-stop / choose epochs and learning rate on `val`.
- Input: `query` and `response` as a sentence pair (e.g. `tokenizer(query, response, truncation=True)`). You can add `source` too, but summarization articles are long, so keep truncation in mind.
- Don't train on dialogue or summarization at all; they're only for testing.

### Rohaan: NLI checker (RoBERTa/DeBERTa-MNLI, no training)
- Premise = `source`, hypothesis = the response. Hallucination score = 1 − P(entailment) (or P(contradiction)).
- For QA, short answers like "Arthur's Magazine" aren't full claims, so use `query + " " + response` as the hypothesis.
- Long summarization articles exceed the model's 512-token limit. Split the article into chunks and take the max entailment over chunks.
- Use `val` only to choose the decision threshold.

### Connor: self-consistency (SelfCheckGPT-style, no training)
- Both rows of a pair share the same question, so **sample the LLM once per `pair_id`** (not per row). That halves the generation cost.
- Prompt with `query` (QA/dialogue) or "Summarize: " + `source` (summarization). Generate N samples, then score each row's `response` by how much the samples agree with it.
- Use `val` only to choose the decision threshold.

## Rules so the comparison is fair

1. **Only `train` is used to fit models.** Never train or tune anything on a `test_*` file.
2. **Everyone picks their threshold on `val`.** Trained and training-free methods get the same tuning data.
3. **Report the same metrics on all three test files:** accuracy, F1 (hallucinated = positive class), and AUROC. Save your predictions as a CSV with `id`, `score`, `pred` so we can combine results.

```python
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
acc = accuracy_score(df["label"], preds)
f1  = f1_score(df["label"], preds)
auc = roc_auc_score(df["label"], scores)   # scores = probability of hallucinated
```

## ⚠️ Known shortcut: answer length

In QA, correct answers average **~2 words** and hallucinated answers **~11 words**. A rule with no ML at all, "more than 4 words → hallucinated", scores:

| test_qa | test_dialogue | test_summarization |
|---|---|---|
| **95%** | 54% | 50% (coin flip) |

So a trained model with ~95% on QA may simply have learned "long = hallucinated". Include this length rule as a reference row in the results table. The drop on dialogue/summarization is part of the answer to our research question.

## Rebuilding from scratch

```bash
python make_splits.py
```

Downloads the raw HaluEval files into `data/raw/` (not committed to git because they're large) and regenerates `data/splits/`. You only need this if we decide to change the split.
