"""TF-IDF + logistic regression hallucination detector (trained on labelled QA data).

The simplest trained detector in the project: bag-of-n-grams features and a
linear classifier. It never looks at the evidence (`source`), so it cannot
check facts; whatever it scores well on, it does from the wording of the
response alone. That makes it the reference point for the other three methods.

    python tfidf_baseline.py --limit 200       # quick smoke test on 200 pairs/split
    python tfidf_baseline.py                   # full run (CPU, well under a minute)
    python tfidf_baseline.py --input response  # fix the input instead of choosing it on val

Model
-----
Each text d becomes a sparse vector x(d) with one entry per n-gram t:

    x_t(d) = tf(t, d) * idf(t),    idf(t) = ln((1 + N) / (1 + df(t))) + 1

where tf is the count of t in d, df(t) the number of *training* texts that
contain t and N the number of training texts; x(d) is then scaled to unit L2
norm. Logistic regression models P(hallucinated | d) = sigmoid(w . x(d) + b)
and is fitted by minimising the L2-regularised cross-entropy

    (1/2) ||w||^2 + C * sum_i log(1 + exp(-y_i (w . x_i + b))),   y_i in {-1, +1}

so small C means strong regularisation. The score of a row is P(hallucinated).

Protocol (same as the other detectors, see data/README.md)
----------------------------------------------------------
* The vocabulary, the idf weights and the classifier are fitted on `train` only.
  `val` and the test splits are only ever passed through `transform`.
* Hyperparameters (input text, n-gram range, C) are chosen by val AUROC, then the
  decision threshold is chosen on val (max accuracy) with the same
  `best_threshold` the NLI and fine-tuned detectors use. Both are then frozen.
* Dialogue and summarization are never trained or tuned on.

Outputs (results/tfidf/):
  preds_<split>.csv   id, score, pred     (score = P(hallucinated))
  metrics.csv         accuracy / F1 / AUROC per split
  grid.csv            train / val AUROC of every configuration tried
  top_features.csv    the n-grams with the largest weights in each direction
  config.json         chosen input, n-gram range, C, threshold
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from data_utils import TEST_SPLITS, load_split
from nli_detector import best_threshold, metrics

METHOD = "TF-IDF + logistic regression"
EVAL_SPLITS = ["val"] + TEST_SPLITS
INPUTS = ["response", "query_response"]
NGRAMS = [(1, 1), (1, 2)]
CS = [0.01, 0.1, 1.0, 10.0, 100.0]
MIN_DF = 2          # ignore n-grams seen in fewer than 2 training texts
MAX_ITER = 5000


def take_pairs(df, n):
    """First n pairs (both rows of each), so a subset stays 50/50."""
    keep = df["pair_id"].drop_duplicates().iloc[:n]
    return df[df["pair_id"].isin(keep)].reset_index(drop=True)


def make_text(df, input_kind):
    """The string that gets vectorised. `source` is deliberately never used."""
    response = df["response"].fillna("").astype(str)
    if input_kind == "response":
        return response.str.strip()
    query = df["query"].fillna("").astype(str)  # empty for summarization
    return (query + " " + response).str.strip()


def fit(train_text, train_labels, ngram_range, C):
    """Fit the vectorizer and the classifier. Call this with `train` only."""
    vec = TfidfVectorizer(ngram_range=ngram_range, min_df=MIN_DF)
    X = vec.fit_transform(train_text)
    clf = LogisticRegression(C=C, max_iter=MAX_ITER).fit(X, train_labels)
    return vec, clf


def score(vec, clf, text):
    """P(hallucinated) for each text; only `transform`, never `fit`."""
    return clf.predict_proba(vec.transform(text))[:, 1]


def search(splits, inputs, ngrams, cs):
    """Every (input, n-gram range, C), judged by val AUROC only.
    Returns (grid rows, best config). Ties keep the earlier = simpler config."""
    y_train, y_val = splits["train"]["label"].values, splits["val"]["label"].values
    rows, best = [], None
    for input_kind in inputs:
        train_text, val_text = make_text(splits["train"], input_kind), make_text(splits["val"], input_kind)
        for ngram_range in ngrams:
            for C in cs:
                vec, clf = fit(train_text, y_train, ngram_range, C)
                row = {"input": input_kind, "ngram_max": ngram_range[1], "C": C,
                       "n_features": len(vec.vocabulary_),
                       "train_auroc": roc_auc_score(y_train, score(vec, clf, train_text)),
                       "val_auroc": roc_auc_score(y_val, score(vec, clf, val_text))}
                rows.append(row)
                print(f"  input={input_kind:<14} ngrams=1-{ngram_range[1]} C={C:<6g} "
                      f"features={row['n_features']:>6}  train AUROC {row['train_auroc']:.4f}  "
                      f"val AUROC {row['val_auroc']:.4f}", flush=True)
                if best is None or row["val_auroc"] > best["val_auroc"]:
                    best = {**row, "ngram_range": ngram_range}
    return rows, best


def top_features(vec, clf, k):
    """The k most 'hallucinated' and k most 'correct' n-grams by weight."""
    w = pd.DataFrame({"feature": vec.get_feature_names_out(), "weight": clf.coef_[0], "idf": vec.idf_})
    w = w.sort_values("weight", ascending=False)
    top = pd.concat([w.head(k).assign(direction="hallucinated"),
                     w.tail(k).iloc[::-1].assign(direction="correct")])
    return top[["direction", "feature", "weight", "idf"]].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=INPUTS, help="fix the input text (default: choose on val)")
    ap.add_argument("--ngram-max", type=int, choices=[1, 2], help="fix the n-gram range to 1..N (default: choose on val)")
    ap.add_argument("--C", type=float, nargs="+", default=CS, help="inverse regularisation strengths to try")
    ap.add_argument("--top-k", type=int, default=30, help="features per direction in top_features.csv")
    ap.add_argument("--limit", type=int, help="only the first N pairs of each split (smoke test)")
    ap.add_argument("--out", help="output folder (default results/tfidf)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else Path(__file__).parent / "results" / "tfidf"
    if args.limit:
        out = out / f"sample_{args.limit}"
    out.mkdir(parents=True, exist_ok=True)

    splits = {s: load_split(s) for s in ["train"] + EVAL_SPLITS}
    if args.limit:
        splits = {s: take_pairs(df, args.limit) for s, df in splits.items()}

    # 1) hyperparameter search: fitted on train, judged by val AUROC only
    inputs = [args.input] if args.input else INPUTS
    ngrams = [(1, args.ngram_max)] if args.ngram_max else NGRAMS
    print(f"searching {len(inputs) * len(ngrams) * len(args.C)} configurations "
          f"({len(splits['train'])} train rows, {len(splits['val'])} val rows)")
    grid, best = search(splits, inputs, ngrams, args.C)
    pd.DataFrame(grid).to_csv(out / "grid.csv", index=False)
    print(f"\nbest on val: input={best['input']}, n-grams 1-{best['ngram_max']}, C={best['C']:g}, "
          f"val AUROC {best['val_auroc']:.4f}")

    # 2) refit the chosen configuration on train and score every split
    vec, clf = fit(make_text(splits["train"], best["input"]), splits["train"]["label"].values,
                   best["ngram_range"], best["C"])
    scores = {s: score(vec, clf, make_text(splits[s], best["input"])) for s in EVAL_SPLITS}

    # 3) threshold on val only, then frozen for the test splits
    thr = best_threshold(splits["val"]["label"].values, scores["val"])
    print(f"threshold (max val accuracy) = {thr:.4f}")
    rows = []
    for s in EVAL_SPLITS:
        pred = (scores[s] > thr).astype(int)
        pd.DataFrame({"id": splits[s]["id"], "score": scores[s], "pred": pred}).to_csv(
            out / f"preds_{s}.csv", index=False)
        rows.append({"method": METHOD, "split": s, **metrics(splits[s]["label"], scores[s], pred)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "metrics.csv", index=False)
    top_features(vec, clf, args.top_k).to_csv(out / "top_features.csv", index=False)
    (out / "config.json").write_text(json.dumps({
        "method": METHOD, "input": best["input"], "ngram_range": list(best["ngram_range"]),
        "C": best["C"], "min_df": MIN_DF, "n_features": best["n_features"],
        "inputs_tried": inputs, "ngram_ranges_tried": [list(n) for n in ngrams], "Cs_tried": args.C,
        "val_auroc": best["val_auroc"], "threshold": thr,
        "intercept": float(clf.intercept_[0]), "sklearn_version": sklearn.__version__,
        "n_rows": {s: len(df) for s, df in splits.items()},
    }, indent=2))
    print("\n" + res.set_index("split").drop(columns="method").round(3).to_string())


if __name__ == "__main__":
    main()
