"""
Build the shared train/val/test splits for all four hallucination detectors.

Design
------
* In-domain task = QA. The trained detectors (TF-IDF, DistilBERT) train on QA only.
* Out-of-domain tasks = dialogue and summarization. Nobody trains on these; they test
  whether a detector still works on a *different type* of hallucination.
* Every HaluEval item has one correct and one hallucinated response. Both rows of a pair
  always land in the same split (otherwise the model sees the question/knowledge in
  train and its twin in test -> leakage). It also makes every split exactly 50/50.
* Fixed seed, so re-running gives byte-identical files. Check manifest.json md5s.

Output (data/splits/), one row per response:
    train.jsonl            QA, used to fit TF-IDF and DistilBERT
    val.jsonl              QA, threshold / hyperparameter tuning for ALL methods
    test_qa.jsonl          QA, in-domain test
    test_dialogue.jsonl    dialogue, out-of-domain test
    test_summarization.jsonl  summarization, out-of-domain test

Columns: id, pair_id, task, split, source, query, response, label (1 = hallucinated)

Run:  python make_splits.py
"""

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/splits")
RAW_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/{}_data.json"

# QA pairs (10,000 total): 80 / 10 / 10
QA_VAL_PAIRS = 1000
QA_TEST_PAIRS = 1000
# Out-of-domain test size, in pairs. Kept equal to the QA test so results are comparable.
OOD_TEST_PAIRS = 1000


def load_raw(task):
    path = RAW_DIR / f"{task}_data.json"
    if not path.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        print(f"downloading {path} ...")
        urllib.request.urlretrieve(RAW_URL.format(task), path)
    return pd.read_json(path, lines=True)


def to_long(df, task, source_col, query_col, right_col, hall_col):
    """One HaluEval item -> two rows (correct response label 0, hallucinated label 1)."""
    rows = []
    for i, r in df.iterrows():
        pair_id = f"{task}_{i:05d}"  # i = line number in the original HaluEval file
        source = r[source_col]
        query = r[query_col] if query_col else ""
        for kind, col, label in [("right", right_col, 0), ("hall", hall_col, 1)]:
            rows.append({
                "id": f"{pair_id}_{kind}",
                "pair_id": pair_id,
                "task": task,
                "source": source,
                "query": query,
                "response": r[col],
                "label": label,
            })
    return pd.DataFrame(rows)


def save(df, name, rng):
    # shuffle rows so the right/hallucinated twin isn't always adjacent
    df = df.sample(frac=1, random_state=rng.integers(2**31)).reset_index(drop=True)
    df.insert(3, "split", name)
    path = OUT_DIR / f"{name}.jsonl"
    df.to_json(path, orient="records", lines=True, force_ascii=False)
    df.to_csv(OUT_DIR / f"{name}.csv", index=False)
    return {
        "file": path.name,
        "task": df["task"].iloc[0],
        "rows": len(df),
        "pairs": df["pair_id"].nunique(),
        "hallucinated_frac": round(df["label"].mean(), 3),
        "md5": hashlib.md5(path.read_bytes()).hexdigest(),
    }


def main():
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- QA: train / val / test_qa ---
    qa = load_raw("qa")
    order = rng.permutation(len(qa))
    test_idx = order[:QA_TEST_PAIRS]
    val_idx = order[QA_TEST_PAIRS:QA_TEST_PAIRS + QA_VAL_PAIRS]
    train_idx = order[QA_TEST_PAIRS + QA_VAL_PAIRS:]
    qa_long = to_long(qa, "qa", "knowledge", "question", "right_answer", "hallucinated_answer")
    pair_of = lambda idx: {f"qa_{i:05d}" for i in idx}

    # --- Dialogue: out-of-domain test ---
    dia = load_raw("dialogue")
    dia = dia.iloc[np.sort(rng.choice(len(dia), OOD_TEST_PAIRS, replace=False))]
    dia_long = to_long(dia, "dialogue", "knowledge", "dialogue_history",
                       "right_response", "hallucinated_response")

    # --- Summarization: out-of-domain test (drop the 2 duplicated documents first) ---
    summ = load_raw("summarization")
    summ = summ[~summ["document"].duplicated()]
    summ = summ.iloc[np.sort(rng.choice(len(summ), OOD_TEST_PAIRS, replace=False))]
    summ_long = to_long(summ, "summarization", "document", None,
                        "right_summary", "hallucinated_summary")

    manifest = {"seed": SEED, "splits": {}}
    for name, df in [
        ("train", qa_long[qa_long["pair_id"].isin(pair_of(train_idx))]),
        ("val", qa_long[qa_long["pair_id"].isin(pair_of(val_idx))]),
        ("test_qa", qa_long[qa_long["pair_id"].isin(pair_of(test_idx))]),
        ("test_dialogue", dia_long),
        ("test_summarization", summ_long),
    ]:
        manifest["splits"][name] = save(df, name, rng)

    # sanity: no QA pair appears in more than one split
    assert not (pair_of(train_idx) & pair_of(val_idx)) and not (pair_of(train_idx) & pair_of(test_idx)) \
        and not (pair_of(val_idx) & pair_of(test_idx))

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(pd.DataFrame(manifest["splits"]).T.to_string())


if __name__ == "__main__":
    main()
