"""NLI hallucination detector (no training).

A pretrained NLI model checks whether the evidence (`source`) entails the
response. Low entailment / high contradiction -> hallucinated.

    python nli_detector.py --limit 50          # quick sanity check on 50 pairs/split
    python nli_detector.py                      # full run: val + 3 test splits
    python nli_detector.py --eval-only          # re-pick threshold & metrics from cached probs

How each task becomes (premise, hypothesis):
  qa             premise = source, hypothesis = query + " " + response
                 (short answers like "Arthur's Magazine" aren't claims on their own)
  dialogue       premise = source, hypothesis = response
  summarization  premise = source split into overlapping token chunks,
                 hypothesis = each sentence of the summary.
                 sentence entailment = max over chunks; summary entailment = min over
                 sentences (one unsupported sentence is enough to be hallucinated).

Two candidate scores per row: 1 - P(entailment) and P(contradiction).
The score type AND the threshold are both picked on `val` only, then frozen.

Outputs (results/nli/):
  raw_<split>.csv      per-row entailment / contradiction probabilities (cache)
  preds_<split>.csv    id, score, pred      (format the team agreed on)
  metrics.csv          accuracy / F1 / AUROC per test split, plus length-rule baseline
  config.json          model, chosen score type, threshold
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data_utils import TEST_SPLITS, load_split

DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
OUT_DIR = Path(__file__).parent / "results" / "nli"
EVAL_SPLITS = ["val"] + TEST_SPLITS
SCORE_TYPES = ["one_minus_entail", "contradiction"]


# ---------------------------------------------------------------- building pairs

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def split_sentences(text):
    sents = [s.strip() for s in _SENT_RE.split(text.strip()) if s.strip()]
    return sents or [text.strip()]


def chunk_premise(tokenizer, premise, hyp_len, max_len, stride):
    """Split premise into overlapping chunks that fit next to the hypothesis."""
    ids = tokenizer(premise, add_special_tokens=False)["input_ids"]
    room = max_len - hyp_len - 4  # 4 = special tokens for a sentence pair
    room = max(room, 64)
    if len(ids) <= room:
        return [premise]
    step = max(room - stride, 1)
    chunks = []
    for start in range(0, len(ids), step):
        chunks.append(tokenizer.decode(ids[start:start + room]))
        if start + room >= len(ids):
            break
    return chunks


def build_jobs(df, tokenizer, max_len, stride):
    """One job = one (premise chunk, hypothesis) pair to run through the model.
    Returns jobs as (row_idx, sent_idx, premise, hypothesis)."""
    jobs = []
    for i, row in enumerate(df.itertuples(index=False)):
        if row.task == "qa":
            hyps = [f"{row.query} {row.response}".strip()]
        elif row.task == "summarization":
            hyps = split_sentences(row.response)
        else:
            hyps = [row.response.strip()]
        for s, hyp in enumerate(hyps):
            hyp_len = len(tokenizer(hyp, add_special_tokens=False)["input_ids"])
            for chunk in chunk_premise(tokenizer, row.source, hyp_len, max_len, stride):
                jobs.append((i, s, chunk, hyp))
    return jobs


# ---------------------------------------------------------------- running the model

def load_model(name, device):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(device).eval()
    labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    ent = next(i for l, i in labels.items() if l.startswith("entail"))
    con = next(i for l, i in labels.items() if l.startswith("contra"))
    return tok, model, ent, con, torch


def run_jobs(jobs, tok, model, ent_idx, con_idx, torch, device, max_len, batch_size):
    """Return arrays of P(entail), P(contradiction) aligned with jobs."""
    # sort by length so each batch has similar padding -> much faster on CPU
    order = sorted(range(len(jobs)), key=lambda j: len(jobs[j][2]) + len(jobs[j][3]))
    ent = np.zeros(len(jobs))
    con = np.zeros(len(jobs))
    t0 = time.time()
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        enc = tok([jobs[j][2] for j in idx], [jobs[j][3] for j in idx],
                  truncation="only_first", max_length=max_len,
                  padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            probs = model(**enc).logits.softmax(-1).cpu().numpy()
        ent[idx] = probs[:, ent_idx]
        con[idx] = probs[:, con_idx]
        done = b + len(idx)
        if (b // batch_size) % 20 == 0 or done == len(order):
            rate = done / (time.time() - t0)
            print(f"    {done}/{len(order)} pairs  ({rate:.1f}/s, "
                  f"~{(len(order) - done) / rate / 60:.1f} min left)", flush=True)
    return ent, con


def aggregate(df, jobs, ent, con):
    """chunks -> sentence: max entail (best supporting chunk), max contradiction.
    sentences -> row: min entail (weakest sentence), max contradiction."""
    j = pd.DataFrame({"row": [x[0] for x in jobs], "sent": [x[1] for x in jobs],
                      "ent": ent, "con": con})
    per_sent = j.groupby(["row", "sent"]).agg(ent=("ent", "max"), con=("con", "max"))
    per_row = per_sent.groupby("row").agg(ent=("ent", "min"), con=("con", "max"))
    per_row = per_row.reindex(range(len(df)))
    return pd.DataFrame({
        "id": df["id"].values,
        "label": df["label"].values,
        "n_words": df["response"].str.split().str.len().values,
        "p_entail": per_row["ent"].values,
        "p_contradiction": per_row["con"].values,
        "n_pairs": j.groupby("row").size().reindex(range(len(df))).values,
    })


# ---------------------------------------------------------------- evaluation

def scores_of(raw, score_type):
    return 1 - raw["p_entail"] if score_type == "one_minus_entail" else raw["p_contradiction"]


def best_threshold(labels, scores):
    """Threshold that maximises accuracy on val (data is 50/50, so this is fair)."""
    cands = np.unique(scores)
    mids = np.concatenate([[cands[0] - 1e-9], (cands[:-1] + cands[1:]) / 2, [cands[-1] + 1e-9]])
    accs = [accuracy_score(labels, scores > t) for t in mids]
    return float(mids[int(np.argmax(accs))])


def metrics(labels, scores, preds):
    return {"accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, zero_division=0),
            "auroc": roc_auc_score(labels, scores)}


def evaluate(raws, model_name):
    val = raws["val"]
    # 1) choose the score type on val by AUROC (threshold-free), 2) threshold on val
    val_auc = {s: roc_auc_score(val["label"], scores_of(val, s)) for s in SCORE_TYPES}
    score_type = max(val_auc, key=val_auc.get)
    thr = best_threshold(val["label"].values, scores_of(val, score_type).values)
    print(f"\nval AUROC per score type: { {k: round(v, 4) for k, v in val_auc.items()} }")
    print(f"chosen score = {score_type}, threshold = {thr:.4f} (frozen for test)")

    rows = []
    for split, raw in raws.items():
        s = scores_of(raw, score_type).values
        p = (s > thr).astype(int)
        pd.DataFrame({"id": raw["id"], "score": s, "pred": p}).to_csv(
            OUT_DIR / f"preds_{split}.csv", index=False)
        rows.append({"method": "NLI", "split": split, **metrics(raw["label"], s, p)})
        # reference row: "more than 4 words -> hallucinated" (see data/README.md)
        lp = (raw["n_words"] > 4).astype(int)
        rows.append({"method": "length rule (>4 words)", "split": split,
                     **metrics(raw["label"], raw["n_words"], lp)})

    res = pd.DataFrame(rows)
    res.to_csv(OUT_DIR / "metrics.csv", index=False)
    (OUT_DIR / "config.json").write_text(json.dumps({
        "model": model_name, "score_type": score_type, "threshold": thr,
        "val_auroc_by_score_type": val_auc,
        "n_rows": {k: len(v) for k, v in raws.items()},
    }, indent=2))
    print("\n" + res.pivot(index="method", columns="split").round(3).to_string())
    return res


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--splits", nargs="+", default=EVAL_SPLITS)
    ap.add_argument("--limit", type=int, help="only the first N pairs of each split (sanity check)")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=128, help="token overlap between chunks")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-only", action="store_true", help="skip the model, use cached raw_*.csv")
    args = ap.parse_args()

    global OUT_DIR
    if args.limit:
        OUT_DIR = OUT_DIR / f"sample_{args.limit}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raws = {}
    if args.eval_only:
        for split in args.splits:
            raws[split] = pd.read_csv(OUT_DIR / f"raw_{split}.csv")
    else:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"loading {args.model} on {device}")
        tok, model, ent_idx, con_idx, torch = load_model(args.model, device)
        for split in args.splits:
            df = load_split(split)
            if args.limit:  # keep whole pairs so the subset stays 50/50
                keep = df["pair_id"].drop_duplicates().iloc[:args.limit]
                df = df[df["pair_id"].isin(keep)].reset_index(drop=True)
            jobs = build_jobs(df, tok, args.max_len, args.stride)
            print(f"\n[{split}] {len(df)} rows -> {len(jobs)} model pairs")
            ent, con = run_jobs(jobs, tok, model, ent_idx, con_idx, torch, device,
                                args.max_len, args.batch_size)
            raws[split] = aggregate(df, jobs, ent, con)
            raws[split].to_csv(OUT_DIR / f"raw_{split}.csv", index=False)

    if "val" in raws:
        evaluate(raws, args.model)
    else:
        print("val not in --splits; skipping threshold selection / metrics")


if __name__ == "__main__":
    main()
