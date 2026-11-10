"""Fine-tuned transformer hallucination classifier (trained on labelled QA data).

A pretrained encoder (DistilBERT by default) gets a classification head and is
fine-tuned on `train` to predict label 1 = hallucinated / 0 = correct.

    python finetune_classifier.py --limit 32 --epochs 1      # quick smoke test (CPU ok)
    python finetune_classifier.py                             # full run (use a GPU, e.g. Colab)
    python finetune_classifier.py --with-source               # also feed the evidence text

Input: a sentence pair (first, second) = (query, response), as the shared protocol
says. With --with-source, first = source + " " + query, so the model can also
compare the response with the evidence (long articles get truncated; the
response is never cut first because the longer side is truncated first).

Model selection uses `val` only: every (learning rate, epoch) checkpoint is scored
by val AUROC, the best one is kept, and the decision threshold is then chosen on
val (max accuracy) exactly like the NLI detector. Dialogue and summarization are
never trained on; they are only scored at the end.

Outputs (results/finetune[_src]/):
  preds_<split>.csv   id, score, pred     (score = P(hallucinated))
  metrics.csv         accuracy / F1 / AUROC per split
  history.csv         val loss / AUROC after every epoch of every run
  config.json         model, chosen lr / epoch, threshold
"""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from data_utils import TEST_SPLITS, load_split
from nli_detector import best_threshold, metrics

DEFAULT_MODEL = "distilbert-base-uncased"
EVAL_SPLITS = ["val"] + TEST_SPLITS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def take_pairs(df, n, seed=0):
    """n random pairs (both rows of each), so a subset stays 50/50."""
    keep = df["pair_id"].drop_duplicates().sample(frac=1, random_state=seed).iloc[:n]
    return df[df["pair_id"].isin(keep)].reset_index(drop=True)


def make_examples(df, with_source):
    first = df["query"].fillna("")
    if with_source:
        first = df["source"].fillna("") + " " + first
    return list(zip(first.str.strip(), df["response"].fillna("").str.strip(), df["label"]))


def make_loader(examples, tok, max_len, batch_size, shuffle):
    def collate(batch):
        enc = tok([b[0] for b in batch], [b[1] for b in batch], truncation="longest_first",
                  max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([b[2] for b in batch])
        return enc
    return DataLoader(examples, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, loader, device):
    """Return (P(hallucinated) per row, mean cross-entropy loss)."""
    model.eval()
    probs, losses = [], []
    for enc in loader:
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            out = model(**enc)
        losses.append(out.loss.float().item() * len(enc["labels"]))
        probs.append(out.logits.float().softmax(-1)[:, 1].cpu().numpy())
    probs = np.concatenate(probs)
    return probs, sum(losses) / len(probs)


def train_one_lr(lr, args, tok, train_loader, val_loader, val_labels, device, history):
    """Fine-tune a fresh model at this lr; return (best val AUROC, epoch, CPU state_dict)."""
    set_seed(args.seed)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = len(train_loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    best = (-1.0, None, None)
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, enc in enumerate(train_loader, 1):
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = model(**enc).loss
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if step % 100 == 0 or step == len(train_loader):
                print(f"    lr={lr:g} epoch {epoch} step {step}/{len(train_loader)} "
                      f"train loss {running / step:.4f} ({time.time() - t0:.0f}s)", flush=True)

        val_probs, val_loss = predict(model, val_loader, device)
        val_auc = roc_auc_score(val_labels, val_probs)
        history.append({"lr": lr, "epoch": epoch, "train_loss": running / len(train_loader),
                        "val_loss": val_loss, "val_auroc": val_auc})
        print(f"  lr={lr:g} epoch {epoch}: val loss {val_loss:.4f}, val AUROC {val_auc:.4f}", flush=True)
        if val_auc > best[0]:  # early stopping: keep the best epoch, judged on val only
            best = (val_auc, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--with-source", action="store_true", help="prepend the evidence text to the input")
    ap.add_argument("--lrs", type=float, nargs="+", default=[2e-5, 5e-5], help="learning rates to try")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, help="default 256, or 512 with --with-source")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="only N pairs per split (smoke test)")
    ap.add_argument("--out", help="output folder (default results/finetune or results/finetune_src)")
    args = ap.parse_args()
    max_len = args.max_len or (512 if args.with_source else 256)

    name = "finetune_src" if args.with_source else "finetune"
    out = Path(args.out) if args.out else Path(__file__).parent / "results" / name
    if args.limit:
        out = out / f"sample_{args.limit}"
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"model {args.model} on {device}, max_len {max_len}, with_source={args.with_source}")
    tok = AutoTokenizer.from_pretrained(args.model)

    splits = {s: load_split(s) for s in ["train"] + EVAL_SPLITS}
    if args.limit:
        splits = {s: take_pairs(df, args.limit) for s, df in splits.items()}
    examples = {s: make_examples(df, args.with_source) for s, df in splits.items()}
    train_loader = make_loader(examples["train"], tok, max_len, args.batch_size, shuffle=True)
    eval_loaders = {s: make_loader(examples[s], tok, max_len, args.batch_size * 2, shuffle=False)
                    for s in EVAL_SPLITS}
    val_labels = splits["val"]["label"].values

    # 1) hyperparameter search: every lr x epoch, judged by val AUROC only
    history, best = [], (-1.0, None, None, None)
    for lr in args.lrs:
        print(f"\n=== training with lr={lr:g} ({len(splits['train'])} train rows) ===")
        auc, epoch, state = train_one_lr(lr, args, tok, train_loader, eval_loaders["val"],
                                         val_labels, device, history)
        if auc > best[0]:
            best = (auc, lr, epoch, state)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    best_auc, best_lr, best_epoch, best_state = best
    print(f"\nbest on val: lr={best_lr:g}, epoch {best_epoch}, val AUROC {best_auc:.4f}")

    # 2) reload the chosen checkpoint and score every split
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    model.load_state_dict(best_state)
    model.to(device)
    scores = {s: predict(model, eval_loaders[s], device)[0] for s in EVAL_SPLITS}

    # 3) threshold on val only, then frozen for the test splits
    thr = best_threshold(val_labels, scores["val"])
    print(f"threshold (max val accuracy) = {thr:.4f}")
    rows = []
    for s in EVAL_SPLITS:
        pred = (scores[s] > thr).astype(int)
        pd.DataFrame({"id": splits[s]["id"], "score": scores[s], "pred": pred}).to_csv(
            out / f"preds_{s}.csv", index=False)
        rows.append({"method": f"fine-tuned {args.model}" + (" + source" if args.with_source else ""),
                     "split": s, **metrics(splits[s]["label"], scores[s], pred)})
    res = pd.DataFrame(rows)
    res.to_csv(out / "metrics.csv", index=False)
    (out / "config.json").write_text(json.dumps({
        "model": args.model, "with_source": args.with_source, "max_len": max_len,
        "lrs_tried": args.lrs, "epochs_max": args.epochs, "batch_size": args.batch_size,
        "seed": args.seed, "chosen_lr": best_lr, "chosen_epoch": best_epoch,
        "val_auroc": best_auc, "threshold": thr,
        "n_rows": {s: len(df) for s, df in splits.items()},
    }, indent=2))
    print("\n" + res.set_index("split").drop(columns="method").round(3).to_string())


if __name__ == "__main__":
    main()
