"""SelfCheckGPT-style, training-free hallucination detector for the HaluEval project.

Run from the repository root:
    python SelfCheckGPT.py

Dependencies:
    pip install torch transformers sentencepiece scikit-learn pandas numpy tqdm

This uses a sampled-response + NLI implementation inspired by SelfCheckGPT,
not the official SelfCheckGPT Python package. No model is fine-tuned.
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

SPLITS = ("val", "test_qa", "test_dialogue", "test_summarization")
REQUIRED_COLUMNS = {"id", "pair_id", "task", "source", "query", "response", "label"}


def parse_args():
    parser = argparse.ArgumentParser(description="SelfCheckGPT-style HaluEval evaluation")
    parser.add_argument("--data-dir", default="data/splits", help="Directory with fixed JSONL splits")
    parser.add_argument("--output-dir", default="results/selfcheckgpt", help="Predictions and generation cache")
    parser.add_argument("--generator", default="google/flan-t5-base", help="Hugging Face seq2seq generator")
    parser.add_argument("--nli-model", default="facebook/bart-large-mnli", help="NLI classifier")
    parser.add_argument("--num-samples", type=int, default=5, help="Samples per unique pair")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8, help="NLI inference batch size")
    return parser.parse_args()


def get_device(choice):
    if choice != "auto":
        if choice == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if choice == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split(name, data_dir):
    path = data_dir / f"{name}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing split: {path}. Clone the repository and run from its root.")
    df = pd.read_json(path, lines=True)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if df["id"].duplicated().any():
        raise ValueError(f"{path}: duplicate row ids")
    if not df["label"].isin([0, 1]).all():
        raise ValueError(f"{path}: labels must be 0 or 1")
    if not df.groupby("pair_id").size().eq(2).all():
        raise ValueError(f"{path}: expected exactly two rows per pair_id")
    if not df.groupby("pair_id")["label"].agg(lambda x: set(x) == {0, 1}).all():
        raise ValueError(f"{path}: each pair must have one correct and one hallucinated row")
    return df.fillna("")


def validate_splits(frames):
    seen = {}
    for split, df in frames.items():
        for pair in df["pair_id"].astype(str).unique():
            if pair in seen:
                raise ValueError(f"pair_id {pair} occurs in both {seen[pair]} and {split}")
            seen[pair] = split


def build_prompt(row):
    task = str(row["task"]).lower()
    source = str(row["source"]).strip()
    query = str(row["query"]).strip()
    if task == "summarization":
        return f"Summarize the following article faithfully and concisely:\n{source}"
    if task == "qa":
        return f"Answer the following question accurately and concisely:\n{query}"
    if task == "dialogue":
        return f"Write a helpful, factual next response to this conversation:\n{query}"
    raise ValueError(f"Unexpected task {task!r}")


def split_sentences(text):
    """Simple dependency-free sentence split; short QA answers remain one unit."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", str(text).strip())
    return [s.strip() for s in sentences if s.strip()]


def stable_seed(base_seed, split, pair_id):
    digest = hashlib.sha256(f"{base_seed}|{split}|{pair_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def load_cache(path, config):
    if not path.exists():
        return {}
    cache = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("config") == config:
                cache[str(record["pair_id"])] = record["samples"]
    return cache


class Sampler:
    def __init__(self, model_id, device, max_input_tokens):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device).eval()
        self.max_input_tokens = max_input_tokens

    @torch.inference_mode()
    def sample(self, prompt, count, temperature, top_p, max_new_tokens, seed):
        tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_tokens)
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        # Per-pair seed makes resumed runs reproducible regardless of skipped cached pairs.
        torch.manual_seed(seed)
        outputs = self.model.generate(
            **tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=count,
            max_new_tokens=max_new_tokens,
        )
        return [x.strip() for x in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]


class NLIScorer:
    def __init__(self, model_id, device, batch_size):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device).eval()
        self.device = device
        self.batch_size = batch_size
        id_to_label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        matches = {name: [idx for idx, label in id_to_label.items() if name in label] for name in ("contradiction", "entailment")}
        if any(len(ids) != 1 for ids in matches.values()):
            raise ValueError(f"NLI model needs contradiction and entailment labels; got {id_to_label}")
        self.contradiction_idx = matches["contradiction"][0]
        self.entailment_idx = matches["entailment"][0]

    @torch.inference_mode()
    def score(self, response, samples):
        sentences = split_sentences(response)
        if not sentences or not samples:
            return 0.5
        # Each independent generated passage is the premise; the candidate sentence
        # is the hypothesis. This does NOT compare against the source document.
        comparisons = [(sample, sent) for sent in sentences for sample in samples]
        values = []
        for offset in range(0, len(comparisons), self.batch_size):
            batch = comparisons[offset:offset + self.batch_size]
            inputs = self.tokenizer(
                [premise for premise, _ in batch],
                [hypothesis for _, hypothesis in batch],
                return_tensors="pt", padding=True, truncation=True, max_length=512,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits.float(), dim=-1)
            contra = probs[:, self.contradiction_idx]
            entail = probs[:, self.entailment_idx]
            # Official SelfCheck-NLI normalizes over contradiction/entailment.
            normalized = contra / (contra + entail).clamp_min(1e-12)
            values.extend(normalized.cpu().tolist())
        return float(np.mean(values))


def generate_samples(frame, split, sampler, args, cache_path):
    config = {
        "generator": args.generator,
        "num_samples": args.num_samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "max_input_tokens": args.max_input_tokens,
        "seed": args.seed,
    }
    cache = load_cache(cache_path, config)
    representatives = frame.drop_duplicates(subset="pair_id", keep="first")
    with cache_path.open("a", encoding="utf-8") as output:
        for _, row in tqdm(representatives.iterrows(), total=len(representatives), desc=f"Generate {split}"):
            pair_id = str(row["pair_id"])
            if pair_id in cache:
                continue
            prompt = build_prompt(row)
            samples = sampler.sample(
                prompt, args.num_samples, args.temperature, args.top_p,
                args.max_new_tokens, stable_seed(args.seed, split, pair_id),
            )
            cache[pair_id] = samples
            output.write(json.dumps({"pair_id": pair_id, "config": config, "samples": samples}, ensure_ascii=False) + "\n")
            output.flush()
    return cache


def score_frame(frame, split, samples, scorer, output_dir):
    rows = []
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc=f"Score {split}"):
        pair_id = str(row["pair_id"])
        rows.append({
            "id": row["id"],
            "score": scorer.score(row["response"], samples[pair_id]),
            "label": int(row["label"]),
        })
    scored = pd.DataFrame(rows)
    scored.to_csv(output_dir / f"{split}_scores.csv", index=False)
    return scored


def choose_threshold(y_true, scores):
    # Pick the threshold that maximizes validation F1; never use test labels here.
    unique_scores = np.unique(scores)
    thresholds = np.unique(np.r_[0.0, (unique_scores[:-1] + unique_scores[1:]) / 2.0, 1.0])
    best_t, best_f1 = 0.5, -1.0
    for threshold in thresholds:
        f1 = f1_score(y_true, (scores >= threshold).astype(int), zero_division=0)
        if f1 > best_f1 or (np.isclose(f1, best_f1) and abs(threshold - 0.5) < abs(best_t - 0.5)):
            best_t, best_f1 = float(threshold), float(f1)
    return best_t


def evaluate(scored, threshold):
    labels = scored["label"].to_numpy(dtype=int)
    scores = scored["score"].to_numpy(dtype=float)
    preds = (scores >= threshold).astype(int)
    return preds, {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "auroc": float(roc_auc_score(labels, scores)),
    }


def main():
    args = parse_args()
    if args.num_samples < 2 or args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("Require num-samples >= 2, temperature > 0, 0 < top-p <= 1")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {name: load_split(name, data_dir) for name in SPLITS}
    validate_splits(frames)
    device = get_device(args.device)
    print(f"Device: {device} | generator: {args.generator} | NLI: {args.nli_model}")
    print("No training is performed. All pairs and all rows will be evaluated.")
    sampler = Sampler(args.generator, device, args.max_input_tokens)
    scorer = NLIScorer(args.nli_model, device, args.batch_size)
    scored = {}
    for name in SPLITS:
        cache_path = output_dir / f"{name}_generations.jsonl"
        samples = generate_samples(frames[name], name, sampler, args, cache_path)
        scored[name] = score_frame(frames[name], name, samples, scorer, output_dir)
    threshold = choose_threshold(scored["val"]["label"].to_numpy(), scored["val"]["score"].to_numpy())
    print(f"Validation-selected threshold: {threshold:.6f}")
    metrics = {}
    for name in SPLITS:
        preds, results = evaluate(scored[name], threshold)
        prediction_file = output_dir / f"{name}_predictions.csv"
        pd.DataFrame({"id": scored[name]["id"], "score": scored[name]["score"], "pred": preds}).to_csv(prediction_file, index=False)
        metrics[name] = results
        print(f"{name:20s} accuracy={results['accuracy']:.4f} F1={results['f1']:.4f} AUROC={results['auroc']:.4f}")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"threshold": threshold, "generator": args.generator, "nli_model": args.nli_model,
                   "num_samples": args.num_samples, "metrics": metrics}, handle, indent=2)
    print(f"Results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
