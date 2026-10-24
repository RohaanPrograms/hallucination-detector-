# hallucination-detector-

**Research question:** Do hallucination detectors trained on labelled data (TF-IDF, fine-tuned DistilBERT) outperform detectors that need no training (NLI, self-consistency), and do they still win when tested on a different type of hallucination?

| Method | Needs training? | Owner |
|---|---|---|
| TF-IDF + logistic regression | yes | Nick |
| Fine-tuned DistilBERT / DeBERTa | yes | CK |
| NLI checker (pretrained RoBERTa/DeBERTa) | no | Rohaan |
| Self-consistency (SelfCheckGPT-style) | no | Connor |
| Shared data splits | – | Rohaan |

Dataset: [HaluEval](https://github.com/RUCAIBox/HaluEval). Trained on QA, tested on QA (in-domain) plus dialogue and summarization (unseen hallucination types).

**Start here → [`data/README.md`](data/README.md)** explains the data, the splits, and how each method should use them.

```
data/splits/       shared train / val / test files (use these, don't re-split)
data_utils.py      load_split("train") etc.
make_splits.py     script that built the splits (seed 42, reproducible)
```

Setup: `pip install pandas numpy scikit-learn`
