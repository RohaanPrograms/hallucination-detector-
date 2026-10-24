"""Shared loader so every method reads the splits the same way.

    from data_utils import load_split
    train = load_split("train")
    test_dia = load_split("test_dialogue")
"""

from pathlib import Path

import pandas as pd

SPLITS_DIR = Path(__file__).parent / "data" / "splits"
SPLITS = ["train", "val", "test_qa", "test_dialogue", "test_summarization"]
TEST_SPLITS = ["test_qa", "test_dialogue", "test_summarization"]


def load_split(name):
    if name not in SPLITS:
        raise ValueError(f"unknown split {name!r}, expected one of {SPLITS}")
    return pd.read_json(SPLITS_DIR / f"{name}.jsonl", lines=True)
