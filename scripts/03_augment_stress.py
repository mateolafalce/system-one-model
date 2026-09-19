#!/usr/bin/env python3
"""Build bucket C from gold + distilled JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataio import load_records, parse_data_arg, write_jsonl
from stress import augment_pool


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/gold")
    p.add_argument("--out", default="data/stress/stress.jsonl")
    p.add_argument("--fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    recs = load_records(parse_data_arg(args.data))
    recs = [r for r in recs if r.split in {"train", "val"}]
    print(f"source records: {len(recs)} (train/val only)")
    out = augment_pool(recs, fraction=args.fraction, seed=args.seed)
    n = write_jsonl(args.out, out)
    print(f"wrote {n} stress rows -> {args.out}")


if __name__ == "__main__":
    main()
