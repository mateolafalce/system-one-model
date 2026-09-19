#!/usr/bin/env python3
"""Fit one temperature per (primitive, K-bucket) on val. Does not train weights."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from config import load_yaml
from dataio import JsonlDecisionDataset, load_records, make_collate, parse_data_arg
from metrics import ece, entropy_confidence, k_bucket
from student import StudentModel, dtype_from_name, load_tokenizer


@torch.no_grad()
def collect_logits(model, loader, device: str):
    model.eval()
    rows = []
    for batch in loader:
        feed = {
            k: batch[k].to(device)
            for k in ("input_ids", "attention_mask", "type_ids", "option_positions", "option_mask")
        }
        logits = model(**feed).float().cpu()
        for i, prim in enumerate(batch["primitives"]):
            if int(batch["targets"][i]) < 0:
                continue
            k = int(batch["option_mask"][i].sum())
            rows.append(
                {
                    "logits": logits[i, :k].tolist(),
                    "target": int(batch["targets"][i]),
                    "primitive": prim,
                    "k": k,
                }
            )
    return rows


def ece_at_T(rows, T: float) -> float:
    confs, correct = [], []
    for r in rows:
        t = torch.tensor(r["logits"]) / max(T, 1e-6)
        probs = torch.softmax(t, dim=0)
        yhat = int(probs.argmax())
        confs.append(entropy_confidence(probs.tolist()))
        correct.append(int(yhat == r["target"]))
    return ece(confs, correct)


def fit_T(rows) -> float:
    if not rows:
        return 1.0
    grid = [i / 10 for i in range(5, 51)]  # 0.5 .. 5.0
    best_t, best = 1.0, 1e9
    for t in grid:
        val = ece_at_T(rows, t)
        if val < best:
            best, best_t = val, t
    return best_t


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="artifacts/phase1/best")
    p.add_argument("--data", default="data/gold")
    p.add_argument("--split", default="val")
    p.add_argument("--student-config", default="configs/student_modernbert_base.yaml")
    p.add_argument("--out", default="artifacts/temps.json")
    args = p.parse_args()

    cfg = load_yaml(args.student_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = dtype_from_name("float32" if device == "cpu" else cfg.get("dtype", "bfloat16"))
    tok = load_tokenizer(cfg["backbone"])
    model = StudentModel.from_pretrained(args.ckpt, dtype=dtype)
    model.to(device)
    recs = [r for r in load_records(parse_data_arg(args.data)) if r.split == args.split and r.label.kind == "hard"]
    if not recs:
        recs = [r for r in load_records(parse_data_arg(args.data)) if r.label.kind == "hard"][:256]
        print(f"no {args.split} split; using {len(recs)} hard rows")
    ds = JsonlDecisionDataset(recs, tok, max_length=int(cfg.get("max_length", 512)), shuffle_options=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, collate_fn=make_collate(tok))
    rows = collect_logits(model, loader, device)
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["primitive"], k_bucket(r["k"]))].append(r)

    temps: dict[str, dict[str, float]] = {}
    for (prim, bucket), group in sorted(grouped.items()):
        t = fit_T(group)
        temps.setdefault(prim, {})[bucket] = t
        print(f"{prim:6s} {bucket:5s} n={len(group):4d} T={t:.2f} ece={ece_at_T(group, t):.4f}")

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(temps, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
