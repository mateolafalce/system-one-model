#!/usr/bin/env python3
"""In-task, distilled-template, zero-shot, and selective-automation tables."""

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
from dataio import load_records, parse_data_arg
from metrics import (
    accuracy,
    brier_mean,
    ece,
    entropy_confidence,
    f1_binary,
    fpr_at_tpr,
    quadratic_weighted_kappa,
    risk_coverage_accuracy,
    spearman,
)
from runtime import load_temperatures, predict_batch
from schema import Record
from student import StudentModel, dtype_from_name, load_tokenizer


def _family(rec: Record) -> str:
    return rec.family or rec.source


def eval_records(model, tok, recs: list[Record], temps, device, max_length: int) -> dict:
    if not recs:
        return {}
    answers = []
    bs = 8
    for i in range(0, len(recs), bs):
        answers.extend(predict_batch(model, tok, recs[i : i + bs], max_length=max_length, temperatures=temps, device=device))
    gold_idx = []
    pred_idx = []
    confs = []
    correct = []
    prob_rows = []
    score_pred = []
    score_gold = []
    noul_pred = []
    noul_gold = []
    for rec, ans in zip(recs, answers):
        ids = rec.question.option_ids()
        if rec.question.type == "choice":
            y = ids.index(rec.label.choice)
            yhat = ids.index(ans["choice"])
            probs = [ans["probabilities"][k] for k in ids]
            gold_idx.append(y)
            pred_idx.append(yhat)
            confs.append(ans["confidence"])
            correct.append(int(y == yhat))
            prob_rows.append(probs)
        elif rec.question.type == "score":
            y = int(rec.label.score)
            probs = ans["probabilities"]
            yhat = int(max(range(len(probs)), key=lambda i: probs[i]))
            gold_idx.append(y)
            pred_idx.append(yhat)
            confs.append(ans["confidence"])
            correct.append(int(y == yhat))
            prob_rows.append(probs)
            score_pred.append(ans["score"])
            score_gold.append(float(y))
        else:
            y = 1 if float(rec.label.noul) >= 0.5 else 0
            p = float(ans["noul"])
            yhat = 1 if p >= 0.5 else 0
            gold_idx.append(y)
            pred_idx.append(yhat)
            confs.append(entropy_confidence([p, 1 - p]))
            correct.append(int(y == yhat))
            prob_rows.append([1 - p, p] if y in (0, 1) else [p])
            noul_pred.append(p)
            noul_gold.append(y)
    out = {
        "n": len(recs),
        "acc": accuracy(pred_idx, gold_idx),
        "ece": ece(confs, correct),
        "brier": brier_mean(prob_rows, gold_idx) if prob_rows else None,
        "acc@50cov": risk_coverage_accuracy(correct, confs, 0.50),
        "acc@80cov": risk_coverage_accuracy(correct, confs, 0.80),
    }
    if score_pred:
        k = len(recs[0].question.option_ids())
        out["spearman"] = spearman(score_pred, score_gold)
        out["qwk"] = quadratic_weighted_kappa(pred_idx, gold_idx, n_classes=k)
    if noul_pred:
        out["f1"] = f1_binary(pred_idx, gold_idx, positive=1)
        try:
            out["fpr@95tpr"] = fpr_at_tpr(noul_pred, noul_gold, 0.95)
        except Exception:
            out["fpr@95tpr"] = None
    return out


def go_nogo(tables: dict) -> list[str]:
    lines = []
    b77 = tables.get("in_task", {}).get("banking77") or tables.get("in_task", {}).get("PolyAI/banking77")
    # families we actually emitted
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="artifacts/phase1/best")
    p.add_argument("--data", default="data/gold")
    p.add_argument("--temperatures", default="artifacts/temps.json")
    p.add_argument("--student-config", default="configs/student_modernbert_base.yaml")
    p.add_argument("--out", default="eval/tables")
    args = p.parse_args()

    cfg = load_yaml(args.student_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = dtype_from_name("float32" if device == "cpu" else cfg.get("dtype", "bfloat16"))
    tok = load_tokenizer(cfg["backbone"])
    model = StudentModel.from_pretrained(args.ckpt, dtype=dtype)
    model.to(device)
    temps = load_temperatures(str(ROOT / args.temperatures) if not Path(args.temperatures).is_absolute() else args.temperatures)
    recs = load_records(parse_data_arg(args.data))

    groups = defaultdict(list)
    for r in recs:
        if r.label.kind != "hard" and r.split != "zeroshot":
            # Distilled holdout uses soft labels; still eval argmax vs teacher later.
            pass
        groups[(r.split, _family(r))].append(r)

    tables = {"in_task": {}, "zeroshot": {}, "distilled": {}, "selective": {}}
    for (split, fam), rows in sorted(groups.items()):
        hard = [r for r in rows if r.label.kind == "hard"]
        if not hard:
            continue
        metrics = eval_records(model, tok, hard, temps, device, int(cfg.get("max_length", 512)))
        bucket = "zeroshot" if split == "zeroshot" else "in_task"
        if split == "test" or split == "zeroshot" or split == "val":
            tables[bucket][f"{fam}:{split}"] = metrics
            print(f"{bucket:9s} {fam:24s} {split:8s} n={metrics['n']:5d} acc={metrics['acc']:.3f} ece={metrics['ece']:.3f} @50={metrics['acc@50cov']:.3f}")

    # Go/no-go using in-task test slices when present.
    checks = []
    def find(substr, split="test"):
        for k, v in tables["in_task"].items():
            if substr in k and k.endswith(":" + split):
                return v
        for k, v in tables["in_task"].items():
            if substr in k:
                return v
        return None

    b77 = find("banking77")
    spam = find("sms") or find("enron")
    sst = find("sst5")
    if b77:
        checks.append(("BANKING77 acc>=0.90", b77["acc"] >= 0.90, b77["acc"]))
        checks.append(("BANKING77 ece<=0.08", b77["ece"] <= 0.08, b77["ece"]))
        checks.append(("BANKING77 acc@50>=0.94", b77["acc@50cov"] >= 0.94, b77["acc@50cov"]))
    if spam:
        checks.append(("spam acc>=0.95", spam["acc"] >= 0.95, spam["acc"]))
        checks.append(("spam ece<=0.06", spam["ece"] <= 0.06, spam["ece"]))
    if sst:
        checks.append(("SST-5 acc>=0.52", sst["acc"] >= 0.52, sst["acc"]))
    tables["go_nogo"] = [{"name": n, "pass": bool(ok), "value": v} for n, ok, v in checks]
    for c in tables["go_nogo"]:
        flag = "PASS" if c["pass"] else "FAIL"
        print(f"go/no-go {flag}: {c['name']} value={c['value']:.4f}")

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(tables, indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
