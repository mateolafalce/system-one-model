#!/usr/bin/env python3
"""Score unlabeled states with frozen Qwen letter logits. Unload the student first."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import load_yaml, resolve_path
from dataio import write_jsonl
from schema import Label, Record, TeacherLabel, serialize_state
from teacher import load_or_build_letter_map, load_teacher, score_options
from unlabeled import collect_unlabeled, questions_for_domain


def _hash(state, question) -> str:
    blob = json.dumps(
        {"state": state, "id": question.id, "type": question.type, "ins": question.instructions, "crit": question.criteria},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/teacher_qwen7b_4bit.yaml")
    p.add_argument("--model", default="", help="override model id")
    p.add_argument("--quant", default="")
    p.add_argument("--out", default="data/distilled")
    p.add_argument("--cap-per-source", type=int, default=1500)
    p.add_argument("--max-pairs", type=int, default=40000)
    p.add_argument("--holdout", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--sources",
        default="gold",
        help="comma list: gold,bitext,multiwoz,enron,boolq. gold reuses data/gold (no download)",
    )
    p.add_argument(
        "--ceiling",
        default="",
        help="JSONL with hard labels: score with the teacher, print accuracy, exit (no distill write)",
    )
    p.add_argument("--limit", type=int, default=0, help="cap rows for --ceiling")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    if args.model:
        cfg["model"] = args.model
    if args.quant:
        cfg["quant"] = args.quant

    if args.ceiling:
        from collections import defaultdict

        from dataio import load_records

        recs = [r for r in load_records(args.ceiling) if r.label.kind == "hard"]
        if args.limit:
            recs = recs[: args.limit]
        print(f"teacher ceiling on {len(recs)} hard rows from {args.ceiling}")
        model, tok, model_id = load_teacher(cfg)
        mapping = load_or_build_letter_map(tok, cfg.get("letter_map_path"))
        temp = float(cfg.get("temperature", 1.0))
        max_seq = int(cfg.get("max_seq", 1024))
        by_fam: dict[str, list[int]] = defaultdict(list)
        for i, rec in enumerate(recs):
            probs = score_options(
                model, tok, mapping, rec.state, rec.question, temperature=temp, max_seq=max_seq
            )
            gold = rec.hard_option_id()
            yhat = max(probs, key=probs.get)
            ok = int(yhat == gold)
            fam = rec.family or rec.source
            by_fam[fam].append(ok)
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  {i+1}/{len(recs)} last={yhat!r} gold={gold!r} p={probs[yhat]:.3f}")
        print(f"teacher={model_id} T={temp}")
        for fam, hits in sorted(by_fam.items()):
            acc = sum(hits) / max(len(hits), 1)
            print(f"  {fam}: acc={acc:.4f} n={len(hits)}")
        overall = [h for hits in by_fam.values() for h in hits]
        print(f"overall acc={sum(overall)/max(len(overall),1):.4f} n={len(overall)}")
        print("unload the teacher before training")
        return

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = resolve_path(cfg.get("cache_dir", out_dir / "cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("collecting unlabeled states (no student on GPU)")
    srcs = [s.strip() for s in args.sources.split(",") if s.strip()]
    states = collect_unlabeled(cap_per_source=args.cap_per_source, seed=args.seed, sources=srcs)
    rng = random.Random(args.seed)
    rng.shuffle(states)
    n_hold = int(len(states) * args.holdout)
    hold = states[:n_hold]
    train_states = states[n_hold:]
    (out_dir / "holdout_states.json").write_text(
        json.dumps([{"id": s[0], "domain": s[1], "source": s[3]} for s in hold], indent=2),
        encoding="utf-8",
    )
    print(f"{len(train_states)} states to label, {len(hold)} held out")

    model, tok, model_id = load_teacher(cfg)
    mapping = load_or_build_letter_map(tok, cfg.get("letter_map_path"))
    temp = float(cfg.get("temperature", 1.0))
    max_seq = int(cfg.get("max_seq", 1024))

    recs: list[Record] = []
    for sid, domain, state, source in train_states:
        for q in questions_for_domain(domain):
            if len(recs) >= args.max_pairs:
                break
            key = _hash(state, q)
            cache_path = cache_dir / f"{key}.json"
            if cache_path.exists():
                probs = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                probs = score_options(model, tok, mapping, state, q, temperature=temp, max_seq=max_seq)
                cache_path.write_text(json.dumps(probs), encoding="utf-8")
            recs.append(
                Record(
                    id=f"{sid}-{q.id}",
                    group_id=sid,
                    source=f"teacher:{source}",
                    split="train",
                    state=state,
                    question=q,
                    label=Label(kind="soft"),
                    teacher=TeacherLabel(model=model_id, probs=probs),
                    family=f"distill_{domain}",
                )
            )
            if len(recs) % 50 == 0:
                print(f"  labeled {len(recs)}")
        if len(recs) >= args.max_pairs:
            break

    n = write_jsonl(out_dir / "teacher.jsonl", recs)
    print(f"wrote {n} distilled pairs -> {out_dir / 'teacher.jsonl'}")
    print("unload the teacher before training")


if __name__ == "__main__":
    main()
