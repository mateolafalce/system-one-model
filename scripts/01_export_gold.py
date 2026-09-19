#!/usr/bin/env python3
"""Export gold public datasets to data/gold/*.jsonl."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataio import write_jsonl
from gold import PROOF_TASKS, TASKS, run_task


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/gold")
    p.add_argument("--preset", choices=["proof", "full"], default="proof")
    p.add_argument("--tasks", default="", help="comma-separated task names; overrides preset")
    p.add_argument("--cap", type=int, default=None, help="override per-task cap")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="global row cap after export (smoke)")
    args = p.parse_args()

    if args.tasks:
        names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    elif args.preset == "proof":
        names = list(PROOF_TASKS)
    else:
        names = list(TASKS.keys())

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for name in names:
        print(f"export {name} ...")
        try:
            recs = run_task(name, args.cap, args.seed)
        except Exception as exc:
            print(f"  FAILED {name}: {exc}")
            summary.append({"task": name, "ok": False, "error": str(exc)})
            continue
        if args.limit is not None:
            recs = recs[: args.limit]
        by_split: dict[str, list] = defaultdict(list)
        for r in recs:
            by_split[r.split].append(r)
        paths = []
        for split, rows in by_split.items():
            path = out_dir / f"{name}_{split}.jsonl"
            write_jsonl(path, rows)
            paths.append(str(path.relative_to(ROOT)))
        print(f"  {len(recs)} records -> {paths}")
        summary.append({"task": name, "ok": True, "n": len(recs), "splits": {k: len(v) for k, v in by_split.items()}})
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ok = sum(1 for s in summary if s.get("ok"))
    print(f"done: {ok}/{len(summary)} tasks")


if __name__ == "__main__":
    main()
