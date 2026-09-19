"""JSONL I/O and torch Dataset over packed records."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from schema import Record, record_from_dict
from pack import PackedExample, collate_packed, pack_record
from config import resolve_path


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    p = resolve_path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[Record | dict[str, Any]]) -> int:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            d = rec.to_dict() if isinstance(rec, Record) else rec
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_records(paths: str | Path | Sequence[str | Path]) -> list[Record]:
    if isinstance(paths, (str, Path)):
        raw = [paths]
    else:
        raw = list(paths)
    expanded: list[Path] = []
    for item in raw:
        p = resolve_path(item)
        if p.is_dir():
            expanded.extend(sorted(x for x in p.glob("*.jsonl") if not x.name.startswith("smoke_")))
        else:
            expanded.append(p)
    out: list[Record] = []
    for p in expanded:
        for d in iter_jsonl(p):
            out.append(record_from_dict(d))
    return out


class JsonlDecisionDataset:
    def __init__(
        self,
        records: Sequence[Record],
        tokenizer: Any,
        max_length: int = 512,
        shuffle_options: bool = True,
        seed: int = 0,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_options = shuffle_options
        self.seed = seed

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> PackedExample:
        rec = self.records[idx]
        rng = random.Random(self.seed + idx * 10007)
        return pack_record(
            self.tokenizer,
            rec,
            max_length=self.max_length,
            shuffle_options=self.shuffle_options,
            rng=rng,
        )


def make_collate(tokenizer: Any):
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

    def _collate(examples: list[PackedExample]) -> dict[str, Any]:
        return collate_packed(examples, pad_id=pad_id)

    return _collate


def parse_data_arg(spec: str) -> list[Path]:
    parts = [s.strip() for s in spec.split(",") if s.strip()]
    return [resolve_path(p) for p in parts]


def split_by_name(records: Sequence[Record], name: str) -> list[Record]:
    return [r for r in records if r.split == name]
