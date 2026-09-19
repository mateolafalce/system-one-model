"""Bucket C: option shuffle, paraphrases, JSON/raw, distractors, truncation, NOTA."""

from __future__ import annotations

import copy
import random
from typing import Iterable

from schema import Label, Question, Record, serialize_state
from templates import PARAPHRASES


def _clone(rec: Record, new_id: str) -> Record:
    return Record(
        id=new_id,
        group_id=rec.group_id,
        source=rec.source,
        split=rec.split,
        state=copy.deepcopy(rec.state),
        question=Question(
            id=rec.question.id,
            type=rec.question.type,
            instructions=rec.question.instructions,
            criteria=copy.deepcopy(rec.question.criteria),
        ),
        label=Label(
            kind=rec.label.kind,
            choice=rec.label.choice,
            score=rec.label.score,
            noul=rec.label.noul,
        ),
        teacher=copy.deepcopy(rec.teacher),
        family=rec.family,
    )


def option_shuffle(rec: Record, rng: random.Random) -> Record | None:
    if rec.question.type == "score":
        return None
    items = rec.question.option_items()
    if len(items) < 3:
        return None
    rng.shuffle(items)
    out = _clone(rec, rec.id + "-shuffle")
    if rec.question.type == "choice":
        out.question.criteria = {k: v for k, v in items}
    # noul: option_items is fixed true/false; shuffling is done at pack time.
    return out


def paraphrase(rec: Record, rng: random.Random) -> Record | None:
    key = f"{rec.family.split('_')[0] if rec.family else rec.source}.{rec.question.id}"
    # Try a few key shapes.
    candidates = []
    fam = rec.family or ""
    qid = rec.question.id
    for k in (f"{fam}.{qid}", key):
        if k in PARAPHRASES:
            candidates = PARAPHRASES[k]
            break
    # Fallback: match by suffix of known keys.
    if not candidates:
        for k, v in PARAPHRASES.items():
            if k.endswith("." + qid):
                candidates = v
                break
    if not candidates:
        return None
    alt = rng.choice([p for p in candidates if p != rec.question.instructions] or candidates)
    if alt == rec.question.instructions:
        return None
    out = _clone(rec, rec.id + "-para")
    out.question.instructions = alt
    return out


def json_vs_raw(rec: Record) -> Record | None:
    state = rec.state
    out = _clone(rec, rec.id + "-rawjson")
    if isinstance(state, dict):
        out.state = serialize_state(state)
        return out
    if isinstance(state, str):
        out.state = {"text": state}
        return out
    return None


def distractor_fields(rec: Record, rng: random.Random) -> Record | None:
    if not isinstance(rec.state, dict):
        return None
    out = _clone(rec, rec.id + "-distractor")
    assert isinstance(out.state, dict)
    out.state = {
        **out.state,
        "internal_id": f"x-{rng.randint(10000, 99999)}",
        "random_note": rng.choice(["n/a", "see runbook", "legacy flag"]),
    }
    return out


def hard_truncate(rec: Record, n_chars: int, tag: str) -> Record:
    text = serialize_state(rec.state)
    out = _clone(rec, rec.id + f"-trunc{tag}")
    out.state = {"text": text[:n_chars]}
    return out


def none_of_the_above(rec: Record, rng: random.Random) -> list[Record]:
    if rec.question.type != "choice" or rec.label.kind != "hard":
        return []
    if rec.source not in {"PolyAI/banking77", "clinc_oos"} and "banking77" not in rec.source and "clinc" not in rec.source:
        return []
    criteria = dict(rec.question.criteria or {})
    gold = rec.label.choice
    if gold is None or gold not in criteria or gold in {"other", "oos"}:
        return []
    keys = [k for k in criteria if k != gold]
    if len(keys) < 2:
        return []
    keep = rng.sample(keys, min(8, len(keys)))
    keep_map = {k: criteria[k] for k in keep}
    keep_map["other"] = "Anything else, including the true intent if it is not listed."

    wrong = _clone(rec, rec.id + "-nota-other")
    wrong.question.criteria = keep_map
    wrong.label.choice = "other"

    sibling_gold = _clone(rec, rec.id + "-nota-keep")
    sibling_gold.question.criteria = {**keep_map, gold: criteria[gold]}
    sibling_gold.label.choice = gold
    return [wrong, sibling_gold]


def augment_pool(records: Iterable[Record], fraction: float = 0.10, seed: int = 0) -> list[Record]:
    """Apply transforms to `fraction` of A+B. Keep original labels."""
    recs = list(records)
    rng = random.Random(seed)
    n = max(1, int(len(recs) * fraction))
    picked = rng.sample(recs, min(n, len(recs)))
    out: list[Record] = []
    transforms = [
        lambda r: option_shuffle(r, rng),
        lambda r: paraphrase(r, rng),
        json_vs_raw,
        lambda r: distractor_fields(r, rng),
        lambda r: hard_truncate(r, 512, "128"),  # ~128 tokens ≈ 512 chars
        lambda r: hard_truncate(r, 1024, "256"),
    ]
    for rec in picked:
        fn = rng.choice(transforms)
        made = fn(rec)
        if made is None:
            continue
        if isinstance(made, list):
            out.extend(made)
        else:
            out.append(made)
        if rng.random() < 0.15:
            out.extend(none_of_the_above(rec, rng))
    return out
