"""Pack a record into the GLiClass-style sequence with one [MASK] per option.

Layout:
  [CLS] <type> question: <instructions> [SEP]
  [MASK] option_0 [MASK] option_1 ... [SEP]
  <state> [SEP]

State is truncated from the tail. Option [MASK] markers always survive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from schema import (
    NOUL_TRUE,
    TYPE_TO_ID,
    Record,
    serialize_state,
)


@dataclass
class PackedExample:
    input_ids: list[int]
    option_positions: list[int]
    option_ids: list[str]
    type_id: int
    target_index: int | None = None
    target_probs: list[float] | None = None
    record_id: str = ""
    primitive: str = ""
    family: str = ""
    source: str = ""
    split: str = ""
    is_soft: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.option_ids)


def _encode(tokenizer: Any, text: str) -> list[int]:
    if not text:
        return []
    ids = tokenizer.encode(text, add_special_tokens=False)
    return [int(x) for x in ids]


def _required_ids(tokenizer: Any) -> tuple[int, int, int]:
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    mask_id = tokenizer.mask_token_id
    if cls_id is None or sep_id is None or mask_id is None:
        raise ValueError("tokenizer must define cls_token_id, sep_token_id, mask_token_id")
    return int(cls_id), int(sep_id), int(mask_id)


def pack_record(
    tokenizer: Any,
    record: Record,
    max_length: int = 512,
    shuffle_options: bool = False,
    rng: Any | None = None,
) -> PackedExample:
    """Tokenize one question-state pair. Score options stay in rubric order."""
    cls_id, sep_id, mask_id = _required_ids(tokenizer)
    q = record.question
    items = list(q.option_items())
    if shuffle_options and q.type != "score" and len(items) > 1:
        import random

        r = rng if rng is not None else random
        r.shuffle(items)

    option_ids = [oid for oid, _ in items]
    prefix_text = f"{q.type} question: {q.instructions}"
    prefix = [cls_id] + _encode(tokenizer, prefix_text) + [sep_id]

    # Compact option text when cardinality is high so 77-way BANKING77 still fits.
    compact = len(items) > 15
    option_bodies: list[list[int]] = []
    for oid, text in items:
        if compact:
            readable = oid.replace("_", " ")
            body_text = f" {readable}"
        else:
            body_text = f" {oid}: {text}" if oid not in (text, str(text)) else f" {text}"
            if q.type == "noul":
                body_text = f" {oid}"
        option_bodies.append(_encode(tokenizer, body_text))

    state_ids = _encode(tokenizer, serialize_state(record.state))

    def assemble(prefix_ids: list[int], bodies: list[list[int]], state: list[int]) -> tuple[list[int], list[int]]:
        ids: list[int] = list(prefix_ids)
        positions: list[int] = []
        for body in bodies:
            positions.append(len(ids))
            ids.append(mask_id)
            ids.extend(body)
        ids.append(sep_id)
        ids.extend(state)
        ids.append(sep_id)
        return ids, positions

    ids, positions = assemble(prefix, option_bodies, state_ids)

    if len(ids) > max_length:
        overhead = len(prefix) + len(items) + 2  # prefix + MASKs + two SEPs
        if overhead >= max_length:
            # Last resort: shrink instructions, keep one token of prefix after CLS.
            keep_prefix = max(2, max_length - (len(items) + 2) - 8)
            prefix = prefix[:keep_prefix]
            if prefix[-1] != sep_id:
                prefix = prefix[:-1] + [sep_id]
            overhead = len(prefix) + len(items) + 2
        room_for_bodies_and_state = max_length - overhead
        body_lens = [len(b) for b in option_bodies]
        body_total = sum(body_lens)
        if body_total > room_for_bodies_and_state:
            # Shrink longest option bodies until they fit; state becomes empty.
            extra = body_total - room_for_bodies_and_state
            order = sorted(range(len(body_lens)), key=lambda i: body_lens[i], reverse=True)
            i = 0
            while extra > 0 and any(body_lens):
                idx = order[i % len(order)]
                if body_lens[idx] > 0:
                    body_lens[idx] -= 1
                    extra -= 1
                i += 1
            option_bodies = [b[:n] for b, n in zip(option_bodies, body_lens)]
            state_ids = []
        else:
            state_budget = room_for_bodies_and_state - body_total
            state_ids = state_ids[:state_budget]
        ids, positions = assemble(prefix, option_bodies, state_ids)

    if len(ids) > max_length:
        ids = ids[:max_length]
        positions = [p for p in positions if p < max_length]
        if len(positions) != len(items):
            raise ValueError(f"{record.id}: packing dropped option markers ({len(positions)}/{len(items)})")

    target_index = None
    target_probs = record.aligned_teacher_probs(option_ids)
    if record.label.kind == "hard":
        oid = record.hard_option_id()
        target_index = option_ids.index(oid)
    elif target_probs is None:
        raise ValueError(f"{record.id}: soft label requires teacher.probs")

    return PackedExample(
        input_ids=ids,
        option_positions=positions,
        option_ids=option_ids,
        type_id=TYPE_TO_ID[q.type],
        target_index=target_index,
        target_probs=target_probs,
        record_id=record.id,
        primitive=q.type,
        family=record.family or record.source,
        source=record.source,
        split=record.split,
        is_soft=record.label.kind == "soft",
        extra={"noul_true_index": option_ids.index(NOUL_TRUE) if q.type == "noul" else None},
    )


def collate_packed(
    examples: Sequence[PackedExample],
    pad_id: int,
) -> dict[str, Any]:
    import torch

    max_t = max(len(ex.input_ids) for ex in examples)
    max_k = max(ex.k for ex in examples)
    bsz = len(examples)
    input_ids = torch.full((bsz, max_t), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_t), dtype=torch.long)
    option_positions = torch.full((bsz, max_k), -1, dtype=torch.long)
    option_mask = torch.zeros((bsz, max_k), dtype=torch.bool)
    type_ids = torch.zeros(bsz, dtype=torch.long)
    targets = torch.full((bsz,), -100, dtype=torch.long)
    teacher_probs = torch.zeros((bsz, max_k), dtype=torch.float)
    is_soft = torch.zeros(bsz, dtype=torch.bool)
    is_score = torch.zeros(bsz, dtype=torch.bool)
    noul_true_index = torch.full((bsz,), -1, dtype=torch.long)

    for i, ex in enumerate(examples):
        t = len(ex.input_ids)
        k = ex.k
        input_ids[i, :t] = torch.tensor(ex.input_ids, dtype=torch.long)
        attention_mask[i, :t] = 1
        option_positions[i, :k] = torch.tensor(ex.option_positions, dtype=torch.long)
        option_mask[i, :k] = True
        type_ids[i] = ex.type_id
        if ex.target_index is not None:
            targets[i] = ex.target_index
        if ex.target_probs is not None:
            teacher_probs[i, :k] = torch.tensor(ex.target_probs, dtype=torch.float)
        is_soft[i] = bool(ex.is_soft)
        is_score[i] = ex.primitive == "score"
        if ex.extra.get("noul_true_index") is not None:
            noul_true_index[i] = int(ex.extra["noul_true_index"])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "option_positions": option_positions,
        "option_mask": option_mask,
        "type_ids": type_ids,
        "targets": targets,
        "teacher_probs": teacher_probs,
        "is_soft": is_soft,
        "is_score": is_score,
        "noul_true_index": noul_true_index,
        "option_ids": [ex.option_ids for ex in examples],
        "record_ids": [ex.record_id for ex in examples],
        "primitives": [ex.primitive for ex in examples],
        "families": [ex.family for ex in examples],
        "examples": list(examples),
    }
