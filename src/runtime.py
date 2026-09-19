"""Pack questions against one state, run the student, apply temperatures."""

from __future__ import annotations

from typing import Any

import torch

from loss import masked_softmax
from metrics import entropy_confidence, expected_value, k_bucket
from pack import PackedExample, collate_packed, pack_record
from schema import Label, Question, Record, TYPE_TO_ID, question_from_dict


def load_temperatures(path: str | None) -> dict[str, dict[str, float]]:
    if not path:
        return {}
    from pathlib import Path
    import json

    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def temperature_for(temps: dict[str, dict[str, float]], primitive: str, k: int) -> float:
    bucket = temps.get(primitive) or temps.get("default") or {}
    return float(bucket.get(k_bucket(k), bucket.get("default", 1.0)))


def decode_row(
    probs: list[float],
    option_ids: list[str],
    primitive: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra = extra or {}
    conf = entropy_confidence(probs)
    if primitive == "choice":
        idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        return {
            "type": "choice",
            "choice": option_ids[idx],
            "probabilities": {oid: float(p) for oid, p in zip(option_ids, probs)},
            "confidence": conf,
        }
    if primitive == "score":
        return {
            "type": "score",
            "score": expected_value(probs),
            "probabilities": [float(p) for p in probs],
            "confidence": conf,
        }
    true_i = extra.get("noul_true_index", option_ids.index("true") if "true" in option_ids else 0)
    return {"type": "noul", "noul": float(probs[int(true_i)])}


@torch.no_grad()
def predict_batch(
    model,
    tokenizer,
    records: list[Record],
    max_length: int = 512,
    temperatures: dict[str, dict[str, float]] | None = None,
    device: str | torch.device = "cuda",
) -> list[dict[str, Any]]:
    temps = temperatures or {}
    packed = [pack_record(tokenizer, rec, max_length=max_length, shuffle_options=False) for rec in records]
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)
    batch = collate_packed(packed, pad_id=pad)
    tensor_keys = [
        "input_ids",
        "attention_mask",
        "option_positions",
        "option_mask",
        "type_ids",
    ]
    feed = {k: batch[k].to(device) for k in tensor_keys}
    logits = model(**feed)
    answers = []
    for i, ex in enumerate(packed):
        k = ex.k
        t = temperature_for(temps, ex.primitive, k)
        row_logits = logits[i, :k].float() / max(t, 1e-6)
        probs = torch.softmax(row_logits, dim=0).tolist()
        answers.append(decode_row(probs, ex.option_ids, ex.primitive, ex.extra))
    return answers


def questions_to_records(state: Any, questions: dict[str, Any]) -> list[Record]:
    recs: list[Record] = []
    for qid, spec in questions.items():
        if isinstance(spec, Question):
            q = spec
        else:
            payload = dict(spec)
            payload.setdefault("id", qid)
            q = question_from_dict(payload)
        recs.append(
            Record(
                id=qid,
                group_id="serve",
                source="serve",
                split="infer",
                state=state,
                question=q,
                label=Label(kind="hard", choice=q.option_ids()[0], score=0, noul=0),
            )
        )
    return recs
