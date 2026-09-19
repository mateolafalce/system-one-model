"""JSONL record schema for choice / score / noul."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Primitive = Literal["choice", "score", "noul"]
LabelKind = Literal["hard", "soft"]
PRIMITIVES: tuple[str, ...] = ("choice", "score", "noul")
TYPE_TO_ID: dict[str, int] = {"choice": 0, "score": 1, "noul": 2}
ID_TO_TYPE: dict[int, str] = {v: k for k, v in TYPE_TO_ID.items()}

NOUL_TRUE = "true"
NOUL_FALSE = "false"
NOUL_OPTIONS: tuple[tuple[str, str], ...] = (
    (NOUL_TRUE, "true"),
    (NOUL_FALSE, "false"),
)


class SchemaError(ValueError):
    pass


@dataclass
class Question:
    id: str
    type: str
    instructions: str
    criteria: dict[str, str] | list[str] | None = None

    def __post_init__(self) -> None:
        if self.type not in PRIMITIVES:
            raise SchemaError(f"unknown question type {self.type!r}")
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or len(self.criteria) < 2:
                raise SchemaError("choice criteria must be a dict with at least 2 options")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or len(self.criteria) < 2:
                raise SchemaError("score criteria must be a list with at least 2 levels")
        elif self.criteria not in (None, {}, []):
            raise SchemaError("noul criteria must be empty; options are true/false")

    def option_items(self) -> list[tuple[str, str]]:
        if self.type == "noul":
            return [tuple(p) for p in NOUL_OPTIONS]
        if self.type == "score":
            assert isinstance(self.criteria, list)
            return [(str(i), text) for i, text in enumerate(self.criteria)]
        assert isinstance(self.criteria, dict)
        return list(self.criteria.items())

    def option_ids(self) -> list[str]:
        return [oid for oid, _ in self.option_items()]


@dataclass
class Label:
    kind: str
    choice: str | None = None
    score: int | None = None
    noul: float | int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("hard", "soft"):
            raise SchemaError(f"unknown label kind {self.kind!r}")


@dataclass
class TeacherLabel:
    model: str
    probs: dict[str, float] | list[float] = field(default_factory=dict)


@dataclass
class Record:
    id: str
    group_id: str
    source: str
    split: str
    state: Any
    question: Question
    label: Label
    teacher: TeacherLabel | None = None
    family: str = ""

    def option_items(self) -> list[tuple[str, str]]:
        return self.question.option_items()

    def hard_option_id(self) -> str | None:
        if self.label.kind != "hard":
            return None
        q = self.question
        if q.type == "choice":
            if self.label.choice is None:
                raise SchemaError(f"{self.id}: hard choice missing label.choice")
            return self.label.choice
        if q.type == "noul":
            if self.label.noul is None:
                raise SchemaError(f"{self.id}: hard noul missing label.noul")
            return NOUL_TRUE if float(self.label.noul) >= 0.5 else NOUL_FALSE
        if self.label.score is None:
            raise SchemaError(f"{self.id}: hard score missing label.score")
        return str(int(self.label.score))

    def aligned_teacher_probs(self, option_ids: list[str]) -> list[float] | None:
        if self.teacher is None or not self.teacher.probs:
            return None
        raw = self.teacher.probs
        if isinstance(raw, list):
            if len(raw) != len(option_ids):
                raise SchemaError(f"{self.id}: teacher.probs length {len(raw)} != K={len(option_ids)}")
            vec = [float(x) for x in raw]
        else:
            vec = [float(raw.get(oid, 0.0)) for oid in option_ids]
        total = sum(vec)
        if total <= 0:
            raise SchemaError(f"{self.id}: teacher.probs sum to 0")
        return [x / total for x in vec]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["teacher"] is None:
            d.pop("teacher")
        if not d.get("family"):
            d.pop("family", None)
        if d["question"]["criteria"] in (None, {}, []):
            d["question"].pop("criteria", None)
        return d


def serialize_state(state: Any) -> str:
    if state is None:
        return ""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        if set(state.keys()) == {"text"}:
            return str(state["text"])
        import json

        return json.dumps(state, ensure_ascii=False, sort_keys=True)
    return str(state)


def question_from_dict(d: dict[str, Any]) -> Question:
    return Question(
        id=str(d["id"]),
        type=str(d["type"]),
        instructions=str(d["instructions"]),
        criteria=d.get("criteria"),
    )


def record_from_dict(d: dict[str, Any]) -> Record:
    q = question_from_dict(d["question"])
    lab_d = d["label"]
    label = Label(
        kind=str(lab_d["kind"]),
        choice=lab_d.get("choice"),
        score=lab_d.get("score"),
        noul=lab_d.get("noul"),
    )
    teacher = None
    if d.get("teacher"):
        t = d["teacher"]
        teacher = TeacherLabel(model=str(t.get("model", "")), probs=t.get("probs") or {})
    rec = Record(
        id=str(d["id"]),
        group_id=str(d["group_id"]),
        source=str(d["source"]),
        split=str(d.get("split", "train")),
        state=d["state"],
        question=q,
        label=label,
        teacher=teacher,
        family=str(d.get("family") or ""),
    )
    _validate_record(rec)
    return rec


def _validate_record(rec: Record) -> None:
    ids = rec.question.option_ids()
    if rec.label.kind == "hard":
        oid = rec.hard_option_id()
        if oid not in ids:
            raise SchemaError(f"{rec.id}: label {oid!r} not in options {ids}")
        if rec.question.type == "score":
            assert rec.label.score is not None
            if not (0 <= int(rec.label.score) < len(ids)):
                raise SchemaError(f"{rec.id}: score index out of range")
    rec.aligned_teacher_probs(ids)
