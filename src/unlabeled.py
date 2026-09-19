"""Unlabeled English states for teacher distillation (bucket B)."""

from __future__ import annotations

import random
from typing import Any, Iterator, Sequence

import config  # noqa: F401
from templates import BUNDLE_BY_DOMAIN


def _clip(text: str, n: int = 1600) -> str:
    text = " ".join(str(text).split())
    return text[:n]


def _load(hf_id: str, config: str | None = None):
    from datasets import load_dataset

    if config:
        return load_dataset(hf_id, config)
    return load_dataset(hf_id)


def iter_support_states(cap: int, seed: int) -> Iterator[tuple[str, Any]]:
    rng = random.Random(seed)
    try:
        ds = _load("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
        split = ds["train"] if hasattr(ds, "keys") else ds
        idx = list(range(len(split)))
        rng.shuffle(idx)
        for i in idx[:cap]:
            row = split[int(i)]
            text = row.get("instruction") or row.get("utterance") or row.get("text") or ""
            yield f"bitext-{i}", {"text": _clip(text)}
    except Exception as exc:
        print(f"skip bitext unlabeled: {exc}")


def iter_email_states(cap: int, seed: int) -> Iterator[tuple[str, Any]]:
    rng = random.Random(seed)
    try:
        ds = _load("SetFit/enron_spam")
        split = ds["train"] if hasattr(ds, "keys") else ds
        idx = list(range(len(split)))
        rng.shuffle(idx)
        for i in idx[:cap]:
            row = split[int(i)]
            text = row.get("text") or row.get("message") or ""
            subj = row.get("subject") or ""
            yield f"enron-{i}", {"from": "", "subject": _clip(subj, 200), "body": _clip(text)}
    except Exception as exc:
        print(f"skip enron unlabeled: {exc}")


def iter_dialogue_states(cap: int, seed: int) -> Iterator[tuple[str, Any]]:
    rng = random.Random(seed)
    try:
        ds = _load("pfb30/multi_woz_2_2")
    except Exception:
        try:
            ds = _load("ConvLab/multiwoz21")
        except Exception as exc:
            print(f"skip multiwoz: {exc}")
            return
    split = ds["train"] if hasattr(ds, "keys") else ds
    n = min(len(split), cap * 4)
    idx = list(range(n))
    rng.shuffle(idx)
    emitted = 0
    for i in idx:
        if emitted >= cap:
            return
        row = split[int(i)]
        turns = row.get("turns") or row.get("dialogue") or row.get("log") or []
        if isinstance(turns, dict):
            utts = turns.get("utterance") or []
            packed = [{"speaker": "user" if j % 2 == 0 else "system", "text": _clip(u, 400)} for j, u in enumerate(utts[-6:])]
        elif isinstance(turns, list):
            packed = []
            for t in turns[-6:]:
                if isinstance(t, dict):
                    packed.append(
                        {
                            "speaker": t.get("speaker") or t.get("role") or "user",
                            "text": _clip(t.get("utterance") or t.get("text") or "", 400),
                        }
                    )
                else:
                    packed.append({"speaker": "user", "text": _clip(t, 400)})
        else:
            continue
        if not packed:
            continue
        yield f"woz-{i}", {"turns": packed}
        emitted += 1


def iter_passage_states(cap: int, seed: int) -> Iterator[tuple[str, Any]]:
    rng = random.Random(seed)
    emitted = 0
    try:
        ds = _load("google/boolq")
        split = ds["train"] if hasattr(ds, "keys") else ds
        idx = list(range(len(split)))
        rng.shuffle(idx)
        for i in idx:
            if emitted >= cap:
                return
            row = split[int(i)]
            yield f"boolq-{i}", {"passage": _clip(row["passage"]), "question": row["question"]}
            emitted += 1
    except Exception as exc:
        print(f"skip boolq passages: {exc}")


def iter_repo_gold_states(
    cap: int, seed: int, families: tuple[str, ...], domain: str
) -> Iterator[tuple[str, Any]]:
    """Reuse already-exported gold train states. No Hub download."""
    from pathlib import Path

    from dataio import load_records

    gold = Path(__file__).resolve().parents[1] / "data" / "gold"
    if not gold.is_dir():
        return
    recs = load_records(gold)
    rng = random.Random(seed)
    seen: set[str] = set()
    candidates: list[tuple[str, Any]] = []
    want = set(families)
    for r in recs:
        if r.split != "train":
            continue
        fam = r.family or r.source
        if want and fam not in want:
            continue
        if r.group_id in seen:
            continue
        seen.add(r.group_id)
        candidates.append((r.group_id, r.state))
    rng.shuffle(candidates)
    for gid, state in candidates[:cap]:
        yield gid, state


def collect_unlabeled(
    cap_per_source: int = 2000,
    seed: int = 0,
    sources: Sequence[str] | None = None,
) -> list[tuple[str, str, Any, str]]:
    """Returns (state_id, domain, state, source).

    Default is `gold` only: Hub dumps (Bitext, Enron, MultiWOZ, BoolQ) wait
    until /home has more than ~4 GB free.
    """
    wanted = [s.strip() for s in (sources or ("gold",)) if s.strip()]
    out: list[tuple[str, str, Any, str]] = []
    if "gold" in wanted:
        for sid, state in iter_repo_gold_states(cap_per_source, seed, ("banking77", "banking77_coarse"), "support"):
            out.append((sid, "support", state, "gold-banking77"))
        for sid, state in iter_repo_gold_states(cap_per_source // 2, seed + 4, ("sms_spam",), "email"):
            out.append((sid, "email", state, "gold-sms"))
    if "bitext" in wanted:
        for sid, state in iter_support_states(cap_per_source, seed):
            out.append((sid, "support", state, "bitext"))
    if "multiwoz" in wanted:
        for sid, state in iter_dialogue_states(cap_per_source // 2, seed + 1):
            out.append((sid, "support", state, "multiwoz"))
    if "enron" in wanted:
        for sid, state in iter_email_states(cap_per_source, seed + 2):
            out.append((sid, "email", state, "enron"))
    if "boolq" in wanted:
        for sid, state in iter_passage_states(cap_per_source // 2, seed + 3):
            out.append((sid, "support", state, "boolq"))
    return out


def questions_for_domain(domain: str):
    return BUNDLE_BY_DOMAIN[domain]
