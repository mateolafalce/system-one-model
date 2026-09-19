"""Constrained option scoring from a frozen Qwen teacher. No JSON sampling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F

import config  # noqa: F401
from config import resolve_path
from schema import Question, Record, serialize_state


# Prefer single-token labels. A-Z then extra ASCII that Qwen encodes as one token.
LETTER_CANDIDATES = (
    [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    + [chr(c) for c in range(ord("a"), ord("z") + 1)]
    + [str(d) for d in range(10)]
    + list("!@#$%^&*()-_=+[]{};:,./<>?")
    + [chr(c) for c in range(0x391, 0x3AA)]  # Greek capitals
    + [chr(c) for c in range(0xC0, 0x100)]
)


def _single_token_id(tokenizer: Any, text: str) -> int | None:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) == 1:
        return int(ids[0])
    return None


def build_letter_map(tokenizer: Any, k_max: int = 96) -> dict[str, Any]:
    """Pick K unique single-token option letters and freeze their ids."""
    letters: list[str] = []
    ids: list[int] = []
    used: set[int] = set()
    for cand in LETTER_CANDIDATES:
        for variant in (cand, f" {cand}"):
            tid = _single_token_id(tokenizer, variant)
            if tid is None or tid in used:
                continue
            letters.append(cand if variant == cand else variant)
            ids.append(tid)
            used.add(tid)
            break
        if len(letters) >= k_max:
            break
    if len(letters) < 8:
        raise RuntimeError(f"teacher tokenizer yielded only {len(letters)} single-token letters")
    return {
        "letters": letters,
        "token_ids": ids,
        "note": "Frozen at smoke time. Prefill ends with 'Answer:'; next-token logits at these ids are the option scores.",
    }


def load_or_build_letter_map(tokenizer: Any, path: str | Path | None) -> dict[str, Any]:
    if path:
        p = resolve_path(path)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        mapping = build_letter_map(tokenizer)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
        return mapping
    return build_letter_map(tokenizer)


def letter_token_ids(mapping: dict[str, Any], k: int) -> list[int]:
    ids = mapping["token_ids"]
    if k > len(ids):
        raise ValueError(f"need {k} letter tokens, map has {len(ids)}")
    return [int(x) for x in ids[:k]]


def build_teacher_prompt(
    tokenizer: Any,
    state: Any,
    question: Question,
    option_items: Sequence[tuple[str, str]],
    mapping: dict[str, Any],
) -> str:
    letters = mapping["letters"]
    if len(option_items) > len(letters):
        raise ValueError(f"K={len(option_items)} exceeds letter map")
    compact = len(option_items) > 15
    lines = []
    for i, (oid, text) in enumerate(option_items):
        letter = letters[i].strip()
        if compact:
            lines.append(f"{letter} {oid.replace('_', ' ')}")
        elif oid == text:
            lines.append(f"{letter} {text}")
        else:
            lines.append(f"{letter} {oid}: {text}")
    user = (
        f"{question.instructions}\n\n"
        f"State:\n{serialize_state(state)}\n\n"
        f"Options:\n" + "\n".join(lines) + "\n\n"
        "Reply with a single letter from the option list."
    )
    messages = [
        {
            "role": "system",
            "content": "You classify by choosing exactly one option letter. Do not explain.",
        },
        {"role": "user", "content": user},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not text.endswith("Answer:"):
        text = text + "Answer:"
    return text


def _ensure_ninja_on_path() -> None:
    import os
    import shutil
    import sys
    from pathlib import Path

    if shutil.which("ninja"):
        return
    venv_bin = Path(sys.executable).resolve().parent
    os.environ["PATH"] = str(venv_bin) + os.pathsep + os.environ.get("PATH", "")


def local_snapshot(model_id: str) -> str | None:
    """Prefer a world-readable Hub snapshot so we do not write into a root-owned cache."""
    from pathlib import Path

    name = "models--" + model_id.replace("/", "--")
    hubs = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path(__file__).resolve().parents[1] / ".hf" / "hub",
    ]
    for hub in hubs:
        snaps = hub / name / "snapshots"
        if not snaps.is_dir():
            continue
        for p in snaps.iterdir():
            if (p / "config.json").exists():
                return str(p.resolve())
    return None


def load_teacher(cfg: dict[str, Any], device: str = "cuda"):
    """Load the frozen 4-bit/AWQ teacher. Unload the student first."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _ensure_ninja_on_path()
    model_id = cfg["model"]
    local = local_snapshot(model_id)
    load_id = local or model_id
    dtype = torch.float16 if cfg.get("dtype", "float16") == "float16" else torch.bfloat16
    tok = AutoTokenizer.from_pretrained(load_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    quant = str(cfg.get("quant", "awq")).lower()
    errors: list[str] = []
    model = None
    if quant == "awq":
        try:
            from awq import AutoAWQForCausalLM

            awq_model = AutoAWQForCausalLM.from_quantized(
                load_id,
                fuse_layers=False,
                device_map="auto" if device == "cuda" else device,
            )
            model = awq_model.model if hasattr(awq_model, "model") else awq_model
        except Exception as exc:
            errors.append(f"autoawq: {exc}")
            print(f"autoawq load failed: {exc}")
    if model is None:
        try:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    load_id,
                    device_map=device,
                    dtype=dtype,
                    trust_remote_code=True,
                )
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(
                    load_id,
                    device_map=device,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                )
        except Exception as exc:
            errors.append(f"transformers: {exc}")
            fallback = cfg.get("fallback_model")
            if not fallback:
                raise RuntimeError("teacher load failed: " + " | ".join(errors)) from exc
            print(f"AWQ load failed; falling back to bitsandbytes 4-bit {fallback}")
            from transformers import BitsAndBytesConfig

            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            tok = AutoTokenizer.from_pretrained(fallback, trust_remote_code=True)
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                fallback,
                quantization_config=bnb,
                device_map=device,
                trust_remote_code=True,
            )
            model_id = fallback
    model.eval()
    return model, tok, model_id


@torch.no_grad()
def score_options(
    model,
    tokenizer,
    mapping: dict[str, Any],
    state: Any,
    question: Question,
    temperature: float = 1.0,
    max_seq: int = 1024,
) -> dict[str, float]:
    items = question.option_items()
    prompt = build_teacher_prompt(tokenizer, state, question, items, mapping)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq)
    device = getattr(model, "device", None) or next(model.parameters()).device
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model(**enc)
    logits = out.logits[0, -1].float()
    letters = mapping["letters"]
    letter_logits = []
    for i in range(len(items)):
        display = str(letters[i]).strip()
        vals = []
        for variant in (display, f" {display}"):
            tid = _single_token_id(tokenizer, variant)
            if tid is not None:
                vals.append(logits[tid])
        if not vals:
            tid = int(mapping["token_ids"][i])
            vals.append(logits[tid])
        letter_logits.append(torch.stack(vals).max())
    letter_logits = torch.stack(letter_logits)
    t = max(float(temperature), 1e-6)
    probs = F.softmax(letter_logits / t, dim=0)
    return {oid: float(probs[i]) for i, (oid, _) in enumerate(items)}


@torch.no_grad()
def score_record(model, tokenizer, mapping: dict[str, Any], record: Record, temperature: float = 1.0, max_seq: int = 1024) -> dict[str, float]:
    return score_options(
        model,
        tokenizer,
        mapping,
        record.state,
        record.question,
        temperature=temperature,
        max_seq=max_seq,
    )
