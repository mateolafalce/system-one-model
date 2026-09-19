#!/usr/bin/env python3
"""Phase 0 go/no-go: packing, dummy forward, 100-row overfit, teacher letter ids."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: F401
import torch

from config import load_yaml, resolve_path
from dataio import load_records, write_jsonl
from gold import export_banking77
from loss import student_loss
from pack import pack_record
from schema import Label, Question, Record, record_from_dict
from student import build_student, infer_lora_targets


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def smoke_pack(tok) -> None:
    rec = Record(
        id="demo",
        group_id="demo",
        source="smoke",
        split="train",
        state={"text": "I still have not received my new card"},
        question=Question(
            id="intent",
            type="choice",
            instructions="Which banking intent matches this customer message?",
            criteria={
                "card_arrival": "The user is waiting for a physical card to arrive",
                "activate_my_card": "The user wants to activate a card",
                "other": "Anything else",
            },
        ),
        label=Label(kind="hard", choice="card_arrival"),
    )
    packed = pack_record(tok, rec, max_length=512)
    assert packed.k == 3
    assert len(packed.option_positions) == 3
    assert packed.input_ids[packed.option_positions[0]] == tok.mask_token_id
    assert packed.target_index == packed.option_ids.index("card_arrival")
    # 77-way still keeps every marker.
    criteria = {f"lab_{i}": f"description for label {i} " * 8 for i in range(77)}
    rec77 = Record(
        id="wide",
        group_id="wide",
        source="smoke",
        split="train",
        state={"text": "hello " * 400},
        question=Question(id="q", type="choice", instructions="Pick a label.", criteria=criteria),
        label=Label(kind="hard", choice="lab_3"),
    )
    packed77 = pack_record(tok, rec77, max_length=512)
    assert packed77.k == 77, packed77.k
    assert len(packed77.input_ids) <= 512
    print(f"pack ok: k=3 and k=77, seq={len(packed77.input_ids)}")


def smoke_forward(model, tok, device: str) -> None:
    rec = Record(
        id="fwd",
        group_id="fwd",
        source="smoke",
        split="train",
        state={"text": "the wifi is down"},
        question=Question(
            id="dept",
            type="choice",
            instructions="Which team should handle this?",
            criteria={"billing": "money", "technical": "bugs", "other": "else"},
        ),
        label=Label(kind="hard", choice="technical"),
    )
    packed = pack_record(tok, rec, max_length=512)
    batch = {
        "input_ids": torch.tensor([packed.input_ids], device=device),
        "attention_mask": torch.ones(1, len(packed.input_ids), dtype=torch.long, device=device),
        "type_ids": torch.tensor([packed.type_id], device=device),
        "option_positions": torch.tensor([packed.option_positions], device=device),
        "option_mask": torch.ones(1, packed.k, dtype=torch.bool, device=device),
        "targets": torch.tensor([packed.target_index], device=device),
        "teacher_probs": torch.zeros(1, packed.k, device=device),
        "is_soft": torch.tensor([False], device=device),
        "is_score": torch.tensor([False], device=device),
    }
    model.eval()
    with torch.no_grad():
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            type_ids=batch["type_ids"],
            option_positions=batch["option_positions"],
            option_mask=batch["option_mask"],
        )
    assert logits.shape == (1, 3), logits.shape
    print(f"forward ok: logits={tuple(logits.shape)} gather_ids={packed.option_positions}")


def _tiny_gold(n: int = 100) -> list[Record]:
    try:
        recs = export_banking77(cap=n, seed=0)
        recs = [r for r in recs if r.question.id == "intent"][:n]
        if len(recs) >= min(32, n):
            # Shrink to 3 options so 50-step overfit is a real go/no-go.
            import random as _rng

            r = _rng.Random(0)
            slim = []
            for rec in recs:
                gold = rec.label.choice
                others = [k for k in rec.question.criteria if k != gold]
                keep = [gold] + r.sample(others, min(2, len(others)))
                rec.question.criteria = {k: rec.question.criteria[k] for k in keep}
                slim.append(rec)
            return slim
    except Exception as exc:
        print(f"BANKING77 download skipped ({exc}); using synthetic rows")
    recs = []
    labels = ["card_arrival", "activate_my_card", "other"]
    texts = [
        "I still have not received my new card",
        "please activate the card you sent me",
        "what is the routing number for my account",
        "my plastic still has not shown up",
        "turn on the new debit card",
        "how do I change my address",
    ]
    criteria = {
        "card_arrival": "waiting for a physical card",
        "activate_my_card": "activate a card",
        "other": "anything else",
    }
    for i in range(n):
        lab = labels[i % 3]
        recs.append(
            Record(
                id=f"syn-{i:03d}",
                group_id=f"syn-{i:03d}",
                source="synthetic",
                split="train",
                state={"text": texts[i % len(texts)]},
                question=Question(
                    id="intent",
                    type="choice",
                    instructions="Which banking intent matches this customer message?",
                    criteria=criteria,
                ),
                label=Label(kind="hard", choice=lab),
                family="banking77",
            )
        )
    return recs


def smoke_overfit(model, tok, device: str, steps: int = 50) -> Path:
    recs = _tiny_gold(100)
    out_path = ROOT / "data" / "gold" / "smoke_banking77.jsonl"
    write_jsonl(out_path, recs)
    roundtrip = load_records(out_path)
    assert len(roundtrip) == len(recs)
    record_from_dict(roundtrip[0].to_dict())
    print(f"gold round-trip ok: {len(roundtrip)} rows -> {out_path}")

    from pack import collate_packed

    packed = [pack_record(tok, r, max_length=512, shuffle_options=False) for r in recs]
    pad = int(tok.pad_token_id)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"trainable tensors={len(trainable)} params={n_train/1e6:.2f}M")
    if not trainable:
        raise SystemExit("NO-GO: no trainable parameters")
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    opt = torch.optim.AdamW(trainable, lr=1e-3)
    model.train()
    losses = []
    for step in range(steps):
        chunk = packed[(step * 4) % (len(packed) - 3) : (step * 4) % (len(packed) - 3) + 4]
        if len(chunk) < 4:
            chunk = packed[:4]
        batch = collate_packed(chunk, pad_id=pad)
        feed = {
            k: batch[k].to(device)
            for k in ("input_ids", "attention_mask", "type_ids", "option_positions", "option_mask")
        }
        logits = model(**feed)
        if not logits.requires_grad:
            raise SystemExit(f"NO-GO: logits have no grad (shape={tuple(logits.shape)})")
        out = student_loss(logits, {k: batch[k].to(device) for k in batch if torch.is_tensor(batch[k])})
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        opt.step()
        losses.append(float(out["loss"].detach().cpu()))
        if step % 10 == 0 or step == steps - 1:
            print(f"  step {step:02d} loss={losses[-1]:.4f}")
    if losses[-1] >= losses[0] * 0.95:
        raise SystemExit(f"NO-GO: loss did not drop ({losses[0]:.4f} -> {losses[-1]:.4f})")
    print(f"overfit ok: {losses[0]:.4f} -> {losses[-1]:.4f}")
    ckpt = ROOT / "artifacts" / "smoke"
    model.save_pretrained(ckpt, extra={"smoke_loss": {"first": losses[0], "last": losses[-1]}})
    tok.save_pretrained(ckpt / "tokenizer")
    print(f"saved {ckpt}")
    return ckpt


def smoke_teacher(cfg_path: str) -> None:
    from teacher import build_letter_map, load_teacher, score_options
    from schema import Question

    cfg = load_yaml(cfg_path)
    print(f"loading teacher {cfg['model']} (unload the student first)")
    model, tok, model_id = load_teacher(cfg, device=_device())
    mapping = build_letter_map(tok, k_max=96)
    path = resolve_path(cfg.get("letter_map_path", "artifacts/teacher_letter_map.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    mapping["model"] = model_id
    path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    print(f"letter map: {len(mapping['letters'])} tokens -> {path}")
    print("  first 8:", list(zip(mapping["letters"][:8], mapping["token_ids"][:8])))
    q = Question(
        id="dept",
        type="choice",
        instructions="Which team should handle this customer request?",
        criteria={
            "billing": "Charges, invoices, refunds",
            "technical": "Bugs, outages, login",
            "other": "Anything else",
        },
    )
    probs = score_options(
        model,
        tok,
        mapping,
        {"text": "The app crashes every time I tap Pay."},
        q,
        temperature=float(cfg.get("temperature", 1.0)),
    )
    print("teacher probs:", {k: round(v, 4) for k, v in probs.items()})
    if max(probs.values()) - min(probs.values()) < 0.02:
        raise SystemExit("NO-GO: teacher letter logits look uniform")
    print("teacher letter logits are peaked (not uniform)")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--student-config", default="configs/student_modernbert_base.yaml")
    p.add_argument("--teacher-config", default="configs/teacher_qwen7b_4bit.yaml")
    p.add_argument("--skip-teacher", action="store_true")
    p.add_argument("--steps", type=int, default=50)
    args = p.parse_args()

    device = _device()
    print(f"device={device}")
    cfg = load_yaml(args.student_config)
    if device == "cpu":
        cfg["dtype"] = "float32"
    model, tok = build_student(cfg)
    model.to(device)
    try:
        targets = infer_lora_targets(model.backbone, cfg.get("lora", {}).get("target_modules"))
        print("lora targets:", targets)
    except Exception as exc:
        print("lora target probe:", exc)

    smoke_pack(tok)
    smoke_forward(model, tok, device)
    smoke_overfit(model, tok, device, steps=args.steps)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.skip_teacher:
        print("skip teacher")
        return
    smoke_teacher(args.teacher_config)
    print("Phase 0 GO")


if __name__ == "__main__":
    main()
