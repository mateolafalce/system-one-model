# System One

A local, open-weight **decision** model for an 8 GB GPU: JSON or text `state` in, typed `choice` / `score` / `noul` out, one forward pass, no text generation.

This is not a TypeSafe Jev clone. Jev is closed, uses unpublished RLCD, and claims a 64k window. This repo is a **domain System One**: a calibrated specialist on short English text. That is the project that fits 8 GB.

Teacher: frozen `Qwen/Qwen2.5-7B-Instruct-AWQ` (letter-logit scoring, never JSON sampling).  
Student: `ModernBERT-base` (149M) + a scoring head. Upgrade to `ModernBERT-large` only if base plateaus.

**Code:** https://github.com/mateolafalce/system-one-model  
**Resume:** see `plan.md` section 0 (2026-09-19). Phases 0–2 done. Ship `artifacts/phase2/best` + `artifacts/temps.json`. Test: BANKING77 88.5%, SMS 98.9%, SST-5 55.9% (ECE 0.033). Teacher ceiling on BANKING77 is 56%. Serve: `python scripts/07_serve.py --ckpt artifacts/phase2/best --port 8010`. Agent conventions: `AGENTS.md`.

## Hardware split

The 8 GB card does **one job at a time**. Do not keep Qwen and the student resident together.

| Job | Model | Precision | Fits 8 GB? |
|---|---|---|---|
| Teacher labeling | Qwen2.5-7B-Instruct AWQ | 4-bit | Yes, inference only |
| Student train | ModernBERT-base + LoRA r=16, seq 512, batch 4 | bf16 | Yes |
| Student train (tight) | ModernBERT-large + LoRA, batch 2, grad checkpoint | bf16 | Tight, yes |
| Student serve | either checkpoint, seq 512, batch 8 | fp16/bf16 | Yes |

Default teacher is the cached AWQ 7B. If that OOMs, drop to `Qwen2.5-3B-Instruct` FP16. Do not use a 0.5B teacher.

## Layout

```
configs/          student + teacher YAML
scripts/          00_smoke … 07_serve
src/              pack, student, teacher, loss, schema
data/gold|distilled|stress|splits/
eval/tables/
```

Sequence layout (same idea as Laya / GLiClass):

```
[CLS] <type> question: <instructions> [SEP]
[MASK] option_0 [MASK] option_1 ... [SEP]
<state> [SEP]
```

The head reads the hidden state at each `[MASK]`, projects to one logit, softmaxes inside that question. `noul` is `{true, false}`. `score` is ordered levels, then `sum(i * p_i)`. Confidence is `1 - H(p) / log(K)`, not a learned head. Cap is **512 tokens**; the state is truncated from the tail.

## Setup (Python 3.12, CUDA)

System Python here is 3.13; the ML stack is pinned to 3.12.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -e ".[dev]"
# teacher scoring (AWQ):
uv pip install autoawq
```

Ampere (RTX 3070) uses `sdpa`. Skip Flash Attention 2 unless you already have it.

If `~/.cache/huggingface` is not writable (this machine: root-owned), the code falls back to `./.hf` and symlinks existing Hub snapshots. Teacher weights load from the local Qwen AWQ snapshot via AutoAWQ.

## First command sequence

```bash
source .venv/bin/activate

python scripts/00_smoke.py
python scripts/01_export_gold.py --preset proof --out data/gold
python scripts/04_train.py --config configs/student_modernbert_base.yaml --data data/gold --phase 1

# unload student, then label:
python scripts/02_label_teacher.py --config configs/teacher_qwen7b_4bit.yaml --out data/distilled
python scripts/03_augment_stress.py --data data/gold,data/distilled --out data/stress/stress.jsonl

python scripts/04_train.py --config configs/student_modernbert_base.yaml \
  --data data/gold,data/distilled,data/stress --phase 2 --init artifacts/phase1/best \
  --output artifacts/phase2

python scripts/05_fit_temperature.py --ckpt artifacts/phase2/best --split val
python scripts/06_eval.py --ckpt artifacts/phase2/best --temperatures artifacts/temps.json
python scripts/07_serve.py --ckpt artifacts/phase2/best --temperatures artifacts/temps.json --port 8010
```

`--preset proof` is BANKING77 + SMS spam + SST-5 (two-day proof). `--preset full` adds HelpSteer2, CLINC, the support/email teacher bundles, and the rest of section 4.1.

## Runtime

```
POST /v1/systemone
{
  "state": {"text": "..."},
  "questions": {
    "dept": {"type": "choice", "instructions": "...", "criteria": {...}},
    "urgent": {"type": "score", "instructions": "...", "criteria": ["...", "..."]},
    "refund": {"type": "noul", "instructions": "The customer asks for a refund."}
  }
}
```

No generation. No JSON parsing of model text. Pack every question against the same state, one batched forward, divide logits by the fitted temperature, return.

On this machine ports 8000–8003 are often taken; the script defaults to **8010**. Example payload: `examples/support.json`.

```bash
curl -sS http://127.0.0.1:8010/health
curl -sS -X POST http://127.0.0.1:8010/v1/systemone \
  -H 'content-type: application/json' \
  --data @examples/support.json
```

## Conventions

See `AGENTS.md` for product, stack, hardware split, and Conventional Commits. PR title and commits are checked in CI (`scripts/check-conventional-commits`).

## Go / no-go

Ship the base student if:

1. BANKING77 test accuracy ≥ 90%, ECE ≤ 0.08
2. Enron or SMS spam accuracy ≥ 95%, ECE ≤ 0.06
3. SST-5 accuracy ≥ 52%
4. At 50% coverage, BANKING77 accuracy ≥ 94%
5. 10-question batch at 512 tokens p95 ≤ 80 ms
6. Option shuffle on val drops accuracy by less than 2 points

If (1) fails and the teacher itself is below 90% on BANKING77 with constrained decoding, the teacher is the bottleneck. If the teacher is above 92%, the packing or `[MASK]` gather is wrong.

## Limits

- 512 tokens. Longer tickets are cropped in code.
- English only.
- Specialist, not Jev. New templates need more gold or more teacher labels.
- No arithmetic. Amounts, dates, and counts stay in Python.
- Distilled share stays ≤ 40%. Human gold (spam, BANKING77, SST-5) pins the student to reality.
