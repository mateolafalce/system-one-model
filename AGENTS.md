# AGENTS.md

Instructions for agents and contributors working in this repository.

## Product

System One is a local, open-weight **decision** model for an 8 GB GPU: JSON or text `state` in, typed `choice` / `score` / `noul` out, one forward pass, no text generation.

It is a domain specialist on short English text, not a TypeSafe Jev clone and not a chatbot. Do not add generation, tool-calling, or JSON sampling of model text.

Teacher: frozen `Qwen/Qwen2.5-7B-Instruct-AWQ` with letter-logit scoring.  
Student: `ModernBERT-base` (149M) + a scoring head. Ship `artifacts/phase2/best` with `artifacts/temps.json`.

Do not upgrade to ModernBERT-large to chase BANKING77 90%. The teacher ceiling on that task is ~56%; the student already beats it.

## Stack

- Python 3.12 via `uv` (system Python is 3.13 — do not use it)
- PyTorch + Transformers + PEFT + FastAPI
- Ampere GPU (RTX 3070, 8 GB): attention is `sdpa`, never Flash Attention 2
- Teacher load: AutoAWQ (`fuse_layers=False`, `device_map="auto"`). Do not install `gptqmodel`

```
state + typed questions
        |
        v
ModernBERT + [MASK] scoring head  (one batched forward)
        |
        v
temperature scaling
        |
        v
POST /v1/systemone  ->  choice | score | noul
```

The 8 GB card does **one job at a time**. Never keep the teacher and the student resident together.

## Layout

```
configs/          student + teacher YAML
scripts/          00_smoke … 07_serve, plus CI helpers
src/              pack, student, teacher, loss, schema, runtime
data/gold|distilled|stress/
artifacts/        checkpoints (local, not committed) and temps.json
eval/tables/      go/no-go numbers
tests/            unit tests (no GPU, no Hub downloads)
```

Sequence layout (GLiClass-style):

```
[CLS] <type> question: <instructions> [SEP]
[MASK] option_0 [MASK] option_1 ... [SEP]
<state> [SEP]
```

Cap is **512 tokens**. Truncate the state from the tail. The decision head is **fp32**; the backbone is bf16.

## Code conventions

- Keep identifiers, comments, and user-facing API fields in English.
- Do not write inline comments unless the developer asks for them.
- `student_loss` must return the live `loss` tensor. Never detach it and splat a detached copy over the real one.
- Disable nested tensor on the head `TransformerEncoder`.
- Hugging Face cache: if `~/.cache/huggingface` is not writable, `src/config.py` sets `HF_HOME` to `./.hf` and symlinks only `models--*` snapshots, never `.locks`.
- Do not download HelpSteer2, Amazon reviews, or a second Qwen copy unless disk is confirmed free.
- Re-running `scripts/01_export_gold.py --tasks X` overwrites `data/gold/manifest.json` to that task only.

## Serve

```bash
source .venv/bin/activate
python scripts/07_serve.py --ckpt artifacts/phase2/best --temperatures artifacts/temps.json --port 8010
```

Default checkpoint is `artifacts/phase2/best`. On this machine ports 8000–8003 are often taken; use `--port 8010`.

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

Support and email question bundles live in `src/templates.py`. Example payloads are under `examples/`.

## Commits and pull requests

Commit messages and pull request titles must follow [Conventional Commits](https://www.conventionalcommits.org/).

Format:

```
<type>(<scope>): <description>
```

Rules:

- `type` is required, lowercase, from the table below.
- `scope` is optional, lowercase (`pack`, `student`, `teacher`, `serve`, `gold`, `distill`, `train`, `eval`, `ci`).
- Breaking change: `!` after the type or scope.
- `description` in English, imperative mood, no trailing period.
- First line at most 100 characters.

Allowed types:

| Type | Use |
| --- | --- |
| `feat` | New behavior |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no behavior change |
| `refactor` | Code change that is neither feat nor fix |
| `perf` | Performance |
| `test` | Add or fix tests |
| `build` | Dependencies, packaging |
| `ci` | CI workflows and scripts |
| `chore` | Maintenance that fits no other type |
| `revert` | Revert a previous commit |

Valid examples:

- `feat(serve): load the phase2 checkpoint by default`
- `fix(pack): keep question tokens when truncating state`
- `docs: explain the support question bundle`
- `ci: validate conventional commits on pull requests`

Invalid examples: `Update`, `Fix bug`, `wip serve`, `feat: add runtime.`.

CI validates the PR title and each commit (merge commits are ignored). Check locally:

```
scripts/check-conventional-commits --title "feat(serve): load the phase2 checkpoint by default"
scripts/check-conventional-commits --from origin/main --to HEAD
```

## Tests

```
source .venv/bin/activate
python -m pytest
```

Re-run `python scripts/00_smoke.py` only if packing, the head, or the teacher loader changed. Smoke and train need the GPU; unit tests do not.

Before finishing a code change, run the affected tests. Do not start a train or teacher-label job if another GPU job is running (`nvidia-smi`).

Handoff for the next session lives in `plan.md` section 0.
