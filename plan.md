# System One from scratch on an 8 GB GPU

Plan for a local, open-weight decision model: state in, typed `choice` / `score` / `noul` out, one forward pass, no text generation. A local Qwen checkpoint is the teacher. The student is a small encoder with a scoring head. English only.

This is not a clone of TypeSafe Jev. Jev is closed, uses unpublished RLCD, and has a 64k context window. This plan builds a **domain System One**: a calibrated specialist that answers a fixed family of questions on short English text. That is the project that fits 8 GB.

## 0. Status — resume here (2026-09-19)

Handoff for the next session. The original spec below is still the source of truth. This section is what already happened on this machine.

**Where you are.** Phases 0–2 + temperature + proof-gold test eval are **done**. Ship checkpoint: `artifacts/phase2/best` (step 1400) with `artifacts/temps.json`. Serve is `python scripts/07_serve.py --ckpt artifacts/phase2/best --port 8010` (`POST /v1/systemone`). Repo conventions live in `AGENTS.md` (same pattern as MetodologiasAgiles PR #5: Conventional Commits, PR template, CI check). Next: 20-ticket sanity set and distilled-template eval on the 225 holdout states. Do not resume Phase 1/2. Do not upgrade to ModernBERT-large to chase BANKING77 90% — the teacher ceiling is 56%.

### Done

| Item | Evidence |
|---|---|
| Repo layout, configs, scripts `00`–`07`, unit tests | `src/`, `scripts/`, `tests/` (21 passed) |
| Python 3.12 venv + CUDA torch | `.venv` (`Python 3.12.13`). Do not use system 3.13. |
| Phase 0 smoke | packing k=3 and k=77 in 512; `[MASK]` gather; 100-row JSONL round-trip; overfit 1.09 → 0.21 in 50 steps |
| Teacher letter map | `artifacts/teacher_letter_map.json` (96 single-token ids, `A=32` …). Frozen. Do not regenerate unless the tokenizer changes. |
| Teacher constrained scoring | Qwen AWQ on “The app crashes every time I tap Pay.” → `{technical: 1.0, billing: 0, other: 0}` |
| Proof gold JSONL | `data/gold/`: BANKING77 train 19986 / test 6152 (intent + coarse); SMS train 5016 / val 279 / test 279; SST-5 train 8544 / val 1101 / test 2210 |
| Smoke student checkpoint | `artifacts/smoke/` (overfit toy, not for eval) |
| Phase 1 train | `artifacts/phase1/best` at step 2000. 2 epochs, ~35 min, 2.5 GB VRAM. Log: `artifacts/phase1/train.log`. |
| Phase 1 val (carved BANKING77 + official SMS/SST-5) | macro 0.833 · B77 0.856 · coarse 0.932 · SMS 0.978 · SST-5 0.565 · ECE 0.077 |
| Phase 1 **test** (`scripts/06_eval.py`) | `eval/tables/phase1.json` |
| Teacher ceiling | `eval/tables/teacher_ceiling.json` — Qwen 56.2% / 60.2% vs student 88% / 95% |
| Distill JSONL | `data/distilled/teacher.jsonl` 13480 pairs (support 10760, email 2720), 225 holdout states |
| Stress JSONL | `data/stress/stress.jsonl` 4756 rows (train/val only) |
| Phase 2 train | `artifacts/phase2/best` step 1400, 1 epoch, mix 1:1 hard:soft. Mid-run B77 dipped to 0.806 then recovered. |
| Temperatures | `artifacts/temps.json` — noul K=2 T=0.6; score K=3–5 T=0.7 (no choice val to fit 77-way T) |
| Phase 2 **test** | `eval/tables/phase2.json` |
| Latency | `eval/tables/latency.json` — 10-q support/email bundle p95 **32.5 ms** (PASS). 10× BANKING77-77way p95 133 ms (pack-bound). |
| Serve | `scripts/07_serve.py --ckpt artifacts/phase2/best --port 8010`. `GET /health` → `{"ok": true, "device": "cuda"}`. ~613 MiB VRAM. |
| Repo conventions | `AGENTS.md`, `scripts/check-conventional-commits`, `.github/workflows/conventional-commits.yml`, PR template. GitHub: https://github.com/mateolafalce/system-one-model (`main`). |

Phase 1 vs Phase 2 **test** go/no-go (section 11):

| Check | Phase 1 | Phase 2 | Result |
|---|---|---|---|
| BANKING77 acc ≥ 90% | 88.0% | **88.5%** | FAIL (1.5 pt short; teacher ceiling 56%) |
| BANKING77 ECE ≤ 0.08 | 0.059 | 0.059 | PASS |
| BANKING77 acc@50% cov ≥ 94% | 98.9% | 99.0% | PASS |
| SMS acc ≥ 95% | 99.3% | 98.9% | PASS |
| SMS ECE ≤ 0.06 | 0.026 | **0.009** | PASS (T=0.6) |
| SST-5 acc ≥ 52% | 57.5% | 55.9% | PASS |
| SST-5 ECE | 0.152 | **0.033** | (T=0.7; not a ship gate) |
| 10-q batch p95 ≤ 80 ms | — | **32.5 ms** bundle / 133 ms 77-way | PASS on product bundle |

88% on 77-way is not “far below” 90%. Packing/`[MASK]` is working (coarse 95.4%, acc@50 98.9%).

Teacher ceiling (constrained letter logits, 1000 BANKING77 test rows, compact 77-way options):

| Family | Teacher acc | Student Phase 1 test |
|---|---|---|
| banking77 (77-way) | **56.2%** (n=500) | **88.0%** |
| banking77_coarse (8-way) | **60.2%** (n=500) | **95.4%** |

The student **beats** the teacher on gold, as section 11 allows. The 90% miss is a teacher-ceiling issue, not a packing bug. Do not scale the student first. Distill is for **new questions** (support/email bundles) on gold states, not for replacing BANKING77 labels. Evidence: `eval/tables/teacher_ceiling.json`.

### Not done

- Distilled-template eval on the 225 holdout states (compare student vs a fresh Qwen pass). Imitation, not gold.
- 20 hand-written English tickets as a sanity set (one example payload is `examples/support.json`). Live POST on four tickets: `dept`/`tone`/`refund`/`cancel` look sane; `phish` noul is biased high (0.54–0.94 on non-phish mail). Treat `phish` as untrusted until the holdout eval.
- `--preset full` gold (HelpSteer2, CLINC, Bitext, …) has **not** been exported. Disk was cleaned 2026-09-19 (`/home` ~83 GB free); still do not download until you mean to train on it.
- Option-shuffle drop on val not measured as a standalone table (stress rows were in Phase 2 train).
- No GitHub remote yet (local git only).

### Next session: exact commands

```bash
cd /home/linesserver/proyectos/system-one
source .venv/bin/activate
nvidia-smi   # student serve only; do not load Qwen at the same time

python scripts/07_serve.py --ckpt artifacts/phase2/best --temperatures artifacts/temps.json --port 8010
curl -sS http://127.0.0.1:8010/health
curl -sS -X POST http://127.0.0.1:8010/v1/systemone \
  -H 'content-type: application/json' --data @examples/support.json
```

Two-day proof is BANKING77 + SMS + SST-5, already exported. Add HelpSteer2 and the support bundle (`--preset full`) **before** calling it a System One, but only after freeing disk (see below).

Re-run smoke only if packing, the head, or the teacher loader changed:

```bash
python scripts/00_smoke.py
python -m pytest
```

### Machine facts the next session must not re-discover

- **GPU:** NVIDIA GeForce RTX 3070, 8192 MiB, compute 8.6 (Ampere). Use `sdpa`, not Flash Attention 2.
- **Disk:** `/home` was cleaned 2026-09-19 (~83 GB free of 128 GB). Teacher AWQ lives at `proyectos/system-one/.hf/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ`. Do not copy it. Still skip extra Hub dumps until you mean to train on them.
- **HF cache:** `~/.cache/huggingface` is **root-owned, not writable**. `src/config.py:ensure_hf_home()` sets `HF_HOME` to `./.hf` and symlinks only `models--*` snapshots. New downloads (ModernBERT, datasets) go to `.hf/`.
- **Teacher weights:** `Qwen/Qwen2.5-7B-Instruct-AWQ` snapshot  
  `/home/linesserver/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct-AWQ/snapshots/b25037543e9394b818fdfca67ab2a00ecc7dd641`  
  Load via **AutoAWQ** (`awq.AutoAWQForCausalLM.from_quantized`, `device_map="auto"`, `fuse_layers=False`), resolved by `teacher.local_snapshot()`. `device_map="cuda"` is invalid for AutoAWQ.
- **Do not install `gptqmodel`.** Transformers 5 wants it for AWQ, but installing it (a) replaced torch with `2.14.0+cu130` and (b) made PEFT require `optimum>=1.24`, which **broke LoRA on ModernBERT**. It was uninstalled. AutoAWQ 0.2.9 works. Current torch is still `2.14.0+cu130` (CUDA works). Leave it.
- **`datasets` 5.x** dropped script-based Hub datasets. BANKING77 still loads (parquet). Label feature is a plain int, no `ClassLabel.names` — names are hardcoded as `BANKING77_LABELS` in `src/gold.py`.
- **Head dtype:** decision head is **fp32**; backbone is bf16. The fused bf16 `TransformerEncoder` path did not record autograd.
- **Loss API:** `student_loss` must return the live `loss` tensor. Do not put a detached `loss` into the same dict and splat it over the real one (that was the Phase 0 backward bug).
- **Smoke overfit:** 50 steps is enough for 3-way BANKING77 (gold label + 2 distractors), not for 77-way or 8-way coarse. `scripts/00_smoke.py` collapses intent to 3 options on purpose.
- **Gold manifest:** `data/gold/manifest.json` lists banking77 + sms + sst5. Re-running `01_export_gold.py --tasks X` overwrites the manifest to that task only; the other jsonl files stay.
- **One GPU job at a time.** Unload the student before `02_label_teacher.py`. Unload Qwen before Phase 1/2 train.

### Code map (what to open)

| Path | Role |
|---|---|
| `src/pack.py` | GLiClass-style sequence, `[MASK]` per option, 512 cap, state truncated from the tail |
| `src/student.py` | ModernBERT + LoRA on `Wqkv`,`Wi`,`Wo` + 2-layer fp32 head |
| `src/teacher.py` | letter-logit scoring, AutoAWQ load, frozen letter map |
| `src/loss.py` | CE, KL(teacher ‖ student), RPS for score |
| `src/gold.py` | public dataset exporters; `--preset proof` = banking77, sms, sst5 |
| `src/templates.py` | frozen support/email bundles; human paraphrases |
| `configs/student_modernbert_base.yaml` | seq 512, batch 4, accum 8, LoRA r=16 |
| `configs/teacher_qwen7b_4bit.yaml` | AWQ 7B, T=1.0 |

### Go / no-go still ahead

Phase 1 val on BANKING77 test / SMS test / SST-5 test. Ship numbers are unchanged (section 11). If BANKING77 test is far below 90% after Phase 1, check packing/`[MASK]` gather before adding data. Teacher ceiling on BANKING77 with constrained decoding has not been measured yet (only the 3-way smoke example).

## 1. Goal and non-goals

**Goal.** After this plan, you can load a ~150M to ~400M student, pass a JSON or text `state` plus a set of typed questions, and get:

| Primitive | Returns |
|---|---|
| `choice` | winning option, probability per option, confidence |
| `score` | expected position on an ordered rubric, level distribution, confidence |
| `noul` | P(true) in `[0, 1]` |

Latency target on the 8 GB GPU: under 50 ms for a batch of 8 questions at 512 tokens.

**Non-goals.**

- Matching Jev on arbitrary new questions (zero-shot generalist).
- 32k or 64k context.
- Generating replies, tool arguments, or summaries.
- Training the teacher. The teacher is frozen.
- Loading teacher and student on the GPU at the same time.

## 2. Hardware split

Do not keep Qwen and the student resident together. The 8 GB card does **one job at a time**.

| Job | Model | Precision | Approx. VRAM | Fits 8 GB? |
|---|---|---|---|---|
| Teacher labeling (offline) | `Qwen2.5-7B-Instruct` | 4-bit (AWQ or Q4_K_M) | 5.0 to 6.0 GB | Yes, inference only |
| Teacher labeling (safer) | `Qwen2.5-3B-Instruct` | FP16 | ~6.0 GB | Yes, more headroom |
| Student train, first pass | `ModernBERT-base` 149M + head | LoRA r=16, seq 512, batch 4 | 4 to 6 GB | Yes |
| Student train, second pass | `ModernBERT-large` 395M + head | LoRA r=16, seq 512, batch 2, grad checkpoint | 6 to 7.5 GB | Tight, yes |
| Student inference | either checkpoint, FP16 | seq 512, batch 8 | ~1 to 2 GB | Yes, including CPU fallback |

Default teacher: **`Qwen2.5-7B-Instruct` in 4-bit**. If the 7B 4-bit run OOMs during long batches, drop to `Qwen2.5-3B-Instruct` FP16. Do not use a 0.5B teacher. The student cannot exceed the teacher.

Serve the teacher with vLLM (AWQ) or llama.cpp / Ollama (Q4_K_M). Unload it before training.

## 3. Architecture

Two processes, one GPU, never overlapping.

```
unlabeled English text + public labeled datasets
                 |
                 v
     Qwen teacher (frozen, 4-bit)
     constrained option scoring
                 |
                 v
     records: (state, questions, soft or hard labels)
                 |
                 v
     student: ModernBERT + decision head
     one forward pass, softmax over option markers
                 |
                 v
     temperature scaling on a held-out set
                 |
                 v
     System One runtime (FastAPI, Jev-shaped JSON)
```

### 3.1 Student

Backbone: [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base) (149M) for the first training run. Upgrade to [`answerdotai/ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) (395M) only if the base model plateaus and VRAM still has room.

Why an encoder, not a small Qwen: bidirectional attention over state and options in one pass, no decoding, ~400M is the right size for 8 GB training, Apache-2.0.

Sequence layout (same idea as Laya / GLiClass):

```
[CLS] <type> question: <instructions> [SEP]
[MASK] option_0_text [MASK] option_1_text ... [SEP]
<state text or JSON> [SEP]
```

- Each option has its own `[MASK]` marker.
- The head reads the hidden state at each marker, projects to one logit, softmaxes inside that question.
- `noul` is two options: `true` and `false`. P(true) is the `true` mass after softmax.
- `score` uses ordered levels as options, then expected value `sum(i * p_i)`.
- Cap: **512 tokens** per question (question + options + state). Truncate the state from the tail, keep the question and options intact.
- Multi-question calls: collate N independent sequences into one batch. One forward pass.

Decision head (train from scratch, ~5 to 25M params):

1. Type embedding (`choice=0`, `score=1`, `noul=2`) added to token 0.
2. Two TransformerEncoder layers, `d=768` (base) or `d=1024` (large).
3. Option scorer: LayerNorm, Linear, GELU, Linear to 1.
4. Confidence is not a learned head. Compute it from the distribution: `1 - H(p) / log(K)`.

### 3.2 Teacher scoring (do not sample JSON)

Do not ask Qwen to write `{"choice": "..."}`. That is slow, uncalibrated, and schema-fragile. Score options as **single-token continuations**.

For a `choice` with options `{billing, technical, other}`:

1. Build a chat prompt that ends with `Answer:` and a letter map (`A billing`, `B technical`, `C other`).
2. Prefill. Read the logits of the tokens `A`, `B`, `C` (or the first token of each option name if you skip letters).
3. Softmax those K logits. That vector is the teacher distribution.
4. Store the full vector, not only argmax.

Same pattern for `noul` (`A yes` / `B no`) and `score` (one letter per rubric level).

If a letter is not a single token in the Qwen tokenizer, use the first subtoken and document the mapping. Test this once and freeze it.

Temperature of the teacher softmax: start at `1.0`. If the teacher is over-peaked (almost one-hot on easy items), raise it to `1.5` before you write labels. Do not train the student on one-hot teacher output. The whole point of the teacher is the **soft** distribution.

## 4. Data mix

Three sources. Do not train only on teacher labels. Public datasets already have human gold. The teacher fills gaps: extra questions on the same state, paraphrased instructions, unlabeled tickets and emails.

| Bucket | Share of training tokens | Label type | Role |
|---|---|---|---|
| A. Gold public datasets | 55% | hard (human) | accuracy on known tasks |
| B. Teacher distillation | 35% | soft (Qwen logits) | instruction following, extra questions, unlabeled text |
| C. Stress / invariance | 10% | hard or soft | option order, paraphrases, distractor JSON, truncation |

Target size after packing: **80k to 150k question-state pairs**. That is enough for a 150M to 400M student. More than ~200k has diminishing returns on 8 GB compute.

Split: 90 / 5 / 5 train / val / test, stratified by task family. Hold out **entire task families** for zero-shot (section 7), not random rows from the same family.

English only. Drop non-English rows (MASSIVE has many languages; keep `en-US`).

### 4.1 Gold datasets by primitive

Use the Hugging Face ids as written. Cap each dataset so no single source dominates (suggested caps in the last column). Shuffle option order at train time even for gold.

#### Choice (unordered closed set)

| Dataset | HF id | What it teaches | Size (use) | License (check card) |
|---|---|---|---|---|
| BANKING77 | `PolyAI/banking77` | fine-grained support intent, 77 labels | 13k, use all | CC-BY-4.0 |
| CLINC150 + OOS | `clinc_oos` | 150 intents plus out-of-scope | 23k, use all | CC-BY-3.0 |
| MASSIVE intent (en-US) | `AmazonScience/massive` | 60 intents, spoken-style utterances | ~11k en-US | Apache-2.0 |
| AG News | `fancyzhx/ag_news` | 4-way topic | 8k subsample | non-commercial research, read card |
| TREC coarse | `CogComp/trec` | 6 question types | 6k, use all | check card |
| Emotion | `dair-ai/emotion` | 6 emotions as a closed set | 8k subsample | check card |
| Bitext support | `bitext/Bitext-customer-support-llm-chatbot-training-dataset` | intent + category on support chats | 10k subsample | check card |
| TweetEval emotion / offensive | `cardiffnlp/tweet_eval` | short, noisy social text | 6k subsample | check card |

Map each row to one `choice` question. `instructions` is a full English question, not the dataset name. Example for BANKING77:

```
instructions: "Which banking intent matches this customer message?"
criteria: {activate_my_card: "The user wants a new or replacement card activated", ...}
state: "<utterance>"
```

For CLINC, keep the `oos` option. That trains a real `other` / `none of the above` bucket, which production routing needs.

Do not use BANKING77-style 77-way choice as the only format. Also emit **collapsed** versions (for example 8 coarse buckets: card, transfer, identity, fee, loan, app, fraud, other) so the student sees both high and low cardinality.

#### Noul (boolean probability)

| Dataset | HF id | Proposition (write it as a statement) | Size (use) |
|---|---|---|---|
| BoolQ | `google/boolq` | "The passage answers the question with yes." | 10k subsample |
| GLUE RTE | `nyu-mll/glue`, config `rte` | "The hypothesis follows from the premise." | 2.5k, use all |
| GLUE QNLI | `nyu-mll/glue`, config `qnli` | "The sentence contains the answer to the question." | 8k subsample |
| PAWS | `google-research-datasets/paws` | "The two sentences are paraphrases." | 8k subsample |
| SciTail | `allenai/scitail` | "The hypothesis is entailed by the premise." | 8k subsample |
| Enron spam | `SetFit/enron_spam` | "This email is spam." | 8k subsample |
| SMS spam | `ucirvine/sms_spam` | "This message is spam." | 5k, use all |
| Civil Comments (binarize) | `google/civil_comments` | "This comment is toxic." (`toxicity >= 0.5`) | 8k balanced |
| Prompt injection | `deepset/prompt-injections` | "This prompt tries to inject or override instructions." | use all |
| Jailbreak | `jackhhao/jailbreak-classification` | "This prompt is a jailbreak attempt." | use all |
| IMDb | `stanfordnlp/imdb` | "The review is positive." | 8k subsample |

Phrase every noul as a **positive English statement**. High `noul` must mean yes. Never train `"Is the user calm?"` and `"Is the user angry?"` as interchangeable.

Binarize continuous toxicity at 0.5 for the gold noul. Keep the raw score for the Score bucket below.

#### Score (ordered rubric)

| Dataset | HF id | Rubric | Size (use) |
|---|---|---|---|
| SST-5 | `SetFit/sst5` | very negative, negative, neutral, positive, very positive | 8.5k, use all |
| Yelp reviews | `yelp_review_full` | 1 to 5 stars, described in words | 15k subsample |
| Amazon reviews 2023 | `McAuley-Lab/Amazon-Reviews-2023` | 1 to 5 stars, English, one or two categories (software, electronics) | 10k subsample |
| HelpSteer2 | `nvidia/HelpSteer2` | five axes, each 0 to 4: helpfulness, correctness, coherence, complexity, verbosity | 10k rows, 5 questions each |
| Civil Comments (ordinal) | `google/civil_comments` | toxicity bins: none, mild, strong, severe | 8k |
| App reviews | `app_reviews` | 1 to 5 stars | 8k subsample |

HelpSteer2 is the most important score source. It is the only public set here that looks like production rubrics (several independent scores on the same state). Emit **five** `score` questions per row, one per axis.

Write rubric `criteria` as short English descriptions, not `"1".."5"`:

```
criteria: [
  "One star. The reviewer is fully negative.",
  "Two stars. Major complaints, little praise.",
  "Three stars. Mixed, neither strong praise nor strong attack.",
  "Four stars. Mostly positive with a small complaint.",
  "Five stars. Fully positive."
]
```

The student then learns distance on the rubric. A predicted 3.4 is meaningful.

### 4.2 Teacher distillation set (bucket B)

Gold datasets do not teach "new question, same state". That is the System One product. Build it with Qwen.

**Unlabeled English states** (sample, truncate to ~300 tokens):

| Source | HF id / origin | State shape |
|---|---|---|
| Bitext support (text only, ignore labels for this bucket) | `bitext/Bitext-customer-support-llm-chatbot-training-dataset` | chat / ticket |
| MultiWOZ 2.2 | `pfb30/multi_woz_2_2` or `ConvLab/multiwoz21` | dialogue turns as JSON |
| Enron (body + subject) | `SetFit/enron_spam` (use all mail, not only spam labels) | email JSON |
| AG News + BoolQ passages | reuse from gold | article / paragraph |
| HelpSteer2 prompts | reuse | instruction + response as JSON |

For each state, ask Qwen a **bundle of 4 to 8 questions**, mixed primitives, written by you (templates, not model-generated questions). Examples of a support bundle:

- choice: department (`billing`, `technical`, `sales`, `other`)
- choice: channel tone (`request`, `complaint`, `question`, `praise`)
- score: urgency (three levels)
- score: frustration (three levels)
- noul: asks for a refund
- noul: reports a bug or outage
- noul: threatens to cancel
- noul: looks like phishing or social engineering

Examples of an email bundle:

- choice: type (`transactional`, `newsletter`, `personal`, `spam`, `phishing`)
- noul: contains a call to action
- noul: requests money or credentials
- score: professionalism (three levels)

Store teacher softmax vectors. Cap at ~40k distilled pairs.

**Do not** let Qwen invent the question text. Frozen templates prevent the student from overfitting to Qwen's phrasing.

### 4.3 Stress set (bucket C)

Apply these transforms to 10% of A+B. Keep the original label or teacher vector.

1. **Option shuffle.** Permute `criteria` keys. The answer must follow the option, not the position.
2. **Instruction paraphrase.** 3 human-written paraphrases per template (not Qwen-written, to avoid style collapse). Example: `"Which team should handle this?"` / `"Route this ticket to a department."` / `"Who owns this request?"`
3. **JSON vs raw.** Same content as a string and as a nested object (`{"from":..., "subject":..., "body":...}`).
4. **Distractor fields.** Add unused JSON keys (`internal_id`, `random_note`) so the student learns to read the field named in `instructions`.
5. **Hard truncation.** Cut the state at 128 and 256 tokens. Labels stay the same. This teaches graceful degradation, not 8k magic.
6. **None-of-the-above.** For a subset of BANKING77 / CLINC, replace the true label with an `other` option and keep a sibling row where `other` is correct (CLINC OOS).

### 4.4 What you are not going to download

- Synthetic questions generated by the teacher about the teacher. That calibrates Qwen to Qwen.
- Multilingual corpora. This plan is English.
- Long-document QA (SQuAD full context, Hotpot). 512 tokens will mutilate them.
- Math, code execution, or date arithmetic. Those stay in deterministic code. System One is a bad calculator, same as Jev.

## 5. Record format

One JSONL line per question, even if several questions share a state. Keep a `group_id` so you can rebuild multi-question batches.

```json
{
  "id": "banking77-train-00412-intent",
  "group_id": "banking77-train-00412",
  "source": "PolyAI/banking77",
  "split": "train",
  "state": {"text": "I still have not received my new card"},
  "question": {
    "id": "intent",
    "type": "choice",
    "instructions": "Which banking intent matches this customer message?",
    "criteria": {
      "card_arrival": "The user is waiting for a physical card to arrive",
      "activate_my_card": "The user wants to activate a card",
      "other": "Anything else"
    }
  },
  "label": {
    "kind": "hard",
    "choice": "card_arrival"
  },
  "teacher": {
    "model": "Qwen2.5-7B-Instruct",
    "probs": {"card_arrival": 0.81, "activate_my_card": 0.11, "other": 0.08}
  }
}
```

For gold rows, `label.kind` is `hard` and `teacher` is optional (you can still score Qwen for analysis). For bucket B, `label.kind` is `soft` and the train target is `teacher.probs`.

For `score`, store `label.score` as an integer level index and, if you have it, a soft vector. For `noul`, store `label.noul` as `0` or `1` (gold) or a float (teacher).

## 6. Training recipe (8 GB)

Environment: Python venv, PyTorch 2.x, `transformers`, `peft`, `bitsandbytes` (teacher only), Flash Attention 2 if the GPU supports it (Ampere or newer). If the 8 GB card is Turing (1660 / 2060), skip FA2 and use `sdpa`.

### Phase 0. Repo and smoke (1 day)

1. Create the venv. Pin CUDA builds.
2. Load ModernBERT-base, run a dummy forward of the packed sequence, confirm `[MASK]` gather.
3. Load Qwen 4-bit, score a 3-option choice, print the letter-token ids. Save that mapping.
4. Convert BANKING77 (100 rows) to JSONL. Round-trip.

Go/no-go: dummy student loss decreases on those 100 rows in 50 steps. Teacher letter logits are not uniform.

### Phase 1. Gold supervised (2 to 3 days)

- Model: ModernBERT-base + head, LoRA r=16 on `Wqkv`, `Wi`, `Wo` (ModernBERT names) plus full train of the head.
- Seq: 512. Batch: 4. Grad accum: 8. Effective batch 32.
- LR: 2e-4 LoRA, 1e-4 head, cosine, 500 warmup steps.
- Epochs: 2 on bucket A + C (gold and stress). No teacher yet.
- Loss:
  - `choice` / `noul`: cross-entropy on gold.
  - `score`: cross-entropy on the level **plus** 0.5 * ranked probability score (RPS) so off-by-one is cheaper than off-by-three.
- Max steps: whatever 2 epochs are. Expect a few hours, not days.

Checkpoint by val macro-accuracy and val ECE, not train loss.

### Phase 2. Distill (2 to 4 days, GPU time dominated by labeling)

1. Unload the student. Load Qwen. Label bucket B. Write JSONL. Unload Qwen.
2. Load the Phase 1 adapter. Train 1 epoch on A+B+C.
3. Loss on soft rows: `KL(teacher_probs || student_probs)` at temperature 1.0. On hard rows: same as Phase 1.
4. Mix each batch 1:1 hard:soft so the teacher cannot wash out gold.

If KL makes the student too smooth (val accuracy drops more than 2 points), add `0.3 * CE(student, teacher_argmax)` as an auxiliary term.

### Phase 3. Upgrade backbone (optional, 1 to 2 days)

If Phase 2 val accuracy on BANKING77 is under 90% or HelpSteer2 Spearman is under 0.45, repeat Phase 1+2 on ModernBERT-large. Lower batch to 2, enable gradient checkpointing, LoRA r=16 still. Do not full-finetune large on 8 GB.

### Phase 4. Calibration (half a day, CPU is enough)

On the validation split, **do not train weights**. Fit one temperature per primitive, and optionally per cardinality `K` for choice:

`T_hat = argmin_T ECE(softmax(logits / T), labels)`

Laya published per-cardinality temperatures. Copy that idea: `K=2`, `K=3-5`, `K=6-15`, `K>15`.

Save `{type, k_bucket -> T}` next to the weights. Inference always divides logits by T before softmax.

Act/escalate rule after calibration, measured on val:

- `choice` / `score`: auto-act if `confidence >= t`
- `noul`: auto-act if `noul >= t_yes` or `noul <= t_no`

Sweep `t` on val. Report accuracy **at 50% coverage** and **at 80% coverage**. Those two numbers are the product. Raw accuracy is not.

## 7. Evaluation

Run four tables every checkpoint. Do not tune on the test split.

### In-task (trained families)

| Slice | Metric |
|---|---|
| BANKING77 test | accuracy, ECE, Brier |
| CLINC150 in-scope / OOS | accuracy, OOS recall |
| SST-5 | accuracy, QWK or Spearman |
| HelpSteer2 helpfulness | Spearman, ECE after binning |
| BoolQ | accuracy, ECE |
| Enron spam | accuracy, ECE, F1 |
| Prompt injection | accuracy, ECE, FPR at 95% TPR |

### Distilled templates

The support bundle and email bundle from section 4.2, on **states the teacher did not label** (hold out 10% of unlabeled states before distillation). Compare student argmax to a fresh Qwen pass. This measures imitation, not truth. Keep it, but do not treat it as gold.

### Zero-shot families (never in train)

Hold out entirely:

- `dair-ai/emotion` **or** TweetEval emotion (pick one for train, the other for zero-shot)
- GLUE CoLA (`cola`) as noul: "This sentence is grammatical."
- A HelpSteer2 axis you did **not** train (if you drop verbosity from train, eval it here)

Expect a drop. Laya-class models fall from ~84% in-task to ~65% zero-shot. If zero-shot is within 5 points of in-task, you leaked the family.

### Selective automation

For each slice, plot risk-coverage. Ship only if, at 50% coverage, in-task accuracy is >= 90% on BANKING77 and ECE <= 0.06 on noul spam/injection.

### Teacher ceiling

On the same test rows, score Qwen with the same constrained decoder. The student should land within ~3 to 8 points of the teacher on distilled templates, and can **beat** the teacher on gold datasets (those have human labels). If the student is far below Qwen on gold BANKING77, the architecture or packing is wrong, not the data.

## 8. Runtime

Small FastAPI service, one GPU worker.

```
POST /v1/systemone
{
  "state": {"text": "..."} or any JSON,
  "questions": {
    "dept": {"type": "choice", "instructions": "...", "criteria": {...}},
    "urgent": {"type": "score", "instructions": "...", "criteria": ["...", "...", "..."]},
    "refund": {"type": "noul", "instructions": "The customer asks for a refund."}
  }
}
```

Response shape aligned with TypeSafe so you can swap later:

```
{
  "answers": {
    "dept": {"type": "choice", "choice": "billing", "probabilities": {...}, "confidence": 0.84},
    "urgent": {"type": "score", "score": 1.6, "probabilities": [...], "confidence": 0.71},
    "refund": {"type": "noul", "noul": 0.92}
  }
}
```

Pack every question against the same serialized state, batch, one forward, apply temperatures, return. No generation. No JSON parsing of model text.

## 9. Suggested repo layout

```
system-one/
  README.md
  configs/
    teacher_qwen7b_4bit.yaml
    student_modernbert_base.yaml
    student_modernbert_large.yaml
  scripts/
    00_smoke.py
    01_export_gold.py
    02_label_teacher.py
    03_augment_stress.py
    04_train.py
    05_fit_temperature.py
    06_eval.py
    07_serve.py
  src/
    pack.py          # sequence + mask indices
    student.py       # ModernBERT + head
    teacher.py       # letter-logit scoring
    loss.py          # CE, KL, RPS
    schema.py        # choice / score / noul records
  data/
    gold/  distilled/  stress/  splits/
  eval/
    tables/
```

## 10. Calendar (one person, evenings, 8 GB)

| Days | Output | Status (2026-09-19) |
|---|---|---|
| 1 | Smoke: packing, teacher letter ids, 100-row overfit | **Done.** Phase 0 GO. See section 0. |
| 2 to 3 | Export gold JSONL (section 4.1), train Phase 1 | **Done.** Proof gold + real Phase 1. Test: B77 88.0% / SMS 99.3% / SST-5 57.5%. Full gold still blocked on disk. |
| 4 to 6 | Teacher labels on unlabeled states (slow, 4-bit). Leave it running. | **Done.** Ceiling + 13480 gold-state pairs (`--sources gold`). |
| 7 to 8 | Phase 2 distill + temperature + eval tables | **Done.** phase2/best + temps.json + eval/tables/phase2.json. |
| 9 | Serve endpoint, 20 hand-written English tickets as a sanity set | **Next.** Script exists; use phase2/best. |
| 10+ | Only if Phase 2 is short of the go/no-go numbers: ModernBERT-large | Optional. |

Teacher labeling is the long pole, not training. Batch Qwen requests. Cache by `hash(state, question)`. Resume commands are in **section 0**.

## 11. Go / no-go

Ship the base student if all of these hold:

1. BANKING77 test accuracy >= 90%, ECE <= 0.08.
2. Enron spam or SMS spam accuracy >= 95%, ECE <= 0.06.
3. SST-5 accuracy >= 52% (5-way is hard; majority is ~20 to 30%).
4. At 50% coverage, BANKING77 accuracy >= 94%.
5. A 10-question batch at 512 tokens p95 latency <= 80 ms on the 8 GB card.
6. Option shuffle on val drops accuracy by less than 2 points.

If (1) fails and the teacher itself is below 90% on BANKING77 with constrained decoding, the teacher is the bottleneck. Switch the labeling model to `Qwen2.5-7B` if you were on 3B, or accept a lower ceiling. Do not scale the student first.

If (1) fails and the teacher is above 92%, the packing or `[MASK]` gather is wrong. Fix that before more data.

## 12. Limits you accept

- **512 tokens.** Longer tickets must be summarized or cropped in code, not in the model.
- **English.** No multilingual claim.
- **Specialist, not Jev.** New question templates need more gold or more teacher labels. There is no free zero-shot for arbitrary instructions.
- **No arithmetic.** Amounts, dates, and counts stay in Python.
- **Teacher bias.** Qwen will be overconfident on jailbreaks and political tone. The human gold buckets (spam, BANKING77, SST-5) exist to pin the student to reality. Do not raise the distilled share above 40%.
- **8 GB forbids** RLCD with G=8 noisy forwards on ModernBERT-large at 512. If you later want a proper-scoring RL stage, run it on the **base** student, G=2, batch 1, or rent a 24 GB box for a weekend. It is optional. CE + Brier/RPS + temperature is the 8 GB path.

## 13. First command sequence

The venv already exists. Do **not** recreate it unless it is gone. Full resume commands: **section 0**.

```bash
cd /home/linesserver/proyectos/system-one
source .venv/bin/activate

# already done:
# python scripts/00_smoke.py
# python scripts/01_export_gold.py --preset proof --out data/gold

python scripts/04_train.py --config configs/student_modernbert_base.yaml --data data/gold --phase 1 --output artifacts/phase1
# unload student, then:
python scripts/02_label_teacher.py --config configs/teacher_qwen7b_4bit.yaml --out data/distilled
python scripts/03_augment_stress.py --data data/gold,data/distilled --out data/stress/stress.jsonl
python scripts/04_train.py --config configs/student_modernbert_base.yaml \
  --data data/gold,data/distilled,data/stress --phase 2 \
  --init artifacts/phase1/best --output artifacts/phase2
python scripts/05_fit_temperature.py --ckpt artifacts/phase2/best --split val
python scripts/06_eval.py --ckpt artifacts/phase2/best --temperatures artifacts/temps.json
python scripts/07_serve.py --ckpt artifacts/phase2/best
```

Proof gold (BANKING77 + SMS spam + SST-5) is already exported. Add HelpSteer2 and the support bundle (`--preset full`) before you call it a System One, after freeing `/home` disk.
