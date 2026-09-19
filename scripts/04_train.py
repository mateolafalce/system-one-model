#!/usr/bin/env python3
"""Train ModernBERT + decision head. One GPU. Student only — teacher must be unloaded."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from config import load_yaml, resolve_path
from dataio import JsonlDecisionDataset, load_records, make_collate, parse_data_arg
from loss import student_loss
from metrics import accuracy, ece, entropy_confidence
from student import build_student, dtype_from_name


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_records(records, val_frac: float = 0.05, seed: int = 42):
    """Keep official val/test. If a source has train but no val, carve by group_id."""
    by = defaultdict(list)
    for r in records:
        key = r.split if r.split in {"train", "val", "test", "zeroshot"} else "train"
        by[key].append(r)
    sources_with_val = {r.source for r in by["val"]}
    rng = random.Random(seed)
    new_train = []
    new_val = list(by["val"])
    by_src: dict[str, list] = defaultdict(list)
    for r in by["train"]:
        by_src[r.source].append(r)
    for src, recs in by_src.items():
        if src in sources_with_val:
            new_train.extend(recs)
            continue
        groups = defaultdict(list)
        for r in recs:
            groups[r.group_id].append(r)
        gids = list(groups)
        rng.shuffle(gids)
        n_val = max(1, int(round(len(gids) * val_frac))) if len(gids) >= 20 else max(1, len(gids) // 20)
        for i, gid in enumerate(gids):
            (new_val if i < n_val else new_train).extend(groups[gid])
    by["train"] = new_train
    by["val"] = new_val
    return by


def _family_counts(records) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        counts[r.family or r.source] += 1
    return dict(sorted(counts.items()))


def cosine_lr(step: int, warmup: int, total: int, base: float) -> float:
    if step < warmup:
        return base * float(step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model, loader, device: str) -> dict:
    model.eval()
    pred, gold, confs, correct = [], [], [], []
    by_fam: dict[str, dict[str, list]] = defaultdict(lambda: {"pred": [], "gold": [], "confs": [], "correct": []})
    tot_loss = 0.0
    n = 0
    for batch in loader:
        feed = {
            k: batch[k].to(device)
            for k in ("input_ids", "attention_mask", "type_ids", "option_positions", "option_mask")
        }
        logits = model(**feed)
        tensors = {k: batch[k].to(device) for k in batch if torch.is_tensor(batch[k])}
        tot_loss += float(student_loss(logits.float(), tensors)["loss"].cpu())
        n += 1
        option_mask = batch["option_mask"]
        targets = batch["targets"]
        families = batch.get("families") or ["?"] * logits.size(0)
        for i in range(logits.size(0)):
            if int(targets[i]) < 0:
                continue
            k = int(option_mask[i].sum())
            probs = torch.softmax(logits[i, :k].float(), dim=0)
            yhat = int(probs.argmax())
            y = int(targets[i])
            c = entropy_confidence(probs.tolist())
            ok = int(yhat == y)
            pred.append(yhat)
            gold.append(y)
            confs.append(c)
            correct.append(ok)
            fam = families[i]
            by_fam[fam]["pred"].append(yhat)
            by_fam[fam]["gold"].append(y)
            by_fam[fam]["confs"].append(c)
            by_fam[fam]["correct"].append(ok)
    model.train()
    families = {}
    for fam, d in sorted(by_fam.items()):
        families[fam] = {
            "acc": accuracy(d["pred"], d["gold"]),
            "ece": ece(d["confs"], d["correct"]),
            "n": len(d["gold"]),
        }
    fam_accs = [v["acc"] for v in families.values() if v["n"]]
    macro = sum(fam_accs) / max(len(fam_accs), 1)
    return {
        "loss": tot_loss / max(n, 1),
        "acc": accuracy(pred, gold),
        "macro_acc": macro,
        "ece": ece(confs, correct),
        "n": len(gold),
        "families": families,
    }


def build_loader(records, tok, cfg, train: bool, mix_hard_soft: bool) -> DataLoader:
    tcfg = cfg["train"]
    ds = JsonlDecisionDataset(
        records,
        tok,
        max_length=int(cfg.get("max_length", 512)),
        shuffle_options=train,
        seed=int(tcfg.get("seed", 42)),
    )
    collate = make_collate(tok)
    kwargs = dict(
        batch_size=int(tcfg["batch_size"]),
        collate_fn=collate,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    if train and mix_hard_soft:
        hard = sum(1 for r in records if r.label.kind == "hard")
        soft = len(records) - hard
        if hard and soft:
            # 1:1 hard:soft regardless of pool size.
            w = []
            for r in records:
                w.append((0.5 / hard) if r.label.kind == "hard" else (0.5 / soft))
            sampler = WeightedRandomSampler(w, num_samples=len(records), replacement=True)
            return DataLoader(ds, sampler=sampler, **kwargs)
    return DataLoader(ds, shuffle=train, **kwargs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/student_modernbert_base.yaml")
    p.add_argument("--data", default="data/gold")
    p.add_argument("--phase", type=int, default=1, help="1=gold CE, 2=distill mix")
    p.add_argument("--init", default="", help="warm start from a previous checkpoint dir")
    p.add_argument("--output", default="")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--epochs", type=int, default=0, help="override config train.epochs (Phase 2 is 1)")
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    tcfg = cfg["train"]
    set_seed(int(tcfg.get("seed", 42)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        cfg["dtype"] = "float32"

    records = load_records(parse_data_arg(args.data))
    records = [r for r in records if r.split != "zeroshot"]
    if args.phase == 1:
        records = [r for r in records if r.label.kind == "hard"]
    splits = split_records(records, seed=int(tcfg.get("seed", 42)))
    print(f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} phase={args.phase}")
    for split_name in ("train", "val", "test"):
        counts = _family_counts(splits[split_name])
        if counts:
            pretty = " ".join(f"{k}={v}" for k, v in counts.items())
            print(f"  {split_name}: {pretty}")
    if not splits["train"]:
        raise SystemExit("no train records")

    if args.init:
        from student import StudentModel, load_tokenizer

        tok = load_tokenizer(cfg["backbone"])
        model = StudentModel.from_pretrained(
            args.init,
            dtype=dtype_from_name(cfg.get("dtype", "bfloat16")),
            is_trainable=True,
        )
    else:
        model, tok = build_student(cfg)
    model.to(device)

    mix = args.phase == 2
    train_loader = build_loader(splits["train"], tok, cfg, train=True, mix_hard_soft=mix)
    val_recs = splits["val"] or splits["train"][: min(64, len(splits["train"]))]
    val_loader = build_loader(val_recs, tok, cfg, train=False, mix_hard_soft=False)

    accum = max(1, int(tcfg["grad_accum"]))
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accum))
    epochs = args.epochs or int(tcfg["epochs"])
    total_steps = args.max_steps or steps_per_epoch * epochs
    print(f"steps_per_epoch={steps_per_epoch} epochs={epochs} total_steps={total_steps}")

    groups = model.trainable_param_groups(float(tcfg["lr_lora"]), float(tcfg["lr_head"]), float(tcfg.get("weight_decay", 0.01)))
    opt = torch.optim.AdamW(groups)
    use_amp = device == "cuda" and cfg.get("dtype") in {"bfloat16", "bf16"}
    log_every = int(tcfg.get("log_every", 20))
    eval_every = int(tcfg.get("eval_every", 200))
    max_norm = float(tcfg.get("max_grad_norm", 1.0))
    out_dir = resolve_path(args.output or cfg.get("output_dir") or f"artifacts/phase{args.phase}")
    out_dir.mkdir(parents=True, exist_ok=True)

    def maybe_save(tag: str, metrics: dict, opt_step: int) -> None:
        model.save_pretrained(out_dir / tag, extra={"val": metrics, "step": opt_step})
        tok.save_pretrained(out_dir / tag / "tokenizer")

    best = {"macro_acc": -1.0, "acc": -1.0, "ece": 9.0}
    model.train()
    opt_step = 0
    micro = 0
    running = 0.0
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    stop = False

    while not stop:
        for batch in train_loader:
            feed = {
                k: batch[k].to(device)
                for k in ("input_ids", "attention_mask", "type_ids", "option_positions", "option_mask")
            }
            tensors = {k: batch[k].to(device) for k in batch if torch.is_tensor(batch[k])}
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(**feed)
            out = student_loss(
                logits.float(),
                tensors,
                score_rps_weight=float(tcfg.get("score_rps_weight", 0.5)),
                aux_ce_weight=float(tcfg.get("aux_ce_weight", 0.0)),
            )
            loss = out["loss"] / accum
            if not loss.requires_grad:
                raise SystemExit(f"NO-GO: loss has no grad at step {opt_step}")
            loss.backward()
            running += float(out["loss"].detach().cpu())
            micro += 1
            if micro % accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            lr_scale = cosine_lr(opt_step, int(tcfg["warmup_steps"]), total_steps, 1.0)
            opt.param_groups[0]["lr"] = float(tcfg["lr_lora"]) * lr_scale
            if len(opt.param_groups) > 1:
                opt.param_groups[1]["lr"] = float(tcfg["lr_head"]) * lr_scale
            opt.step()
            opt.zero_grad(set_to_none=True)
            opt_step += 1
            if opt_step % log_every == 0:
                avg = running / (log_every * accum)
                running = 0.0
                print(
                    f"step {opt_step}/{total_steps} loss={avg:.4f} "
                    f"lr_lora={opt.param_groups[0]['lr']:.2e} dt={time.time()-t0:.0f}s"
                )
            if not args.skip_eval and opt_step % eval_every == 0:
                metrics = evaluate(model, val_loader, device)
                fam_s = " ".join(
                    f"{k}={v['acc']:.3f}/{v['n']}" for k, v in (metrics.get("families") or {}).items()
                )
                print(
                    f"  val macro={metrics['macro_acc']:.4f} acc={metrics['acc']:.4f} "
                    f"ece={metrics['ece']:.4f} n={metrics['n']} {fam_s}"
                )
                better = metrics["macro_acc"] > best["macro_acc"] + 1e-4 or (
                    abs(metrics["macro_acc"] - best["macro_acc"]) < 1e-4 and metrics["ece"] < best["ece"]
                )
                if better:
                    best = {
                        "macro_acc": metrics["macro_acc"],
                        "acc": metrics["acc"],
                        "ece": metrics["ece"],
                        "families": metrics.get("families", {}),
                    }
                    maybe_save("best", metrics, opt_step)
                    print(f"  saved best -> {out_dir / 'best'}")
                if device == "cuda":
                    torch.cuda.empty_cache()
            if opt_step >= total_steps:
                stop = True
                break

    metrics = (
        evaluate(model, val_loader, device)
        if not args.skip_eval
        else {"acc": 0, "macro_acc": 0, "ece": 0, "n": 0, "families": {}}
    )
    fam_s = " ".join(f"{k}={v['acc']:.3f}/{v['n']}" for k, v in (metrics.get("families") or {}).items())
    print(f"final val macro={metrics.get('macro_acc', 0):.4f} acc={metrics['acc']:.4f} ece={metrics['ece']:.4f} {fam_s}")
    maybe_save("last", metrics, opt_step)
    if best["macro_acc"] < 0:
        maybe_save("best", metrics, opt_step)
    (out_dir / "train_meta.json").write_text(
        json.dumps({"best": best, "final": metrics, "phase": args.phase}, indent=2),
        encoding="utf-8",
    )
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
