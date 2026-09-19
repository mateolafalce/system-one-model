#!/usr/bin/env python3
"""FastAPI runtime: POST /v1/systemone — one forward, no generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import load_yaml
from runtime import load_temperatures, predict_batch, questions_to_records
from student import StudentModel, dtype_from_name, load_tokenizer


class ServeState:
    model = None
    tok = None
    temps = None
    device = "cpu"
    max_length = 512


S = ServeState()


class Query(BaseModel):
    state: Any
    questions: dict[str, Any] = Field(..., min_length=1)


def create_app() -> FastAPI:
    app = FastAPI(title="System One", version="0.1.0")

    @app.get("/health")
    def health():
        return {"ok": True, "device": S.device}

    @app.post("/v1/systemone")
    def systemone(q: Query):
        if S.model is None:
            raise HTTPException(503, "model not loaded")
        try:
            recs = questions_to_records(q.state, q.questions)
        except Exception as exc:
            raise HTTPException(400, str(exc)) from exc
        answers = predict_batch(
            S.model,
            S.tok,
            recs,
            max_length=S.max_length,
            temperatures=S.temps,
            device=S.device,
        )
        return {"answers": {rec.id: ans for rec, ans in zip(recs, answers)}}

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="artifacts/phase2/best")
    p.add_argument("--student-config", default="configs/student_modernbert_base.yaml")
    p.add_argument("--temperatures", default="artifacts/temps.json")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8010)
    args = p.parse_args()

    cfg = load_yaml(args.student_config)
    S.device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = dtype_from_name("float32" if S.device == "cpu" else cfg.get("dtype", "bfloat16"))
    S.max_length = int(cfg.get("max_length", 512))
    S.tok = load_tokenizer(cfg["backbone"])
    S.model = StudentModel.from_pretrained(args.ckpt, dtype=dtype)
    S.model.to(S.device)
    S.model.eval()
    S.temps = load_temperatures(args.temperatures)
    print(f"serving {args.ckpt} on {S.device} at http://{args.host}:{args.port}")
    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
