"""ModernBERT backbone + decision head that scores [MASK] option markers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

import config  # noqa: F401  — sets HF_HOME before transformers
from transformers import AutoModel, AutoTokenizer

from config import resolve_path
from schema import TYPE_TO_ID


def _nhead(hidden: int) -> int:
    for h in (16, 12, 8, 4):
        if hidden % h == 0 and hidden // h >= 32:
            return h
    return 8


class DecisionHead(nn.Module):
    def __init__(self, hidden_size: int, n_layers: int = 2, dropout: float = 0.1, n_types: int = 3) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.type_emb = nn.Embedding(n_types, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=_nhead(hidden_size),
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        type_ids: torch.Tensor,
        option_positions: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.type_emb(type_ids).to(dtype=hidden.dtype).unsqueeze(1)
        hidden = torch.cat([hidden[:, :1] + emb, hidden[:, 1:]], dim=1)
        pad = attention_mask == 0
        hidden = self.encoder(hidden, src_key_padding_mask=pad)

        bsz, max_k = option_positions.shape
        dim = hidden.size(-1)
        safe = option_positions.clamp(min=0)
        gathered = hidden.gather(1, safe.unsqueeze(-1).expand(bsz, max_k, dim))
        gathered = gathered * option_mask.unsqueeze(-1).to(gathered.dtype)
        logits = self.scorer(gathered).squeeze(-1)
        logits = logits.masked_fill(~option_mask, torch.finfo(logits.dtype).min)
        return logits


def infer_lora_targets(model: nn.Module, preferred: list[str] | None = None) -> list[str]:
    leaves: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            # ModernBERT Wqkv is a Linear; still collect by leaf name.
            pass
        leaf = name.split(".")[-1]
        if leaf in {"Wqkv", "Wi", "Wo", "wo", "wi", "Wq", "Wk", "Wv", "query", "key", "value", "dense"}:
            leaves.add(leaf)
    if preferred:
        hit = [p for p in preferred if p in leaves]
        if hit:
            return hit
    if not leaves:
        raise ValueError("could not infer LoRA target modules; inspect named_modules()")
    return sorted(leaves)


class StudentModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        lora: dict[str, Any] | None = None,
        head_layers: int = 2,
        head_dropout: float = 0.1,
        dtype: torch.dtype | None = None,
        attn_implementation: str = "sdpa",
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        kw: dict[str, Any] = {}
        if dtype is not None:
            kw["dtype"] = dtype
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        try:
            self.backbone = AutoModel.from_pretrained(backbone_name, **kw)
        except TypeError:
            kw.pop("attn_implementation", None)
            if "dtype" in kw:
                kw["torch_dtype"] = kw.pop("dtype")
            self.backbone = AutoModel.from_pretrained(backbone_name, **kw)
        hidden = int(self.backbone.config.hidden_size)
        # Head stays fp32. The fused bf16 TransformerEncoder path does not
        # always record autograd; fp32 is cheap at ~15M params.
        self.head = DecisionHead(hidden, n_layers=head_layers, dropout=head_dropout)
        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()
        self.lora_cfg = lora
        if lora:
            from peft import LoraConfig, get_peft_model

            targets = infer_lora_targets(self.backbone, lora.get("target_modules"))
            lora_kwargs = dict(
                r=int(lora.get("r", 16)),
                lora_alpha=int(lora.get("alpha", 32)),
                lora_dropout=float(lora.get("dropout", 0.05)),
                target_modules=targets,
                bias="none",
            )
            try:
                from peft import TaskType

                lora_kwargs["task_type"] = TaskType.FEATURE_EXTRACTION
            except Exception:
                pass
            self.backbone = get_peft_model(self.backbone, LoraConfig(**lora_kwargs))

    def forward(self, input_ids, attention_mask, type_ids, option_positions, option_mask, **_):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Head is fp32. Leaving it under bf16 autocast uses a fused
        # TransformerEncoder path that can drop autograd.
        hidden = out.last_hidden_state.float()
        ctx = torch.autocast(device_type=hidden.device.type, enabled=False)
        with ctx:
            return self.head(hidden, attention_mask, type_ids, option_positions, option_mask)

    def trainable_param_groups(self, lr_lora: float, lr_head: float, weight_decay: float) -> list[dict]:
        lora_params = [p for n, p in self.backbone.named_parameters() if p.requires_grad]
        head_params = [p for p in self.head.parameters() if p.requires_grad]
        groups = []
        if lora_params:
            groups.append({"params": lora_params, "lr": lr_lora, "weight_decay": weight_decay})
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})
        return groups

    def save_pretrained(self, directory: str | Path, extra: dict[str, Any] | None = None) -> None:
        d = resolve_path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(d / "backbone")
        torch.save(self.head.state_dict(), d / "head.pt")
        meta = {
            "backbone_name": self.backbone_name,
            "hidden_size": self.head.hidden_size,
            "head_layers": len(self.head.encoder.layers),
            "type_to_id": TYPE_TO_ID,
            **(extra or {}),
        }
        (d / "student.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def from_pretrained(
        cls,
        directory: str | Path,
        dtype: torch.dtype | None = None,
        attn_implementation: str = "sdpa",
        is_trainable: bool = False,
    ) -> "StudentModel":
        d = resolve_path(directory)
        meta = json.loads((d / "student.json").read_text(encoding="utf-8"))
        obj = cls(
            backbone_name=meta["backbone_name"],
            lora=None,
            head_layers=int(meta.get("head_layers", 2)),
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
        adapter_cfg = d / "backbone" / "adapter_config.json"
        if adapter_cfg.exists():
            from peft import PeftModel

            obj.backbone = PeftModel.from_pretrained(
                obj.backbone,
                str(d / "backbone"),
                is_trainable=is_trainable,
            )
        else:
            kw = {}
            if dtype is not None:
                kw["torch_dtype"] = dtype
            obj.backbone = AutoModel.from_pretrained(d / "backbone", **kw)
        state = torch.load(d / "head.pt", map_location="cpu", weights_only=True)
        obj.head.load_state_dict(state)
        if not is_trainable:
            obj.eval()
        return obj


def load_tokenizer(backbone_name: str):
    tok = AutoTokenizer.from_pretrained(backbone_name)
    if tok.cls_token_id is None or tok.sep_token_id is None or tok.mask_token_id is None:
        raise ValueError(f"{backbone_name} tokenizer is missing CLS/SEP/MASK")
    if tok.pad_token_id is None:
        tok.pad_token = tok.sep_token
    return tok


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[name]


def build_student(cfg: dict[str, Any]) -> tuple[StudentModel, Any]:
    dtype = dtype_from_name(cfg.get("dtype", "bfloat16"))
    tok = load_tokenizer(cfg["backbone"])
    model = StudentModel(
        backbone_name=cfg["backbone"],
        lora=cfg.get("lora"),
        head_layers=int(cfg.get("head", {}).get("n_layers", 2)),
        head_dropout=float(cfg.get("head", {}).get("dropout", 0.1)),
        dtype=dtype,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
        gradient_checkpointing=bool(cfg.get("train", {}).get("gradient_checkpointing", False)),
    )
    return model, tok
