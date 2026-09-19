"""Calibration and selective-prediction metrics."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def entropy_confidence(probs: Sequence[float]) -> float:
    """1 - H(p) / log(K). One-hot -> 1, uniform -> 0."""
    p = np.asarray(probs, dtype=np.float64)
    p = p[p > 0]
    k = max(len(probs), 1)
    if k <= 1:
        return 1.0
    p = p / p.sum()
    h = float(-(p * np.log(p)).sum())
    return float(1.0 - h / math.log(k))


def expected_value(probs: Sequence[float]) -> float:
    return float(sum(i * float(p) for i, p in enumerate(probs)))


def brier_multiclass(probs: Sequence[float], target: int) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.zeros_like(p)
    y[int(target)] = 1.0
    return float(np.sum((p - y) ** 2))


def ece(confidences: Sequence[float], correctness: Sequence[int | bool], n_bins: int = 15) -> float:
    """Expected calibration error on max-prob (or task confidence)."""
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correctness, dtype=np.float64)
    if conf.size == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    acc = 0.0
    n = conf.size
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == 0:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf > lo) & (conf <= hi)
        if not mask.any():
            continue
        acc += (mask.sum() / n) * abs(corr[mask].mean() - conf[mask].mean())
    return float(acc)


def accuracy(preds: Sequence[int], labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    return float(np.mean(np.asarray(preds) == np.asarray(labels)))


def brier_mean(prob_rows: Sequence[Sequence[float]], labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    return float(np.mean([brier_multiclass(p, y) for p, y in zip(prob_rows, labels)]))


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman(pred: Sequence[float], gold: Sequence[float]) -> float:
    if len(pred) < 2:
        return 0.0
    a = _rank(np.asarray(pred, dtype=np.float64))
    b = _rank(np.asarray(gold, dtype=np.float64))
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def quadratic_weighted_kappa(pred: Sequence[int], gold: Sequence[int], n_classes: int) -> float:
    """QWK for ordinal score."""
    from sklearn.metrics import cohen_kappa_score

    if not gold:
        return 0.0
    return float(cohen_kappa_score(gold, pred, weights="quadratic", labels=list(range(n_classes))))


def risk_coverage_accuracy(
    correctness: Sequence[int | bool],
    confidences: Sequence[float],
    coverage: float,
) -> float:
    """Accuracy among the top `coverage` fraction by confidence."""
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(correctness, dtype=np.float64)
    if conf.size == 0:
        return 0.0
    k = max(1, int(math.ceil(coverage * conf.size)))
    idx = np.argsort(-conf)[:k]
    return float(corr[idx].mean())


def k_bucket(k: int) -> str:
    if k <= 2:
        return "2"
    if k <= 5:
        return "3-5"
    if k <= 15:
        return "6-15"
    return ">15"


def f1_binary(preds: Sequence[int], labels: Sequence[int], positive: int = 1) -> float:
    from sklearn.metrics import f1_score

    if not labels:
        return 0.0
    return float(f1_score(labels, preds, pos_label=positive, zero_division=0.0))


def fpr_at_tpr(scores: Sequence[float], labels: Sequence[int], tpr: float = 0.95) -> float:
    """False positive rate at a given true positive rate. Higher score => positive."""
    from sklearn.metrics import roc_curve

    y = np.asarray(labels)
    s = np.asarray(scores)
    if y.min() == y.max():
        return 0.0
    fpr, tpr_arr, _ = roc_curve(y, s)
    hits = np.where(tpr_arr >= tpr)[0]
    if hits.size == 0:
        return float(fpr[-1])
    return float(fpr[hits[0]])
