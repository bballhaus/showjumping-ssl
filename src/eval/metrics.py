"""Evaluation metrics shared by the downstream heads, baselines, and analysis.

Outcome classification: accuracy + macro-F1 (and per-type breakdown).
Distance regression: MAE in meters, overall and per fence type.
"""

from __future__ import annotations

import numpy as np

from ..downstream.dataset import OUTCOMES, TYPES


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = len(OUTCOMES)) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if tp + fp + fn == 0:
            continue  # class absent from both -> undefined, skip
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def classification_report(y_true, y_pred, types=None) -> dict:
    out = {"accuracy": accuracy(y_true, y_pred), "macro_f1": macro_f1(y_true, y_pred)}
    if types is not None:
        types = np.asarray(types)
        for t in TYPES:
            mask = types == t
            if mask.sum():
                out[f"accuracy_{t}"] = accuracy(np.asarray(y_true)[mask], np.asarray(y_pred)[mask])
                out[f"macro_f1_{t}"] = macro_f1(np.asarray(y_true)[mask], np.asarray(y_pred)[mask])
    return out


def d_mae(d_true, d_pred, valid=None, types=None) -> dict:
    d_true, d_pred = np.asarray(d_true, float), np.asarray(d_pred, float)
    valid = np.ones_like(d_true, bool) if valid is None else np.asarray(valid, bool)
    out = {}
    m = valid
    out["d_mae"] = float(np.abs(d_true[m] - d_pred[m]).mean()) if m.sum() else float("nan")
    if types is not None:
        types = np.asarray(types)
        for t in TYPES:
            mt = m & (types == t)
            out[f"d_mae_{t}"] = (float(np.abs(d_true[mt] - d_pred[mt]).mean())
                                 if mt.sum() else float("nan"))
    return out
