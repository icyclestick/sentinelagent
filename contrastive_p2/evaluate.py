#!/usr/bin/env python3
"""
Evaluation Utilities for Contrastive P2 Pipeline
==================================================
Computes TPR, FPR, Precision, F1, and separate metrics for the 26
adversarial paraphrases.

Author: Thesis Project
Date: May 2026
"""

import sys
import os
import numpy as np

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from sentinelagent_nli_finetune import TRAINING_DATA  # noqa: E402
from contrastive_p2.data_utils import ADVERSARIAL_INDICES  # noqa: E402


def evaluate_contrastive_p2(
    model,
    test_pairs: list[tuple[str, str, int]],
    decomposer=None,
    adversarial_subtasks: set[str] | None = None,
) -> dict:
    """
    Evaluate the contrastive P2 model on a set of (root_goal, subtask, label)
    triples.

    When *decomposer* is given, the subtask is first decomposed and
    serialised, and **only that serialised string** (not the raw instruction)
    is fed to the contrastive model.

    Parameters
    ----------
    model : ContrastiveP2Model
    test_pairs : list of (root_goal, subtask, label)
        label 0 = malicious, label 1 or 2 = benign.
    decomposer : DeBERTaDecomposer or None
    adversarial_subtasks : set of str or None
        If provided, compute separate metrics for these subtasks.

    Returns
    -------
    dict with keys:
        accuracy, tpr, fpr, precision, f1,
        tp, fp, tn, fn, total,
        adv_tp, adv_fn, adv_tpr (if adversarial_subtasks given),
        details (list of per-example dicts).
    """
    if adversarial_subtasks is None:
        adversarial_subtasks = {TRAINING_DATA[i][1] for i in ADVERSARIAL_INDICES}

    tp = fp = tn = fn = 0
    adv_tp = adv_fn = 0
    correct = 0
    details = []

    for goal, subtask, true_label in test_pairs:
        # Optionally decompose
        if decomposer is not None:
            slots = decomposer.decompose(subtask)
            candidate_text = decomposer.serialize(slots)
        else:
            candidate_text = subtask

        pred = model.predict(goal, candidate_text)
        true_malicious = (true_label == 0)
        pred_malicious = pred.is_malicious

        if true_malicious and pred_malicious:
            tp += 1
        elif true_malicious and not pred_malicious:
            fn += 1
        elif not true_malicious and pred_malicious:
            fp += 1
        else:
            tn += 1

        # Map to 3-class for accuracy
        if true_malicious == pred_malicious:
            correct += 1

        # Adversarial subset
        is_adv = subtask in adversarial_subtasks
        if is_adv and true_malicious:
            if pred_malicious:
                adv_tp += 1
            else:
                adv_fn += 1

        details.append({
            "goal": goal[:40],
            "subtask": subtask[:80],
            "true_label": true_label,
            "true_malicious": true_malicious,
            "pred_malicious": pred_malicious,
            "verdict": pred.verdict,
            "cosine": pred.cosine_similarity,
            "is_adversarial": is_adv,
            "correct": true_malicious == pred_malicious,
        })

    total = len(test_pairs)
    accuracy = correct / total * 100 if total else 0
    tpr = tp / (tp + fn) * 100 if (tp + fn) else 0
    fpr = fp / (fp + tn) * 100 if (fp + tn) else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) else 0
    f1 = 2 * tp / (2 * tp + fp + fn) * 100 if (2 * tp + fp + fn) else 0

    adv_total = adv_tp + adv_fn
    adv_tpr = adv_tp / adv_total * 100 if adv_total else 0

    return {
        "accuracy": accuracy,
        "tpr": tpr,
        "fpr": fpr,
        "precision": precision,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "total": total,
        "adv_tp": adv_tp,
        "adv_fn": adv_fn,
        "adv_tpr": adv_tpr,
        "details": details,
    }


def print_metrics(label: str, metrics: dict, show_adversarial: bool = True):
    """Pretty-print evaluation metrics."""
    print(f"\n  {label}")
    print(f"  {'─' * len(label)}")
    print(f"    Accuracy:  {metrics['accuracy']:.1f}%")
    print(f"    TPR:       {metrics['tpr']:.1f}%  ({metrics['tp']}/{metrics['tp']+metrics['fn']})")
    print(f"    FPR:       {metrics['fpr']:.1f}%  ({metrics['fp']}/{metrics['fp']+metrics['tn']})")
    print(f"    Precision: {metrics['precision']:.1f}%")
    print(f"    F1:        {metrics['f1']:.1f}%")
    if show_adversarial and (metrics.get("adv_tp", 0) + metrics.get("adv_fn", 0)) > 0:
        print(f"    Adv TPR:   {metrics['adv_tpr']:.1f}%  "
              f"({metrics['adv_tp']}/{metrics['adv_tp']+metrics['adv_fn']})")
