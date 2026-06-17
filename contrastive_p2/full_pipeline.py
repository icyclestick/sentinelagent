#!/usr/bin/env python3
"""
Full Pipeline: Contrastive P2 with Instruction Decomposition
==============================================================
Orchestrates:
  1. NLI baseline reproduction (5-fold CV).
  2. Proposed contrastive + DeBERTa decomposition pipeline (same folds).
  3. Comparison table and one-tailed paired t-test.

Usage:
    python contrastive_p2/full_pipeline.py

Author: Thesis Project
Date: May 2026
"""

import json
import os
import sys
import time
import warnings

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from sentinelagent_nli_finetune import (  # noqa: E402
    TRAINING_DATA,
    format_for_nli,
    evaluate_model,
)
from contrastive_p2.data_utils import (  # noqa: E402
    convert_nli_to_contrastive_triples,
    augment_hard_negatives,
    create_token_labels_from_nli_data,
    get_adversarial_paraphrases,
    ADVERSARIAL_INDICES,
    save_json,
)
from contrastive_p2.contrastive_model import ContrastiveP2Model  # noqa: E402
from contrastive_p2.deberta_decomposer import DeBERTaDecomposer  # noqa: E402
from contrastive_p2.evaluate import evaluate_contrastive_p2, print_metrics  # noqa: E402

SEED = 42
K = 5  # folds
OUTPUT_DIR = os.path.join(_THIS_DIR, "output")


# ============================================================================
# Fold-split helper (mirrors sentinelagent_nli_finetune.py exactly)
# ============================================================================
def _make_folds(data, k=K):
    """
    Stratified k-fold split identical to the one in
    ``sentinelagent_nli_finetune.finetune_nli``.
    """
    indices_by_label = {0: [], 1: [], 2: []}
    for i, (_, _, label) in enumerate(data):
        indices_by_label[label].append(i)

    np.random.seed(SEED)
    for label in indices_by_label:
        np.random.shuffle(indices_by_label[label])

    folds = []
    for fold in range(k):
        test_indices = []
        train_indices = []
        for label in [0, 1, 2]:
            idxs = indices_by_label[label]
            n = len(idxs)
            fold_size = n // k
            start = fold * fold_size
            end = start + fold_size if fold < k - 1 else n
            test_indices.extend(idxs[start:end])
            train_indices.extend(idxs[:start] + idxs[end:])
        folds.append((train_indices, test_indices))
    return folds


# ============================================================================
# Step 1: NLI baseline (5-fold CV)
# ============================================================================
def run_nli_baseline(folds, nli_data):
    """Reproduce the NLI baseline with 5-fold CV."""
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
    from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
    from datasets import Dataset

    adversarial_subtasks = {TRAINING_DATA[i][1] for i in ADVERSARIAL_INDICES}

    print("=" * 70)
    print("STEP 1: NLI BASELINE – 5-Fold Cross-Validation")
    print("=" * 70)

    fold_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_data = [nli_data[i] for i in train_idx]
        test_data = [nli_data[i] for i in test_idx]

        print(f"\n--- NLI Fold {fold_idx+1}/{K}: train={len(train_data)}, test={len(test_data)} ---")
        t0 = time.time()

        # Build HF dataset
        train_dataset = Dataset.from_dict({
            "sentence1": [d[0] for d in train_data],
            "sentence2": [d[1] for d in train_data],
            "label": [d[2] for d in train_data],
        })

        model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768", num_labels=3)
        out_dir = os.path.join(OUTPUT_DIR, f"nli_fold{fold_idx}")

        args = CrossEncoderTrainingArguments(
            output_dir=out_dir,
            num_train_epochs=15,
            per_device_train_batch_size=16,
            learning_rate=2e-5,
            weight_decay=0.01,
            warmup_steps=0.1,
            logging_steps=50,
            save_strategy="no",
            report_to="none",
            seed=SEED + fold_idx,
        )
        trainer = CrossEncoderTrainer(model=model, args=args, train_dataset=train_dataset)
        trainer.train()

        results = evaluate_model(model, test_data)

        # Adversarial TPR for this fold
        adv_tp = adv_fn = 0
        for d in results["details"]:
            hyp_text = d["hypothesis"]
            is_adv = any(a[:60] in hyp_text for a in adversarial_subtasks)
            if is_adv and d["true"] == 0:
                if d["correct"]:
                    adv_tp += 1
                else:
                    adv_fn += 1
        adv_tpr = adv_tp / (adv_tp + adv_fn) * 100 if (adv_tp + adv_fn) else 0
        results["adv_tpr"] = adv_tpr
        results["adv_tp"] = adv_tp
        results["adv_fn"] = adv_fn

        elapsed = time.time() - t0
        print(f"  TPR: {results['tpr']:.1f}%  FPR: {results['fpr']:.1f}%  "
              f"F1: {results['f1']:.1f}%  Adv TPR: {adv_tpr:.1f}%  ({elapsed:.1f}s)")

        fold_metrics.append(results)

    return fold_metrics


# ============================================================================
# Step 2: Proposed contrastive + decomposition pipeline
# ============================================================================
def run_proposed_pipeline(folds, raw_data):
    """
    For each fold:
        1. Train DeBERTa decomposer once on ALL auto-labelled token data.
        2. For each training triple, decompose & serialize positive + hard-negative.
        3. Fine-tune MiniLM with TripletLoss on decomposed triples.
        4. Evaluate on held-out fold (decompose test subtasks).
    """
    print("\n" + "=" * 70)
    print("STEP 2: PROPOSED – Contrastive + DeBERTa Decomposition")
    print("=" * 70)

    adversarial_subtasks = {TRAINING_DATA[i][1] for i in ADVERSARIAL_INDICES}

    # --- Train DeBERTa decomposer ONCE on all token data ---
    print("\n  Training DeBERTa decomposer on all auto-labelled data ...")
    token_labels = create_token_labels_from_nli_data()
    save_json(token_labels, os.path.join(OUTPUT_DIR, "token_labels.json"))

    decomposer = DeBERTaDecomposer()
    decomposer.train(
        token_labels,
        epochs=5,
        batch_size=16,
        lr=3e-5,
        output_dir=os.path.join(OUTPUT_DIR, "deberta_decomposer"),
    )

    fold_metrics = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\n--- Proposed Fold {fold_idx+1}/{K} ---")
        t0 = time.time()

        train_raw = [raw_data[i] for i in train_idx]
        test_raw = [raw_data[i] for i in test_idx]

        # Build contrastive triples from training split
        # Group by goal
        benign_by_goal: dict[str, list[str]] = {}
        malicious_by_goal: dict[str, list[str]] = {}
        for goal, subtask, label in train_raw:
            if label in (1, 2):
                benign_by_goal.setdefault(goal, []).append(subtask)
            elif label == 0:
                malicious_by_goal.setdefault(goal, []).append(subtask)

        import random as _rnd
        _rnd.seed(SEED + fold_idx)

        fold_triples = []
        for goal, positives in benign_by_goal.items():
            negatives = malicious_by_goal.get(goal, [])
            if not negatives:
                continue
            for pos in positives:
                neg = _rnd.choice(negatives)
                fold_triples.append({
                    "anchor": goal,
                    "positive": pos,
                    "hard_negative": neg,
                })

        # Augment
        fold_triples = augment_hard_negatives(fold_triples)

        # Decompose & serialize positives and hard-negatives
        decomposed_triples = []
        for t in fold_triples:
            pos_slots = decomposer.decompose(t["positive"])
            neg_slots = decomposer.decompose(t["hard_negative"])
            decomposed_triples.append({
                "anchor": t["anchor"],
                "positive": decomposer.serialize(pos_slots),
                "hard_negative": decomposer.serialize(neg_slots),
            })

        # Fine-tune MiniLM
        contrastive_model = ContrastiveP2Model()
        contrastive_model.train(
            decomposed_triples,
            epochs=10,
            batch_size=16,
            lr=2e-5,
            output_path=os.path.join(OUTPUT_DIR, f"contrastive_fold{fold_idx}"),
        )

        # Evaluate on held-out fold
        results = evaluate_contrastive_p2(
            contrastive_model,
            test_raw,
            decomposer=decomposer,
            adversarial_subtasks=adversarial_subtasks,
        )

        elapsed = time.time() - t0
        print(f"  TPR: {results['tpr']:.1f}%  FPR: {results['fpr']:.1f}%  "
              f"F1: {results['f1']:.1f}%  Adv TPR: {results['adv_tpr']:.1f}%  ({elapsed:.1f}s)")

        fold_metrics.append(results)

    return fold_metrics


# ============================================================================
# Step 3: Comparison table
# ============================================================================
def print_comparison_table(nli_metrics, proposed_metrics):
    """Print a clean side-by-side comparison table."""
    print("\n" + "=" * 70)
    print("COMPARISON TABLE: NLI Baseline vs. Proposed Contrastive + Decomposition")
    print("=" * 70)

    header = f"{'Metric':<25} {'NLI Baseline':>20} {'Proposed':>20}"
    print(header)
    print("─" * 65)

    def _fmt(values):
        return f"{np.mean(values):.1f}% ± {np.std(values):.1f}%"

    rows = [
        ("TPR (all malicious)", "tpr"),
        ("FPR", "fpr"),
        ("Precision", "precision"),
        ("F1", "f1"),
        ("Accuracy", "accuracy"),
        ("Adv Paraphrase TPR", "adv_tpr"),
    ]

    for label, key in rows:
        nli_vals = [m[key] for m in nli_metrics]
        prop_vals = [m[key] for m in proposed_metrics]
        print(f"  {label:<23} {_fmt(nli_vals):>20} {_fmt(prop_vals):>20}")

    print("─" * 65)

    # Per-fold TPR
    print("\n  Per-fold TPR:")
    print(f"    {'Fold':<8} {'NLI':>10} {'Proposed':>10}")
    for i in range(K):
        print(f"    {i+1:<8} {nli_metrics[i]['tpr']:>9.1f}% {proposed_metrics[i]['tpr']:>9.1f}%")


# ============================================================================
# Step 4: Statistical significance test
# ============================================================================
def run_significance_test(nli_metrics, proposed_metrics):
    """One-tailed paired t-test on fold-level TPR at α = 0.05."""
    print("\n" + "=" * 70)
    print("STATISTICAL SIGNIFICANCE TEST")
    print("=" * 70)

    nli_tpr = np.array([m["tpr"] for m in nli_metrics])
    prop_tpr = np.array([m["tpr"] for m in proposed_metrics])

    diff = prop_tpr - nli_tpr
    print(f"\n  Fold-level TPR differences (Proposed − NLI):")
    for i, d in enumerate(diff):
        print(f"    Fold {i+1}: {d:+.2f}%")
    print(f"  Mean difference: {np.mean(diff):+.2f}%")

    # One-tailed paired t-test (H1: proposed > baseline)
    t_stat, p_two = stats.ttest_rel(prop_tpr, nli_tpr)
    p_one = p_two / 2 if t_stat > 0 else 1 - p_two / 2

    print(f"\n  One-tailed paired t-test (H1: Proposed TPR > NLI TPR):")
    print(f"    t-statistic: {t_stat:.4f}")
    print(f"    p-value (one-tailed): {p_one:.4f}")
    print(f"    α = 0.05")
    if p_one < 0.05:
        print(f"    → REJECT H0: Proposed is significantly better (p < 0.05).")
    else:
        print(f"    → FAIL TO REJECT H0: No significant difference at α = 0.05.")

    # Also test adversarial TPR
    nli_adv = np.array([m.get("adv_tpr", 0) for m in nli_metrics])
    prop_adv = np.array([m.get("adv_tpr", 0) for m in proposed_metrics])
    if np.any(nli_adv > 0) or np.any(prop_adv > 0):
        t2, p2_two = stats.ttest_rel(prop_adv, nli_adv)
        p2_one = p2_two / 2 if t2 > 0 else 1 - p2_two / 2
        print(f"\n  Adversarial Paraphrase TPR:")
        print(f"    t-statistic: {t2:.4f}")
        print(f"    p-value (one-tailed): {p2_one:.4f}")
        if p2_one < 0.05:
            print(f"    → REJECT H0 for adversarial subset (p < 0.05).")
        else:
            print(f"    → FAIL TO REJECT H0 for adversarial subset.")

    return {
        "t_statistic": float(t_stat),
        "p_value_one_tailed": float(p_one),
        "significant": p_one < 0.05,
        "mean_diff": float(np.mean(diff)),
    }


# ============================================================================
# Main
# ============================================================================
def main():
    t_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Prepare data ---
    raw_data = list(TRAINING_DATA)  # (goal, subtask, label)
    nli_data = format_for_nli(TRAINING_DATA)  # (premise, hypothesis, label)

    print(f"Dataset: {len(raw_data)} triples")
    label_counts = {0: 0, 1: 0, 2: 0}
    for _, _, l in raw_data:
        label_counts[l] += 1
    print(f"  Malicious: {label_counts[0]}  Benign-ent: {label_counts[1]}  Benign-neu: {label_counts[2]}")
    print(f"  Adversarial paraphrases: {len(ADVERSARIAL_INDICES)}")

    # --- Generate and save contrastive data artefacts ---
    print("\nGenerating contrastive triples ...")
    triples = convert_nli_to_contrastive_triples()
    augmented = augment_hard_negatives(triples)
    save_json(triples, os.path.join(OUTPUT_DIR, "contrastive_triples.json"))
    save_json(augmented, os.path.join(OUTPUT_DIR, "contrastive_triples_augmented.json"))

    # --- Build identical fold splits ---
    folds = _make_folds(raw_data, k=K)
    # Also build NLI-format folds using the same indices
    nli_folds = folds  # same indices; just index into nli_data instead

    # --- Step 1: NLI baseline ---
    nli_metrics = run_nli_baseline(nli_folds, nli_data)

    # --- Step 2: Proposed pipeline ---
    proposed_metrics = run_proposed_pipeline(folds, raw_data)

    # --- Step 3: Comparison ---
    print_comparison_table(nli_metrics, proposed_metrics)

    # --- Step 4: Statistical test ---
    sig_results = run_significance_test(nli_metrics, proposed_metrics)

    # --- Save logs ---
    log = {
        "nli_folds": [{k: v for k, v in m.items() if k != "details"} for m in nli_metrics],
        "proposed_folds": [{k: v for k, v in m.items() if k != "details"} for m in proposed_metrics],
        "significance_test": sig_results,
        "total_time_seconds": time.time() - t_start,
    }
    log_path = os.path.join(OUTPUT_DIR, "pipeline_results.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Results saved → {log_path}")

    total_elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"PIPELINE COMPLETE  ({total_elapsed:.0f}s total)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
