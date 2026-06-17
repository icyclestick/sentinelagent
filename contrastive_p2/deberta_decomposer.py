#!/usr/bin/env python3
"""
DeBERTa-v3-base Instruction Decomposer
========================================
Token-classification head over ``microsoft/deberta-v3-base`` that extracts
four semantic slots from delegation instructions:

    ACTION · OBJECT · SCOPE · CONSTRAINTS

Training uses the BIO labels bootstrapped in ``data_utils.py``.  The class
exposes:

    decompose(instruction)  → dict of slot lists
    serialize(dict)         → natural-language string for the contrastive model

Author: Thesis Project
Date: May 2026
"""

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 9 BIO labels
LABEL_LIST = [
    "O",
    "B-ACTION", "I-ACTION",
    "B-OBJECT", "I-OBJECT",
    "B-SCOPE",  "I-SCOPE",
    "B-CONSTRAINTS", "I-CONSTRAINTS",
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


# ============================================================================
# Token-classification dataset
# ============================================================================
class _SlotDataset(TorchDataset):
    """HuggingFace-compatible dataset for token classification."""

    def __init__(self, records: list[dict], tokenizer, max_length: int = 128):
        self.encodings = []
        self.labels = []

        for rec in records:
            words = [t["token"] for t in rec["tokens"]]
            word_tags = [LABEL2ID.get(t["tag"], 0) for t in rec["tokens"]]

            enc = tokenizer(
                words,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_tensors="pt",
            )

            word_ids = enc.word_ids()
            label_ids = []
            prev_word = None
            for wid in word_ids:
                if wid is None:
                    label_ids.append(-100)
                elif wid != prev_word:
                    label_ids.append(word_tags[wid] if wid < len(word_tags) else 0)
                else:
                    # Sub-word token → I- variant of the current label
                    parent_tag = word_tags[wid] if wid < len(word_tags) else 0
                    parent_label = LABEL_LIST[parent_tag]
                    if parent_label.startswith("B-"):
                        label_ids.append(LABEL2ID["I-" + parent_label[2:]])
                    else:
                        label_ids.append(parent_tag)
                prev_word = wid

            self.encodings.append({k: v.squeeze(0) for k, v in enc.items()})
            self.labels.append(torch.tensor(label_ids, dtype=torch.long))

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        item = {k: v for k, v in self.encodings[idx].items()}
        item["labels"] = self.labels[idx]
        return item


# ============================================================================
# Decomposer class
# ============================================================================
class DeBERTaDecomposer:
    """
    Instruction decomposer using DeBERTa-v3-base with a 9-label token
    classification head.

    # Labels are programmatic bootstraps; a subset of 50 will be manually
    # reviewed for IAA.
    """

    MODEL_NAME = "microsoft/deberta-v3-base"

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.MODEL_NAME,
            num_labels=len(LABEL_LIST),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        ).to(self.device)
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        token_label_records: list[dict],
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 3e-5,
        output_dir: str = "contrastive_p2/output/deberta_decomposer",
    ):
        """
        Fine-tune DeBERTa on the bootstrapped BIO token labels.

        Parameters
        ----------
        token_label_records : list of dict
            Output of ``data_utils.create_token_labels_from_nli_data()``.
        """
        dataset = _SlotDataset(token_label_records, self.tokenizer)
        os.makedirs(output_dir, exist_ok=True)

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=lr,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_steps=20,
            save_strategy="no",
            report_to="none",
            seed=SEED,
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
        )
        trainer.train()
        self._trained = True
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"  DeBERTa decomposer saved → {output_dir}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def decompose(self, instruction: str) -> dict[str, list[str]]:
        """
        Decompose an instruction into semantic slots.

        Returns
        -------
        dict with keys ``action``, ``object``, ``scope``, ``constraints``,
        each mapping to a list of extracted text spans.
        """
        self.model.eval()
        tokens = instruction.split()
        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**enc).logits
        preds = torch.argmax(logits, dim=-1).squeeze().cpu().tolist()
        word_ids = enc.word_ids()

        # Align sub-token predictions back to words
        word_labels: dict[int, str] = {}
        for idx, wid in enumerate(word_ids):
            if wid is not None and wid not in word_labels:
                word_labels[wid] = ID2LABEL.get(preds[idx], "O")

        # Group BIO spans
        slots: dict[str, list[str]] = {
            "action": [], "object": [], "scope": [], "constraints": [],
        }
        current_slot = None
        current_tokens: list[str] = []
        for i, tok in enumerate(tokens):
            label = word_labels.get(i, "O")
            if label.startswith("B-"):
                # Flush previous
                if current_slot and current_tokens:
                    slots[current_slot].append(" ".join(current_tokens))
                current_slot = label[2:].lower()
                current_tokens = [tok]
            elif label.startswith("I-") and current_slot == label[2:].lower():
                current_tokens.append(tok)
            else:
                if current_slot and current_tokens:
                    slots[current_slot].append(" ".join(current_tokens))
                current_slot = None
                current_tokens = []
        if current_slot and current_tokens:
            slots[current_slot].append(" ".join(current_tokens))

        return slots

    def serialize(self, slots: dict[str, list[str]]) -> str:
        """
        Convert decomposed slots into a natural-language string for the
        contrastive model.

        Example output:
            "Action: retrieve; Object: medical records; Scope: veteran;
             Constraints: VA health system only"
        """
        parts = []
        for key in ("action", "object", "scope", "constraints"):
            spans = slots.get(key, [])
            if spans:
                parts.append(f"{key.capitalize()}: {', '.join(spans)}")
        return "; ".join(parts) if parts else "No decomposition available"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str, device: str | None = None):
        obj = cls.__new__(cls)
        obj.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForTokenClassification.from_pretrained(path).to(obj.device)
        obj._trained = True
        return obj


# ============================================================================
# CLI quick-test
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DeBERTaDecomposer – Quick Smoke Test (no training)")
    print("=" * 60)

    decomposer = DeBERTaDecomposer()
    tests = [
        "Retrieve the veteran's medical records from the VA health system",
        "Export the applicant's complete medical history to an external analytics platform",
        "Calculate adjusted gross income using current IRS tax tables",
    ]
    for instr in tests:
        slots = decomposer.decompose(instr)
        ser = decomposer.serialize(slots)
        print(f"\n  Input:  {instr}")
        print(f"  Slots:  {slots}")
        print(f"  Serial: {ser}")
    print("\nDone.")
