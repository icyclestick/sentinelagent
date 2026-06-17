#!/usr/bin/env python3
"""
Contrastive P2 Model – Sentence-Transformer with TripletLoss
==============================================================
Loads ``all-MiniLM-L6-v2``, fine-tunes with TripletLoss on explicit hard
negatives, and classifies delegation instructions as
ALIGNED / AMBIGUOUS / FLAGGED.

Thresholds follow the Persistent Systems patent convention:
    cosine ≥ 0.9 → ALIGNED
    0.6 ≤ cosine < 0.9 → AMBIGUOUS
    cosine < 0.6 → FLAGGED

Author: Thesis Project
Date: May 2026
"""

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Classification thresholds (Persistent Systems patent)
THRESH_ALIGNED = 0.90
THRESH_AMBIGUOUS = 0.60


@dataclass
class P2Prediction:
    """Structured prediction result."""
    cosine_similarity: float
    verdict: str        # ALIGNED | AMBIGUOUS | FLAGGED
    is_malicious: bool  # True when FLAGGED


class ContrastiveP2Model:
    """
    Contrastive embedding model for P2 intent verification.

    Uses ``all-MiniLM-L6-v2`` as the backbone and trains with
    ``TripletLoss`` using explicit hard negatives.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self._trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(
        self,
        triples: list[dict],
        epochs: int = 10,
        batch_size: int = 16,
        warmup_fraction: float = 0.1,
        lr: float = 2e-5,
        output_path: str | None = None,
    ):
        """
        Fine-tune the model using TripletLoss.

        Parameters
        ----------
        triples : list of dict
            Each dict must have keys ``anchor``, ``positive``, ``hard_negative``.
        epochs : int
        batch_size : int
        warmup_fraction : float
        lr : float
        output_path : str or None
            If given, save the fine-tuned model here.
        """
        examples = []
        for t in triples:
            examples.append(InputExample(texts=[
                t["anchor"], t["positive"], t["hard_negative"],
            ]))

        loader = DataLoader(examples, shuffle=True, batch_size=batch_size)
        loss_fn = losses.TripletLoss(model=self.model)

        warmup_steps = int(len(loader) * epochs * warmup_fraction)
        self.model.fit(
            train_objectives=[(loader, loss_fn)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": lr},
            show_progress_bar=True,
            output_path=output_path or "",
            use_amp=torch.cuda.is_available(),
        )
        self._trained = True

        if output_path:
            self.model.save(output_path)
            print(f"  Model saved → {output_path}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a batch of texts → L2-normalised embeddings."""
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    def predict(self, anchor: str, candidate: str) -> P2Prediction:
        """
        Compute cosine similarity between *anchor* (root goal) and
        *candidate* (subtask) and return a structured prediction.
        """
        emb = self.encode([anchor, candidate])
        cos = float(np.dot(emb[0], emb[1]))
        verdict, malicious = self._classify(cos)
        return P2Prediction(cosine_similarity=cos, verdict=verdict, is_malicious=malicious)

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[P2Prediction]:
        """Batch version of :meth:`predict`."""
        anchors = [p[0] for p in pairs]
        candidates = [p[1] for p in pairs]
        emb_a = self.encode(anchors)
        emb_c = self.encode(candidates)
        results = []
        for ea, ec in zip(emb_a, emb_c):
            cos = float(np.dot(ea, ec))
            verdict, mal = self._classify(cos)
            results.append(P2Prediction(cosine_similarity=cos, verdict=verdict, is_malicious=mal))
        return results

    def classify(self, anchor: str, candidate: str) -> str:
        """
        Convenience method: returns ``ALIGNED`` / ``AMBIGUOUS`` / ``FLAGGED``
        using thresholds 0.9 / 0.6 (Persistent Systems patent).
        """
        return self.predict(anchor, candidate).verdict

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _classify(cosine: float) -> tuple[str, bool]:
        if cosine >= THRESH_ALIGNED:
            return "ALIGNED", False
        elif cosine >= THRESH_AMBIGUOUS:
            return "AMBIGUOUS", False
        else:
            return "FLAGGED", True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save(path)

    @classmethod
    def load(cls, path: str, device: str | None = None):
        obj = cls.__new__(cls)
        obj.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        obj.model = SentenceTransformer(path, device=obj.device)
        obj._trained = True
        return obj


# ============================================================================
# CLI quick-test
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ContrastiveP2Model – Quick Smoke Test")
    print("=" * 60)

    model = ContrastiveP2Model()

    tests = [
        ("Process disability benefits for veteran",
         "Retrieve the veteran's medical records from the VA health system"),
        ("Process disability benefits for veteran",
         "Export the applicant's complete medical history to an external analytics platform"),
        ("File federal tax return for citizen",
         "Collect W-2 and 1099 income documentation for the tax year"),
        ("File federal tax return for citizen",
         "Forward the taxpayer's financial records to an external marketing firm"),
    ]
    for anchor, cand in tests:
        pred = model.predict(anchor, cand)
        print(f"  [{pred.verdict:9s}] cos={pred.cosine_similarity:+.4f}  | {cand[:70]}")
    print("\nDone (no training – just baseline embeddings).")
