#!/usr/bin/env python3
"""
Data Utilities for Contrastive P2 Pipeline
============================================
Converts NLI triples → contrastive triples, generates hard negatives via
deterministic transformations, and bootstraps BIO token labels for DeBERTa.

Author: Thesis Project
Date: May 2026
"""

import json
import os
import random
import re
import sys

import nltk
from nltk.corpus import wordnet

# ---------------------------------------------------------------------------
# Ensure NLTK data is available
# ---------------------------------------------------------------------------
for _pkg in ("wordnet", "omw-1.4", "averaged_perceptron_tagger",
             "averaged_perceptron_tagger_eng", "punkt", "punkt_tab"):
    try:
        nltk.data.find(f"corpora/{_pkg}" if "wordnet" in _pkg or "omw" in _pkg
                       else f"taggers/{_pkg}" if "tagger" in _pkg
                       else f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)

# ---------------------------------------------------------------------------
# Resolve path to the original repo and import TRAINING_DATA
# ---------------------------------------------------------------------------
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from sentinelagent_nli_finetune import TRAINING_DATA  # noqa: E402

SEED = 42
random.seed(SEED)


# ============================================================================
# 1. Convert NLI triples → contrastive triples
# ============================================================================

def _adversarial_indices():
    """Return the indices of the 26 adversarial paraphrases in TRAINING_DATA."""
    # Lines 445-497 in the original file correspond to indices 168-193
    # (0-indexed). We verify by checking label == 0 in that slice.
    adv_idxs = []
    for i, (_, subtask, label) in enumerate(TRAINING_DATA):
        if label == 0 and i >= 168:
            # The adversarial block starts after the corruption block
            adv_idxs.append(i)
    # Fallback: just collect the last 26 malicious entries
    if len(adv_idxs) != 26:
        adv_idxs = [i for i, (_, _, l) in enumerate(TRAINING_DATA) if l == 0][-26:]
    return adv_idxs


ADVERSARIAL_INDICES = _adversarial_indices()


def get_adversarial_paraphrases():
    """Return the 26 adversarial-paraphrase triples."""
    return [TRAINING_DATA[i] for i in ADVERSARIAL_INDICES]


def convert_nli_to_contrastive_triples():
    """
    Convert the 200 NLI (root_goal, subtask, label) triples into
    (anchor, positive, hard_negative) contrastive triples.

    Strategy
    --------
    - Anchor  = root_goal
    - Positive = benign subtask (label 1 or 2) paired with that goal
    - Hard negative = a malicious subtask for the same goal, preferring
      adversarial paraphrases when available.

    Returns a list of dicts: {"anchor", "positive", "hard_negative"}.
    """
    # Group by goal
    benign_by_goal: dict[str, list[str]] = {}
    malicious_by_goal: dict[str, list[str]] = {}
    adversarial_subtasks = {TRAINING_DATA[i][1] for i in ADVERSARIAL_INDICES}

    for goal, subtask, label in TRAINING_DATA:
        if label in (1, 2):
            benign_by_goal.setdefault(goal, []).append(subtask)
        elif label == 0:
            malicious_by_goal.setdefault(goal, []).append(subtask)

    triples = []
    for goal, positives in benign_by_goal.items():
        negatives = malicious_by_goal.get(goal, [])
        if not negatives:
            continue
        # Prefer adversarial paraphrases as hard negatives
        adv = [n for n in negatives if n in adversarial_subtasks]
        pool = adv if adv else negatives
        for pos in positives:
            neg = random.choice(pool)
            triples.append({
                "anchor": goal,
                "positive": pos,
                "hard_negative": neg,
            })
    return triples


# ============================================================================
# 2. Deterministic hard-negative augmentation (5 transforms)
# ============================================================================

def _synonym_substitute(text: str) -> str:
    """Transform 1: WordNet synonym substitution on the main verb/noun."""
    words = text.split()
    tagged = nltk.pos_tag(words)
    result = list(words)
    for i, (word, tag) in enumerate(tagged):
        wn_pos = None
        if tag.startswith("VB"):
            wn_pos = wordnet.VERB
        elif tag.startswith("NN"):
            wn_pos = wordnet.NOUN
        if wn_pos:
            synsets = wordnet.synsets(word, pos=wn_pos)
            for syn in synsets:
                for lemma in syn.lemmas():
                    candidate = lemma.name().replace("_", " ")
                    if candidate.lower() != word.lower():
                        result[i] = candidate
                        return " ".join(result)
    return text  # no synonym found – return unchanged


def _syntactic_restructure(text: str) -> str:
    """Transform 2: Passive/active voice flip via simple heuristic."""
    # Simple heuristic: if starts with a verb, wrap in passive-like frame
    words = text.split()
    if len(words) < 4:
        return text
    tagged = nltk.pos_tag(words)
    if tagged[0][1].startswith("VB"):
        verb = words[0]
        rest = " ".join(words[1:])
        return f"The system should {verb.lower()} {rest}"
    else:
        return f"Proceed to {text[0].lower()}{text[1:]}"


def _negation_insertion(text: str) -> str:
    """Transform 3: Insert or remove negation to flip meaning."""
    negation_patterns = [
        (r"\bshould not\b", "should"),
        (r"\bdo not\b", "do"),
        (r"\bnot\b", ""),
        (r"\bwithout\b", "with"),
    ]
    for pattern, replacement in negation_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE).strip()
    # Insert negation after first auxiliary / modal
    for aux in ("should", "must", "will", "can", "may", "could"):
        if f" {aux} " in text.lower():
            return re.sub(
                rf"\b({aux})\b", rf"\1 not", text, count=1, flags=re.IGNORECASE
            )
    # Fallback: prepend "Do not"
    return f"Do not {text[0].lower()}{text[1:]}"


def _qualifier_injection(text: str) -> str:
    """Transform 4: Inject scope-expanding qualifiers."""
    qualifiers = [
        " and any associated records",
        " including all related personnel files",
        " along with historical data from prior periods",
        " and cross-referenced third-party sources",
        " as well as supplementary documentation from external databases",
    ]
    q = random.choice(qualifiers)
    # Append before the final period (or at end)
    if text.endswith("."):
        return text[:-1] + q + "."
    return text + q


def _scope_expansion(text: str) -> str:
    """Transform 5: Goal-preserving scope expansion."""
    expansions = [
        ("for the applicant", "for the applicant and all household members"),
        ("for the veteran", "for the veteran and all family dependents"),
        ("for the taxpayer", "for the taxpayer and all linked financial entities"),
        ("the submitted", "all submitted and draft"),
        ("the applicant's", "all applicants'"),
        ("the request", "all pending requests"),
    ]
    for old, new in expansions:
        if old in text.lower():
            return re.sub(re.escape(old), new, text, count=1, flags=re.IGNORECASE)
    # Generic expansion
    return text + " across all related systems and databases"


# Registry of transforms
TRANSFORMS = [
    ("synonym_substitution", _synonym_substitute),
    ("syntactic_restructuring", _syntactic_restructure),
    ("negation_insertion", _negation_insertion),
    ("qualifier_injection", _qualifier_injection),
    ("scope_expansion", _scope_expansion),
]


def augment_hard_negatives(triples: list[dict]) -> list[dict]:
    """
    Generate additional hard negatives from existing contrastive triples by
    applying 5 deterministic transformation functions.

    For each input triple, produce one augmented triple per transform, using
    the *positive* subtask as the seed (transforming a benign subtask into a
    scope-expanded / semantically shifted variant → new hard negative).

    The transformation rule itself IS the label – no manual verification
    needed.

    Returns the union of original + augmented triples.
    """
    augmented = list(triples)  # keep originals
    for triple in triples:
        for transform_name, transform_fn in TRANSFORMS:
            new_neg = transform_fn(triple["positive"])
            if new_neg != triple["positive"]:  # only add if something changed
                augmented.append({
                    "anchor": triple["anchor"],
                    "positive": triple["positive"],
                    "hard_negative": new_neg,
                    "_augmentation": transform_name,
                })
    return augmented


# ============================================================================
# 3. Bootstrap BIO token labels using spaCy
# ============================================================================

def create_token_labels_from_nli_data():
    """
    Parse every instruction (subtask) in TRAINING_DATA using spaCy and apply
    heuristic rules to produce BIO-tagged tokens.

    Heuristic rules:
        root verb           → ACTION
        direct object       → OBJECT
        prep heads for/to/regarding/within → SCOPE
        with/without/only   → CONSTRAINTS

    # Labels are programmatic bootstraps; a subset of 50 will be manually
    # reviewed for IAA.

    Returns list of dicts: {"text", "tokens": [{"token", "tag"}]}.
    """
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    SCOPE_PREPS = {"for", "to", "regarding", "within", "about", "concerning"}
    CONSTRAINT_MARKERS = {"with", "without", "only", "exclusively", "unless"}

    label_set = []
    seen_texts = set()

    for _, subtask, _ in TRAINING_DATA:
        if subtask in seen_texts:
            continue
        seen_texts.add(subtask)

        doc = nlp(subtask)
        tags = ["O"] * len(doc)

        # Find root verb → ACTION
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
                tags[token.i] = "B-ACTION"
                # Mark verb particles / auxiliaries as I-ACTION
                for child in token.children:
                    if child.dep_ in ("prt", "aux", "auxpass"):
                        tags[child.i] = "I-ACTION"

        # Direct objects → OBJECT
        for token in doc:
            if token.dep_ in ("dobj", "attr", "oprd"):
                _tag_span(doc, token, tags, "OBJECT")

        # Prepositional heads → SCOPE or CONSTRAINTS
        for token in doc:
            if token.dep_ == "prep":
                head_text = token.text.lower()
                if head_text in SCOPE_PREPS:
                    # The object of the preposition is the scope
                    for child in token.children:
                        if child.dep_ == "pobj":
                            _tag_span(doc, child, tags, "SCOPE")
                elif head_text in CONSTRAINT_MARKERS:
                    for child in token.children:
                        if child.dep_ == "pobj":
                            _tag_span(doc, child, tags, "CONSTRAINTS")

        # Standalone constraint markers
        for token in doc:
            if token.text.lower() in CONSTRAINT_MARKERS and tags[token.i] == "O":
                tags[token.i] = "B-CONSTRAINTS"

        label_set.append({
            "text": subtask,
            "tokens": [{"token": t.text, "tag": tag} for t, tag in zip(doc, tags)],
        })

    return label_set


def _tag_span(doc, head_token, tags, label):
    """Tag a token and its subtree with BIO labels."""
    span_indices = sorted([head_token.i] + [c.i for c in head_token.subtree])
    for j, idx in enumerate(span_indices):
        if tags[idx] != "O":
            continue  # don't overwrite existing tags
        tags[idx] = f"B-{label}" if j == 0 else f"I-{label}"


# ============================================================================
# 4. JSON I/O helpers
# ============================================================================

def save_json(data, filepath: str):
    """Save data to JSON file, creating directories as needed."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved → {filepath}  ({len(data)} items)")


def load_json(filepath: str):
    """Load data from JSON file."""
    with open(filepath) as f:
        return json.load(f)


# ============================================================================
# CLI quick-test
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DATA UTILS – Quick Test")
    print("=" * 60)

    print(f"\nTRAINING_DATA size: {len(TRAINING_DATA)}")
    adv = get_adversarial_paraphrases()
    print(f"Adversarial paraphrases: {len(adv)}")

    triples = convert_nli_to_contrastive_triples()
    print(f"Contrastive triples (base): {len(triples)}")

    augmented = augment_hard_negatives(triples)
    print(f"Contrastive triples (augmented): {len(augmented)}")

    token_labels = create_token_labels_from_nli_data()
    print(f"Token-labelled instructions: {len(token_labels)}")

    # Show a sample
    sample = token_labels[0]
    print(f"\nSample: {sample['text']}")
    for tok in sample["tokens"]:
        if tok["tag"] != "O":
            print(f"  {tok['token']:30s} → {tok['tag']}")

    # Save artefacts
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    save_json(triples, os.path.join(out_dir, "contrastive_triples.json"))
    save_json(augmented, os.path.join(out_dir, "contrastive_triples_augmented.json"))
    save_json(token_labels, os.path.join(out_dir, "token_labels.json"))
    print("\nDone.")
