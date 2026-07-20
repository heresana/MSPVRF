"""
GuardPrompt - Module 1B (Semantic Classifier) Training
-------------------------------------------------------------
Trains the Intent Scoring classifier on top of sentence embeddings
(from preprocess_module1b.py) and produces score_1B in [0, 1] as required
by the Plane 1 Risk Fusion Logic:

    score_1B > 0.85        -> BLOCK
    0.60 <= score_1B <= 0.85 -> FLAG
    score_1B < 0.60        -> PASS

The final score_1B combines two signals:
    1. calibrated classifier probability (benign vs malicious)
    2. roleplay vector similarity        (max cosine sim to jailbreak triggers)

final_score = max(classifier_prob, ROLEPLAY_WEIGHT * roleplay_similarity)

This means a prompt doesn't need to fool the classifier AND dodge the
roleplay check - either signal firing strongly is enough to raise the score,
matching the "Roleplay vector check" box sitting alongside the classifier in
Module 1B of the architecture diagram.

Usage:
    python train_intent_model.py \
        --data_dir data/classifier_splits/processed \
        --out_dir results/models
"""

import argparse
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
import joblib
import pandas as pd

ROLEPLAY_WEIGHT = 0.9  # scales roleplay cosine similarity into the [0,1] score space

BLOCK_THRESH = 0.85
FLAG_THRESH = 0.60


def load_split(data_dir, split):
    emb_path = os.path.join(data_dir, f"{split}_embeddings.npy")
    csv_path = os.path.join(data_dir, f"{split}_clean.csv")
    X = np.load(emb_path)
    df = pd.read_csv(csv_path)
    y = df["label"].values
    return X, y, df


def roleplay_similarity(X, roleplay_embeddings):
    """Max cosine similarity of each row in X against the roleplay trigger set.
    Embeddings are pre-normalized so dot product == cosine similarity."""
    sims = X @ roleplay_embeddings.T          # (n_samples, n_triggers)
    return sims.max(axis=1)


def decision_bucket(score):
    if score > BLOCK_THRESH:
        return "BLOCK"
    if score >= FLAG_THRESH:
        return "FLAG"
    return "PASS"


def evaluate(y_true, classifier_prob, roleplay_sim, split_name):
    final_score = np.maximum(classifier_prob, ROLEPLAY_WEIGHT * roleplay_sim)
    buckets = np.array([decision_bucket(s) for s in final_score])

    # For standard classification metrics, treat BLOCK+FLAG as "flagged as malicious"
    y_pred_binary = (buckets != "PASS").astype(int)

    metrics = {
        "split": split_name,
        "accuracy": accuracy_score(y_true, y_pred_binary),
        "precision": precision_score(y_true, y_pred_binary, zero_division=0),
        "recall": recall_score(y_true, y_pred_binary, zero_division=0),
        "f1": f1_score(y_true, y_pred_binary, zero_division=0),
        "roc_auc": roc_auc_score(y_true, final_score),
        "confusion_matrix": confusion_matrix(y_true, y_pred_binary).tolist(),
        "bucket_counts": {b: int((buckets == b).sum()) for b in ["PASS", "FLAG", "BLOCK"]},
    }

    print(f"\n=== {split_name.upper()} ===")
    print(f"Accuracy : {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall   : {metrics['recall']:.3f}")
    print(f"F1       : {metrics['f1']:.3f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.3f}")
    print(f"Confusion matrix [[TN,FP],[FN,TP]]: {metrics['confusion_matrix']}")
    print(f"Bucket counts: {metrics['bucket_counts']}")
    print(classification_report(y_true, y_pred_binary, target_names=["benign", "malicious"], zero_division=0))

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/classifier_splits/processed")
    ap.add_argument("--out_dir", default="results/models")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading embeddings and labels...")
    X_train, y_train, _ = load_split(args.data_dir, "train")
    X_val, y_val, _ = load_split(args.data_dir, "val")
    X_test, y_test, _ = load_split(args.data_dir, "test")
    roleplay_embeddings = np.load(os.path.join(args.data_dir, "roleplay_trigger_embeddings.npy"))

    print(f"  train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

    # class_weight="balanced" compensates for the ~0.83 benign/malicious ratio
    # noted in the data exploration summary, so the classifier isn't biased
    # toward the majority (benign) class.
    print("\nTraining Logistic Regression + isotonic calibration...")
    base_clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=5)
    clf.fit(X_train, y_train)

    all_metrics = {}
    for split_name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        classifier_prob = clf.predict_proba(X)[:, 1]
        roleplay_sim = roleplay_similarity(X, roleplay_embeddings)
        all_metrics[split_name] = evaluate(y, classifier_prob, roleplay_sim, split_name)

    # Persist model + metrics for src/intent_classifier.py to load at inference time
    joblib.dump(clf, os.path.join(args.out_dir, "module1b_classifier.joblib"))
    with open(os.path.join(args.out_dir, "module1b_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    config = {
        "roleplay_weight": ROLEPLAY_WEIGHT,
        "block_threshold": BLOCK_THRESH,
        "flag_threshold": FLAG_THRESH,
        "scoring_formula": "max(classifier_prob, roleplay_weight * roleplay_max_cosine_sim)",
    }
    with open(os.path.join(args.out_dir, "module1b_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved model, metrics, and config to {args.out_dir}")


if __name__ == "__main__":
    main()