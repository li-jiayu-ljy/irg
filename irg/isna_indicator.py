import json
import os

import numpy as np
import torch
from xgboost import XGBClassifier


def train_isna_indicator(context: np.ndarray, isna: np.ndarray, model_dir: str, **kwargs):
    os.makedirs(model_dir, exist_ok=True)

    classifier = XGBClassifier(**kwargs)
    classifier.fit(context, isna)
    torch.save(classifier, os.path.join(model_dir, "clf.pt"))
    with open(os.path.join(model_dir, "info.json"), "w") as f:
        json.dump({"true ratio": float(isna.astype(np.float64).mean())}, f)


def predict_isna_indicator(context: np.ndarray, model_dir: str) -> np.ndarray:
    classifier = torch.load(os.path.join(model_dir, "clf.pt"))
    proba = classifier.predict_proba(context)
    if len(proba.shape) == 2:
        if proba.shape[-1] == 1:
            proba = proba[:, 0]
        else:
            proba = proba[:, 1]
    with open(os.path.join(model_dir, "info.json"), "r") as f:
        true_ratio = json.load(f).get("true ratio")
    proba += np.random.normal(0, 1e-3, proba.shape)
    threshold = np.quantile(proba, 1 - true_ratio)
    return proba > threshold
