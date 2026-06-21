from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from detectors.keypoint_features import FEATURE_VERSION, features_for_indices
from train.dataset import VideoRecord, cache_path, load_records


RANDOM_SEED = 42
SAMPLE_HZ = 10.0
POSITIVE_DURATION_SEC = 2.0

# Recorded once on this exact seed-42 held-out split before the rule detector
# was retired. Keeping the numbers preserves the model-selection comparison
# without carrying obsolete production code.
RETIRED_RULE_METRICS = {
    "tp": 16, "fp": 3, "tn": 17, "fn": 4,
    "accuracy": 0.825, "precision": 16 / 19, "recall": 0.8,
    "specificity": 0.85, "f1": 32 / 39,
}


def _split(records: Sequence[VideoRecord], test_size: float) -> tuple[list, list]:
    labels = [record.label for record in records]
    train, test = train_test_split(
        list(records),
        test_size=test_size,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    return sorted(train, key=lambda item: item.key), sorted(test, key=lambda item: item.key)


def _load_cache(cache_root: Path, record: VideoRecord) -> tuple[np.ndarray, np.ndarray, float]:
    path = cache_path(cache_root, record)
    if not path.exists():
        raise FileNotFoundError(f"Missing cache {path}; run extract_keypoints.py first")
    with np.load(path) as data:
        return data["keypoints"], data["pose_scores"], float(data["fps"])


def _sample_indices(record: VideoRecord, frame_count: int, fps: float) -> tuple[np.ndarray, np.ndarray]:
    step = max(1, int(round(fps / SAMPLE_HZ)))
    indices = np.arange(0, frame_count, step, dtype=np.int32)
    times = indices / fps
    if not record.label:
        return indices, np.zeros(len(indices), dtype=np.int8)

    assert record.fall_start is not None and record.fall_end is not None
    positive_end = min(record.fall_end, record.fall_start + POSITIVE_DURATION_SEC)
    negative = times < record.fall_start
    positive = (times >= record.fall_start) & (times <= positive_end)
    keep = negative | positive
    return indices[keep], positive[keep].astype(np.int8)


def build_samples(cache_root: Path, records: Sequence[VideoRecord]) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for record in records:
        keypoints, pose_scores, fps = _load_cache(cache_root, record)
        indices, target = _sample_indices(record, len(keypoints), fps)
        features.append(features_for_indices(keypoints, pose_scores, fps, indices))
        labels.append(target)
    return np.concatenate(features), np.concatenate(labels)


def _video_probabilities(model, cache_root: Path, record: VideoRecord) -> np.ndarray:
    keypoints, pose_scores, fps = _load_cache(cache_root, record)
    indices = np.arange(len(keypoints), dtype=np.int32)
    features = features_for_indices(keypoints, pose_scores, fps, indices)
    return model.predict_proba(features)[:, 1]


def _event_prediction(probabilities: np.ndarray, threshold: float, confirmations: int) -> bool:
    run = 0
    for probability in probabilities:
        run = run + 1 if probability >= threshold else 0
        if run >= confirmations:
            return True
    return False


def _metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "accuracy": float((tp + tn) / max(1, tp + tn + fp + fn)),
        "precision": float(precision), "recall": float(recall),
        "specificity": float(specificity), "f1": float(f1),
    }


def evaluate_model(model, cache_root: Path, records: Sequence[VideoRecord], threshold: float, confirmations: int):
    truth, prediction = [], []
    for record in records:
        probabilities = _video_probabilities(model, cache_root, record)
        truth.append(record.label)
        prediction.append(int(_event_prediction(probabilities, threshold, confirmations)))
    return _metrics(truth, prediction)


def tune_event_rule(model, cache_root: Path, records: Sequence[VideoRecord]) -> tuple[float, int, Dict[str, float]]:
    best = None
    for threshold in np.arange(0.35, 0.81, 0.05):
        for confirmations in (2, 3, 4, 5):
            metrics = evaluate_model(model, cache_root, records, float(threshold), confirmations)
            score = (metrics["f1"], metrics["specificity"], metrics["accuracy"])
            if best is None or score > best[0]:
                best = (score, float(threshold), confirmations, metrics)
    assert best is not None
    return best[1], best[2], best[3]


def _export_forest(model, threshold: float, confirmations: int, path: Path, name: str) -> None:
    trees = []
    for estimator in model.estimators_:
        tree = estimator.tree_
        values = tree.value[:, 0, :]
        totals = np.sum(values, axis=1)
        positive = np.divide(values[:, 1], totals, out=np.zeros(len(totals)), where=totals > 0)
        trees.append({
            "feature": tree.feature.astype(int).tolist(),
            "threshold": tree.threshold.astype(float).tolist(),
            "left": tree.children_left.astype(int).tolist(),
            "right": tree.children_right.astype(int).tolist(),
            "positive_probability": positive.astype(float).tolist(),
        })
    artifact = {
        "format_version": 1,
        "feature_version": FEATURE_VERSION,
        "classifier": "forest",
        "name": name,
        "threshold": threshold,
        "confirmations": confirmations,
        "trees": trees,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--cache", type=Path, default=Path("train/cache"))
    parser.add_argument("--artifact", type=Path, default=Path("models/fall_classifier.json"))
    parser.add_argument("--report", type=Path, default=Path("train/report.json"))
    args = parser.parse_args()

    records = load_records(args.dataset)
    train_records, test_records = _split(records, 0.25)
    fit_records, validation_records = _split(train_records, 0.25)
    x_fit, y_fit = build_samples(args.cache, fit_records)

    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=160, max_depth=9, min_samples_leaf=4,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_SEED,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=160, max_depth=10, min_samples_leaf=4,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED,
        ),
    }

    trials = {}
    best = None
    for name, model in candidates.items():
        model.fit(x_fit, y_fit)
        threshold, confirmations, metrics = tune_event_rule(
            model, args.cache, validation_records
        )
        trials[name] = {
            "threshold": threshold,
            "confirmations": confirmations,
            "validation": metrics,
        }
        score = (metrics["f1"], metrics["specificity"], metrics["accuracy"])
        print(name, trials[name])
        if best is None or score > best[0]:
            best = (score, name, threshold, confirmations)

    assert best is not None
    _, winner_name, threshold, confirmations = best
    x_train, y_train = build_samples(args.cache, train_records)
    winner = candidates[winner_name]
    winner.fit(x_train, y_train)

    classifier_metrics = evaluate_model(
        winner, args.cache, test_records, threshold, confirmations
    )
    _export_forest(winner, threshold, confirmations, args.artifact, winner_name)

    report = {
        "seed": RANDOM_SEED,
        "split": {"train_videos": len(train_records), "test_videos": len(test_records)},
        "winner": winner_name,
        "threshold": threshold,
        "confirmations": confirmations,
        "validation_trials": trials,
        "held_out_classifier": classifier_metrics,
        "held_out_rules": RETIRED_RULE_METRICS,
        "kept": classifier_metrics["f1"] > RETIRED_RULE_METRICS["f1"],
        "train_videos": [record.key for record in train_records],
        "test_videos": [record.key for record in test_records],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
