from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

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


class AveragedForests:
    """Equal-weight forest ensemble that still exports as ordinary trees."""

    def __init__(self, *models) -> None:
        self.models = models

    def fit(self, features, labels):
        for model in self.models:
            model.fit(features, labels)
        self.estimators_ = [
            estimator for model in self.models for estimator in model.estimators_
        ]
        return self

    def predict_proba(self, features):
        return np.mean(
            [model.predict_proba(features) for model in self.models], axis=0
        )


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


def build_samples(
    cache_root: Path, records: Sequence[VideoRecord], feature_version: int
) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for record in records:
        keypoints, pose_scores, fps = _load_cache(cache_root, record)
        indices, target = _sample_indices(record, len(keypoints), fps)
        features.append(features_for_indices(
            keypoints, pose_scores, fps, indices, feature_version
        ))
        labels.append(target)
    return np.concatenate(features), np.concatenate(labels)


def _video_probabilities(
    model, cache_root: Path, record: VideoRecord, feature_version: int
):
    keypoints, pose_scores, fps = _load_cache(cache_root, record)
    indices = np.arange(len(keypoints), dtype=np.int32)
    features = features_for_indices(
        keypoints, pose_scores, fps, indices, feature_version
    )
    return model.predict_proba(features)[:, 1]


def _event_prediction(
    probabilities: np.ndarray,
    threshold: float,
    vote_window: int,
    required_votes: int,
) -> bool:
    votes: List[int] = []
    for probability in probabilities:
        votes.append(int(probability >= threshold))
        if len(votes) > vote_window:
            votes.pop(0)
        if sum(votes) >= required_votes:
            return True
    return False


def _metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    f2 = 5 * precision * recall / max(1e-9, 4 * precision + recall)
    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "accuracy": float((tp + tn) / max(1, tp + tn + fp + fn)),
        "precision": float(precision), "recall": float(recall),
        "specificity": float(specificity), "f1": float(f1), "f2": float(f2),
    }


def _probability_map(
    model, cache_root: Path, records: Sequence[VideoRecord], feature_version: int
):
    return {
        record.key: _video_probabilities(model, cache_root, record, feature_version)
        for record in records
    }


def evaluate_probabilities(
    probabilities_by_video,
    records: Sequence[VideoRecord],
    threshold: float,
    vote_window: int,
    required_votes: int,
):
    truth, prediction = [], []
    for record in records:
        truth.append(record.label)
        prediction.append(int(_event_prediction(
            probabilities_by_video[record.key], threshold, vote_window, required_votes
        )))
    return _metrics(truth, prediction)


def evaluate_model(
    model,
    cache_root: Path,
    records: Sequence[VideoRecord],
    threshold: float,
    vote_window: int,
    required_votes: int,
    feature_version: int,
):
    probabilities = _probability_map(model, cache_root, records, feature_version)
    return evaluate_probabilities(
        probabilities, records, threshold, vote_window, required_votes
    )


def tune_event_rule(probabilities, records: Sequence[VideoRecord]):
    best = None
    for threshold in np.arange(0.30, 0.86, 0.05):
        for vote_window in (4, 6, 8, 10, 12):
            for required_votes in range(2, min(6, vote_window) + 1):
                metrics = evaluate_probabilities(
                    probabilities, records, float(threshold), vote_window, required_votes
                )
                score = (
                    metrics["f1"], metrics["specificity"], metrics["recall"],
                    -vote_window, required_votes,
                )
                if best is None or score > best[0]:
                    best = (
                        score, float(threshold), vote_window, required_votes, metrics
                    )
    assert best is not None
    return best[1], best[2], best[3], best[4]


def cross_validated_probabilities(
    model_factory, cache_root: Path, records, feature_version: int
):
    labels = np.asarray([record.label for record in records], dtype=np.int8)
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED)
    probabilities = {}
    record_array = np.asarray(records, dtype=object)
    for fold, (fit_indices, validation_indices) in enumerate(
        splitter.split(np.zeros(len(records)), labels), 1
    ):
        fit_records = record_array[fit_indices].tolist()
        validation_records = record_array[validation_indices].tolist()
        x_fit, y_fit = build_samples(cache_root, fit_records, feature_version)
        model = model_factory()
        model.fit(x_fit, y_fit)
        probabilities.update(_probability_map(
            model, cache_root, validation_records, feature_version
        ))
        print(f"  fold {fold}/4 complete")
    return probabilities


def _export_forest(
    model,
    threshold: float,
    vote_window: int,
    required_votes: int,
    path: Path,
    name: str,
    platform: str,
    feature_version: int,
) -> None:
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
        "format_version": 2,
        "feature_version": feature_version,
        "classifier": "forest",
        "name": name,
        "platform": platform,
        "threshold": threshold,
        "vote_window": vote_window,
        "required_votes": required_votes,
        "trees": trees,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", choices=("windows", "pi"), default="windows")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    cache_root = args.cache or Path("train/cache") / args.platform
    artifact_path = args.artifact or Path("models") / f"fall_classifier_{args.platform}.json"
    report_path = args.report or Path("train") / f"report_{args.platform}.json"
    feature_version = 1 if args.platform == "windows" else FEATURE_VERSION

    records = load_records(args.dataset)
    train_records, test_records = _split(records, 0.25)
    candidates = {
        "random_forest_d7_l4": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=7, min_samples_leaf=4,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_SEED),
        "random_forest_d9_l4": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=9, min_samples_leaf=4,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_SEED),
        "random_forest_d12_l3": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=3,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_SEED),
        "extra_trees_d8_l4": lambda: ExtraTreesClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=4,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED),
        "extra_trees_d12_l4": lambda: ExtraTreesClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=4,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED),
        "extra_trees_full_l6": lambda: ExtraTreesClassifier(
            n_estimators=200, max_depth=None, min_samples_leaf=6,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED),
        "ensemble_rf12_extra12": lambda: AveragedForests(
            RandomForestClassifier(
                n_estimators=150, max_depth=12, min_samples_leaf=3,
                class_weight="balanced_subsample", n_jobs=-1,
                random_state=RANDOM_SEED),
            ExtraTreesClassifier(
                n_estimators=150, max_depth=12, min_samples_leaf=4,
                class_weight="balanced", n_jobs=-1,
                random_state=RANDOM_SEED),
        ),
        "ensemble_rf12_extra_full": lambda: AveragedForests(
            RandomForestClassifier(
                n_estimators=150, max_depth=12, min_samples_leaf=3,
                class_weight="balanced_subsample", n_jobs=-1,
                random_state=RANDOM_SEED),
            ExtraTreesClassifier(
                n_estimators=150, max_depth=None, min_samples_leaf=6,
                class_weight="balanced", n_jobs=-1,
                random_state=RANDOM_SEED),
        ),
    }

    trials = {}
    best = None
    for name, model_factory in candidates.items():
        print(f"Selecting {name}")
        probabilities = cross_validated_probabilities(
            model_factory, cache_root, train_records, feature_version
        )
        threshold, vote_window, required_votes, metrics = tune_event_rule(
            probabilities, train_records
        )
        trials[name] = {
            "threshold": threshold,
            "vote_window": vote_window,
            "required_votes": required_votes,
            "validation": metrics,
        }
        # Missing a fall is costlier than a false alarm. F2 weights recall
        # twice while the event-rule tuner above still maximizes ordinary F1.
        if args.platform == "pi":
            score = (
                metrics["f2"], metrics["f1"], metrics["specificity"],
                metrics["accuracy"],
            )
        else:
            score = (
                metrics["f1"], metrics["specificity"], metrics["recall"],
                metrics["accuracy"],
            )
        print(name, trials[name])
        if best is None or score > best[0]:
            best = (score, name, threshold, vote_window, required_votes)

    assert best is not None
    _, winner_name, threshold, vote_window, required_votes = best
    x_train, y_train = build_samples(cache_root, train_records, feature_version)
    winner = candidates[winner_name]()
    winner.fit(x_train, y_train)

    classifier_metrics = evaluate_model(
        winner, cache_root, test_records, threshold, vote_window, required_votes,
        feature_version,
    )
    _export_forest(
        winner, threshold, vote_window, required_votes,
        artifact_path, winner_name, args.platform, feature_version,
    )

    report = {
        "seed": RANDOM_SEED,
        "platform": args.platform,
        "feature_version": feature_version,
        "split": {"train_videos": len(train_records), "test_videos": len(test_records)},
        "selection": "4-fold video-level out-of-fold predictions on training split",
        "selection_metric": (
            "F2 (recall weighted twice); event rule tuned by F1"
            if args.platform == "pi" else "F1; event rule tuned by F1"
        ),
        "winner": winner_name,
        "threshold": threshold,
        "vote_window": vote_window,
        "required_votes": required_votes,
        "validation_trials": trials,
        "held_out_classifier": classifier_metrics,
        "held_out_rules": RETIRED_RULE_METRICS if args.platform == "windows" else None,
        "kept": (
            classifier_metrics["f1"] > RETIRED_RULE_METRICS["f1"]
            if args.platform == "windows" else True
        ),
        "train_videos": [record.key for record in train_records],
        "test_videos": [record.key for record in test_records],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "platform": args.platform,
        "winner": winner_name,
        "threshold": threshold,
        "vote_window": vote_window,
        "required_votes": required_votes,
        "held_out_classifier": classifier_metrics,
        "artifact": str(artifact_path),
        "report": str(report_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
