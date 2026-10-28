"""
Machine-learning models that predict how hard a clip is for the ASR system.

Planned contents:
- regression: predict the clip's WER (baseline, linear regression, decision tree, random forest, SVR, small neural net)
- classification: predict whether the clip is "hard" (WER above a threshold)
  (baseline, logistic regression, decision tree, random forest, SVM, small neural net)
- three feature sets: all features, correlation-reduced features, PCA (95% variance)
- group-aware train/test split so clean and noisy copies of one clip never leak across the split
- save outputs/model_results.csv, outputs/test_predictions.csv and the best models (joblib)
Run with: python -m src.train_model
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, mean_absolute_error, mean_squared_error,
                             precision_recall_fscore_support, r2_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import config, features

SEED = config.RANDOM_SEED


def estimators(task):
    if task == "regression":
        return {"baseline_mean": DummyRegressor(),
                "linear_regression": LinearRegression(),
                "decision_tree": DecisionTreeRegressor(max_depth=5, random_state=SEED),
                "random_forest": RandomForestRegressor(n_estimators=200, random_state=SEED),
                "svr": SVR(C=1.0),
                "neural_net": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=3000, random_state=SEED)}
    return {"baseline_majority": DummyClassifier(strategy="most_frequent"),
            "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "decision_tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=SEED),
            "random_forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED),
            "svm": SVC(class_weight="balanced"),
            "neural_net": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=3000, random_state=SEED)}


def make_pipeline(estimator, use_pca):
    steps = [("scale", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=0.95, random_state=SEED)))
    return Pipeline(steps + [("model", estimator)])


def evaluate(task, y_true, y_pred):
    if task == "regression":
        return {"MAE": mean_absolute_error(y_true, y_pred),
                "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "R2": r2_score(y_true, y_pred)}
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "F1": f}


def main():
    config.ensure_dirs()
    df, cols = features.build_feature_table()
    train_idx, test_idx = next(GroupShuffleSplit(test_size=0.25, random_state=SEED)
                               .split(df, groups=df["source_clip_id"]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    kept, _ = features.drop_correlated(train[cols])
    feature_sets = {"all": (cols, False), "reduced": (kept, False), "pca95": (kept, True)}
    print(f"train={len(train)} test={len(test)} | all={len(cols)} features, reduced={len(kept)}")

    results, best = [], {}
    for task, target in [("regression", "wer"), ("classification", "hard")]:
        if task == "classification" and train[target].nunique() < 2:
            print("Skipping classification: only one class in the training data.")
            continue
        for set_name, (fcols, use_pca) in feature_sets.items():
            for name, est in estimators(task).items():
                if name.startswith("baseline") and set_name != "all":
                    continue
                pipe = make_pipeline(est, use_pca).fit(train[fcols], train[target])
                pred = pipe.predict(test[fcols])
                m = evaluate(task, test[target], pred)
                results.append({"task": task, "feature_set": set_name, "model": name,
                                "n_features": len(fcols), **{k: round(float(v), 4) for k, v in m.items()}})
                key = "MAE" if task == "regression" else "F1"
                better = (lambda a, b: a < b) if task == "regression" else (lambda a, b: a > b)
                if not name.startswith("baseline") and (task not in best or better(m[key], best[task][0])):
                    best[task] = (m[key], pipe, fcols, name, set_name, pred)

    res = pd.DataFrame(results)
    res.to_csv(config.OUTPUT_DIR / "model_results.csv", index=False)
    pd.set_option("display.width", 160)
    print(res.to_string(index=False))

    out = test[["clip_id", "asr_model", "condition", "wer", "hard"]].copy()
    for task, (score, pipe, fcols, name, set_name, pred) in best.items():
        joblib.dump({"pipeline": pipe, "features": fcols}, config.MODELS_DIR / f"best_{task}.joblib")
        out["pred_wer" if task == "regression" else "pred_hard"] = pred
        print(f"Best {task}: {name} on '{set_name}' features ({score:.3f})")
    out.to_csv(config.OUTPUT_DIR / "test_predictions.csv", index=False)


if __name__ == "__main__":
    main()
