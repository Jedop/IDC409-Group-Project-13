"""
Final evaluation: collect every metric into one report and draw the result plots.

Planned contents:
- ASR metric: WER per model and condition (from wer_summary.csv)
- text task metrics: keyword precision/recall/F1 and transcript-matching accuracy
- ML metrics: regression (MAE, RMSE, R2) and classification (accuracy, precision, recall, F1)
- plots: predicted vs actual WER, confusion matrix, model comparison
- save outputs/final_report.md
Run with: python -m src.evaluate
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from . import config


def df_to_md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def load(name):
    p = config.OUTPUT_DIR / name
    return pd.read_csv(p) if p.exists() else None


def main():
    config.ensure_dirs()
    report = ["# Final evaluation report", ""]

    wer = load("wer_summary.csv")
    if wer is not None:
        report += ["## 1. ASR - word error rate", df_to_md(wer[["asr_model", "condition", "n_clips", "mean_wer", "corpus_wer"]]), ""]
    kw = load("keyword_metrics.csv")
    if kw is not None:
        report += ["## 2a. Keyword detection", df_to_md(kw[["asr_model", "condition", "precision", "recall", "f1"]]), ""]
    mt = load("matching_summary.csv")
    if mt is not None:
        report += ["## 2b. Matching to the expected transcript", df_to_md(mt), ""]
    res = load("model_results.csv")
    if res is not None:
        for task, key, asc in [("regression", "MAE", True), ("classification", "F1", False)]:
            sub = res[res["task"] == task].sort_values(key, ascending=asc).drop(columns="task").dropna(axis=1, how="all")
            report += [f"## 3. ML - {task} (sorted by {key})", df_to_md(sub), ""]
            top = sub[~sub["model"].str.startswith("baseline")].head(8)
            plt.figure(figsize=(7, 4))
            labels = top["model"] + " (" + top["feature_set"] + ")"
            plt.barh(labels[::-1], top[key].values[::-1], color="#4C72B0")
            plt.xlabel(key); plt.title(f"Best {task} models")
            plt.tight_layout(); plt.savefig(config.PLOTS_DIR / f"eval_{task}_models.png", dpi=130); plt.close()

    pred = load("test_predictions.csv")
    if pred is not None and "pred_wer" in pred:
        plt.figure(figsize=(5, 5))
        plt.scatter(pred["wer"], pred["pred_wer"], alpha=0.6)
        lim = max(pred["wer"].max(), pred["pred_wer"].max()) * 1.05
        plt.plot([0, lim], [0, lim], "k--", lw=1)
        plt.xlabel("Actual WER"); plt.ylabel("Predicted WER"); plt.title("Predicted vs actual WER (test set)")
        plt.tight_layout(); plt.savefig(config.PLOTS_DIR / "eval_pred_vs_actual.png", dpi=130); plt.close()
    if pred is not None and "pred_hard" in pred:
        ConfusionMatrixDisplay.from_predictions(pred["hard"], pred["pred_hard"], cmap="Blues")
        plt.title("Hard-clip classifier (test set)")
        plt.tight_layout(); plt.savefig(config.PLOTS_DIR / "eval_confusion_matrix.png", dpi=130); plt.close()
        report += ["## 4. Best classifier - detailed report", "```",
                   classification_report(pred["hard"], pred["pred_hard"], zero_division=0), "```", ""]

    text = "\n".join(report)
    (config.OUTPUT_DIR / "final_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
