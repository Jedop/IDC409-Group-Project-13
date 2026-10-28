"""
Exploratory data analysis (EDA).

Planned contents:
- summary statistics: clips per condition, duration, word counts, transcript length, missing values
- plots: duration histogram, word-count histogram, transcript length vs duration,
  WER per condition, confidence vs WER
- error examples: the worst clips with true vs predicted text
- outputs go to outputs/eda_summary.txt, outputs/error_examples.csv and outputs/plots/
Run with: python -m src.eda
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, database, metrics
from .text_utils import tokenize, normalize_text, condition_sort_key


def savefig(name):
    plt.tight_layout()
    plt.savefig(config.PLOTS_DIR / name, dpi=130)
    plt.close()


def describe_clips(clips):
    clean = clips[clips["condition"] == "clean"].copy()
    clean["n_words"] = clean["true_text"].map(lambda t: len(tokenize(t)))
    clean["n_chars"] = clean["true_text"].map(lambda t: len(normalize_text(t)))
    clean["words_per_sec"] = clean["n_words"] / clean["duration_s"]
    return clean


def main():
    config.ensure_dirs()
    clips = database.read_clips()
    if clips.empty:
        raise SystemExit("No clips found - run python -m src.data_prep first.")
    clean = describe_clips(clips)
    lines = ["=== EDA SUMMARY ===", "", "Clips per condition:",
             clips.groupby("condition").size().to_string(), "",
             "Clean-clip statistics:",
             clean[["duration_s", "n_words", "n_chars", "words_per_sec"]].describe().round(2).to_string(), "",
             "Missing values in clips table:", clips.isna().sum().to_string()]

    plt.figure(figsize=(6, 4)); plt.hist(clean["duration_s"], bins=15, color="#4C72B0", edgecolor="white")
    plt.xlabel("Clip duration (s)"); plt.ylabel("Number of clips"); plt.title("Clip duration")
    savefig("eda_duration_hist.png")

    plt.figure(figsize=(6, 4)); plt.hist(clean["n_words"], bins=15, color="#55A868", edgecolor="white")
    plt.xlabel("Words in true transcript"); plt.ylabel("Number of clips"); plt.title("Word count")
    savefig("eda_wordcount_hist.png")

    plt.figure(figsize=(6, 4)); plt.scatter(clean["duration_s"], clean["n_chars"], alpha=0.7)
    plt.xlabel("Duration (s)"); plt.ylabel("Transcript length (characters)")
    plt.title("Transcript length vs duration"); savefig("eda_length_vs_duration.png")

    joined = database.read_joined()
    if not joined.empty:
        df = metrics.add_wer(joined)
        df["confidence"] = df["confidence"].fillna(0)
        conds = sorted(df["condition"].unique(), key=condition_sort_key)
        models = sorted(df["asr_model"].unique())

        fig, ax = plt.subplots(figsize=(7, 4))
        width = 0.8 / len(models)
        for i, m in enumerate(models):
            data = [df[(df.asr_model == m) & (df.condition == c)]["wer"].dropna() for c in conds]
            pos = np.arange(len(conds)) + i * width
            ax.boxplot(data, positions=pos, widths=width * 0.9)
            ax.plot([], [], label=m)
        ax.set_xticks(np.arange(len(conds)) + width * (len(models) - 1) / 2); ax.set_xticklabels(conds)
        ax.set_ylabel("Per-clip WER"); ax.set_title("WER by condition and ASR model"); ax.legend()
        savefig("eda_wer_by_condition.png")

        plt.figure(figsize=(6, 4))
        plt.scatter(df["confidence"], df["wer"], alpha=0.5)
        plt.xlabel("ASR confidence"); plt.ylabel("WER"); plt.title("Confidence vs WER")
        savefig("eda_confidence_vs_wer.png")

        worst = df.sort_values("wer", ascending=False).head(15)
        worst[["clip_id", "asr_model", "condition", "wer", "confidence", "true_text", "predicted_text"]] \
            .to_csv(config.OUTPUT_DIR / "error_examples.csv", index=False)
        lines += ["", "Mean WER per model and condition:",
                  df.pivot_table(index="condition", columns="asr_model", values="wer", aggfunc="mean")
                    .loc[conds].round(3).to_string(),
                  "", "Correlation of confidence with WER: "
                  f"{df['confidence'].corr(df['wer']):.3f}",
                  "", "Top 3 error examples:"]
        for _, r in worst.head(3).iterrows():
            lines += [f"[{r.asr_model} | {r.condition} | WER {r.wer:.2f}]",
                      f"  true: {r.true_text}", f"  pred: {r.predicted_text}"]

    text = "\n".join(lines)
    (config.OUTPUT_DIR / "eda_summary.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
